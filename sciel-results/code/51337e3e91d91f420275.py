import time
import random
import numpy as np

BASE = 51975
DEADLINE = time.time() + 145  # 5s safety margin

def _get_divisors():
    """Get all odd divisors > 1 of BASE, sorted ascending."""
    divs = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divs.append(d)
    return divs

def _build_coverage(moduli_residues):
    """Build a coverage array from a list of [m, r] pairs."""
    covered = bytearray(BASE)
    for m, r in moduli_residues:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return covered

def _score(covered):
    """Number of uncovered residues."""
    return covered.count(0)

def _greedy_construct(divs, randomize=False, seed=None):
    """Greedy construction with optional randomization."""
    if seed is not None:
        random.seed(seed)
    
    covered = bytearray(BASE)
    classes = []
    used_moduli = set()
    
    # Sort divisors: prioritize small moduli, with some randomization
    if randomize:
        # Weight by 1/m (smaller moduli get higher weight)
        weights = [1.0 / m for m in divs]
        total_w = sum(weights)
        remaining = list(divs)
        rem_weights = list(weights)
    else:
        remaining = sorted(divs)
    
    while remaining:
        if time.time() > DEADLINE:
            break
        
        if randomize:
            # Adaptive: try top-k candidates, pick best marginal gain
            k = min(10, len(remaining))
            # Sample k candidates weighted by 1/m
            candidates = []
            for _ in range(k):
                r = random.random() * sum(rem_weights)
                cum = 0
                for i, w in enumerate(rem_weights):
                    cum += w
                    if r <= cum:
                        candidates.append(i)
                        break
            # Deduplicate
            candidates = list(set(candidates))
            if not candidates:
                candidates = [0]
        else:
            candidates = list(range(min(50, len(remaining))))
        
        best_gain = -1
        best_idx = -1
        best_res = -1
        
        for idx in candidates:
            m = remaining[idx]
            # Find best residue for this modulus
            best_r = -1
            best_r_gain = -1
            for r in range(m):
                # Count newly covered
                gain = sum(1 for i in range(r, BASE, m) if covered[i] == 0)
                if gain > best_r_gain:
                    best_r_gain = gain
                    best_r = r
            if best_r_gain > best_gain:
                best_gain = best_r_gain
                best_idx = idx
                best_res = best_r
        
        if best_gain <= 0:
            break
        
        m = remaining.pop(best_idx)
        rem_weights.pop(best_idx)
        used_moduli.add(m)
        classes.append([m, best_res])
        covered[best_res::m] = b'\x01' * len(range(best_res, BASE, m))
    
    return classes

def _local_search(classes, divs, covered):
    """Try to improve by swapping residues or adding/removing classes."""
    if time.time() > DEADLINE:
        return classes, covered
    
    used_moduli = set(m for m, r in classes)
    available = [d for d in divs if d not in used_moduli]
    if not available:
        return classes, covered
    
    best_classes = classes
    best_covered = covered
    best_score = _score(covered)
    
    # Try adding a new class
    for m in available[:30]:  # Limit to smaller moduli
        if time.time() > DEADLINE:
            break
        best_r = -1
        best_gain = -1
        for r in range(m):
            gain = sum(1 for i in range(r, BASE, m) if covered[i] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            new_classes = classes + [[m, best_r]]
            new_covered = bytearray(covered)
            new_covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
            new_score = _score(new_covered)
            if new_score < best_score:
                best_score = new_score
                best_classes = new_classes
                best_covered = new_covered
                used_moduli.add(m)
                available = [d for d in available if d not in used_moduli]
    
    # Try swapping residues for existing classes
    for i, (m, r) in enumerate(best_classes):
        if time.time() > DEADLINE:
            break
        best_r_new = r
        best_local_score = best_score
        for r_new in range(m):
            if r_new == r:
                continue
            # Temporarily remove old, add new
            new_covered = bytearray(best_covered)
            # Clear old
            for pos in range(r, BASE, m):
                new_covered[pos] = 0
            # Set new
            for pos in range(r_new, BASE, m):
                new_covered[pos] = 1
            new_score = _score(new_covered)
            if new_score < best_local_score:
                best_local_score = new_score
                best_r_new = r_new
        
        if best_r_new != r:
            best_classes[i] = [m, best_r_new]
            best_covered = bytearray(best_covered)
            for pos in range(r, BASE, m):
                best_covered[pos] = 0
            for pos in range(best_r_new, BASE, m):
                best_covered[pos] = 1
            best_score = best_local_score
    
    return best_classes, best_covered

def construct():
    global DEADLINE
    DEADLINE = time.time() + 145
    
    divs = _get_divisors()
    
    # Start with deterministic greedy
    best_classes = _greedy_construct(divs, randomize=False)
    best_covered = _build_coverage(best_classes)
    best_score = _score(best_covered)
    
    # Local search
    best_classes, best_covered = _local_search(best_classes, divs, best_covered)
    best_score = _score(best_covered)
    
    # Randomized restarts
    seed = 42
    while time.time() < DEADLINE:
        seed += 1
        classes = _greedy_construct(divs, randomize=True, seed=seed)
        covered = _build_coverage(classes)
        score = _score(covered)
        
        if score < best_score:
            best_score = score
            best_classes = classes
            best_covered = covered
        
        # Local search on this
        classes_ls, covered_ls = _local_search(classes, divs, covered)
        score_ls = _score(covered_ls)
        if score_ls < best_score:
            best_score = score_ls
            best_classes = classes_ls
            best_covered = covered_ls
    
    return best_classes
