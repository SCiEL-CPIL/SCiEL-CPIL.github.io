"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Divisors:
# 3^a * 5^b * 7^c * 11^d
# a in 0..3, b in 0..2, c in 0..1, d in 0..1

def _get_divisors():
    divs = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    val = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if val > 1:
                        divs.append(val)
    return sorted(divs)

DIVISORS = _get_divisors()
N_DIVS = len(DIVISORS)

def _score(classes):
    """Count uncovered residues."""
    covered = bytearray(BASE)
    for m, r in classes:
        if r < m:
            covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return covered.count(0)

def _compute_covered(classes):
    """Compute covered bytearray for a list of classes."""
    covered = bytearray(BASE)
    for m, r in classes:
        if r < m:
            covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return covered

def _greedy_order(moduli_order, covered=None):
    """Greedy: pick residue for each modulus in given order to maximize new coverage."""
    if covered is None:
        covered = bytearray(BASE)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_gain = -1
        # For small m, try all residues
        for r in range(m):
            # Count how many uncovered residues we'd cover
            gain = covered[r::m].count(0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        if best_gain > 0:
            covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
    return classes, _score(classes)

def _local_search(classes, max_iter=10000, deadline=None, temperature=0.0):
    """Local search with optional simulated annealing.
    
    Tries single-residue changes and pairwise residue changes.
    If temperature > 0, uses simulated annealing to accept worse moves.
    """
    score = _score(classes)
    best_score = score
    best_classes = [c[:] for c in classes]
    covered = _compute_covered(classes)
    
    # Precompute for each modulus the list of residues
    m_residues = {m: list(range(m)) for m in DIVISORS}
    
    # Index moduli for fast access
    m_to_idx = {c[0]: i for i, c in enumerate(classes)}
    
    it = 0
    t = temperature
    cooling = 0.999 if temperature > 0 else 1.0
    
    while it < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random class to modify
        idx = random.randint(0, len(classes) - 1)
        m, old_r = classes[idx]
        
        # Decide whether to do a single move or a pair move
        if len(classes) >= 2 and random.random() < 0.3:
            # Pairwise move: change two residues simultaneously
            idx2 = random.randint(0, len(classes) - 1)
            if idx2 == idx:
                continue
            m2, old_r2 = classes[idx2]
            new_r = random.randint(0, m - 1)
            new_r2 = random.randint(0, m2 - 1)
            if new_r == old_r and new_r2 == old_r2:
                continue
            
            # Compute new covered state
            new_covered = bytearray(covered)
            # Remove old contributions (need to recompute from scratch for correctness)
            # Actually, let's just recompute covered for the pair change
            new_classes = [c[:] for c in classes]
            new_classes[idx][1] = new_r
            new_classes[idx2][1] = new_r2
            new_covered = _compute_covered(new_classes)
            new_score = new_covered.count(0)
            
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in new_classes]
                classes = new_classes
                covered = new_covered
            elif new_score < score or (temperature > 0 and random.random() < np.exp(-(new_score - score) / max(t, 1e-10))):
                classes = new_classes
                covered = new_covered
                score = new_score
        else:
            # Single move
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            # Compute new covered state
            new_covered = bytearray(covered)
            # This is tricky: we need to remove old contribution and add new
            # For simplicity, recompute from scratch
            new_classes = [c[:] for c in classes]
            new_classes[idx][1] = new_r
            new_covered = _compute_covered(new_classes)
            new_score = new_covered.count(0)
            
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in new_classes]
                classes = new_classes
                covered = new_covered
                score = new_score
            elif new_score < score or (temperature > 0 and random.random() < np.exp(-(new_score - score) / max(t, 1e-10))):
                classes = new_classes
                covered = new_covered
                score = new_score
        
        if temperature > 0:
            t *= cooling
        it += 1
    
    return best_classes, best_score

def _hill_climb_polish(classes, deadline=None, max_iter=5000):
    """Final high-precision polish: try all possible single-residue changes exhaustively."""
    score = _score(classes)
    best_score = score
    best_classes = [c[:] for c in classes]
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for idx in range(len(classes)):
            m, old_r = classes[idx]
            best_new_r = old_r
            best_new_score = score
            for new_r in range(m):
                if new_r == old_r:
                    continue
                new_classes = [c[:] for c in classes]
                new_classes[idx][1] = new_r
                new_score = _score(new_classes)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_r = new_r
            if best_new_r != old_r:
                classes[idx][1] = best_new_r
                score = best_new_score
                improved = True
                if score < best_score:
                    best_score = score
                    best_classes = [c[:] for c in classes]
                if deadline is not None and time.time() > deadline:
                    break
        if deadline is not None and time.time() > deadline:
            break
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with greedy in increasing modulus order
    classes, score = _greedy_order(DIVISORS)
    best_classes = [c[:] for c in classes]
    best_score = score
    
    # Try greedy in decreasing modulus order
    classes2, score2 = _greedy_order(list(reversed(DIVISORS)))
    if score2 < best_score:
        best_score = score2
        best_classes = [c[:] for c in classes2]
    
    # Try greedy with random shuffles
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        shuffled = DIVISORS[:]
        random.shuffle(shuffled)
        classes3, score3 = _greedy_order(shuffled)
        if score3 < best_score:
            best_score = score3
            best_classes = [c[:] for c in classes3]
    
    # Local search with simulated annealing
    if time.time() < deadline - 10.0:
        best_classes, best_score = _local_search(
            best_classes, max_iter=30000, deadline=deadline - 10.0, temperature=5.0
        )
    
    # Additional local search passes with lower temperature
    for temp in [2.0, 1.0, 0.5, 0.1]:
        if time.time() < deadline - 5.0:
            before = best_score
            best_classes, best_score = _local_search(
                best_classes, max_iter=10000, deadline=deadline - 5.0, temperature=temp
            )
            if best_score == before and temp < 0.5:
                break
    
    # Final polish with exhaustive single-residue search
    if time.time() < deadline - 2.0:
        best_classes, best_score = _hill_climb_polish(
            best_classes, deadline=deadline - 1.0, max_iter=100
        )
    
    return best_classes
# EVOLVE-BLOCK-END
