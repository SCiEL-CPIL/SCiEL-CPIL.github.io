"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set_fast(U):
    """Compute score using set operations for speed."""
    U_sorted = sorted(set(U))
    n = len(U_sorted)
    if n == 0:
        return 0.0
    max_u = U_sorted[-1]
    if max_u == 0:
        return 1.0
    
    # Compute sumset using set operations
    sumset = set()
    for i in range(n):
        for j in range(i, n):  # symmetric
            sumset.add(U_sorted[i] + U_sorted[j])
    sumset_size = len(sumset)
    
    # Compute difference set
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U_sorted[i] - U_sorted[j])
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Fallback score computation (uses fast version)."""
    return _score_set_fast(U)

def _generate_golomb_rulers(max_len=8, deadline=None):
    """Generate known Golomb ruler constructions."""
    rulers = []
    # Known optimal Golomb rulers
    known = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 45],
    ]
    for r in known:
        if len(r) <= max_len:
            rulers.append(r[:])
    
    # Generate more via greedy construction
    for target_len in range(3, max_len + 1):
        if deadline and time.time() > deadline:
            break
        # Greedy Golomb ruler
        U = [0]
        last = 0
        for i in range(1, target_len):
            best_pos = last + 1
            best_diff = float('inf')
            for pos in range(last + 1, last + 30):
                # Check if this creates a duplicate difference
                valid = True
                for u in U:
                    d = abs(pos - u)
                    for v in U:
                        if v != u and abs(pos - v) == d:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_pos = pos
                    break
            U.append(best_pos)
        if len(set(U)) == len(U):
            rulers.append(sorted(U))
    
    return rulers

def _generate_sidon_sets(max_n=10, deadline=None):
    """Generate Sidon/B_2 sets which have large difference sets."""
    candidates = []
    rng = random.Random(42)
    
    for trial in range(200):
        if deadline and time.time() > deadline:
            break
        n = rng.randint(3, max_n)
        max_val = rng.randint(n * 2, n * 5)
        U = [0] + sorted(rng.sample(range(1, max_val + 1), n - 1))
        candidates.append(U)
    
    # Also try modular constructions
    for p in [7, 11, 13, 17, 19, 23, 29, 31]:
        if deadline and time.time() > deadline:
            break
        # Quadratic residues mod p
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        if 0 in residues:
            residues.discard(0)
        U = [0] + sorted(residues)
        candidates.append(U)
    
    return candidates

def _search_small_sets(max_n=7, max_val=50, deadline=None):
    """Exhaustive search for small sets U using fast scoring."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            s = _score_set_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Structure 1: Almost AP
    for k in range(3, 12):
        for M in range(k, 80):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Structure 2: Golomb rulers
    candidates.extend(_generate_golomb_rulers(max_len=8, deadline=deadline))
    
    # Structure 3: Sidon sets
    candidates.extend(_generate_sidon_sets(max_n=8, deadline=deadline))
    
    # Structure 4: Random perturbations of AP
    rng = random.Random(42)
    for trial in range(100):
        if deadline and time.time() > deadline:
            break
        n = rng.randint(3, 10)
        base_ap = list(range(n))
        num_perturb = rng.randint(1, min(2, n - 1))
        indices = rng.sample(range(1, n), num_perturb)
        U = base_ap[:]
        for i in indices:
            new_val = rng.randint(1, 100)
            U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
            
    return candidates

def _local_search(U, max_iters=300, deadline=None):
    """Improve set U via local search with sampled neighborhood."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 50
    rng = random.Random()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element to a sampled new value
        for i in range(len(U)):
            old_val = U[i]
            if old_val == 0:
                continue
            
            # Sample candidate values: nearby + random jumps
            candidates_new = set()
            # Nearby values
            for delta in range(-10, 11):
                v = old_val + delta
                if v >= 1:
                    candidates_new.add(v)
            # Random values in range
            for _ in range(20):
                candidates_new.add(rng.randint(1, upper_bound))
            # Some larger jumps
            for _ in range(5):
                candidates_new.add(rng.randint(1, upper_bound + 100))
            
            for new_val in candidates_new:
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    upper_bound = max(best_U) + 50
                    improved = True
                    break
            if improved:
                break
        
        # Try removing an element (if n > 2)
        if not improved and len(U) > 2:
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    upper_bound = max(best_U) + 50
                    improved = True
                    break
        
        # Try adding a new element
        if not improved and len(U) < 12:
            for _ in range(30):
                new_val = rng.randint(1, upper_bound + 20)
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    upper_bound = max(best_U) + 50
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # 1. Exhaustive search for small sets
    try:
        U_sol, s_sol = _search_small_sets(max_n=6, max_val=45, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Heuristic candidates
    candidates = _construct_heuristic_sets(deadline=deadline)
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 3. Local search with random restarts
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=200, deadline=deadline)
        
        # Random restarts: try local search from different starting points
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            rng = random.Random(123)
            num_restarts = min(10, int(remaining_time / 5))
            for restart in range(num_restarts):
                if time.time() > deadline:
                    break
                # Generate a random starting set
                n = rng.randint(3, 9)
                start_set = [0] + sorted(rng.sample(range(1, 80), n - 1))
                U_restart, s_restart = _local_search(start_set, max_iters=100, deadline=deadline)
                if s_restart > best_score:
                    best_score = s_restart
                    best_U = U_restart
                    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
