"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from collections import Counter

def _count_collisions(cands):
    """Count the number of duplicate pair sums (i<=j) in the set. Higher is better for minimization."""
    s = sorted(cands)
    n = len(s)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(s[i] + s[j])
    c = Counter(sums)
    collisions = 0
    for v in c.values():
        if v > 1:
            collisions += v - 1
    return collisions

def _build_candidates():
    """Generate a small, high-quality set of candidates with high additive structure."""
    candidates = []
    seen = set()
    
    def add_candidate(c):
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            seen.add(key)
            candidates.append(list(c))
    
    # 1. Arithmetic Progressions (known to have small Sidon subsets)
    for start in range(0, 10):
        for diff in range(1, 15):
            c = [start + i * diff for i in range(12)]
            if max(c) <= 128:
                add_candidate(c)
    
    # 2. Two-cluster APs
    for start1 in range(0, 5):
        for diff1 in range(1, 5):
            for start2 in range(50, 100):
                for diff2 in range(1, 5):
                    c1 = [start1 + i * diff1 for i in range(6)]
                    c2 = [start2 + i * diff2 for i in range(6)]
                    c = c1 + c2
                    if len(set(c)) == 12 and max(c) <= 128:
                        add_candidate(c)
    
    # 3. Random perturbations of APs
    rng = random.Random(123)
    for _ in range(200):
        base = [i * 10 for i in range(12)] # Large gaps to allow perturbation
        c = []
        for i, b in enumerate(base):
            p = rng.randint(-5, 5)
            c.append(b + p)
        # Ensure distinct and in range
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)
            
    # 4. Specific "collision-heavy" constructions
    # e.g. pairs summing to same value
    for _ in range(100):
        c = set()
        while len(c) < 12:
            # Pick a sum S and split it
            S = rng.randint(0, 256)
            a = rng.randint(0, 128)
            b = S - a
            if 0 <= b <= 128:
                c.add(a)
                c.add(b)
        if len(c) == 12:
            add_candidate(list(c))
            
    return candidates

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin (evaluator kills at 90s)
    
    # Generate candidates
    candidates = _build_candidates()
    
    if not candidates:
        return list(range(12))
        
    # Score candidates by collision count (higher is better for minimizing Sidon size)
    # We use this as a proxy for "hard to find large Sidon subset"
    best_collisions = -1
    best_cand = candidates[0]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        col = _count_collisions(cand)
        if col > best_collisions:
            best_collisions = col
            best_cand = cand
            
    # Optional: Quick local search on the best candidate to see if we can increase collisions further
    # This is fast because _count_collisions is O(N^2)
    if time.time() < deadline - 5:
        current = list(best_cand)
        current_col = best_collisions
        rng = random.Random(999)
        pool = list(range(129))
        
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            idx = rng.randint(0, 11)
            old_val = current[idx]
            current_set = set(current)
            available = [x for x in pool if x not in current_set]
            if not available:
                break
            new_val = rng.choice(available)
            current[idx] = new_val
            new_col = _count_collisions(current)
            if new_col > current_col:
                current_col = new_col
                best_collisions = new_col
                best_cand = list(current)
            else:
                current[idx] = old_val
                
    return best_cand
# EVOLVE-BLOCK-END
