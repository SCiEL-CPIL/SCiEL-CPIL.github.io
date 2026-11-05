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
    
    # 1. Two-cluster APs (Optimized for collisions)
    # Cluster 1: start1, diff1, len 6
    # Cluster 2: start2, diff2, len 6
    # Ensure separation to avoid cross-collisions interfering with internal ones? 
    # Actually, cross-collisions are good too if they increase total collisions.
    for start1 in range(0, 20):
        for diff1 in range(1, 8):
            c1 = [start1 + i * diff1 for i in range(6)]
            if max(c1) > 128: continue
            for start2 in range(40, 129):
                for diff2 in range(1, 8):
                    c2 = [start2 + i * diff2 for i in range(6)]
                    if max(c2) > 128: continue
                    c = c1 + c2
                    if len(set(c)) == 12:
                        add_candidate(c)
                        
    # 2. Three-cluster APs (4 elements each)
    for s1 in range(0, 10):
        for d1 in range(1, 5):
            c1 = [s1 + i * d1 for i in range(4)]
            if max(c1) > 128: continue
            for s2 in range(20, 60):
                for d2 in range(1, 5):
                    c2 = [s2 + i * d2 for i in range(4)]
                    if max(c2) > 128: continue
                    for s3 in range(70, 129):
                        for d3 in range(1, 5):
                            c3 = [s3 + i * d3 for i in range(4)]
                            if max(c3) > 128: continue
                            c = c1 + c2 + c3
                            if len(set(c)) == 12:
                                add_candidate(c)

    # 3. Random perturbations of the best two-cluster structure
    rng = random.Random(42)
    base = [0, 1, 2, 3, 4, 5, 50, 51, 52, 53, 54, 55]
    for _ in range(500):
        c = base.copy()
        for _ in range(3):
            idx = rng.randint(0, 11)
            shift = rng.randint(-5, 5)
            c[idx] += shift
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)

    return candidates

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin (evaluator kills at 90s)
    
    # Generate candidates
    candidates = _build_candidates()
    
    if not candidates:
        return [0, 1, 2, 3, 4, 5, 50, 51, 52, 53, 54, 55]
        
    # Score candidates by collision count (higher is better for minimizing Sidon size)
    best_collisions = -1
    best_cand = candidates[0]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        col = _count_collisions(cand)
        if col > best_collisions:
            best_collisions = col
            best_cand = cand
            
    # Local search: Simulated Annealing / Hill Climbing
    if time.time() < deadline - 5:
        current = list(best_cand)
        current_col = best_collisions
        rng = random.Random(999)
        pool = list(range(129))
        
        # Hill climbing with random restarts if stuck
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            idx = rng.randint(0, 11)
            old_val = current[idx]
            current_set = set(current)
            # Choose a new value from the pool, preferring values close to existing ones or random
            if rng.random() < 0.5:
                # Local move
                new_val = old_val + rng.randint(-5, 5)
                if new_val < 0 or new_val > 128 or new_val in current_set:
                    continue
            else:
                # Global move
                available = [x for x in pool if x not in current_set]
                if not available:
                    continue
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
