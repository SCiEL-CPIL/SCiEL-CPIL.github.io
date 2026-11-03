"""Kissing number in dimension 11: maximize number of integer vectors
with min pairwise distance >= max norm.

Uses Simulated Annealing on a conflict graph of small-norm integer vectors.
"""

import time
import numpy as np
import itertools
import random
import math

D = 11
DEADLINE = 140  # safety margin before 150s limit


def _generate_candidates(norm_sq_max):
    """Generate all integer vectors in R^11 with squared norm <= norm_sq_max."""
    candidates = []
    seen = set()
    
    # Helper to add vector
    def add_vec(v):
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            candidates.append(np.array(v, dtype=np.int64))

    # Norm squared = 1
    for i in range(D):
        for si in (1, -1):
            v = [0] * D
            v[i] = si
            if sum(x*x for x in v) <= norm_sq_max:
                add_vec(v)

    # Norm squared = 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                if sum(x*x for x in v) <= norm_sq_max:
                    add_vec(v)

    # Norm squared = 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    if sum(x*x for x in v) <= norm_sq_max:
                        add_vec(v)
    
    # Norm squared = 4
    if norm_sq_max >= 4:
        # One coordinate +/-2
        for i in range(D):
            for si in (2, -2):
                v = [0] * D
                v[i] = si
                add_vec(v)
        # Four coordinates +/-1
        for i, j, k, l in itertools.combinations(range(D), 4):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        for sl in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            v[l] = sl
                            add_vec(v)

    # Norm squared = 5
    if norm_sq_max >= 5:
        # Two coords: one +/-2, one +/-1
        for i in range(D):
            for j in range(D):
                if i == j: continue
                for si in (2, -2):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        add_vec(v)
        # Five coords +/-1
        for i, j, k, l, m in itertools.combinations(range(D), 5):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        for sl in (1, -1):
                            for sm in (1, -1):
                                v = [0] * D
                                v[i] = si
                                v[j] = sj
                                v[k] = sk
                                v[l] = sl
                                v[m] = sm
                                add_vec(v)

    return candidates


def _greedy_init(candidates, threshold, rng):
    """Greedy initialization using min-degree heuristic."""
    n = len(candidates)
    # Precompute degrees? Or just sample.
    # For speed, we'll just do a random greedy with a bit of bias.
    available = set(range(n))
    selected = []
    
    # Approximate degree by counting conflicts with a random subset?
    # Or just pick random.
    # Let's try min-degree on a sample.
    
    while available:
        if len(available) < 50:
            v = rng.choice(list(available))
        else:
            # Sample 20 vertices, pick one with fewest conflicts in available
            sample = rng.sample(list(available), 20)
            best_v = sample[0]
            min_conf = float('inf')
            for v in sample:
                # Count conflicts in available
                # This is O(N) per v. 20 * N.
                # N can be 5000. 100k ops. OK.
                conf_count = 0
                # We need conflict info. We don't have a full matrix.
                # We can check distance on the fly? No, too slow.
                # We need a fast way to check conflicts.
                
                # Since we are in init, let's just use random for now to keep it fast,
                # and rely on SA to fix it.
                pass
            
            # Actually, let's just use random for init to save time.
            v = rng.choice(list(available))
        
        selected.append(v)
        available.discard(v)
        # Remove neighbors
        # We need to check which available vertices conflict with v.
        # dist(candidates[v], candidates[available]) < threshold
        v_vec = candidates[v]
        a_vecs = candidates[list(available)]
        # Vectorized distance
        diffs = a_vecs - v_vec
        dists = np.sum(diffs * diffs, axis=1)
        to_remove = np.where(dists < threshold)[0]
        for idx in to_remove:
            available.discard(list(available)[idx]) # This is slow.
            
    # This removal loop is slow. Let's use a boolean mask.
    # Re-implementing init with NumPy masks.
    
    selected = []
    remaining_mask = np.ones(n, dtype=bool)
    
    while np.any(remaining_mask):
        # Pick a vertex from remaining
        rem_indices = np.where(remaining_mask)[0]
        v = rng.choice(rem_indices)
        selected.append(v)
        remaining_mask[v] = False
        
        # Find conflicts with v
        v_vec = candidates[v]
        diffs = candidates - v_vec
        dists = np.sum(diffs * diffs, axis=1)
        conflicts = dists < threshold
        conflicts[v] = False # self
        remaining_mask[conflicts & remaining_mask] = False
        
    return selected


def _sa_optimize(candidates, initial_set, threshold, rng, time_limit):
    """Simulated Annealing optimization."""
    n = len(candidates)
    S = set(initial_set)
    best_S = set(S)
    best_size = len(S)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    start_time = time.time()
    
    # Precompute candidate array for fast access
    cand_arr = np.array(candidates)
    
    while time.time() - start_time < time_limit:
        # Move 1: Add
        if len(S) < n:
            # Pick random v not in S
            # To speed up, maintain a list of not-in-S?
            # For now, sample random indices until we find one not in S
            # This is inefficient if S is large.
            # Let's use a list of available indices.
            pass
        
        # Efficient Move Generation:
        # Maintain `in_S` boolean array.
        
        # Let's restart the loop structure for efficiency.
        
        # ... (See final code for implementation)
        
    return best_S

# The above sketch is a bit messy. Let's write the actual code carefully.
