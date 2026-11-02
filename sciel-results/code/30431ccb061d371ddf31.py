"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

DEADLINE_MARGIN = 5.0  # seconds before hard limit

def _time_left(start):
    return 120.0 - DEADLINE_MARGIN - (time.time() - start)

def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)

    # Generate candidate vectors: all nonzero integer vectors with small norms
    # We'll use vectors with entries in {-1, 0, 1} (norm 1 or sqrt(2)) and some with a 2.
    # Actually, let's be systematic: generate all vectors with entries in {-1,0,1} that are nonzero.
    # That's 3^11 - 1 = 177146 vectors. But we need min pairwise distance >= max norm.
    # 
    # Key insight: if we normalize to norm 2, the condition is:
    #   min_{i!=j} ||v_i - v_j|| >= max_i ||v_i||
    # 
    # For vectors with entries in {-1,0,1}, the max norm is sqrt(k) where k is the max support size.
    # The pairwise distance ||v_i - v_j||^2 = sum (v_i[a] - v_j[a])^2.
    # Each term is 0, 1, or 4.
    # 
    # Let's think about this differently. The kissing number problem in dimension 11.
    # Known kissing numbers: k(11) is between 504 and 556 (or similar ranges).
    # 
    # Strategy: Use a greedy/local search approach to find a large set of vectors
    # satisfying the constraint. We'll work with unit-norm normalized vectors conceptually.
    #
    # Let's use the following approach:
    # 1. Generate a pool of candidate vectors (integer, small entries).
    # 2. Use local search to maximize the size of a valid subset.
    
    # Generate candidate pool: vectors with entries in {-1, 0, 1}, nonzero
    # To keep it manageable, let's use a subset. Actually 177k is fine.
    
    # Let's generate all {-1,0,1} vectors
    candidates = []
    for mask in range(1, 3**D):
        v = [0] * D
        m = mask
        valid = True
        for i in range(D):
            digit = m % 3
            m //= 3
            if digit == 1:
                v[i] = 1
            elif digit == 2:
                v[i] = -1
        if any(x != 0 for x in v):
            candidates.append(tuple(v))
    
    # Remove duplicates and also remove v and -v pairs? No, we can have both.
    # Actually, let's deduplicate
    candidates = list(set(candidates))
    
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.array([sum(x*x for x in c) for c in candidates], dtype=np.float64)
    
    # The constraint: for all i != j in our set S:
    #   ||v_i - v_j||^2 >= max_{k in S} ||v_k||^2
    # 
    # This is tricky because the RHS depends on the set. Let's reframe:
    # If we fix the maximum norm squared M in our set, then we need:
    #   ||v_i - v_j||^2 >= M for all i != j in S, and all ||v_k||^2 <= M.
    # 
    # So we can iterate over possible M values (which are integers from 1 to D for {-1,0,1} vectors).
    # For a given M, we want the largest subset S such that:
    #   1. ||v_k||^2 <= M for all k in S
    #   2. ||v_i - v_j||^2 >= M for all i != j in S
    #
    # This is an independent set problem in a graph where vertices are candidates with norm^2 <= M
    # and edges connect pairs with distance^2 < M.
    
    # Let's try different M values and use local search / greedy to find large independent sets.
    
    best_set = []
    best_size = 0
    
    # For each possible M (max norm squared), find a large independent set
    for M in range(1, D + 1):
        if _time_left(start) < 10:
            break
        
        # Filter candidates with norm^2 <= M
        valid_idx = [i for i in range(n_cand) if norms_sq[i] <= M]
        if len(valid_idx) < best_size:
            continue
        
        # Build conflict graph: edge between i,j if ||v_i - v_j||^2 < M
        # We want independent set in this graph.
        # Use greedy + local search.
        
        # Greedy: pick vertices with minimum degree first
        # First compute degrees
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue
        
        # Map to local indices
        idx_map = {i: j for j, i in enumerate(valid_idx)}
        
        # Compute conflict matrix (only for valid candidates)
        # This could be large. Let's be smart: only compute on the fly.
        
        # Greedy independent set
        in_set = [False] * n_valid
        degree = [0] * n_valid
        
        # Initialize degrees
        for i in range(n_valid):
            vi = candidates[valid_idx[i]]
            for j in range(i+1, n_valid):
                vj = candidates[valid_idx[j]]
                dist_sq = sum((vi[k]-vj[k])**2 for k in range(D))
                if dist_sq < M:
                    degree[i] += 1
                    degree[j] += 1
        
        # Greedy: repeatedly pick vertex with min degree, remove it and its neighbors
        remaining = list(range(n_valid))
        remaining_set = set(remaining)
        chosen = []
        
        while remaining and _time_left(start) > 5:
            # Find min degree
            min_deg = min(degree[i] for i in remaining)
            # Pick first with min degree (or random among them)
            candidates_min = [i for i in remaining if degree[i] == min_deg]
            pick = random.choice(candidates_min)
            chosen.append(pick)
            remaining_set.discard(pick)
            # Remove neighbors of pick
            vi = candidates[valid_idx[pick]]
            new_remaining = []
            for j in remaining:
                if j == pick:
                    continue
                vj = candidates[valid_idx[j]]
                dist_sq = sum((vi[k]-vj[k])**2 for k in range(D))
                if dist_sq < M:
                    # j is neighbor of pick, remove j
                    degree[j] = -1  # mark as removed
                else:
                    new_remaining.append(j)
                    degree[j] -= 1  # pick was not a neighbor, but we removed pick from graph
            remaining = new_remaining
        
        # Local search: try to add more vertices or swap
        # Try adding vertices not in set
        for attempt in range(50):
            if _time_left(start) < 5:
                break
            improved = False
            for j in range(n_valid):
                if j in set(chosen):
                    continue
                # Check if j can be added
                vj = candidates[valid_idx[j]]
                can_add = True
                for i in chosen:
                    vi = candidates[valid_idx[i]]
                    dist_sq = sum((vi[k]-vj[k])**2 for k in range(D))
                    if dist_sq < M:
                        can_add = False
                        break
                if can_add:
                    chosen.append(j)
                    improved = True
                    break
            if not improved:
                break
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = [valid_idx[i] for i in chosen]
    
    # Convert best_set to array
    if best_set:
        result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    else:
        # Fallback: use the seed program's approach
        from itertools import combinations as comb
        vectors = []
        for i, j in comb(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
