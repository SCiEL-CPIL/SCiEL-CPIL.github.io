"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate Weight 3 and Weight 4 vectors
    candidates = []
    norms_sq = []
    
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products and squared distances
    dots = np.dot(candidates, candidates.T)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict: dist_sq[i,j] < 4 means vectors are too close
    # (max norm is sqrt(4)=2, so min pairwise distance must be >= 2, i.e. dist_sq >= 4)
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute conflict neighbor lists as arrays for speed
    conflict_neighbors = [np.where(conflicts[i])[0] for i in range(n_candidates)]
    
    def max_min_greedy(seed_vec=None):
        """
        Max-min greedy: repeatedly pick the candidate that maximizes
        its minimum squared distance to all already-selected vectors.
        This directly optimizes the kissing constraint.
        """
        selected = []
        selected_set = set()
        
        # Optional: start with a seed vector
        if seed_vec is not None and seed_vec < n_candidates:
            selected.append(seed_vec)
            selected_set.add(seed_vec)
        
        while True:
            if time.time() > deadline - 2.0:
                break
            
            best_v = -1
            best_min_dist = -1
            
            # For each candidate not yet selected and not conflicting with selected,
            # compute min dist to selected set; pick the one with max min-dist
            # Use vectorized computation for speed
            remaining = np.ones(n_candidates, dtype=bool)
            for s in selected_set:
                remaining[s] = False
                remaining[conflict_neighbors[s]] = False
            
            if not remaining.any():
                break
            
            # Get indices of remaining candidates
            rem_indices = np.where(remaining)[0]
            
            # Compute min dist from each remaining to selected set
            if len(selected_set) == 0:
                # No selected yet, just pick first remaining (or random)
                best_v = rem_indices[0]
            else:
                sel_arr = np.array(list(selected_set))
                # dist_sq[rem_indices][:, sel_arr] gives distances from each remaining to each selected
                d_matrix = dist_sq[np.ix_(rem_indices, sel_arr)]
                min_dists = d_matrix.min(axis=1)
                # Pick the one with maximum minimum distance
                best_local = np.argmax(min_dists)
                best_v = rem_indices[best_local]
                best_min_dist = min_dists[best_local]
            
            if best_v < 0:
                break
            
            selected.append(best_v)
            selected_set.add(best_v)
        
        return selected
    
    def local_search(selected):
        """
        Try to improve the selected set by:
        1. Removing a vector and trying to insert a better one
        2. Swapping one vector for another that allows more additions
        Returns improved selected list.
        """
        selected_set = set(selected)
        improved = True
        max_iters = 50
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            found_improvement = False
            
            # Try removal + reinsertion of a different vector
            for i, s in enumerate(selected):
                # Try removing s
                trial_set = set(selected)
                trial_set.discard(s)
                
                # Find best candidate to add
                best_add = -1
                best_score = -1
                for c in range(n_candidates):
                    if c in trial_set:
                        continue
                    # Check if c conflicts with any in trial_set
                    if any(c in conflict_neighbors[t] for t in trial_set):
                        continue
                    # Score: how many additional vectors could be added after inserting c
                    # Approximate: count non-conflicting candidates not in trial_set ∪ {c}
                    added_set = trial_set | {c}
                    compat = 0
                    for t in range(n_candidates):
                        if t in added_set:
                            continue
                        if any(t in conflict_neighbors[x] for x in added_set):
                            continue
                        compat += 1
                    if compat > best_score:
                        best_score = compat
                        best_add = c
                
                if best_add >= 0 and best_score > 0:
                    # Replace s with best_add
                    selected[i] = best_add
                    selected_set = set(selected)
                    found_improvement = True
                    break
            
            if not found_improvement:
                break
        
        return selected
    
    np.random.seed(42)
    random.seed(42)
    
    best_selected = []
    best_len = 0
    
    # Phase 1: Multiple max-min greedy runs with different seeds
    num_greedy = 200
    for trial in range(num_greedy):
        if time.time() > deadline - 30.0:
            break
        
        # Random seed for diversity
        seed_vec = random.randint(0, n_candidates - 1)
        sel = max_min_greedy(seed_vec)
        
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel
    
    # Phase 2: Local search on the best solution
    if best_selected:
        improved = local_search(best_selected)
        if len(improved) > best_len:
            best_selected = improved
            best_len = len(best_selected)
        
        # Try local search on a few other good solutions
        # (we don't store all, so just do a few more greedy + local search)
        for extra in range(10):
            if time.time() > deadline - 5.0:
                break
            seed_vec = random.randint(0, n_candidates - 1)
            sel = max_min_greedy(seed_vec)
            if len(sel) >= best_len - 5:  # only improve near-best solutions
                imp = local_search(sel)
                if len(imp) > best_len:
                    best_len = len(imp)
                    best_selected = imp
    
    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
# EVOLVE-BLOCK-END
