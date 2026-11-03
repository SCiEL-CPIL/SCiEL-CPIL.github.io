<<<<<<< SEARCH
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit
    
    # Generate Weight 2, 3, and 4 vectors
    candidates = []
    norms_sq = []
    
    # Weight 2: C(11,2)*4 = 220 vectors
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(2)
    
    # Weight 3: C(11,3)*8 = 1320 vectors
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4: C(11,4)*16 = 5280 vectors
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
    
    # Compute dot products and squared distances in blocks to save memory
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dots[i,j]
    # Conflict if dist_sq < 4, i.e., dots > (norms_sq[i] + norms_sq[j])/2 - 2
    # For weight-4 vs weight-4: conflict if dot > (4+4)/2 - 2 = 2, i.e. dot >= 3
    # For weight-4 vs weight-3: conflict if dot > (4+3)/2 - 2 = 1.5, i.e. dot >= 2
    # For weight-4 vs weight-2: conflict if dot > (4+2)/2 - 2 = 1, i.e. dot >= 2
    # For weight-3 vs weight-3: conflict if dot > (3+3)/2 - 2 = 1, i.e. dot >= 2
    # For weight-3 vs weight-2: conflict if dot > (3+2)/2 - 2 = 0.5, i.e. dot >= 1
    # For weight-2 vs weight-2: conflict if dot > (2+2)/2 - 2 = 0, i.e. dot >= 1
    
    # Build conflict bitsets using integer bit operations
    # n_candidates can be up to 220+1320+5280 = 6820
    # Use Python integers as bitsets
    n_bits = n_candidates
    all_ones = (1 << n_bits) - 1
    
    # Compute dots in blocks
    block_size = 2000
    conflicts_bitset = [0] * n_candidates  # conflicts_bitset[i] has bit j set if i conflicts with j
    
    for i_start in range(0, n_candidates, block_size):
        i_end = min(i_start + block_size, n_candidates)
        dots_block = np.dot(candidates[i_start:i_end], candidates.T)
        ns_i = norms_sq[i_start:i_end, None]
        ns_j = norms_sq[None, :]
        dist_sq_block = ns_i + ns_j - 2 * dots_block
        conflict_block = dist_sq_block < 4
        np.fill_diagonal(conflict_block, False)
        
        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            # Get conflicting indices
            conf_indices = np.where(conflict_block[local_i])[0]
            if len(conf_indices) > 0:
                # Convert to bitset
                bits = 0
                for j in conf_indices:
                    bits |= (1 << j)
                conflicts_bitset[global_i] = bits
    
    # Also build reverse: for each j, which i conflict with j
    # conflicts_bitset[i] has bit j if i~j, so conflicts_bitset[j] should have bit i
    # But we built it symmetrically already since dist_sq is symmetric
    # Verify: conflicts_bitset[i] & (1<<j) == conflicts_bitset[j] & (1<<i)
    
    def greedy_max_min(seed_vec=None, randomize=False):
        """
        Greedy: maintain a bitset of available (non-conflicting with selected) candidates.
        At each step, pick the candidate that maximizes minimum distance to selected,
        or randomly from top-k for diversity.
        """
        selected = []
        if seed_vec is not None and 0 <= seed_vec < n_candidates:
            selected.append(seed_vec)
        
        # Available bitset: all candidates except selected and their conflicts
        avail = all_ones
        for s in selected:
            avail &= ~(1 << s)
            avail &= ~conflicts_bitset[s]
        
        while True:
            if time.time() > deadline - 1.0:
                break
            
            # Check if any available
            if avail == 0:
                break
            
            # Get list of available indices
            # This is expensive for large n, but we need it for distance computation
            avail_indices = []
            a = avail
            idx = 0
            while a:
                if a & 1:
                    avail_indices.append(idx)
                a >>= 1
                idx += 1
            
            if not avail_indices:
                break
            
            if len(selected) == 0:
                if randomize:
                    best_v = random.choice(avail_indices)
                else:
                    best_v = avail_indices[0]
            else:
                sel_arr = np.array(selected)
                # Compute min distance from each available to selected
                d_matrix = dist_sq_block_full[np.ix_(avail_indices, sel_arr)] if False else None
                
                # We need full dist_sq; compute on the fly
                # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*candidates[i]@candidates[j]
                dots_sel = np.dot(candidates[avail_indices], candidates[sel_arr].T)
                ns_a = norms_sq[avail_indices, None]
                ns_s = norms_sq[sel_arr][None, :]
                d_mat = ns_a + ns_s - 2 * dots_sel
                min_dists = d_mat.min(axis=1)
                
                if randomize:
                    # Pick from top 20% or top 10
                    k = max(1, min(10, len(min_dists)))
                    top_k = np.argpartition(min_dists, -k)[-k:]
                    best_v = avail_indices[random.choice(top_k)]
                else:
                    best_v = avail_indices[np.argmax(min_dists)]
            
            selected.append(best_v)
            # Update available
            avail &= ~(1 << best_v)
            avail &= ~conflicts_bitset[best_v]
        
        return selected
    
    # Precompute full dist_sq for speed in greedy (memory: 6820^2 * 8 bytes = ~370MB, too much)
    # Instead, compute dots on the fly as above
    
    np.random.seed(42)
    random.seed(42)
    
    best_selected = []
    best_len = 0
    
    # Phase 1: Multiple greedy runs
    # Use different strategies: deterministic, random seed, randomized pick
    num_trials = 0
    while time.time() < deadline - 20.0:
        num_trials += 1
        strategy = num_trials % 3
        if strategy == 0:
            sel = greedy_max_min(seed_vec=None, randomize=False)
        elif strategy == 1:
            seed_v = random.randint(0, n_candidates - 1)
            sel = greedy_max_min(seed_vec=seed_v, randomize=False)
        else:
            seed_v = random.randint(0, n_candidates - 1)
            sel = greedy_max_min(seed_vec=seed_v, randomize=True)
        
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel
    
    # Phase 2: Try to improve by adding more vectors
    # For the best solution, try to find any additional compatible vector
    if best_selected:
        sel_set = set(best_selected)
        # Build bitset of selected
        sel_bits = 0
        for s in best_selected:
            sel_bits |= (1 << s)
        
        # Available = all - selected - conflicts with selected
        avail = all_ones & ~sel_bits
        for s in best_selected:
            avail &= ~conflicts_bitset[s]
        
        # Try to greedily add more
        while avail != 0 and time.time() < deadline - 2.0:
            # Pick any available (or the one with max min-dist)
            avail_indices = []
            a = avail
            idx = 0
            while a:
                if a & 1:
                    avail_indices.append(idx)
                a >>= 1
                idx += 1
            
            if not avail_indices:
                break
            
            # Pick one with max min-dist to selected
            sel_arr = np.array(best_selected)
            dots_sel = np.dot(candidates[avail_indices], candidates[sel_arr].T)
            ns_a = norms_sq[avail_indices, None]
            ns_s = norms_sq[sel_arr][None, :]
            d_mat = ns_a + ns_s - 2 * dots_sel
            min_dists = d_mat.min(axis=1)
            best_v = avail_indices[np.argmax(min_dists)]
            
            best_selected.append(best_v)
            best_len = len(best_selected)
            sel_bits |= (1 << best_v)
            avail &= ~(1 << best_v)
            avail &= ~conflicts_bitset[best_v]
    
    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize I'm referencing `dist_sq_block_full` which doesn't exist. Let me fix that and also the bitset extraction is O(n) per iteration which is slow. Let me use a different approach - use numpy boolean masks but update them incrementally.

Let me rewrite more carefully:

