N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290
    
    n_pts = N * N
    all_points = [(x, y) for y in range(N) for x in range(N)]
    pts_arr = np.array(all_points, dtype=np.int32)
    
    # Precompute squared distances
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    sq_dists = (diff ** 2).sum(axis=2).astype(np.int64)
    
    # For each pair (a, b), precompute the set of points p such that
    # triangle (p, a, b) is isosceles. We represent this as a bitset.
    # n_pts = 4096, so we need 4096/64 = 64 int64 words per bitset
    n_words = (n_pts + 63) // 64
    
    # conflict_mask[a, b] = bitset of points that form isosceles with a and b
    # Memory: 4096 * 4096 * 64 bytes = 1GB - too much!
    # Instead, we'll compute on the fly using NumPy vectorization.
    
    # Alternative: for each point p, precompute the set of distance values
    # dist_from_p[p, d] = number of points at distance d from p
    # Then p can be added to S iff for all pairs (a,b) in S, dist(p,a) != dist(p,b)
    # and dist(p,a) != dist(a,b) and dist(p,b) != dist(a,b)
    
    # Let's use a different approach: precompute for each point p, the distance
    # to all other points, and use a hash-based check.
    
    def try_greedy_random(rng, time_budget):
        """Greedy construction with random ordering, using fast conflict check."""
        S = set()
        # dist_sets[p] = set of squared distances from p to points in S
        dist_sets = {}
        
        order = list(range(n_pts))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            # Check if idx can be added to S
            valid = True
            
            # Check 1: all distances from idx to points in S must be distinct
            # Check 2: for each point j in S, dist(idx, j) must not be in dist_sets[j]
            
            dists_to_idx = set()
            for j in S:
                d = sq_dists[idx, j]
                if d in dists_to_idx:
                    valid = False
                    break
                dists_to_idx.add(d)
                if d in dist_sets.get(j, set()):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add idx to S
            S.add(idx)
            dist_sets[idx] = set(dists_to_idx)
            for j in S:
                if j == idx:
                    continue
                dist_sets[j].add(sq_dists[idx, j])
        
        return S, dist_sets
    
    def local_search_swap(S, dist_sets, rng, max_steps=200):
        """Local search: try removing one point and adding another."""
        improved = True
        steps = 0
        current_size = len(S)
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            if time.time() > deadline:
                break
            
            # Pick a random point to try removing
            if not S:
                break
            idx_to_remove = rng.choice(list(S))
            
            # Temporarily remove
            S.remove(idx_to_remove)
            saved_dist_set = dist_sets.pop(idx_to_remove, set())
            for j in list(S):
                d = sq_dists[idx_to_remove, j]
                if d in dist_sets.get(j, set()):
                    dist_sets[j].discard(d)
            
            # Try to add a random candidate
            candidates = [i for i in range(n_pts) if i not in S]
            if candidates:
                # Try up to 20 random candidates
                rng.shuffle(candidates)
                for idx_to_add in candidates[:20]:
                    valid = True
                    dists_to_new = set()
                    for j in S:
                        d = sq_dists[idx_to_add, j]
                        if d in dists_to_new:
                            valid = False
                            break
                        dists_to_new.add(d)
                        if d in dist_sets.get(j, set()):
                            valid = False
                            break
                    
                    if valid:
                        S.add(idx_to_add)
                        dist_sets[idx_to_add] = set(dists_to_new)
                        for j in S:
                            if j == idx_to_add:
                                continue
                            dist_sets[j].add(sq_dists[idx_to_add, j])
                        if len(S) > current_size:
                            current_size = len(S)
                        improved = True
                        break
            
            if not improved:
                # Restore
                S.add(idx_to_remove)
                dist_sets[idx_to_remove] = saved_dist_set
                for j in S:
                    if j != idx_to_remove:
                        dist_sets[j].add(sq_dists[idx_to_remove, j])
        
        return S, dist_sets
    
    def greedy_expand(S, dist_sets, rng):
        """Try to add more points in random order."""
        candidates = [i for i in range(n_pts) if i not in S]
        rng.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline:
                break
            
            valid = True
            dists_to_new = set()
            for j in S:
                d = sq_dists[idx, j]
                if d in dists_to_new:
                    valid = False
                    break
                dists_to_new.add(d)
                if d in dist_sets.get(j, set()):
                    valid = False
                    break
            
            if valid:
                S.add(idx)
                dist_sets[idx] = set(dists_to_new)
                for j in S:
                    if j == idx:
                        continue
                    dist_sets[j].add(sq_dists[idx, j])
        
        return S, dist_sets
    
    best_S = set()
    best_dist = {}
    best_size = 0
    
    rng = random.Random(42)
    
    # Multiple restarts with different random seeds
    while time.time() < deadline - 40:
        seed = rng.randint(0, 2**31)
        rng_trial = random.Random(seed)
        
        S, dist_sets = try_greedy_random(rng_trial, deadline)
        if len(S) > best_size:
            best_S = S.copy()
            best_dist = {k: v.copy() for k, v in dist_sets.items()}
            best_size = len(S)
    
    # Local search refinement
    if best_size > 0 and time.time() < deadline - 20:
        best_S, best_dist = local_search_swap(best_S, best_dist, rng)
        best_size = len(best_S)
    
    # Final expansion
    if best_size > 0 and time.time() < deadline - 5:
        best_S, best_dist = greedy_expand(best_S, best_dist, rng)
        best_size = len(best_S)
    
    result = [all_points[i] for i in best_S]
    return result
# EVOLVE-BLOCK-END
