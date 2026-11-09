import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 300s, aim to finish by 290s
    deadline = start_time + 290.0

    # 1. Precompute all grid points and the distance matrix
    # points[i] = (x, y) for i in [0, M)
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # D[i, j] = squared distance between point i and point j
    # Using broadcasting: (M, 1, 2) - (1, M, 2) -> (M, M, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)
    
    # Helper function to check if a point can be added to a set
    # S_arr: numpy array of indices of points in the current set
    def can_add(p_idx, S_arr):
        if len(S_arr) == 0:
            return True
        
        # 1. Distances from p to all points in S must be distinct
        # d_p_S[k] is distance from p to S_arr[k]
        d_p_S = D[p_idx, S_arr]
        
        # If any distance is repeated, we have an isosceles triangle with apex at p
        # We can check uniqueness efficiently
        if len(np.unique(d_p_S)) != len(S_arr):
            return False
        
        # 2. For each point s_k in S, the distance d(p, s_k) must not equal 
        #    the distance d(s_k, s_j) for any j != k in S.
        #    This prevents isosceles triangles with apex at s_k.
        
        # Get the submatrix of distances among points in S
        # d_S_S[i, j] is distance between S_arr[i] and S_arr[j]
        d_S_S = D[np.ix_(S_arr, S_arr)]
        
        # We need to check if d_p_S[i] is present in d_S_S[i, :]
        # Note: d_S_S[i, i] is 0. d_p_S[i] is distance from p to S_arr[i].
        # Since p is not in S (we are checking if we can add it), d_p_S[i] > 0.
        # So if d_p_S[i] == d_S_S[i, j] for some j != i, it's a conflict.
        # Actually, if d_p_S[i] == d_S_S[i, j], it means dist(p, s_i) == dist(s_i, s_j).
        # This forms an isosceles triangle with vertices (p, s_i, s_j) and apex at s_i.
        
        # Vectorized check:
        # Create a mask where d_S_S[i, j] == d_p_S[i]
        # d_p_S is shape (K,). We need to compare d_S_S (KxK) with d_p_S (K,) broadcasted to rows.
        # d_p_S[:, np.newaxis] is shape (K, 1)
        # d_S_S == d_p_S[:, np.newaxis] checks if d_S_S[i, j] == d_p_S[i]
        
        matches = (d_S_S == d_p_S[:, np.newaxis])
        
        # If any match is found, it means there exists i, j such that dist(s_i, s_j) == dist(p, s_i).
        # This is a conflict.
        if matches.any():
            return False
            
        return True

    def greedy_fill(S_list, candidates_pool, time_limit):
        """
        Greedily adds points from candidates_pool to S_list.
        S_list is a list of indices.
        candidates_pool is a list of indices to consider.
        Returns the updated S_list.
        """
        S_arr = np.array(S_list, dtype=np.int32)
        
        # We iterate through candidates. 
        # For efficiency, we could sort candidates by some heuristic, but random order is standard for ILS restarts.
        # However, checking can_add is O(K^2) due to matrix extraction.
        
        for p_idx in candidates_pool:
            if time.time() > time_limit:
                break
            
            # Quick check: if p_idx is already in S, skip
            if p_idx in S_arr:
                continue
                
            if can_add(p_idx, S_arr):
                S_list.append(p_idx)
                S_arr = np.append(S_arr, p_idx)
                
        return S_list

    best_S = []
    best_size = 0
    
    # Phase 1: Multiple Random Restarts
    # We do a few full restarts to get a decent baseline
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline - 60: # Leave time for ILS
            break
            
        S = []
        candidates = list(range(M))
        random.shuffle(candidates)
        
        # Use a tighter deadline for this specific run to ensure we finish
        run_deadline = deadline - 60
        S = greedy_fill(S, candidates, run_deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            
    # Phase 2: Iterated Local Search (ILS)
    # Start from best_S, perturb, and re-greedy
    current_S = best_S[:]
    current_size = best_size
    
    # We will run ILS until time is up
    # Each ILS step involves:
    # 1. Removing k random points from current_S
    # 2. Greedily adding points from the remaining grid
    
    # To make greedy_fill faster in ILS, we can pre-filter candidates or just use random order.
    # Since M=4096, iterating through all candidates is fast enough if can_add is fast.
    # can_add is O(K^2) vectorized. K ~ 10-20. So it's very fast.
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Perturbation: Remove k random points
        if current_size == 0:
            # If we have no points, do a full restart
            S = []
            candidates = list(range(M))
            random.shuffle(candidates)
            S = greedy_fill(S, candidates, deadline - 5)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S[:]
            current_S = S[:]
            current_size = len(S)
            continue
            
        # Choose k: smaller k for finer tuning, larger k for bigger jumps
        # Adaptive k: start small, increase if stuck? 
        # For simplicity, random k between 1 and 3
        k = random.randint(1, min(3, current_size))
        
        # Remove k random points
        to_remove = random.sample(current_S, k)
        remove_set = set(to_remove)
        
        S = [p for p in current_S if p not in remove_set]
        
        # Candidates: All points not in S
        # We can shuffle the remaining points to try different orders
        # Or we can prioritize points that were removed? 
        # Usually, random shuffle of the complement set works well.
        
        # Optimization: Instead of creating a list of 4096-k elements and shuffling,
        # we can just iterate through all M points and check if they are in S.
        # But shuffling ensures we don't always try to add the same "easy" points first.
        
        candidates = [p for p in range(M) if p not in remove_set and p not in set(S)]
        random.shuffle(candidates)
        
        # Greedy fill
        S = greedy_fill(S, candidates, deadline - 5)
        
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            current_S = S[:]
            current_size = best_size
        else:
            # If we didn't improve, we might still want to keep this S as the starting point
            # for the next perturbation if it's "good enough" or just revert to best?
            # Standard ILS: if no improvement, revert to best or keep current?
            # Keeping current allows for local wandering. Reverting to best focuses on best.
            # Let's keep the best as the anchor for future perturbations to avoid drifting too far.
            # But sometimes a slightly worse solution is the key to a better one.
            # Let's use a probabilistic acceptance or just keep best.
            # Given the short time, sticking to the best known solution for perturbation base is safer.
            pass

    # Convert indices back to (x, y) tuples
    return [tuple(points[i]) for i in best_S]

# Ensure the function is accessible as required
if __name__ == "__main__":
    # Quick self-test (optional, usually not run in evaluator)
    pass
