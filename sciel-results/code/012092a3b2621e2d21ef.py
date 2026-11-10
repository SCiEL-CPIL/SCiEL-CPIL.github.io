"""Program for the no-isosceles set task using greedy with hill climbing and precomputed distances."""
import random
import numpy as np

N = 64

def construct():
    # Precompute points and distances
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    coords = np.array(points)
    
    # Distance matrix D[i, j] = squared distance between point i and point j
    # Using float64 for precision, though int32 is sufficient for squared distances up to 64^2+64^2 ~ 8000
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)
    
    # Helper to get index from point
    # point (x, y) -> y * N + x
    def get_idx(p):
        return p[1] * N + p[0]

    def can_add(p_idx, S_indices, distances_from):
        # S_indices is a list of indices
        # distances_from is a dict: idx -> set of distances to other points in S
        
        # 1. Check if p_idx is apex
        # Get distances from p_idx to all points in S
        # We need to check if any distance is repeated
        seen_dists = set()
        for a_idx in S_indices:
            d = D[p_idx, a_idx]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
        # 2. Check if any a in S is apex
        # For each a in S, check if dist(p, a) is already in distances_from[a]
        for a_idx in S_indices:
            d = D[p_idx, a_idx]
            if d in distances_from[a_idx]:
                return False
                
        return True

    def add_point(p_idx, S_indices, distances_from):
        S_indices.append(p_idx)
        distances_from[p_idx] = set()
        for a_idx in S_indices[:-1]:
            d = D[p_idx, a_idx]
            distances_from[a_idx].add(d)
            distances_from[p_idx].add(d)

    def remove_point(r_idx, S_indices, distances_from):
        idx_pos = S_indices.index(r_idx)
        S_indices.pop(idx_pos)
        for a_idx in S_indices:
            d = D[r_idx, a_idx]
            distances_from[a_idx].discard(d)
        del distances_from[r_idx]

    def greedy(order_indices):
        S_indices = []
        dist_from = {}
        for p_idx in order_indices:
            if can_add(p_idx, S_indices, dist_from):
                add_point(p_idx, S_indices, dist_from)
        return S_indices, dist_from

    def hill_climb(S_indices, dist_from, max_steps=200):
        best_S = S_indices[:]
        best_dist = {k: v.copy() for k, v in dist_from.items()}
        best_size = len(best_S)
        
        for _ in range(max_steps):
            if not best_S:
                break
            
            # Randomly pick a point to remove
            r_idx = random.choice(best_S)
            
            exp_S = best_S[:]
            exp_dist = {k: v.copy() for k, v in best_dist.items()}
            
            remove_point(r_idx, exp_S, exp_dist)
            
            # Try to add any point that fits
            # To speed up, we can shuffle the list of candidates or just iterate all
            # Since N^2 is 4096, iterating all is fine
            for p_idx in range(num_points):
                if p_idx in exp_S:
                    continue
                if can_add(p_idx, exp_S, exp_dist):
                    add_point(p_idx, exp_S, exp_dist)
            
            if len(exp_S) > best_size:
                best_S = exp_S
                best_dist = exp_dist
                best_size = len(exp_S)
                
        return best_S

    best_set_indices = []
    best_size = 0
    
    # Try multiple random orders
    for seed in range(50):
        random.seed(seed)
        order = list(range(num_points))
        random.shuffle(order)
        
        S, dist_from = greedy(order)
        S = hill_climb(S, dist_from, max_steps=100)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set_indices = S[:]
            
    # Convert indices back to points
    best_set = [points[idx] for idx in best_set_indices]
    return best_set
