"""Program for the no-isosceles set task using Iterated Local Search with optimized checks."""
import random
import numpy as np
import time

N = 64

def construct():
    # Precompute points and distances
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    coords = np.array(points)
    
    # Distance matrix D[i, j] = squared distance between point i and point j
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)
    
    def can_add(p_idx, S_set, S_list, distances_from):
        # 1. Check if p_idx is apex
        seen_dists = set()
        for a_idx in S_list:
            d = D[p_idx, a_idx]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
        # 2. Check if any a in S is apex
        for a_idx in S_list:
            d = D[p_idx, a_idx]
            if d in distances_from[a_idx]:
                return False
                
        return True

    def add_point(p_idx, S_set, S_list, distances_from):
        S_list.append(p_idx)
        S_set.add(p_idx)
        distances_from[p_idx] = set()
        for a_idx in S_list[:-1]:
            d = D[p_idx, a_idx]
            distances_from[a_idx].add(d)
            distances_from[p_idx].add(d)

    def remove_point(r_idx, S_set, S_list, distances_from):
        S_list.remove(r_idx)
        S_set.remove(r_idx)
        for a_idx in S_list:
            d = D[r_idx, a_idx]
            distances_from[a_idx].discard(d)
        del distances_from[r_idx]

    def greedy(order_indices):
        S_list = []
        S_set = set()
        dist_from = {}
        for p_idx in order_indices:
            if can_add(p_idx, S_set, S_list, dist_from):
                add_point(p_idx, S_set, S_list, dist_from)
        return S_list, S_set, dist_from

    def hill_climb(S_list, S_set, dist_from, max_steps=200):
        best_S_list = S_list[:]
        best_S_set = set(S_set)
        best_dist = {k: v.copy() for k, v in dist_from.items()}
        best_size = len(best_S_list)
        
        for _ in range(max_steps):
            if not best_S_list:
                break
            
            # Randomly pick a point to remove
            r_idx = random.choice(best_S_list)
            
            # Create copies for experimentation
            exp_S_list = best_S_list[:]
            exp_S_set = set(best_S_set)
            exp_dist = {k: v.copy() for k, v in best_dist.items()}
            
            remove_point(r_idx, exp_S_set, exp_S_list, exp_dist)
            
            # Try to add any point that fits
            # Iterate through all points
            for p_idx in range(num_points):
                if p_idx in exp_S_set:
                    continue
                if can_add(p_idx, exp_S_set, exp_S_list, exp_dist):
                    add_point(p_idx, exp_S_set, exp_S_list, exp_dist)
            
            if len(exp_S_list) > best_size:
                best_S_list = exp_S_list
                best_S_set = exp_S_set
                best_dist = exp_dist
                best_size = len(exp_S_list)
                
        return best_S_list, best_S_set, best_dist

    start_time = time.time()
    timeout = 290.0
    best_set_indices = []
    best_size = 0
    
    # Iterated Local Search
    restarts = 0
    while time.time() - start_time < timeout:
        restarts += 1
        random.seed(restarts)
        
        # Greedy initialization
        order = list(range(num_points))
        random.shuffle(order)
        S_list, S_set, dist_from = greedy(order)
        
        # Local Search
        S_list, S_set, dist_from = hill_climb(S_list, S_set, dist_from, max_steps=100)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set_indices = S_list[:]
            
    # Convert indices back to points
    best_set = [points[idx] for idx in best_set_indices]
    return best_set
