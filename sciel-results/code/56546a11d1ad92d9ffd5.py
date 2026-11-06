import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for quad in k4s:
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to K4s
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def run_search(K, time_limit, seed_offset=0):
        # State
        colors = [0] * num_edges
        # k4_counts[k4_idx][color] = count
        k4_counts = [[0] * K for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        cost = 0
        best_cost = cost
        best_colors = colors[:]
        
        def init_state():
            nonlocal colors, k4_counts, k4_distinct, cost
            colors = [random.randint(0, K - 1) for _ in range(num_edges)]
            k4_counts = [[0] * K for _ in range(num_k4s)]
            k4_distinct = [0] * num_k4s
            cost = 0
            
            for k4_idx in range(num_k4s):
                e_indices = k4_edge_indices[k4_idx]
                for ei in e_indices:
                    c = colors[ei]
                    k4_counts[k4_idx][c] += 1
                # Count distinct
                distinct = 0
                for c in range(K):
                    if k4_counts[k4_idx][c] > 0:
                        distinct += 1
                k4_distinct[k4_idx] = distinct
                if distinct < 5:
                    cost += 1
                    
        def targeted_search(max_iterations):
            """Local search focusing on violating K4s."""
            nonlocal cost, best_cost, best_colors
            no_improve_count = 0
            max_no_improve = 5000
            
            for _ in range(max_iterations):
                if time.time() > time_limit:
                    break
                    
                violating = [k4_idx for k4_idx in range(num_k4s) if k4_distinct[k4_idx] < 5]
                if not violating:
                    return True
                    
                # Pick a random violating K4
                k4_idx = random.choice(violating)
                e_indices = k4_edge_indices[k4_idx]
                
                # Try to fix by changing one edge in this K4
                best_delta = 1  # We need delta < 1 to improve
                best_e = -1
                best_c = -1
                
                for ei in e_indices:
                    old_c = colors[ei]
                    for new_c in range(K):
                        if new_c == old_c:
                            continue
                        
                        # Calculate delta
                        delta = 0
                        for k4_i in edge_to_k4s[ei]:
                            old_dist = k4_distinct[k4_i]
                            old_count = k4_counts[k4_i][old_c]
                            new_count = k4_counts[k4_i][new_c]
                            
                            new_dist = old_dist
                            if old_count == 1:
                                new_dist -= 1
                            if new_count == 0:
                                new_dist += 1
                            
                            if old_dist < 5 and new_dist >= 5:
                                delta -= 1
                            elif old_dist >= 5 and new_dist < 5:
                                delta += 1
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_e = ei
                            best_c = new_c
                            if delta <= -2:
                                break
                    if best_delta <= -2:
                        break
                
                if best_e >= 0 and best_delta < 0:
                    # Apply the change
                    old_c = colors[best_e]
                    for k4_i in edge_to_k4s[best_e]:
                        k4_counts[k4_i][old_c] -= 1
                        k4_counts[k4_i][best_c] += 1
                        old_dist = k4_distinct[k4_i]
                        new_dist = old_dist
                        if k4_counts[k4_i][old_c] == 0:
                            new_dist -= 1
                        if k4_counts[k4_i][best_c] == 1:
                            new_dist += 1
                        k4_distinct[k4_i] = new_dist
                    
                    colors[best_e] = best_c
                    cost += best_delta
                    no_improve_count = 0
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
                        
                    if cost == 0:
                        return True
                else:
                    # No improvement found; try random move to escape local optimum
                    no_improve_count += 1
                    if no_improve_count % 100 == 0:
                        # Random restart of a few edges
                        for _ in range(3):
                            e_idx = random.randint(0, num_edges - 1)
                            new_color = random.randint(0, K - 1)
                            if new_color != colors[e_idx]:
                                old_color = colors[e_idx]
                                for k4_idx2 in edge_to_k4s[e_idx]:
                                    k4_counts[k4_idx2][old_color] -= 1
                                    k4_counts[k4_idx2][new_color] += 1
                                    old_dist = k4_distinct[k4_idx2]
                                    new_dist = old_dist
                                    if k4_counts[k4_idx2][old_color] == 0:
                                        new_dist -= 1
                                    if k4_counts[k4_idx2][new_color] == 1:
                                        new_dist += 1
                                    k4_distinct[k4_idx2] = new_dist
                                colors[e_idx] = new_color
                        # Recompute cost to stay accurate
                        cost = sum(1 for k4_idx2 in range(num_k4s) if k4_distinct[k4_idx2] < 5)
                    
                    if no_improve_count > max_no_improve:
                        return False
                        
            return cost == 0

        init_state()
        if cost == 0:
            return colors[:]
            
        best_cost = cost
        best_colors = colors[:]
        
        # Multiple restarts with targeted search
        while time.time() < time_limit:
            remaining = time_limit - time.time()
            if remaining < 1:
                break
                
            # Try targeted search
            iterations = min(50000, int(remaining * 5000))
            if iterations < 1000:
                iterations = 1000
                
            success = targeted_search(iterations)
            if success:
                return colors[:]
                
            # If stuck, restart
            if time.time() < time_limit - 2:
                init_state()
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                if cost == 0:
                    return colors[:]
                    
        return best_colors if best_cost == 0 else None

    # Try K=13, 14, 15
    # Time allocation: prioritize 13, then 14, then 15
    
    # If 13 fails, try 14.
    # If 14 fails, try 15.
    
    # We should check if time is running out before starting a new K.
    
    for K in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        # Allocate remaining time minus safety margin
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
            
        # Give it a chunk of time
        # If we have lots of time, give more to 13?
        # Let's just give 40s to each if available.
        limit = min(remaining, 40.0)
        
        res = run_search(K, limit)
        if res:
            return res
            
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
