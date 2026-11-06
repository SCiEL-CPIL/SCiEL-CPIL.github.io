import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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
            
    def run_search(K, time_limit):
        # State
        colors = [0] * num_edges
        # k4_counts[k4_idx][color] = count
        # Use list of lists for speed
        k4_counts = [[0] * K for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        cost = 0
        
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
                    
        def apply_change(e_idx, new_color):
            nonlocal cost
            old_color = colors[e_idx]
            if old_color == new_color:
                return 0
                
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_dist = k4_distinct[k4_idx]
                
                # Update counts
                k4_counts[k4_idx][old_color] -= 1
                k4_counts[k4_idx][new_color] += 1
                
                # Update distinct
                new_dist = old_dist
                if k4_counts[k4_idx][old_color] == 0 and old_dist > 0:
                    new_dist -= 1
                if k4_counts[k4_idx][new_color] == 1:
                    new_dist += 1
                    
                k4_distinct[k4_idx] = new_dist
                
                # Update cost
                if old_dist < 5 and new_dist >= 5:
                    delta -= 1
                elif old_dist >= 5 and new_dist < 5:
                    delta += 1
                    
            colors[e_idx] = new_color
            cost += delta
            return delta

        init_state()
        if cost == 0:
            return colors[:]
            
        best_cost = cost
        best_colors = colors[:]
        
        # Local Search
        while time.time() < time_limit:
            e_idx = random.randint(0, num_edges - 1)
            new_color = random.randint(0, K - 1)
            
            if new_color == colors[e_idx]:
                continue
                
            # We need to calculate delta without applying first?
            # Or apply, check, revert if bad?
            # Reverting is expensive.
            # Let's calculate delta first.
            
            old_color = colors[e_idx]
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_dist = k4_distinct[k4_idx]
                
                # Simulate
                old_count = k4_counts[k4_idx][old_color]
                new_count = k4_counts[k4_idx][new_color]
                
                new_dist = old_dist
                if old_count == 1: # color disappears
                    new_dist -= 1
                if new_count == 0: # color appears
                    new_dist += 1
                    
                if old_dist < 5 and new_dist >= 5:
                    delta -= 1
                elif old_dist >= 5 and new_dist < 5:
                    delta += 1
                    
            if delta <= 0:
                # Apply
                for k4_idx in edge_to_k4s[e_idx]:
                    k4_counts[k4_idx][old_color] -= 1
                    k4_counts[k4_idx][new_color] += 1
                    old_dist = k4_distinct[k4_idx]
                    new_dist = old_dist
                    if k4_counts[k4_idx][old_color] == 0:
                        new_dist -= 1
                    if k4_counts[k4_idx][new_color] == 1:
                        new_dist += 1
                    k4_distinct[k4_idx] = new_dist
                    
                colors[e_idx] = new_color
                cost += delta
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    
                if cost == 0:
                    return colors[:]
                    
            # Optional: Random restart if stuck
            # We can add a counter
            
        return best_colors if best_cost == 0 else None

    # Try K=13, 14, 15
    # Time allocation: 45s each?
    # Total 135s.
    
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
