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
    
    # Precompute: for each edge, list of (k4_idx, other 5 edges in that K4)
    # This helps with faster violation checking
    
    def run_search(K, time_limit, init_seed=None):
        if init_seed is not None:
            random.seed(init_seed)
        
        # State
        colors = [0] * num_edges
        k4_counts = [[0] * K for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        violating_k4s = set()
        cost = 0
        
        def init_state():
            nonlocal colors, k4_counts, k4_distinct, violating_k4s, cost
            # Use a structured initialization: color edge (i,j) based on (i+j) % K or similar
            # Try modular coloring
            colors = [0] * num_edges
            for idx, (i, j) in enumerate(edges):
                colors[idx] = (i + j) % K
            
            k4_counts = [[0] * K for _ in range(num_k4s)]
            k4_distinct = [0] * num_k4s
            violating_k4s = set()
            cost = 0
            
            for k4_idx in range(num_k4s):
                e_indices = k4_edge_indices[k4_idx]
                for ei in e_indices:
                    c = colors[ei]
                    k4_counts[k4_idx][c] += 1
                distinct = sum(1 for c in range(K) if k4_counts[k4_idx][c] > 0)
                k4_distinct[k4_idx] = distinct
                if distinct < 5:
                    violating_k4s.add(k4_idx)
                    cost += 1
                    
        def compute_delta(e_idx, old_color, new_color):
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_dist = k4_distinct[k4_idx]
                old_count = k4_counts[k4_idx][old_color]
                new_count = k4_counts[k4_idx][new_color]
                
                new_dist = old_dist
                if old_count == 1:
                    new_dist -= 1
                if new_count == 0:
                    new_dist += 1
                    
                if old_dist < 5 and new_dist >= 5:
                    delta -= 1
                elif old_dist >= 5 and new_dist < 5:
                    delta += 1
            return delta

        def apply_change(e_idx, new_color):
            nonlocal cost
            old_color = colors[e_idx]
            if old_color == new_color:
                return 0
                
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                old_dist = k4_distinct[k4_idx]
                
                k4_counts[k4_idx][old_color] -= 1
                k4_counts[k4_idx][new_color] += 1
                
                new_dist = old_dist
                if k4_counts[k4_idx][old_color] == 0 and old_dist > 0:
                    new_dist -= 1
                if k4_counts[k4_idx][new_color] == 1:
                    new_dist += 1
                    
                k4_distinct[k4_idx] = new_dist
                
                if old_dist < 5 and new_dist >= 5:
                    delta -= 1
                elif old_dist >= 5 and new_dist < 5:
                    delta += 1
                    
                if old_dist < 5 and new_dist >= 5:
                    violating_k4s.discard(k4_idx)
                elif old_dist >= 5 and new_dist < 5:
                    violating_k4s.add(k4_idx)
                    
            colors[e_idx] = new_color
            cost += delta
            return delta

        init_state()
        if cost == 0:
            return colors[:]
            
        best_cost = cost
        best_colors = colors[:]
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < time_limit:
            # Targeted search: pick an edge from a violating K4
            if violating_k4s and random.random() < 0.7:
                # Pick a random violating K4
                k4_idx = random.choice(list(violating_k4s))
                e_indices = k4_edge_indices[k4_idx]
                e_idx = random.choice(e_indices)
            else:
                e_idx = random.randint(0, num_edges - 1)
                
            new_color = random.randint(0, K - 1)
            if new_color == colors[e_idx]:
                continue
                
            old_color = colors[e_idx]
            delta = compute_delta(e_idx, old_color, new_color)
            
            if delta <= 0:
                apply_change(e_idx, new_color)
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if cost == 0:
                    return colors[:]
            else:
                # Occasional random acceptance to escape local optima
                if random.random() < 0.05:
                    apply_change(e_idx, new_color)
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                # Restart with new random initialization
                init_state()
                best_cost = cost
                best_colors = colors[:]
                no_improve_count = 0
                if cost == 0:
                    return colors[:]
                    
        return best_colors if best_cost == 0 else None

    # Try K values from 13 upward
    # Allocate time: 50s for K=13, 40s for K=14, 30s for K=15
    time_alloc = {13: 50.0, 14: 40.0, 15: 30.0}
    
    for K in [13, 14, 15]:
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
            
        limit = min(time_alloc.get(K, 30.0), remaining)
        if limit < 3:
            break
            
        # Try multiple restarts with different seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time() - 5
            if remaining < 3:
                break
            chunk_limit = min(limit / 4.0, remaining)
            if chunk_limit < 2:
                break
                
            res = run_search(K, chunk_limit, init_seed=seed)
            if res:
                return res
                
    # Fallback: try a simple modular coloring that might work
    # Try (i * 3 + j * 5) % 13
    colors = [(i * 3 + j * 5) % 13 for i in range(16) for j in range(i + 1, 16)]
    # Verify
    valid = True
    for quad in k4s:
        edge_colors = set()
        for i in range(4):
            for j in range(i + 1, 4):
                ei = edge_index[(quad[i], quad[j])]
                edge_colors.add(colors[ei])
        if len(edge_colors) < 5:
            valid = False
            break
    if valid:
        return colors
        
    # Last resort fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
