import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    # All possible edges
    all_edges = list(combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute edge endpoints
    edge_u = np.array([e[0] for e in all_edges])
    edge_v = np.array([e[1] for e in all_edges])
    
    # Adjacency bitmask for each vertex (as integer bits)
    # adj_mask[i] has bit j set if edge (i,j) exists
    
    def build_adj_mask(edge_set):
        """Build adjacency bitmasks from a set of edge indices."""
        masks = [0] * n
        for idx in edge_set:
            u = all_edges[idx][0]
            v = all_edges[idx][1]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def max_triangle_sum(masks):
        """Compute max degree sum over all triangles using bitmasks."""
        deg = [bin(m).count('1') for m in masks]
        max_s = 0
        for u in range(n):
            for v in range(u+1, n):
                if not (masks[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = masks[u] & masks[v]
                # Only consider w > v to avoid duplicates
                common &= ~((1 << (v+1)) - 1)
                while common:
                    lsb = common & (-common)
                    w = lsb.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_s:
                        max_s = s
                    common ^= lsb
        return max_s
    
    # Initial configuration: K_{8,8} + one internal edge
    # K_{8,8} edges: all (u,v) where u in [0,8), v in [8,16)
    k88_edges = set()
    for u in range(8):
        for v in range(8, 16):
            k88_edges.add(edge_to_idx[(u, v)])
    
    # Add internal edge (0,1)
    init_edge_set = set(k88_edges)
    init_edge_set.add(edge_to_idx[(0, 1)])
    
    assert len(init_edge_set) == target_edges
    
    init_masks = build_adj_mask(init_edge_set)
    init_score = max_triangle_sum(init_masks)
    
    best_score = init_score
    best_edge_set = set(init_edge_set)
    
    # Simulated annealing
    # Move: remove one edge, add one different edge (keep count at 65)
    
    # Precompute for each edge, which other edges could replace it
    # We'll just pick random pairs
    
    current_edge_set = set(init_edge_set)
    current_masks = build_adj_mask(current_edge_set)
    current_score = init_score
    
    # SA parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.9995
    max_iterations = 500000
    
    T = T_start
    rng = np.random.default_rng(42)
    
    iter_count = 0
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Pick a random edge to remove
        remove_idx = rng.integers(0, len(all_edges))
        
        # Pick a random edge to add (different from removed)
        add_idx = rng.integers(0, len(all_edges))
        while add_idx == remove_idx:
            add_idx = rng.integers(0, len(all_edges))
        
        # Check if removal is valid (edge exists) and addition is valid (edge doesn't exist)
        if remove_idx not in current_edge_set:
            continue
        if add_idx in current_edge_set:
            continue
        
        # Apply move
        new_edge_set = set(current_edge_set)
        new_edge_set.discard(remove_idx)
        new_edge_set.add(add_idx)
        
        if len(new_edge_set) != target_edges:
            continue
        
        new_masks = build_adj_mask(new_edge_set)
        new_score = max_triangle_sum(new_masks)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_edge_set = new_edge_set
            current_masks = new_masks
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(current_edge_set)
        
        T = max(T * cooling_rate, T_end)
        iter_count += 1
    
    # Final hill-climbing polish: only accept improvements
    if time.time() < deadline - 5:
        current_edge_set = set(best_edge_set)
        current_masks = build_adj_mask(current_edge_set)
        current_score = best_score
        
        polish_iter = 0
        for _ in range(100000):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try all single swaps that improve
            edges_list = list(current_edge_set)
            rng.shuffle(edges_list)
            
            for remove_idx in edges_list:
                if time.time() > deadline - 3:
                    break
                
                # Try adding each non-present edge
                for add_idx in range(len(all_edges)):
                    if add_idx in current_edge_set:
                        continue
                    
                    new_edge_set = set(current_edge_set)
                    new_edge_set.discard(remove_idx)
                    new_edge_set.add(add_idx)
                    
                    if len(new_edge_set) != target_edges:
                        continue
                    
                    new_masks = build_adj_mask(new_edge_set)
                    new_score = max_triangle_sum(new_masks)
                    
                    if new_score < current_score:
                        current_edge_set = new_edge_set
                        current_masks = new_masks
                        current_score = new_score
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(current_edge_set)
                        
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Return best found
    edges_out = [[all_edges[e][0], all_edges[e][1]] for e in best_edge_set]
    return {'n': n, 'edges': edges_out}
