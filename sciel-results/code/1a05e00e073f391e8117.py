import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    n = 18
    
    # Initialize with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.add((min(a,b), max(a,b)))
    
    # Bitmask representation for speed
    # adj_mask[i] is a bitmask of neighbors of i
    def build_adj_mask(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_k4_free(edge_set):
        adj = build_adj_mask(edge_set)
        # Check all 4-combinations
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)) and \
               (adj[b] & (1 << c)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                return False
        return True

    def max_triangle_free_size(edge_set, timeout=0.5):
        adj = build_adj_mask(edge_set)
        best = 0
        start = time.time()
        
        # Branch and bound with bitmasks
        # remaining_mask: bits set for vertices not yet decided
        # current_mask: bits set for vertices in current triangle-free set
        def bt(remaining_mask, current_size, current_mask):
            nonlocal best
            if time.time() - start > timeout:
                return
            if current_size + bin(remaining_mask).count('1') <= best:
                return
            if remaining_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick first vertex in remaining
            v = (remaining_mask & -remaining_mask).bit_length() - 1
            rest_mask = remaining_mask ^ (1 << v)
            
            # Check if v can be added
            # v can be added if no triangle is formed with any two in current_mask
            # Triangle formed if there exists u, w in current_mask such that u-v, v-w, u-w are edges.
            # This is equivalent to: neighbors of v in current_mask must be an independent set in the subgraph induced by current_mask?
            # No, simpler: for each u in current_mask connected to v, u must not be connected to any w in current_mask connected to v?
            # Actually, if u, w in current_mask and u-v, v-w, u-w are edges, triangle.
            # So the set of neighbors of v in current_mask must have NO edges between them.
            
            neighbors_in_current = adj[v] & current_mask
            # Check if neighbors_in_current is independent
            is_independent = True
            temp = neighbors_in_current
            while temp:
                u_bit = temp & -temp
                u = u_bit.bit_length() - 1
                # Check if u has any neighbor in neighbors_in_current (excluding u itself, but u is not neighbor of itself)
                if adj[u] & neighbors_in_current:
                    is_independent = False
                    break
                temp ^= u_bit
            
            if is_independent:
                bt(rest_mask, current_size + 1, current_mask | (1 << v))
            
            # Exclude v
            bt(rest_mask, current_size, current_mask)
        
        bt((1 << n) - 1, 0, 0)
        return best

    current_edges = set(edges)
    current_score = max_triangle_free_size(current_edges)
    best_edges = set(current_edges)
    best_score = current_score
    
    random.seed(42)
    all_possible_edges = set(itertools.combinations(range(n), 2))
    
    while time.time() < deadline:
        # Try adding edges
        non_edges = list(all_possible_edges - current_edges)
        random.shuffle(non_edges)
        
        improved = False
        for e in non_edges[:100]:
            if time.time() > deadline:
                break
            new_edges = current_edges | {e}
            if is_k4_free(new_edges):
                s = max_triangle_free_size(new_edges)
                if s < current_score:
                    current_edges = new_edges
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_edges = set(current_edges)
                    improved = True
                    break
        
        if not improved:
            # Try removing edges
            current_edge_list = list(current_edges)
            random.shuffle(current_edge_list)
            for e in current_edge_list[:100]:
                if time.time() > deadline:
                    break
                new_edges = current_edges - {e}
                s = max_triangle_free_size(new_edges)
                if s < current_score:
                    current_edges = new_edges
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_edges = set(current_edges)
                    break

    return [[u, v] for u, v in sorted(best_edges)]
