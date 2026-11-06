import time
import random
import math

_START = time.time()
_DEADLINE = _START + 140.0

def construct():
    n = 16
    target_edges = 65
    
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Initialize with K_8,8 + (0,1)
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
    add_edge(0, 1)
    
    edges = []
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append((i, j))
            else:
                non_edges.append((i, j))
                
    def get_score_and_worst_tri():
        deg = [bin(a).count('1') for a in adj]
        max_sum = 0
        worst_tri = None
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    continue
                for w in range(v+1, n):
                    if (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
        return max_sum, worst_tri

    current_score, _ = get_score_and_worst_tri()
    best_score = current_score
    best_adj = adj.copy()
    best_edges = edges.copy()
    best_non_edges = non_edges.copy()
    
    rng = random.Random(42)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < _DEADLINE:
        if not edges or not non_edges:
            break
            
        # Generate Move
        if rng.random() < 0.5:
            # Targeted: remove edge from worst triangle
            _, worst_tri = get_score_and_worst_tri()
            if worst_tri:
                u, v, w = worst_tri
                tri_edges = [(u, v), (v, w), (u, w)]
                rem_u, rem_v = rng.choice(tri_edges)
            else:
                idx_rm = rng.randrange(len(edges))
                rem_u, rem_v = edges[idx_rm]
            # We need the index of (rem_u, rem_v) in edges to update the list
            # Linear search is O(E) ~ 65. Acceptable.
            idx_rm = -1
            for i, e in enumerate(edges):
                if e == (rem_u, rem_v):
                    idx_rm = i
                    break
            if idx_rm == -1:
                # Should not happen, but fallback
                idx_rm = rng.randrange(len(edges))
                rem_u, rem_v = edges[idx_rm]
        else:
            idx_rm = rng.randrange(len(edges))
            rem_u, rem_v = edges[idx_rm]
            
        idx_add = rng.randrange(len(non_edges))
        add_u, add_v = non_edges[idx_add]
        
        # Perform Swap on adj
        remove_edge(rem_u, rem_v)
        add_edge(add_u, add_v)
        
        # Update Lists
        edges[idx_rm] = (add_u, add_v)
        non_edges[idx_add] = (rem_u, rem_v)
        
        new_score, _ = get_score_and_worst_tri()
        
        # Acceptance
        if new_score < best_score:
            best_score = new_score
            best_adj = adj.copy()
            best_edges = edges.copy()
            best_non_edges = non_edges.copy()
            
        if new_score <= current_score:
            current_score = new_score
        else:
            delta = new_score - current_score
            if rng.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                remove_edge(add_u, add_v)
                add_edge(rem_u, rem_v)
                edges[idx_rm] = (rem_u, rem_v)
                non_edges[idx_add] = (add_u, add_v)
                current_score = get_score_and_worst_tri()[0]
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
