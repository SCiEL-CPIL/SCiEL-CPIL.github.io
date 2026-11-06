<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _build_adjacency(n, edges):
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _compute_score(n, adj):
    """Compute the max triangle degree sum."""
    deg = adj.sum(axis=1)
    max_score = 0
    # Find all triangles
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                common = np.where(adj[i] & adj[j])[0]
                for k in common:
                    if k > j:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_score:
                            max_score = s
    return max_score

def _get_all_edges(n):
    return list(combinations(range(n), 2))

def _get_non_edges(n, adj):
    return [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]

def _local_search(n, edges, max_time=100):
    start_time = time.time()
    deadline = start_time + max_time
    
    adj = _build_adjacency(n, edges)
    deg = adj.sum(axis=1)
    current_score = _compute_score(n, adj)
    best_score = current_score
    best_edges = list(edges)
    
    all_edges = _get_all_edges(n)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Precompute for speed
    n2 = n * n
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, adj)
        if not non_edges:
            break
        
        improved = False
        # Try edge swaps: remove one edge, add another
        # Focus on swaps that might reduce max triangle degree sum
        for e_remove in list(edge_set):
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            # Try adding each non-edge
            for e_add in non_edges:
                u2, v2 = e_add
                if (u2, v2) in edge_set or (v2, u2) in edge_set:
                    continue
                # Skip if this would create parallel structure issues
                # Check if swap is valid (simple graph)
                if u2 == u1 and v2 == v1:
                    continue
                
                # Tentative swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_deg = adj.sum(axis=1)
                new_score = _compute_score(n, adj)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in edge_set | {tuple(sorted(e_add))} - {tuple(sorted(e_remove))}]
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    continue
                
                # If improved, update edge_set
                edge_set.remove(tuple(sorted(e_remove)))
                edge_set.add(tuple(sorted(e_add)))
                break
        
        if not improved:
            # Try single edge moves if no swaps help
            found_move = False
            for e_remove in list(edge_set):
                if time.time() >= deadline:
                    break
                u1, v1 = e_remove
                for e_add in non_edges:
                    u2, v2 = e_add
                    if (u2, v2) in edge_set or (v2, u2) in edge_set:
                        continue
                    if u2 == u1 and v2 == v1:
                        continue
                    
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_score = _compute_score(n, adj)
                    
                    if new_score < best_score:
                        best_score = new_score
                        edge_set.remove(tuple(sorted(e_remove)))
                        edge_set.add(tuple(sorted(e_add)))
                        best_edges = [list(e) for e in edge_set]
                        found_move = True
                        break
                    else:
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0
                
                if found_move:
                    break
            
            if not found_move:
                break
    
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    
    # Start from K_{8,8} + one internal edge (balanced construction)
    initial_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    # Run local search
    best_edges, best_score = _local_search(n, initial_edges, max_time=110)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _build_bitmasks(n, edges):
    """Build adjacency as bitmasks for fast triangle detection."""
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _compute_score_bitmask(n, masks):
    """Compute max triangle degree sum using bitmask operations."""
    deg = [bin(m).count('1') for m in masks]
    max_score = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (masks[i] >> j) & 1:
                common = masks[i] & masks[j]
                # Only consider k > j to avoid duplicates
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    k = (common & -common).bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_score:
                        max_score = s
                    common &= common - 1
    return max_score

def _local_search_bitmask(n, edges, max_time=120):
    """Fast local search using bitmask-based scoring."""
    start_time = time.time()
    deadline = start_time + max_time
    
    masks = _build_bitmasks(n, edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    current_score = _compute_score_bitmask(n, masks)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    # Precompute all possible non-edges
    all_pairs = set((i, j) for i in range(n) for j in range(i + 1, n))
    
    while time.time() < deadline:
        non_edges = all_pairs - edge_set
        if not non_edges:
            break
        
        improved = False
        
        # Try edge swaps: remove one edge, add another
        for e_remove in list(edge_set):
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            
            for e_add in list(non_edges):
                u2, v2 = e_add
                
                # Tentative swap
                masks[u1] &= ~(1 << v1)
                masks[v1] &= ~(1 << u1)
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                
                new_score = _compute_score_bitmask(n, masks)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(e_add)
                    edge_set = set(best_edge_set)
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    masks[u1] |= (1 << v1)
                    masks[v1] |= (1 << u1)
                    masks[u2] &= ~(1 << v2)
                    masks[v2] &= ~(1 << u2)
                
                if improved:
                    break
        
        if not improved:
            # Try 2-edge swaps (remove 2, add 2) for more flexibility
            found_2swap = False
            edge_list = list(edge_set)
            for idx1 in range(len(edge_list)):
                if time.time() >= deadline:
                    break
                e_rem1 = edge_list[idx1]
                u1, v1 = e_rem1
                
                for idx2 in range(idx1 + 1, len(edge_list)):
                    if time.time() >= deadline:
                        break
                    e_rem2 = edge_list[idx2]
                    u3, v3 = e_rem2
                    
                    for e_add1 in list(non_edges):
                        if time.time() >= deadline:
                            break
                        u2, v2 = e_add1
                        if e_add1 in edge_set:
                            continue
                        
                        for e_add2 in list(non_edges):
                            if time.time() >= deadline:
                                break
                            u4, v4 = e_add2
                            if e_add2 in edge_set or e_add2 == e_add1:
                                continue
                            
                            # Tentative 2-swap
                            masks[u1] &= ~(1 << v1)
                            masks[v1] &= ~(1 << u1)
                            masks[u3] &= ~(1 << v3)
                            masks[v3] &= ~(1 << u3)
                            masks[u2] |= (1 << v2)
                            masks[v2] |= (1 << u2)
                            masks[u4] |= (1 << v4)
                            masks[v4] |= (1 << u4)
                            
                            new_score = _compute_score_bitmask(n, masks)
                            
                            # Revert
                            masks[u1] |= (1 << v1)
                            masks[v1] |= (1 << u1)
                            masks[u3] |= (1 << v3)
                            masks[v3] |= (1 << u3)
                            masks[u2] &= ~(1 << v2)
                            masks[v2] &= ~(1 << u2)
                            masks[u4] &= ~(1 << v4)
                            masks[v4] &= ~(1 << u4)
                            
                            if new_score < best_score:
                                best_score = new_score
                                best_edge_set = set(edge_set)
                                best_edge_set.discard(e_rem1)
                                best_edge_set.discard(e_rem2)
                                best_edge_set.add(e_add1)
                                best_edge_set.add(e_add2)
                                edge_set = set(best_edge_set)
                                current_score = new_score
                                found_2swap = True
                                break
                        
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            
            if not found_2swap:
                break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    
    # Start from K_{6,10} + matching on the 10-side
    # Part A: vertices 0-5, Part B: vertices 6-15
    # All cross edges: 6*10 = 60
    # Plus 5 internal edges in B to make 65
    initial_edges = [[u, v] for u in range(6) for v in range(6, 16)]
    # Add 5 internal edges in B (matching)
    internal_b = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    initial_edges.extend(internal_b)
    
    # Run local search with time budget
    best_edges, best_score = _local_search_bitmask(n, initial_edges, max_time=135)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
