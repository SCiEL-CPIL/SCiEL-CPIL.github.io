import time
import random
import math
from itertools import combinations

DEADLINE = 145  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    Returns (max_sum, list_of_worst_triangles)
    """
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj_masks[i]).count('1')
    
    max_sum = 0
    max_triangles = []

    for i in range(n):
        mi = adj_masks[i]
        # Only consider j > i to avoid duplicates
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)

        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb

            # Triangles containing i and j must have k in common neighbors
            common = mi & adj_masks[j]
            # Restrict k > j to avoid duplicates
            mask_k = common & ~((1 << (j + 1)) - 1)
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                mask_k ^= lsb_k
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
                    max_triangles = [(i, j, k)]
                elif s == max_sum:
                    max_triangles.append((i, j, k))

    return max_sum, max_triangles

def _construct_edges_from_masks(adj_masks, n):
    edges = []
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            edges.append([i, j])
            mask_j_greater ^= lsb
    return edges

def _generate_initial_graphs(n, target_edges):
    """Generate diverse initial graphs with exactly target_edges edges."""
    graphs = []
    
    # 1. K_{8,8} + 1 edge (baseline)
    adj1 = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj1[u] |= (1 << v)
            adj1[v] |= (1 << u)
    adj1[0] |= (1 << 1)
    adj1[1] |= (1 << 0)
    graphs.append(adj1)
    
    # 2. Random graphs with balanced degrees
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for seed in [1, 2, 3, 4, 5]:
        rng = random.Random(seed)
        adj = [0] * n
        degs = [0] * n
        shuffled_edges = list(all_possible)
        rng.shuffle(shuffled_edges)
        
        # Greedy add to balance degrees
        # Target degrees: 8 or 9
        # 2 vertices of 9, 14 of 8
        target_degs = [9]*2 + [8]*14
        rng.shuffle(target_degs)
        
        # Sort edges by some heuristic? Or just random.
        # To ensure exactly target_edges, we just pick random ones.
        # But we want them to be "good".
        
        # Let's just pick random edges for now, local search will fix it.
        for u, v in shuffled_edges[:target_edges]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        graphs.append(adj)
        
    return graphs

def _local_search(adj_masks, n, time_budget, seed=0):
    """
    Greedy local search: try all single swaps that improve score.
    """
    start_time = time.time()
    current_score, _ = _compute_score(n, adj_masks)
    best_score = current_score
    best_adj_masks = list(adj_masks)
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices for faster access?
    # We maintain a set of current edges and non-edges.
    current_edge_set = set()
    non_edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if (adj_masks[u] >> v) & 1:
                current_edge_set.add((u, v))
            else:
                non_edge_set.add((u, v))
    
    non_edge_list = list(non_edge_set)
    edge_list = list(current_edge_set)
    
    improved = True
    while improved and (time.time() - start_time < time_budget):
        improved = False
        current_score, _ = _compute_score(n, best_adj_masks)
        
        # Rebuild sets from best_adj_masks to ensure consistency
        current_edge_set = set()
        non_edge_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if (best_adj_masks[u] >> v) & 1:
                    current_edge_set.add((u, v))
                else:
                    non_edge_set.add((u, v))
        
        non_edge_list = list(non_edge_set)
        
        # Try all swaps
        best_swap_score = best_score
        best_swap = None
        
        for rem_edge in edge_list:
            if time.time() - start_time >= time_budget:
                break
            rem_u, rem_v = rem_edge
            
            for add_edge in non_edge_list:
                if time.time() - start_time >= time_budget:
                    break
                add_u, add_v = add_edge
                
                # Apply swap to a copy
                test_masks = list(best_adj_masks)
                test_masks[rem_u] ^= (1 << rem_v)
                test_masks[rem_v] ^= (1 << rem_u)
                test_masks[add_u] ^= (1 << add_v)
                test_masks[add_v] ^= (1 << add_u)
                
                new_score, _ = _compute_score(n, test_masks)
                
                if new_score < best_swap_score:
                    best_swap_score = new_score
                    best_swap = (rem_edge, add_edge, test_masks)
        
        if best_swap:
            rem_edge, add_edge, new_masks = best_swap
            best_adj_masks = new_masks
            best_score = best_swap_score
            improved = True
    
    return best_score, best_adj_masks

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Generate diverse initial graphs
    initial_graphs = _generate_initial_graphs(n, target_edges)
    
    best_overall_score = float('inf')
    best_overall_adj = None
    
    # Evaluate each initial graph
    for idx, adj in enumerate(initial_graphs):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        score, _ = _compute_score(n, adj)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_adj = list(adj)
    
    # Run local search from the best initial graph and a few others
    # Since local search is deterministic and potentially slow, we run it on a few candidates
    time_per_ls = (DEADLINE - 10) / 3  # Reserve 10s for final checks
    
    # Sort initial graphs by score to prioritize better starting points
    scored_graphs = []
    for adj in initial_graphs:
        score, _ = _compute_score(n, adj)
        scored_graphs.append((score, adj))
    scored_graphs.sort(key=lambda x: x[0])
    
    for idx in range(min(3, len(scored_graphs))):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        score, adj = scored_graphs[idx]
        adj_copy = list(adj)
        
        ls_score, ls_adj = _local_search(adj_copy, n, time_per_ls)
        
        if ls_score < best_overall_score:
            best_overall_score = ls_score
            best_overall_adj = ls_adj
    
    # If we have time, do a second round of local search on the best found
    if best_overall_adj is not None and time.time() - start_time < DEADLINE - 5:
        polish_time = min(10, DEADLINE - 5 - (time.time() - start_time))
        if polish_time > 2:
            final_score, final_adj = _local_search(best_overall_adj, n, polish_time)
            if final_score < best_overall_score:
                best_overall_score = final_score
                best_overall_adj = final_adj
    
    # Fallback
    if best_overall_adj is None:
        best_overall_adj = initial_graphs[0]
    
    best_edges = _construct_edges_from_masks(best_overall_adj, n)
    
    # Verify edge count
    assert len(best_edges) == target_edges, f"Expected {target_edges} edges, got {len(best_edges)}"
    
    return {'n': n, 'edges': best_edges}
