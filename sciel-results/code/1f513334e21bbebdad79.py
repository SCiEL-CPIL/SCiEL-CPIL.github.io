import time
import random
import math
from itertools import combinations

DEADLINE = 140  # Safety margin for 150s kill

def _build_adjacency(n, edges):
    """Build adjacency sets and degree list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    return adj, deg

def _compute_score(n, edges):
    """Compute the largest degree sum over triangles using bitmask for speed."""
    adj, deg = _build_adjacency(n, edges)
    
    # Build bitmask adjacency for fast common neighbor checks
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    max_triangle_sum = 0
    for u, v in edges:
        # Find common neighbors w using bitmask AND
        common_mask = adj_mask[u] & adj_mask[v]
        # Iterate over set bits in common_mask
        mask = common_mask
        while mask:
            # Get lowest set bit
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            mask ^= lb
            
            # Only count each triangle once: enforce u < v < w ordering
            if u < v < w:
                s = deg[u] + deg[v] + deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif u < w < v:
                s = deg[u] + deg[v] + deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif v < u < w:
                s = deg[u] + deg[v] + deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif v < w < u:
                s = deg[u] + deg[v] + deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif w < u < v:
                s = deg[u] + deg[v] + deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif w < v < u:
                s = deg[u] + deg[v] + deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
    
    return max_triangle_sum

def _compute_score_incremental(n, edges_set, adj, deg, adj_mask, e1, e2):
    """
    Compute new score after removing e1 and adding e2, using precomputed state.
    Returns (new_score, new_adj, new_deg, new_adj_mask).
    """
    u1, v1 = e1
    u2, v2 = e2
    
    # Update degrees
    new_deg = list(deg)
    new_deg[u1] -= 1
    new_deg[v1] -= 1
    new_deg[u2] += 1
    new_deg[v2] += 1
    
    # Update adjacency masks
    new_adj_mask = list(adj_mask)
    new_adj_mask[u1] &= ~(1 << v1)
    new_adj_mask[v1] &= ~(1 << u1)
    new_adj_mask[u2] |= (1 << v2)
    new_adj_mask[v2] |= (1 << u2)
    
    # Update adjacency sets (shallow copy then modify)
    new_adj = [s.copy() for s in adj]
    new_adj[u1].discard(v1)
    new_adj[v1].discard(u1)
    new_adj[u2].add(v2)
    new_adj[v2].add(u2)
    
    # Compute max triangle sum with new degrees and masks
    max_triangle_sum = 0
    for u, v in edges_set:
        common_mask = new_adj_mask[u] & new_adj_mask[v]
        mask = common_mask
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            mask ^= lb
            
            if u < v < w:
                s = new_deg[u] + new_deg[v] + new_deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif u < w < v:
                s = new_deg[u] + new_deg[v] + new_deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif v < u < w:
                s = new_deg[u] + new_deg[v] + new_deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif v < w < u:
                s = new_deg[u] + new_deg[v] + new_deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif w < u < v:
                s = new_deg[u] + new_deg[v] + new_deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
            elif w < v < u:
                s = new_deg[u] + new_deg[v] + new_deg[w]
                if s > max_triangle_sum:
                    max_triangle_sum = s
    
    return max_triangle_sum, new_adj, new_deg, new_adj_mask

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _local_search_polish(n, edges_set, max_iterations=20000):
    """
    Final high-precision polish: try all single-edge swaps that improve score.
    """
    adj, deg = _build_adjacency(n, list(edges_set))
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    current_score = _compute_score(n, list(edges_set))
    all_pairs = _get_all_pairs(n)
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        if time.time() > DEADLINE - 3:
            break
        
        best_swap = None
        best_new_score = current_score
        
        # Try removing each edge
        for e1 in list(edges_set):
            # Try adding each non-edge
            for e2 in all_pairs:
                if e2 in edges_set or e2 == e1:
                    continue
                
                new_score, _, _, _ = _compute_score_incremental(
                    n, edges_set, adj, deg, adj_mask, e1, e2
                )
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_swap = (e1, e2)
        
        if best_swap is not None:
            e1, e2 = best_swap
            edges_set.remove(e1)
            edges_set.add(e2)
            current_score = best_new_score
            improved = True
            # Rebuild adjacency for next iteration
            adj, deg = _build_adjacency(n, list(edges_set))
            adj_mask = [0] * n
            for i in range(n):
                mask = 0
                for j in adj[i]:
                    mask |= (1 << j)
                adj_mask[i] = mask
    
    return edges_set, current_score

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    
    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts of SA
    max_restarts = 15
    iterations_per_restart = 30000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        # Initialize graph
        if restart == 0:
            # Use the known good structure: K_8,8 + 1 edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
        elif restart == 1:
            # K_8,8 + different internal edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 7))
        elif restart == 2:
            # K_8,8 + two internal edges, remove one cross edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
            edges_set.add((2, 3))
            # Remove one cross edge to keep 65
            edges_set.discard((0, 8))
        else:
            # Random graph with 65 edges
            random.shuffle(all_pairs)
            edges_set = set(all_pairs[:65])
        
        edges_set = set(edges_set)
        
        # Build initial adjacency structures
        adj, deg = _build_adjacency(n, list(edges_set))
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        current_score = _compute_score(n, list(edges_set))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edges_set)
        
        # Simulated Annealing
        T_start = 5.0
        T_end = 0.1
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > DEADLINE - 15:
                break
                
            progress = i / iterations_per_restart
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Get a random edge to remove
            current_edge_list = list(edges_set)
            e1 = random.choice(current_edge_list)
            
            # Get a random non-edge to add
            attempts = 0
            while attempts < 10:
                e2 = random.choice(all_pairs)
                if e2 not in edges_set and e2 != e1:
                    break
                attempts += 1
            else:
                continue
                
            # Calculate new score incrementally
            new_score, new_adj, new_deg, new_adj_mask = _compute_score_incremental(
                n, edges_set, adj, deg, adj_mask, e1, e2
            )
            
            # Acceptance
            if new_score <= current_score:
                # Apply move
                edges_set.remove(e1)
                edges_set.add(e2)
                current_score = new_score
                adj = new_adj
                deg = new_deg
                adj_mask = new_adj_mask
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        edges_set.remove(e1)
                        edges_set.add(e2)
                        current_score = new_score
                        adj = new_adj
                        deg = new_deg
                        adj_mask = new_adj_mask
    
    # Final local search polish on the best solution found
    if best_edges is not None:
        if time.time() - start_time < DEADLINE - 5:
            best_edges_set = set(best_edges)
            polished_edges, polished_score = _local_search_polish(n, best_edges_set)
            if polished_score < best_score:
                best_score = polished_score
                best_edges = list(polished_edges)
    else:
        # Fallback to initial construction
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
