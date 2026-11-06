import time
import random
import math
from itertools import combinations

DEADLINE = 145  # Safety margin for 150s kill

def _compute_score(n, edges):
    """Compute the largest degree sum over triangles using bitmasks."""
    if not edges:
        return 0
        
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    max_triangle_sum = 0
    for u, v in edges:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        common &= ~((1 << (v + 1)) - 1)
        while common:
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    return max_triangle_sum

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score_and_deg(n, adj_mask, deg):
    """Compute max triangle sum."""
    max_triangle_sum = 0
    for u in range(n):
        for v_bit in range(u + 1, n):
            if adj_mask[u] & (1 << v_bit):
                v = v_bit
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w_bit = common & (-common)
                    w = w_bit.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_triangle_sum:
                        max_triangle_sum = s
                    common ^= w_bit
    return max_triangle_sum

def _local_search_il_s(n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score, time_budget=5.0):
    """
    Local search minimizing Max Triangle Sum.
    Tie-breaker: Minimize sum of squared degrees (variance).
    """
    start_time = time.time()
    # Precompute degree variance for tie-breaking
    avg_deg = 2 * len(edges_set) / n
    def _variance(deg_list):
        return sum((d - avg_deg)**2 for d in deg_list)
    
    current_var = _variance(deg)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Find the worst triangle
        worst_tri = None
        worst_sum = current_score
        for u, v in edges_list:
            if u > v:
                u, v = v, u
            common = adj_mask[u] & adj_mask[v]
            common &= ~((1 << (v + 1)) - 1)
            while common:
                w_bit = common & (-common)
                w = w_bit.bit_length() - 1
                s = deg[u] + deg[v] + deg[w]
                if s == worst_sum:
                    worst_tri = (u, v, w)
                    break
                common ^= w_bit
            if worst_tri:
                break
        
        if not worst_tri:
            break
            
        u, v, w = worst_tri
        edges_to_try = [(u, v), (v, w), (u, w)]
        
        found_swap = False
        for e_rm in edges_to_try:
            a, b = sorted(e_rm)
            
            # Try all non-edges
            for e_add in non_edges_list:
                x, y = e_add
                
                # Check if adding (x,y) creates a triangle with sum >= current_score
                # Quick check: if deg[x]+1 + deg[y]+1 + max(deg[common]) >= current_score, skip?
                # We just do the full check for correctness in the loop below.
                
                common_xy = adj_mask[x] & adj_mask[y]
                
                is_valid = True
                new_max_tri = current_score
                # We need to calculate the new max triangle sum if we accept this swap.
                # But doing full _compute_score for every candidate is slow.
                # Optimization: Only accept if new max tri < current_score OR (new max tri == current_score AND variance decreases).
                
                # Calculate potential new triangles involving x or y
                # This is complex. Let's stick to the previous heuristic: 
                # Only accept if we break the worst triangle and don't create a new one with sum >= current_score.
                
                # Check all triangles involving x or y in the NEW graph
                # Triangles involving x: (x, z, w) where z,w are neighbors of x in new graph.
                # This is effectively checking if max tri increases.
                
                # Let's use the fast filter from the original code:
                # If adding (x,y) creates ANY triangle with sum >= current_score, reject.
                while common_xy:
                    z_bit = common_xy & (-common)
                    z = z_bit.bit_length() - 1
                    
                    deg_new_x = deg[x] + 1
                    deg_new_y = deg[y] + 1
                    deg_new_z = deg[z]
                    
                    if x == a or x == b:
                        deg_new_x -= 1
                    if y == a or y == b:
                        deg_new_y -= 1
                    if z == a or z == b:
                        deg_new_z -= 1
                        
                    s_new = deg_new_x + deg_new_y + deg_new_z
                    
                    if s_new >= current_score:
                        is_valid = False
                        break
                    common_xy ^= z_bit
                
                if is_valid:
                    # Perform swap
                    edges_set.remove(e_rm)
                    edges_set.add(e_add)
                    edges_list.remove(e_rm)
                    edges_list.append(e_add)
                    non_edges_list.remove(e_add)
                    non_edges_list.append(e_rm)
                    
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
                    deg[a] -= 1
                    deg[b] -= 1
                    
                    adj_mask[x] |= (1 << y)
                    adj_mask[y] |= (1 << x)
                    deg[x] += 1
                    deg[y] += 1
                    
                    new_score = _compute_score(n, edges_list)
                    new_var = _variance(deg)
                    
                    if new_score < current_score or (new_score == current_score and new_var < current_var):
                        current_score = new_score
                        current_var = new_var
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        edges_set.remove(e_add)
                        edges_set.add(e_rm)
                        edges_list.remove(e_add)
                        edges_list.append(e_rm)
                        non_edges_list.remove(e_rm)
                        non_edges_list.append(e_add)
                        
                        adj_mask[x] &= ~(1 << y)
                        adj_mask[y] &= ~(1 << x)
                        deg[x] -= 1
                        deg[y] -= 1
                        
                        adj_mask[a] |= (1 << b)
                        adj_mask[b] |= (1 << a)
                        deg[a] += 1
                        deg[b] += 1
            
            if found_swap:
                break
        
        if not improved:
            break
    
    return edges_set, current_score, edges_list, non_edges_list, adj_mask, deg

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    
    best_score = float('inf')
    best_edges = None
    
    # ILS Parameters
    max_iterations = 200
    perturbation_size = 10
    
    # Initial solution: K_{8,8} + edge
    edges_set = set()
    for u in range(8):
        for v in range(8, 16):
            edges_set.add((u, v))
    edges_set.add((0, 1))
    
    edges_list = list(edges_set)
    non_edges_list = [pe for pe in all_pairs if pe not in edges_set]
    
    current_score = _compute_score(n, edges_list)
    
    adj_mask = [0] * n
    for u, v in edges_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    if current_score < best_score:
        best_score = current_score
        best_edges = list(edges_set)
        best_adj_mask = adj_mask[:]
        best_deg = deg[:]
        best_edges_set = set(edges_set)
        best_edges_list = edges_list[:]
        best_non_edges_list = non_edges_list[:]
        
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Perturbation: Perform K random swaps
        # We work on a copy of the current best
        curr_edges_set = set(best_edges_set)
        curr_edges_list = list(best_edges_set)
        curr_non_edges_list = [pe for pe in all_pairs if pe not in curr_edges_set]
        
        curr_adj_mask = [0] * n
        for u, v in curr_edges_set:
            curr_adj_mask[u] |= (1 << v)
            curr_adj_mask[v] |= (1 << u)
        curr_deg = [bin(curr_adj_mask[k]).count('1') for k in range(n)]
        
        # Apply K random swaps
        for k in range(perturbation_size):
            if not curr_edges_list or not curr_non_edges_list:
                break
            e1 = random.choice(curr_edges_list)
            e2 = random.choice(curr_non_edges_list)
            if e1 == e2:
                continue
                
            # Swap
            curr_edges_set.remove(e1)
            curr_edges_set.add(e2)
            curr_edges_list.remove(e1)
            curr_edges_list.append(e2)
            curr_non_edges_list.remove(e2)
            curr_non_edges_list.append(e1)
            
            a1, b1 = e1
            curr_adj_mask[a1] &= ~(1 << b1)
            curr_adj_mask[b1] &= ~(1 << a1)
            curr_deg[a1] -= 1
            curr_deg[b1] -= 1
            
            a2, b2 = e2
            curr_adj_mask[a2] |= (1 << b2)
            curr_adj_mask[b2] |= (1 << a2)
            curr_deg[a2] += 1
            curr_deg[b2] += 1
            
        # Local Search on the perturbed solution
        # Time budget for local search: 0.5s per iteration
        curr_edges_set, curr_score, curr_edges_list, curr_non_edges_list, curr_adj_mask, curr_deg = _local_search_il_s(
            n, curr_edges_set, curr_adj_mask, curr_deg, curr_edges_list, curr_non_edges_list, _compute_score(n, curr_edges_list), time_budget=0.5
        )
        
        if curr_score < best_score:
            best_score = curr_score
            best_edges = list(curr_edges_set)
            best_adj_mask = curr_adj_mask[:]
            best_deg = curr_deg[:]
            best_edges_set = set(curr_edges_set)
            best_edges_list = curr_edges_list[:]
            best_non_edges_list = curr_non_edges_list[:]
            
    # Final polish on the best
    if best_edges is not None and time.time() - start_time < DEADLINE - 2:
        best_edges_set, best_score, _, _, _, _ = _local_search_il_s(
            n, best_edges_set, best_adj_mask, best_deg, list(best_edges_set), [pe for pe in all_pairs if pe not in best_edges_set], best_score, time_budget=5.0
        )
        best_edges = list(best_edges_set)
        
    if best_edges is None:
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
