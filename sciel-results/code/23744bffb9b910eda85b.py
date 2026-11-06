import time
import random
import math
from itertools import combinations

DEADLINE = 130  # Safety margin for 150s kill

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

def _try_break_worst_triangle_optimized(n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score):
    """
    Degree-ordered edge-replacement heuristic.
    1. Find worst triangle.
    2. Try removing each edge of it.
    3. Try adding non-edges, prioritizing those between low-degree vertices.
    4. Accept if the new max score is strictly less than current_score.
    """
    if current_score == 0:
        return None
        
    # Find a triangle with sum == current_score
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
        return None
        
    u, v, w = worst_tri
    # Identify the vertex with max degree in the worst triangle
    # This helps prioritize which edge to break? 
    # Actually, breaking any edge in the triangle removes that triangle.
    # But we want to ensure we don't create a *new* triangle with sum >= current_score.
    
    edges_to_try = [(u, v), (v, w), (u, w)]
    
    # Sort non-edges by degree sum of endpoints to try "safe" additions first
    # This is a heuristic: adding an edge between low-degree nodes is less likely
    # to create a high-degree-sum triangle.
    sorted_non_edges = sorted(non_edges_list, key=lambda e: deg[e[0]] + deg[e[1]])
    
    for e_rm in edges_to_try:
        a, b = sorted(e_rm)
        
        # Simulate removal of (a,b)
        # We need to check if adding (x,y) creates a triangle with sum >= current_score
        # in the graph where (a,b) is removed.
        
        # Pre-compute "potential" issues? 
        # For each candidate (x,y), we need to check common neighbors in G - (a,b).
        
        for e_add in sorted_non_edges:
            x, y = e_add
            
            # Quick check: If x or y is a or b, and the other is connected to the third vertex of the triangle?
            # Just do the full check for correctness, it's fast enough for n=16.
            
            is_valid = True
            # Common neighbors of x and y in the ORIGINAL graph
            # But we must exclude neighbors that are only connected via the removed edge?
            # No, common neighbors are determined by adj_mask.
            # If we remove (a,b), the adj_mask changes.
            # Specifically, a is no longer neighbor of b, and b no longer of a.
            
            # Let's manually compute common neighbors for x,y in the modified graph.
            # Modified adj for x: adj_mask[x] with (a,b) removed effect?
            # If x==a, adj_x loses b. If x==b, adj_x loses a.
            # If x not in {a,b}, adj_x unchanged.
            
            mask_x = adj_mask[x]
            mask_y = adj_mask[y]
            
            if x == a: mask_x &= ~(1 << b)
            elif x == b: mask_x &= ~(1 << a)
            if y == a: mask_y &= ~(1 << b)
            elif y == b: mask_y &= ~(1 << a)
            
            common = mask_x & mask_y
            
            # We need to check all common neighbors z.
            # For each z, calculate the new degree sum.
            # New degrees:
            # deg[x] + 1 (unless x==a or b, then -1 + 1 = 0 change? No.)
            # Let's calculate new degrees precisely.
            
            deg_x_new = deg[x]
            deg_y_new = deg[y]
            
            # Effect of removing (a,b):
            if x == a or x == b: deg_x_new -= 1
            if y == a or y == b: deg_y_new -= 1
            
            # Effect of adding (x,y):
            deg_x_new += 1
            deg_y_new += 1
            
            # Note: if x==a and y==b, this is adding back the same edge, which is invalid (e_add != e_rm is checked by set logic usually, but here we should ensure e_add != e_rm)
            if e_add == e_rm:
                continue
            
            # Check triangles formed by (x,y,z)
            while common:
                z_bit = common & (-common)
                z = z_bit.bit_length() - 1
                
                deg_z_new = deg[z]
                if z == a or z == b:
                    deg_z_new -= 1
                if z == x or z == y:
                    # z is a neighbor of x and y, so z != x and z != y (simple graph)
                    pass
                
                s_new = deg_x_new + deg_y_new + deg_z_new
                
                if s_new >= current_score:
                    is_valid = False
                    break
                common ^= z_bit
            
            if is_valid:
                # Perform swap
                new_edges_set = set(edges_set)
                new_edges_set.remove(e_rm)
                new_edges_set.add(e_add)
                
                new_edges_list = [e for e in edges_list if e != e_rm]
                new_edges_list.append(e_add)
                
                new_non_edges_list = [e for e in non_edges_list if e != e_add]
                new_non_edges_list.append(e_rm)
                
                new_score = _compute_score(n, new_edges_list)
                
                if new_score < current_score:
                    # Update masks
                    adj_mask_new = adj_mask[:]
                    deg_new = deg[:]
                    
                    adj_mask_new[a] &= ~(1 << b)
                    adj_mask_new[b] &= ~(1 << a)
                    deg_new[a] -= 1
                    deg_new[b] -= 1
                    
                    adj_mask_new[x] |= (1 << y)
                    adj_mask_new[y] |= (1 << x)
                    deg_new[x] += 1
                    deg_new[y] += 1
                    
                    return new_edges_set, new_score, new_edges_list, new_non_edges_list, adj_mask_new, deg_new
                    
    return None

def _local_search_polish(n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score, time_budget=5.0):
    """
    Final high-precision polish pass: try all possible single-edge swaps
    to reduce the maximum triangle sum.
    """
    start_time = time.time()
    
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
                common_xy = adj_mask[x] & adj_mask[y]
                
                is_valid = True
                while common_xy:
                    z_bit = common_xy & (-common_xy)
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
                    
                    current_score = _compute_score(n, edges_list)
                    improved = True
                    found_swap = True
                    break
            
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
    
    max_restarts = 20
    iterations_per_restart = 25000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Initialize graph
        if restart == 0:
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
        elif restart == 1:
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((3, 5))
        elif restart == 2:
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
            edges_set.add((2, 3))
            edges_set.discard((0, 8))
        elif restart == 3:
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((8, 9))
        elif restart == 4:
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
            edges_set.add((4, 5))
            edges_set.discard((1, 8))
        elif restart == 5:
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((2, 4))
            edges_set.add((6, 7))
            edges_set.discard((3, 10))
        else:
            random.shuffle(all_pairs)
            edges_set = set(all_pairs[:65])
        
        edges_set = set(edges_set)
        edges_list = list(edges_set)
        non_edges_list = [pe for pe in all_pairs if pe not in edges_set]
        
        current_score = _compute_score(n, edges_list)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edges_set)
        
        # Precompute adj_mask and deg
        adj_mask = [0] * n
        for u, v in edges_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        deg = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        T_start = 2.0
        T_end = 0.003
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > DEADLINE - 15:
                break
                
            progress = i / iterations_per_restart
            T = T_start * math.pow(T_end / T_start, progress)
            
            # 30% chance of targeted swap using the new degree-ordered heuristic
            if random.random() < 0.30 and current_score > 0:
                res = _try_break_worst_triangle_optimized(n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score)
                if res:
                    new_edges_set, new_score, new_edges_list, new_non_edges_list, new_adj_mask, new_deg = res
                    edges_set = new_edges_set
                    edges_list = new_edges_list
                    non_edges_list = new_non_edges_list
                    adj_mask = new_adj_mask
                    deg = new_deg
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = list(edges_set)
                    continue
            
            # Standard SA move
            e1 = random.choice(edges_list)
            
            e2 = None
            attempts = 0
            while attempts < 10:
                candidate = random.choice(non_edges_list)
                if candidate != e1:
                    e2 = candidate
                    break
                attempts += 1
            
            if e2 is None:
                continue
            
            # Perform swap
            edges_set.remove(e1)
            edges_set.add(e2)
            edges_list.remove(e1)
            edges_list.append(e2)
            non_edges_list.remove(e2)
            non_edges_list.append(e1)
            
            # Update masks
            a1, b1 = e1
            adj_mask[a1] &= ~(1 << b1)
            adj_mask[b1] &= ~(1 << a1)
            deg[a1] -= 1
            deg[b1] -= 1
            
            a2, b2 = e2
            adj_mask[a2] |= (1 << b2)
            adj_mask[b2] |= (1 << a2)
            deg[a2] += 1
            deg[b2] += 1
            
            new_score = _compute_score(n, edges_list)
            
            if new_score <= current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        edges_set.remove(e2)
                        edges_set.add(e1)
                        edges_list.remove(e2)
                        edges_list.append(e1)
                        non_edges_list.remove(e1)
                        non_edges_list.append(e2)
                        
                        adj_mask[a2] &= ~(1 << b2)
                        adj_mask[b2] &= ~(1 << a2)
                        deg[a2] -= 1
                        deg[b2] -= 1
                        
                        adj_mask[a1] |= (1 << b1)
                        adj_mask[b1] |= (1 << a1)
                        deg[a1] += 1
                        deg[b1] += 1
        
        # Final polish pass for this restart
        if time.time() - start_time < DEADLINE - 10:
            edges_set, current_score, edges_list, non_edges_list, adj_mask, deg = _local_search_polish(
                n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score, time_budget=3.0
            )
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edges_set)
    
    # Final global polish on the best edges found
    if best_edges is not None and time.time() - start_time < DEADLINE - 5:
        best_edges_set = set(tuple(sorted(e)) for e in best_edges)
        best_edges_list = list(best_edges_set)
        best_non_edges = [pe for pe in all_pairs if pe not in best_edges_set]
        
        best_adj_mask = [0] * n
        for u, v in best_edges_set:
            best_adj_mask[u] |= (1 << v)
            best_adj_mask[v] |= (1 << u)
        best_deg = [bin(best_adj_mask[i]).count('1') for i in range(n)]
        
        best_current_score = _compute_score(n, best_edges_list)
        
        best_edges_set, best_current_score, best_edges_list, _, _, _ = _local_search_polish(
            n, best_edges_set, best_adj_mask, best_deg, best_edges_list, best_non_edges, best_current_score, time_budget=5.0
        )
        
        if best_current_score < best_score:
            best_score = best_current_score
            best_edges = list(best_edges_set)
    
    # Fallback
    if best_edges is None:
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
