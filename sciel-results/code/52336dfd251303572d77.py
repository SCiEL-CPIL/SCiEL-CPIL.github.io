import time
import random
import math
from itertools import combinations

DEADLINE = 135  # Safety margin for 150s kill

def _compute_score(n, adj_mask, deg):
    """Compute the largest degree sum over triangles using precomputed masks and degrees."""
    max_triangle_sum = 0
    # Iterate over all triples u < v < w
    for u in range(n):
        mu = adj_mask[u]
        du = deg[u]
        # v must be > u
        for v in range(u + 1, n):
            if not (mu >> v) & 1:
                continue
            mv = adj_mask[v]
            dv = deg[v]
            # w must be > v
            # Check if w is connected to both u and v
            # common = mu & mv & ~( (1 << (v+1)) - 1 )
            # We need w > v
            # So we can mask out bits <= v
            common = mu & mv
            # Clear bits 0..v
            common &= ~((1 << (v + 1)) - 1)
            while common:
                w_bit = common & (-common)
                w = w_bit.bit_length() - 1
                dw = deg[w]
                s = du + dv + dw
                if s > max_triangle_sum:
                    max_triangle_sum = s
                common ^= w_bit
    return max_triangle_sum

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _try_break_worst_triangle(n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score):
    """
    Attempts to find a swap that reduces the maximum triangle sum.
    Returns (new_edges_set, new_score, new_edges_list, new_non_edges_list, new_adj_mask, new_deg) or None.
    """
    if current_score == 0:
        return None
        
    # Find a triangle with sum == current_score
    worst_tri = None
    for u in range(n):
        mu = adj_mask[u]
        du = deg[u]
        for v in range(u + 1, n):
            if not (mu >> v) & 1:
                continue
            mv = adj_mask[v]
            dv = deg[v]
            common = mu & mv
            common &= ~((1 << (v + 1)) - 1)
            while common:
                w_bit = common & (-common)
                w = w_bit.bit_length() - 1
                dw = deg[w]
                s = du + dv + dw
                if s == current_score:
                    worst_tri = (u, v, w)
                    break
                common ^= w_bit
            if worst_tri:
                break
        if worst_tri:
            break
            
    if not worst_tri:
        return None
        
    u, v, w = worst_tri
    
    # Try removing each edge of the worst triangle
    edges_to_try = [(u, v), (v, w), (u, w)]
    
    for e_rm in edges_to_try:
        a, b = sorted(e_rm)
        
        # Try adding non-edges
        # We prioritize non-edges that might break the triangle or reduce degrees of u,v,w
        # But simple random sampling is okay, we just need to check validity quickly
        for _ in range(10): # Reduced attempts per edge to speed up
            if not non_edges_list:
                break
            e_add = random.choice(non_edges_list)
            x, y = e_add
            
            # Quick check: Does adding (x,y) create a new triangle with sum >= current_score?
            # We only need to check triangles involving x or y that are formed by the new edge.
            # The new triangles are (x, y, z) where z is a common neighbor of x and y.
            # Note: Removing (a,b) might have removed some common neighbors, but we are checking the state *after* swap?
            # No, we are checking if the *new* graph has a triangle >= current_score.
            # The new triangles are exactly those involving the new edge (x,y).
            # Triangles not involving (x,y) are unchanged, so they are < current_score (since we picked a worst triangle).
            # So we only need to check triangles (x, y, z).
            
            common_xy = adj_mask[x] & adj_mask[y]
            # Exclude x and y themselves? No, adj_mask doesn't have self-loops.
            # But we must consider that (a,b) is removed.
            # If z is a or b, and (a,b) was removed, does it affect the triangle (x,y,z)?
            # If z=a, triangle is (x,y,a). Edges: (x,a), (y,a), (x,y).
            # (x,a) and (y,a) are unchanged by removing (a,b) unless x=b or y=b.
            # If x=b, then (x,a) is (b,a) which is removed. So (b,y,a) is not a triangle if (b,a) is removed.
            # So we need to be careful.
            
            is_valid = True
            while common_xy:
                z_bit = common_xy & (-common_xy)
                z = z_bit.bit_length() - 1
                
                # Check if (x,z) and (y,z) are still edges after removing (a,b)
                # (x,z) is removed only if {x,z} == {a,b}
                # (y,z) is removed only if {y,z} == {a,b}
                
                edge_xz_removed = (x == a and z == b) or (x == b and z == a)
                edge_yz_removed = (y == a and z == b) or (y == b and z == a)
                
                if edge_xz_removed or edge_yz_removed:
                    common_xy ^= z_bit
                    continue
                
                # Calculate new degrees
                deg_new_x = deg[x] + 1
                deg_new_y = deg[y] + 1
                deg_new_z = deg[z]
                
                # Adjust for removed edge (a,b)
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
                new_edges_set = set(edges_set)
                new_edges_set.remove(e_rm)
                new_edges_set.add(e_add)
                
                new_edges_list = [e for e in edges_list if e != e_rm]
                new_edges_list.append(e_add)
                
                new_non_edges_list = [e for e in non_edges_list if e != e_add]
                new_non_edges_list.append(e_rm)
                
                # Update masks and deg for score calculation
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
                
                new_score = _compute_score(n, adj_mask_new, deg_new)
                
                if new_score < current_score:
                    return new_edges_set, new_score, new_edges_list, new_non_edges_list, adj_mask_new, deg_new
                    
    return None

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    
    best_score = float('inf')
    best_edges = None
    
    max_restarts = 15
    iterations_per_restart = 50000 # Increased iterations since score calc is faster
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Initialize graph
        if restart == 0:
            # Turan T_2(16) is 8x8 bipartite (64 edges). Add one intra-part edge.
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
        else:
            random.shuffle(all_pairs)
            edges_set = set(all_pairs[:65])
        
        edges_set = set(edges_set)
        edges_list = list(edges_set)
        non_edges_list = [pe for pe in all_pairs if pe not in edges_set]
        
        # Precompute adj_mask and deg
        adj_mask = [0] * n
        for u, v in edges_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        deg = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        current_score = _compute_score(n, adj_mask, deg)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edges_set)
        
        T_start = 2.0
        T_end = 0.01
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            progress = i / iterations_per_restart
            T = T_start * math.pow(T_end / T_start, progress)
            
            # 30% chance of targeted swap
            if random.random() < 0.30 and current_score > 0:
                res = _try_break_worst_triangle(n, edges_set, adj_mask, deg, edges_list, non_edges_list, current_score)
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
            
            new_score = _compute_score(n, adj_mask, deg)
            
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
    
    # Fallback
    if best_edges is None:
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
