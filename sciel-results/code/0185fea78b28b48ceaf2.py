import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses fast local search with optimized energy updates.
    """
    deadline = time.time() + 112
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                adj_i = adj[c][i]
                for j in range(i + 1, n):
                    common = adj_i & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def make_initial_colors(n, strategy=0):
        """Generate initial coloring using different strategies."""
        num_edges = n * (n - 1) // 2
        colors = [0] * num_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if strategy == 0:
                    # Random
                    colors[idx] = random.randint(0, 4)
                elif strategy == 1:
                    # Based on sum mod 5
                    colors[idx] = (i + j) % 5
                elif strategy == 2:
                    # Based on difference mod 5
                    colors[idx] = (j - i) % 5
                elif strategy == 3:
                    # Based on product mod 5
                    colors[idx] = (i * j) % 5
                elif strategy == 4:
                    # Mixed: half random, half structured
                    if random.random() < 0.5:
                        colors[idx] = (i + j) % 5
                    else:
                        colors[idx] = random.randint(0, 4)
                else:
                    colors[idx] = random.randint(0, 4)
                idx += 1
        return colors

    best_n = 1
    best_colors = [0]
    
    # Try from largest n down to smaller ones
    # Allocate time proportionally to difficulty
    n_values = list(range(28, 7, -1))
    
    # Time allocation: give more time to larger n values
    # But also ensure we try smaller n if larger ones fail quickly
    time_per_n = {}
    remaining_time = 110
    for n_val in n_values:
        # Larger n gets more time
        weight = (n_val - 7) / sum((nv - 7) for nv in n_values)
        time_per_n[n_val] = remaining_time * weight
        remaining_time -= time_per_n[n_val]
    
    for target_n in n_values:
        if time_left() < 5:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list with indices
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        
        # Try multiple restarts with different initialization strategies
        max_restarts = 100
        restart_time_limit = time_per_n.get(n, 5) / max_restarts
        
        for restart in range(max_restarts):
            if time_left() < 3:
                break
            
            # Vary initialization strategy
            strategy = restart % 5
            colors = make_initial_colors(n, strategy)
            
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            # Simulated annealing parameters tuned for this problem
            temp = 1.5
            cooling = 0.99998
            min_temp = 0.01
            max_steps = 300000
            stuck_count = 0
            max_stuck = 15000
            
            restart_start = time.time()
            restart_deadline = restart_start + restart_time_limit
            
            for step in range(max_steps):
                if time_left() < 2:
                    break
                if time.time() > restart_deadline:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                
                ei = random.randint(0, num_edges - 1)
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                # Fast delta calculation
                delta = 0
                
                # Remove from c_old: iterate over neighbors of u and v in c_old
                adj_u_old = adj[c_old][u]
                adj_v_old = adj[c_old][v]
                
                # For each w that is neighbor of u in c_old (w != v)
                # and neighbor of v in c_old, the pair (u,v) contributes to C4
                # More precisely: for each w != u,v:
                #   if w~u and w~v in c_old, then (u,v) was part of C4s
                #   removing edge (u,v) reduces C4 count by (common_neighbors - 1) for each such w? 
                #   Actually, the energy counts pairs of common neighbors.
                
                # Let's use the formula:
                # When removing edge (u,v) from color c:
                # For each w != u,v where w is adjacent to both u and v in c:
                #   The pair (u,v) had common neighbors including w.
                #   The number of C4s involving edge (u,v) and vertex w is (common(u,v,w) - 1)
                #   where common(u,v,w) = |N_c(u) ∩ N_c(v) ∩ {w}|... 
                #   
                # Simpler: delta = -(number of C4s containing edge (u,v) in c_old)
                #            + (number of C4s containing edge (u,v) in c_new)
                #
                # Number of C4s containing edge (u,v) in color c:
                # = sum over w != u,v of [1 if w~u and w~v in c, else 0] * (common_neighbors(u,v) - 1)
                # Wait, that's not right either.
                #
                # Let me reconsider. The energy is:
                # E = sum over all pairs (i,j) of C(|N(i) ∩ N(j)|, 2)
                # where C(k,2) = k*(k-1)/2
                #
                # When we change edge (u,v) from c_old to c_new:
                # For c_old: removing edge (u,v)
                #   For each w != u,v:
                #     If w~u in c_old and w~v in c_old:
                #       common(u,w) decreases by 1 (v was a common neighbor)
                #       common(v,w) decreases by 1 (u was a common neighbor)
                #     So delta for pair (u,w): C(k-1, 2) - C(k, 2) = -(k-1) where k = common(u,w)
                #     Similarly for pair (v,w): delta = -(k-1) where k = common(v,w)
                #
                # For c_new: adding edge (u,v)
                #   For each w != u,v:
                #     If w~u in c_new and w~v in c_new:
                #       common(u,w) increases by 1
                #       common(v,w) increases by 1
                #     So delta for pair (u,w): C(k+1, 2) - C(k, 2) = k+1 where k = common(u,w)
                #     Similarly for pair (v,w): delta = k+1 where k = common(v,w)
                
                # Remove from c_old
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    mask_w = adj[c_old][w]
                    
                    # If v is neighbor of w in c_old, then u and w share v as common neighbor
                    if mask_w & (1 << v):
                        k = (adj_u_old & mask_w).bit_count()
                        # k is the number of common neighbors of u and w BEFORE removal
                        # After removal, it becomes k-1
                        # Delta = C(k-1,2) - C(k,2) = (k-1)(k-2)/2 - k(k-1)/2 = -(k-1)
                        delta -= (k - 1)
                    
                    # If u is neighbor of w in c_old, then v and w share u as common neighbor
                    if mask_w & (1 << u):
                        k = (adj_v_old & mask_w).bit_count()
                        delta -= (k - 1)
                
                # Add to c_new
                adj_u_new = adj[c_new][u]
                adj_v_new = adj[c_new][v]
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    mask_w = adj[c_new][w]
                    
                    # If v is neighbor of w in c_new
                    if mask_w & (1 << v):
                        k = (adj_u_new & mask_w).bit_count()
                        # After addition, common(u,w) becomes k+1
                        # Delta = C(k+1,2) - C(k,2) = k+1
                        delta += (k + 1)
                    
                    # If u is neighbor of w in c_new
                    if mask_w & (1 << u):
                        k = (adj_v_new & mask_w).bit_count()
                        delta += (k + 1)
                
                new_energy = energy + delta
                
                # Acceptance criterion
                if delta <= 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                    # Accept
                    energy = new_energy
                    
                    # Update adj
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    colors[e_idx] = c_new
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    stuck_count += 1
                
                temp *= cooling
            
            if found:
                break

        if found:
            continue

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
