import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses incremental energy update for fast local search with a deterministic polish phase.
    """
    deadline = time.time() + 115
    
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
    
    def compute_delta_fast(u, v, c_old, c_new, adj, n):
        """Compute energy delta for changing edge (u,v) from c_old to c_new."""
        delta = 0
        mask_v = 1 << v
        mask_u = 1 << u
        
        # Remove from c_old: for each w, if w connected to v in c_old, count common neighbors of u and w
        adj_u_old = adj[c_old][u]
        adj_v_old = adj[c_old][v]
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[c_old][w]
            if mask_w & mask_v:
                k = (adj_u_old & mask_w).bit_count()
                delta -= (k - 1)
            if mask_w & mask_u:
                k = (adj_v_old & mask_w).bit_count()
                delta -= (k - 1)
        
        # Add to c_new
        adj_u_new = adj[c_new][u]
        adj_v_new = adj[c_new][v]
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[c_new][w]
            if mask_w & mask_v:
                k = (adj_u_new & mask_w).bit_count()
                delta += k
            if mask_w & mask_u:
                k = (adj_v_new & mask_w).bit_count()
                delta += k
        
        return delta
    
    def polish(adj, colors, n, edge_list, energy):
        """
        Deterministic hill climbing with targeted violation fixing.
        First pass: try to fix edges involved in C4 violations.
        Second pass: standard best-improvement.
        """
        improved = True
        max_passes = 30
        for _ in range(max_passes):
            if not improved:
                break
            if time_left() < 1:
                break
            
            best_delta = 0
            best_ei = -1
            best_c_new = -1
            
            num_edges = len(colors)
            
            # Phase 1: Prioritize edges that are part of violations
            # An edge (u,v) is "bad" if there exists w such that u-w and v-w are same color as u-v
            # We'll just do best-improvement but with a bias toward high-energy regions
            
            for ei in range(num_edges):
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                
                for c_new in range(5):
                    if c_new == c_old:
                        continue
                    
                    delta = compute_delta_fast(u, v, c_old, c_new, adj, n)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_ei = ei
                        best_c_new = c_new
            
            if best_ei != -1:
                u, v, e_idx = edge_list[best_ei]
                c_old = colors[e_idx]
                c_new = best_c_new
                
                energy += best_delta
                adj[c_old][u] &= ~(1 << v)
                adj[c_old][v] &= ~(1 << u)
                adj[c_new][u] |= (1 << v)
                adj[c_new][v] |= (1 << u)
                colors[e_idx] = c_new
                improved = True
            else:
                improved = False
        
        return energy, colors, adj, improved
    
    def targeted_polish(adj, colors, n, edge_list, energy):
        """
        Focus on edges that are part of C4 violations.
        For each color, find pairs of edges that form a C4 and try to recolor one.
        """
        if time_left() < 2:
            return energy, colors, adj, False
        
        improved = True
        max_rounds = 10
        for _ in range(max_rounds):
            if not improved:
                break
            if time_left() < 1:
                break
            
            best_delta = 0
            best_ei = -1
            best_c_new = -1
            
            # Find edges involved in violations
            candidate_edges = set()
            for c in range(5):
                for i in range(n):
                    adj_i = adj[c][i]
                    for j in range(i + 1, n):
                        if adj_i & (1 << j):
                            common = adj_i & adj[c][j]
                            if common.bit_count() >= 2:
                                # Edge (i,j) is in a violation
                                candidate_edges.add(edge_idx(i, j, n))
                                # Also edges from common neighbors
                                for w in range(n):
                                    if w != i and w != j and (common >> w) & 1:
                                        candidate_edges.add(edge_idx(i, w, n) if i < w else edge_idx(w, i, n))
                                        candidate_edges.add(edge_idx(j, w, n) if j < w else edge_idx(w, j, n))
            
            if not candidate_edges:
                break
            
            # Try recoloring candidate edges
            for ei in candidate_edges:
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                
                for c_new in range(5):
                    if c_new == c_old:
                        continue
                    
                    delta = compute_delta_fast(u, v, c_old, c_new, adj, n)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_ei = ei
                        best_c_new = c_new
            
            if best_ei != -1:
                u, v, e_idx = edge_list[best_ei]
                c_old = colors[e_idx]
                c_new = best_c_new
                
                energy += best_delta
                adj[c_old][u] &= ~(1 << v)
                adj[c_old][v] &= ~(1 << u)
                adj[c_new][u] |= (1 << v)
                adj[c_new][v] |= (1 << u)
                colors[e_idx] = c_new
                improved = True
            else:
                improved = False
        
        return energy, colors, adj, improved

    best_n = 1
    best_colors = [0]
    
    # Time allocation: spend more time on larger n
    time_per_n = {}
    total_time = 115
    # Allocate time proportional to difficulty
    for target_n in range(8, 29):
        if target_n <= 15:
            time_per_n[target_n] = 3
        elif target_n <= 20:
            time_per_n[target_n] = 5
        elif target_n <= 24:
            time_per_n[target_n] = 8
        elif target_n <= 26:
            time_per_n[target_n] = 12
        else:
            time_per_n[target_n] = 15
    
    for target_n in range(8, 29):
        if time_left() < 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        max_restarts = 50
        n_start_time = time.time()
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            if time.time() - n_start_time > time_per_n.get(n, 10):
                break
            
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
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
            best_adj_local = [row[:] for row in adj]
            
            # Simulated annealing with faster delta computation
            temp = 3.0
            cooling = 0.9995
            min_temp = 0.01
            max_steps = 50000
            stuck_count = 0
            max_stuck = 3000
            
            for step in range(max_steps):
                if time_left() < 3:
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
                
                delta = compute_delta_fast(u, v, c_old, c_new, adj, n)
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
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
                        best_adj_local = [row[:] for row in adj]
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
            
            # Targeted polish: focus on violations
            if time_left() > 2:
                energy, colors, adj, improved = targeted_polish(adj, colors, n, edge_list, energy)
                if energy == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                    best_adj_local = [row[:] for row in adj]
            
            # Standard polish
            if time_left() > 2:
                energy, colors, adj, improved = polish(adj, colors, n, edge_list, energy)
                if energy == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                    best_adj_local = [row[:] for row in adj]
            
            # If close to solution, do more targeted search
            if best_energy <= 5 and time_left() > 5:
                # Do a few more targeted polish rounds
                for _ in range(3):
                    if time_left() < 2:
                        break
                    energy, colors, adj, improved = targeted_polish(adj, colors, n, edge_list, energy)
                    if energy == 0:
                        best_n = n
                        best_colors = colors[:]
                        found = True
                        break
                    if energy < best_energy:
                        best_energy = energy
                        best_colors_local = colors[:]
                        best_adj_local = [row[:] for row in adj]
                
                if found:
                    break

        if found:
            continue

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Found n = {result['n']} in {elapsed:.2f}s")
    print(f"Number of colors: {len(result['colors'])}")
    # Verify no C4
    n = result['n']
    colors = result['colors']
    adj = [[0] * n for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    c4_count = 0
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[c][i] & adj[c][j]
                k = common.bit_count()
                if k >= 2:
                    c4_count += k * (k - 1) // 2
    print(f"C4 violations: {c4_count}")
