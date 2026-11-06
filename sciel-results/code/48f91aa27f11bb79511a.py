import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def solve_fast_sa(n, time_limit):
        """
        Local search with incremental cost updates using NumPy.
        Returns a valid coloring list or None if not found within time_limit.
        """
        m = n * (n - 1) // 2
        if m == 0:
            return []
        
        # Map (i, j) to index in colors array
        # idx = i * (2*n - i - 1) // 2 + j - i - 1
        # We need fast access to edge colors.
        # colors is a flat list/array of length m.
        
        # neighbors[c, u, v] = 1 if edge (u,v) has color c
        neighbors = np.zeros((5, n, n), dtype=np.uint8)
        # common_count[c, u, v] = number of common neighbors of u and v in color c
        common_count = np.zeros((5, n, n), dtype=np.uint8)
        
        # Initialize random colors
        colors = np.random.randint(0, 5, size=m, dtype=np.uint8)
        
        # Build initial structures
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c, i, j] = 1
                neighbors[c, j, i] = 1
                idx += 1
        
        # Calculate initial common_count
        # For each color c, common_count[c] = neighbors[c] @ neighbors[c]
        # But we only care about pairs u != v.
        # Note: neighbors[c] @ neighbors[c] gives count of common neighbors including self-loops?
        # No, neighbors[c, u, v] is 1 if edge exists.
        # (neighbors[c] @ neighbors[c])[u, v] = sum_w neighbors[c, u, w] * neighbors[c, w, v]
        # This counts paths of length 2.
        # Since the graph is simple, this is exactly the number of common neighbors.
        for c in range(5):
            common_count[c] = neighbors[c] @ neighbors[c]
            
        # Subtract self-loops?
        # The diagonal of neighbors[c] is 0, so (neighbors @ neighbors)[u, u] is degree.
        # We only care about u != v.
        # The definition of C4 violation is: two vertices u, v have >= 2 common neighbors.
        # So we check common_count[c, u, v] >= 2 for u != v.
        
        # Calculate initial violations
        violations = 0
        for c in range(5):
            # Get upper triangle
            # np.triu_indices
            # Or just iterate
            # Vectorized:
            cc = common_count[c]
            # We need count of elements >= 2 in the upper triangle (excluding diagonal)
            # Mask: u < v
            # Simple loop is faster for n=26?
            # n=26, 26*25/2 = 325 pairs.
            # 5 colors. 1625 checks.
            for u in range(n):
                for v in range(u + 1, n):
                    if cc[u, v] >= 2:
                        violations += 1
                        
        if violations == 0:
            return colors.tolist()
            
        best_violations = violations
        best_colors_snapshot = colors.copy()
        
        search_start = time.time()
        temperature = 1.0
        cooling_rate = 0.9999
        min_temperature = 0.01
        stagnation_counter = 0
        
        # Precompute edge indices for fast access
        # edge_idx[i, j] = index in colors array
        edge_idx = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[i, j] = idx
                edge_idx[j, i] = idx
                idx += 1
                
        while time.time() - search_start < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx_e = edge_idx[i, j]
            c_old = colors[idx_e]
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            delta_violations = 0
            
            # Update common_count and violations
            # Affected pairs: (i, w) and (j, w) for all w != i, j
            
            # We need to revert if we decide not to accept.
            # Store changes: (c, u, v, old_val)
            changes = []
            
            for w in range(n):
                if w == i or w == j:
                    continue
                
                # Pair (i, w)
                # Common neighbors in c_old:
                # j was a common neighbor if neighbors[c_old, j, w] == 1.
                # Now edge (i,j) is c_new, so j is NOT a common neighbor in c_old anymore.
                if neighbors[c_old, j, w] == 1:
                    old_val = common_count[c_old, i, w]
                    new_val = old_val - 1
                    common_count[c_old, i, w] = new_val
                    common_count[c_old, w, i] = new_val
                    if old_val == 2:
                        delta_violations -= 1
                    changes.append((c_old, i, w, old_val))
                    
                # Common neighbors in c_new:
                # j was NOT a common neighbor in c_new (since edge was c_old).
                # Now edge (i,j) is c_new, so j IS a common neighbor in c_new if neighbors[c_new, j, w] == 1.
                if neighbors[c_new, j, w] == 1:
                    old_val = common_count[c_new, i, w]
                    new_val = old_val + 1
                    common_count[c_new, i, w] = new_val
                    common_count[c_new, w, i] = new_val
                    if old_val == 1:
                        delta_violations += 1
                    changes.append((c_new, i, w, old_val))
                    
                # Pair (j, w)
                # Common neighbors in c_old:
                # i was a common neighbor if neighbors[c_old, i, w] == 1.
                if neighbors[c_old, i, w] == 1:
                    old_val = common_count[c_old, j, w]
                    new_val = old_val - 1
                    common_count[c_old, j, w] = new_val
                    common_count[c_old, w, j] = new_val
                    if old_val == 2:
                        delta_violations -= 1
                    changes.append((c_old, j, w, old_val))
                    
                # Common neighbors in c_new:
                # i was NOT a common neighbor in c_new.
                if neighbors[c_new, i, w] == 1:
                    old_val = common_count[c_new, j, w]
                    new_val = old_val + 1
                    common_count[c_new, j, w] = new_val
                    common_count[c_new, w, j] = new_val
                    if old_val == 1:
                        delta_violations += 1
                    changes.append((c_new, j, w, old_val))
                
            # Check pair (i, j) itself?
            # Common neighbors of i and j.
            # The edge (i,j) itself is not a common neighbor.
            # But the change of color of (i,j) affects common_count[c, i, j]?
            # common_count[c, i, j] counts w such that (i,w) and (j,w) are color c.
            # The edge (i,j) is not involved in this count (w != i, j).
            # So common_count[c, i, j] does NOT change directly due to the color of (i,j).
            # Wait, does it?
            # No. The definition is sum_w N_c(i,w) N_c(j,w).
            # w ranges over all vertices except i, j.
            # The color of (i,j) is not used in this sum.
            # So common_count[c, i, j] is invariant under changing color of (i,j).
            
            new_violations = violations + delta_violations
            
            # Acceptance
            if new_violations < violations or (temperature > min_temperature and random.random() < np.exp(-delta_violations / temperature)):
                # Apply change
                neighbors[c_old, i, j] = 0
                neighbors[c_old, j, i] = 0
                neighbors[c_new, i, j] = 1
                neighbors[c_new, j, i] = 1
                colors[idx_e] = c_new
                violations = new_violations
                stagnation_counter = 0
                
                if violations < best_violations:
                    best_violations = violations
                    best_colors_snapshot = colors.copy()
                    
                if violations == 0:
                    return colors.tolist()
            else:
                # Revert changes
                for c, u, v, old_val in changes:
                    # We need to restore the value
                    # Note: we updated both (u,v) and (v,u)
                    # And we stored the old value
                    # We just set it back
                    common_count[c, u, v] = old_val
                    common_count[c, v, u] = old_val
                stagnation_counter += 1
                
            temperature = max(min_temperature, temperature * cooling_rate)
            
            # Perturbation
            if stagnation_counter > 50:
                for _ in range(3):
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    if ri == rj: continue
                    if ri > rj: ri, rj = rj, ri
                    ridx = edge_idx[ri, rj]
                    rc_old = colors[ridx]
                    rc_new = random.randint(0, 4)
                    if rc_new == rc_old: continue
                    
                    # Apply blindly and recalc cost
                    neighbors[rc_old, ri, rj] = 0
                    neighbors[rc_old, rj, ri] = 0
                    neighbors[rc_new, ri, rj] = 1
                    neighbors[rc_new, rj, ri] = 1
                    colors[ridx] = rc_new
                # Recalculate common_count and violations
                for c in range(5):
                    common_count[c] = neighbors[c] @ neighbors[c]
                violations = 0
                for c in range(5):
                    cc = common_count[c]
                    for u in range(n):
                        for v in range(u + 1, n):
                            if cc[u, v] >= 2:
                                violations += 1
                stagnation_counter = 0
                temperature = 1.0

        return None

    targets = [26, 25, 24, 23, 22, 21, 20]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        if n >= 26:
            t_limit = min(40.0, deadline - time.time() - 2)
        elif n >= 24:
            t_limit = min(20.0, deadline - time.time() - 2)
        else:
            t_limit = min(10.0, deadline - time.time() - 2)
            
        if t_limit < 1.0:
            break
            
        sol = solve_fast_sa(n, t_limit)
        
        if sol is not None:
            # Verify
            # Re-verify with a simple check
            neighbors = [[0]*n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = sol[idx]
                    neighbors[c][i] |= (1 << j)
                    neighbors[c][j] |= (1 << i)
                    idx += 1
            valid = True
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0: continue
                    for v in range(u+1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                best_n = n
                best_colors = sol
                break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
