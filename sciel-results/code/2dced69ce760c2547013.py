import time
import random
import numpy as np

DEADLINE = 115  # seconds, safety margin before 150s kill

def _time_left():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    np.random.seed(42)

    # ---- Build the n=32 seed colouring (Greenwood-Gleason doubled) ----
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    n_seed = 32
    colors_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if i // 16 == j // 16:
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors_seed.append(3)

    def _validate(colors, n):
        """Check no monochromatic triangle."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                neighbors = adj[c][i]
                while neighbors:
                    lsb = neighbors & (-neighbors)
                    j = lsb.bit_length() - 1
                    neighbors ^= lsb
                    common = adj[c][i] & adj[c][j]
                    if common:
                        return False
        return True

    if not _validate(colors_seed, n_seed):
        return {'n': 1, 'colors': []}

    best_n = n_seed
    best_colors = colors_seed[:]

    # Precompute edge colors matrix for fast access
    # edge_color[i][j] is the color of edge (i, j)
    # We will maintain this as we extend.
    
    def extend_by_one(current_colors, current_n):
        """Try to extend coloring from current_n to current_n+1 using SA."""
        if not _time_left():
            return None
        
        new_n = current_n + 1
        if new_n > 62:
            return None
        
        # Build M_c matrices: M_c[i][j] = 1 if edge(i,j) has color c
        # i, j in 0..current_n-1
        M = [np.zeros((current_n, current_n), dtype=np.int8) for _ in range(4)]
        idx = 0
        for i in range(current_n):
            for j in range(i + 1, current_n):
                c = current_colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                idx += 1
        
        def compute_cost(x):
            # x is numpy array of shape (current_n,) with values 0..3
            cost = 0
            for c in range(4):
                # v_c is 1 where x == c
                v_c = (x == c).astype(np.int8)
                # cost_c = sum(v_c[i] * v_c[j] * M_c[i][j]) / 2
                # Since M is symmetric, we can just sum and divide by 2
                # Or use dot product
                # cost_c = 0.5 * (v_c @ (M[c] @ v_c))
                cost += np.dot(v_c, M[c].dot(v_c))
            return cost // 2

        # Simulated Annealing
        best_x = None
        best_cost = float('inf')
        
        # Multiple restarts
        for restart in range(20):
            if not _time_left():
                break
            
            x = np.random.randint(0, 4, size=current_n)
            current_cost = compute_cost(x)
            
            if current_cost == 0:
                return current_colors + [int(c) for c in x]
            
            # SA Parameters
            T = 1.0
            cooling_rate = 0.95
            steps = 500
            
            for step in range(steps):
                if not _time_left():
                    break
                
                # Pick random vertex to change
                i = np.random.randint(current_n)
                old_c = x[i]
                new_c = np.random.randint(4)
                if new_c == old_c:
                    continue
                
                # Compute delta cost
                # Delta = cost(x with x[i]=new_c) - cost(x)
                # This is O(n) if we do it smartly, but O(n^2) is fine for n=50
                # Let's just recompute or use delta formula.
                # Delta formula:
                # Removing old_c from i:
                # - For each j: if x[j]==old_c and M[old_c][i][j]==1, cost decreases by 1
                # Adding new_c to i:
                # + For each j: if x[j]==new_c and M[new_c][i][j]==1, cost increases by 1
                
                delta = 0
                # Subtract contribution of old_c at i
                # v_old had 1 at i. Now 0.
                # Term was v_old[i] * (M[old_c] @ v_old)[i]
                # (M[old_c] @ v_old)[i] = sum_j M[old_c][i][j] v_old[j]
                # This sum includes j=i? M[i][i]=0.
                # So we subtract sum_{j != i} M[old_c][i][j] * [x[j]==old_c]
                # Note: The total cost was 0.5 * sum_i v[i] (Mv)[i].
                # Change in (Mv)[i] for all j?
                # It's easier to just compute the difference in the full sum.
                
                # Let's use the O(n) delta calculation
                # cost = sum_c 0.5 * v_c^T M_c v_c
                # Change v_old: v_old[i] goes 1 -> 0.
                # Change v_new: v_new[i] goes 0 -> 1.
                
                # Contribution of v_old change:
                # -0.5 * ( (M_old v_old)[i] + sum_j M_old[i][j] v_old[j] ) ?
                # Actually, d(v^T M v) = 2 v^T M dv (if M symmetric).
                # Here v changes at index i by -1.
                # Delta_old = -1 * (M_old v_old)[i]
                # Delta_new = +1 * (M_new v_new)[i]
                
                # (M_old v_old)[i] = sum_j M_old[i][j] v_old[j]
                # Note v_old[j] is 1 if x[j]==old_c.
                
                sum_old = np.dot(M[old_c][i], (x == old_c).astype(np.int8))
                sum_new = np.dot(M[new_c][i], (x == new_c).astype(np.int8))
                
                # Wait, the factor 0.5 applies to the whole sum.
                # cost = 0.5 * Sum.
                # Delta cost = 0.5 * (Delta Sum).
                # Delta Sum = - (M_old v_old)[i] + (M_new v_new)[i]
                
                delta_cost = 0.5 * (sum_new - sum_old)
                
                # Note: sum_old includes j=i? M[i][i]=0. So no.
                # But v_old[i] was 1.
                # When we change x[i], v_old[i] becomes 0.
                # The term (M_old v_old)[i] uses the *old* v_old.
                # So sum_old is correct using current x.
                
                # However, we must be careful:
                # The term for j=i in the sum for other indices?
                # The formula v^T M v = sum_i sum_j v[i] M[i][j] v[j].
                # Changing v[i] affects terms where i is the first index AND where i is the second index.
                # Since M is symmetric, it's 2 * sum_j v[i] M[i][j] v[j].
                # So the change is -2 * sum_{j != i} M[old][i][j] v_old[j] + 2 * sum_{j != i} M[new][i][j] v_new[j].
                # Multiplied by 0.5, it is - sum_{j != i} ... + sum_{j != i} ...
                # My sum_old and sum_new include j=i? M[i][i]=0, so it's fine.
                
                if delta_cost <= 0 or np.random.random() < np.exp(-delta_cost / T):
                    x[i] = new_c
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
                    if current_cost == 0:
                        return current_colors + [int(c) for c in x]
                
                T *= cooling_rate
                
        if best_cost == 0:
            return current_colors + [int(c) for c in best_x]
        return None

    current_colors = colors_seed[:]
    current_n = n_seed
    
    for target in range(current_n + 1, 51):
        if not _time_left():
            break
        
        new_colors = extend_by_one(current_colors, current_n)
        if new_colors is not None:
            if _validate(new_colors, current_n + 1):
                current_colors = new_colors
                current_n += 1
                best_n = current_n
                best_colors = current_colors[:]
            else:
                break
        else:
            break
    
    return {'n': best_n, 'colors': best_colors}
