import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit

    # Known best solution: Paley tournament on p=67
    p_known = 67
    squares = {x * x % p_known for x in range(1, p_known)}
    orientation_known = []
    for i in range(p_known):
        for j in range(i + 1, p_known):
            if (j - i) % p_known in squares:
                orientation_known.append(0)
            else:
                orientation_known.append(1)
    
    best_n = p_known
    best_orientation = orientation_known

    # Attempt to find a solution for n = 66
    def try_find_n66(max_time):
        n = 66
        m = n * (n - 1) // 2
        
        # Precompute edge endpoints
        edge_ends = [None] * m
        idx_map = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_ends[idx] = (i, j)
                idx_map[i][j] = idx
                idx_map[j][i] = idx
                idx += 1

        # Helper to check if a set of 4 is good
        def is_good(orient, a, b, c, d):
            # Check if any vertex dominates the other 3
            # v_dom_w(v, w)
            def v_dom_w(v, w):
                if v < w:
                    return orient[idx_map[v][w]] == 0
                else:
                    return orient[idx_map[v][w]] == 1
            
            if (v_dom_w(a,b) and v_dom_w(a,c) and v_dom_w(a,d)): return True
            if (v_dom_w(b,a) and v_dom_w(b,c) and v_dom_w(b,d)): return True
            if (v_dom_w(c,a) and v_dom_w(c,b) and v_dom_w(c,d)): return True
            if (v_dom_w(d,a) and v_dom_w(d,b) and v_dom_w(d,c)): return True
            return False

        # Initialize orientation from Paley(67) minus vertex 0
        # Paley(67) orientation:
        # We need to map vertices 1..66 of Paley(67) to 0..65 of new tournament.
        # Let new vertex i correspond to old vertex i+1.
        orient = [0] * m
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Old vertices are i+1 and j+1
                old_i = i + 1
                old_j = j + 1
                # In Paley(67), edge (old_i, old_j)
                # Bit is 0 if old_i -> old_j, which happens if (old_j - old_i) % 67 is a square.
                if (old_j - old_i) % p_known in squares:
                    orient[idx] = 0
                else:
                    orient[idx] = 1
                idx += 1

        # Calculate initial bad count
        bad_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not is_good(orient, a, b, c, d):
                            bad_count += 1
        
        if bad_count == 0:
            return orient

        best_bad = bad_count
        best_orient = orient[:]
        
        T = 1.0
        min_T = 0.01
        cooling = 0.9999
        
        while time.time() - start_time < max_time:
            if T < min_T:
                break
                
            u, v = edge_ends[random.randint(0, m - 1)]
            
            # Find all quads containing u and v
            # Iterate x, y from remaining vertices
            delta_bad = 0
            
            # Optimization: Precompute list of other vertices?
            others = [i for i in range(n) if i != u and i != v]
            
            for i in range(len(others)):
                x = others[i]
                for j in range(i + 1, len(others)):
                    y = others[j]
                    
                    q = (u, v, x, y)
                    
                    was_good = is_good(orient, *q)
                    
                    # Flip
                    idx_e = idx_map[u][v]
                    old_val = orient[idx_e]
                    orient[idx_e] = 1 - old_val
                    
                    is_good_now = is_good(orient, *q)
                    
                    # Restore
                    orient[idx_e] = old_val
                    
                    if was_good and not is_good_now:
                        delta_bad += 1
                    elif not was_good and is_good_now:
                        delta_bad -= 1
            
            if delta_bad < 0:
                orient[idx_map[u][v]] ^= 1
                bad_count += delta_bad
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_orient = orient[:]
                    if best_bad == 0:
                        return best_orient
            else:
                if random.random() < math.exp(-delta_bad / T):
                    orient[idx_map[u][v]] ^= 1
                    bad_count += delta_bad
            
            T *= cooling

        return None

    # Try to find n=66 with a limited time budget
    # If it fails, we fall back to n=67
    if time.time() - start_time < 20: # Give it 20 seconds
        sol = try_find_n66(20)
        if sol is not None:
            # Verify solution
            n = 66
            idx_map = [[-1] * n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    idx_map[i][j] = idx
                    idx_map[j][i] = idx
                    idx += 1
                    
            def v_dom_w(v, w, orient):
                if v < w:
                    return orient[idx_map[v][w]] == 0
                else:
                    return orient[idx_map[v][w]] == 1

            ok = True
            for a in range(n):
                if not ok: break
                for b in range(a + 1, n):
                    if not ok: break
                    for c in range(b + 1, n):
                        if not ok: break
                        for d in range(c + 1, n):
                            if not (
                                (v_dom_w(a,b,sol) and v_dom_w(a,c,sol) and v_dom_w(a,d,sol)) or
                                (v_dom_w(b,a,sol) and v_dom_w(b,c,sol) and v_dom_w(b,d,sol)) or
                                (v_dom_w(c,a,sol) and v_dom_w(c,b,sol) and v_dom_w(c,d,sol)) or
                                (v_dom_w(d,a,sol) and v_dom_w(d,b,sol) and v_dom_w(d,c,sol))
                            ):
                                ok = False
                                break
            if ok:
                return {'n': 66, 'orientation': sol}

    return {'n': best_n, 'orientation': best_orientation}
