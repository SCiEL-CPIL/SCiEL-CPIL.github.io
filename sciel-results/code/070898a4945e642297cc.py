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

    # We will attempt to find a solution for n = 66.
    # If successful, we return n=66. Otherwise, we return n=67.
    
    def try_find_n66(max_time):
        n = 66
        m = n * (n - 1) // 2
        
        # Precompute edge endpoints and index mapping
        edge_ends = [None] * m
        idx_map = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_ends[idx] = (i, j)
                idx_map[i][j] = idx
                idx_map[j][i] = idx
                idx += 1

        # Precompute all 4-tuples
        # C(66, 4) = 7,362,825
        # This is a large list, but we need it for evaluation.
        # We can store them as tuples of 4 ints.
        quads = []
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        quads.append((a, b, c, d))
        
        if time.time() - start_time > max_time * 0.1:
            return None

        # Helper to check if a set of 4 is good
        # A set {a,b,c,d} is good if one vertex dominates the other 3.
        # v dominates w if v->w.
        # orient[idx_map[v][w]] == 0 means v->w if v<w.
        # orient[idx_map[v][w]] == 1 means w->v if v<w, i.e., v->w is false.
        # Wait, definition:
        # bit 0: i -> j
        # bit 1: j -> i
        # So if v < w:
        #   orient[idx] == 0 => v->w
        #   orient[idx] == 1 => w->v
        # If v > w:
        #   idx = idx_map[w][v]
        #   orient[idx] == 0 => w->v
        #   orient[idx] == 1 => v->w
        
        def is_good(orient, quad):
            a, b, c, d = quad
            # Check a
            if (orient[idx_map[a][b]] == 0 and orient[idx_map[a][c]] == 0 and orient[idx_map[a][d]] == 0):
                return True
            # Check b
            # b->a? if b>a, idx_map[a][b]. bit 1 means a->b is false? No.
            # Let's just use a unified check function to avoid mistakes.
            
            def v_dom_w(v, w, orient):
                if v < w:
                    return orient[idx_map[v][w]] == 0
                else:
                    return orient[idx_map[v][w]] == 1
            
            if v_dom_w(b, a, orient) and v_dom_w(b, c, orient) and v_dom_w(b, d, orient):
                return True
            if v_dom_w(c, a, orient) and v_dom_w(c, b, orient) and v_dom_w(c, d, orient):
                return True
            if v_dom_w(d, a, orient) and v_dom_w(d, b, orient) and v_dom_w(d, c, orient):
                return True
            return False

        # Initial evaluation is too slow if we do it from scratch for every restart.
        # We need a fast local search.
        # Strategy:
        # 1. Start with random orient.
        # 2. Compute bad_count.
        # 3. Loop:
        #    a. Pick random edge (u, v).
        #    b. Identify all quads containing u and v.
        #       There are C(n-2, 2) such quads.
        #       For each pair (x, y) with x < y, x, y not in {u, v}:
        #       Quad is {u, v, x, y}.
        #    c. For each such quad, check if it was good before and is good after.
        #       Update bad_count accordingly.
        #    d. If bad_count decreases, accept. Else, accept with probability exp(-delta/T) or just reject (hill climbing).
        
        # To speed up "identify all quads containing u and v", we can just iterate x, y.
        # C(64, 2) = 2016 quads per edge flip.
        # For each quad, we check 4 vertices * 3 edges = 12 lookups.
        # ~24k operations per flip.
        # In Python, this is maybe 1-2 ms per flip?
        # If we can do 500 flips/second, in 60s we do 30k flips.
        # This might be enough to escape local minima if we use simulated annealing.
        
        def get_bad_count(orient):
            count = 0
            for q in quads:
                if not is_good(orient, q):
                    count += 1
            return count

        # We will try a few random restarts.
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() - start_time > max_time:
                break
            
            orient = [random.randint(0, 1) for _ in range(m)]
            bad_count = get_bad_count(orient)
            
            if bad_count == 0:
                return orient
            
            best_bad = bad_count
            best_orient = orient[:]
            
            T = 0.5
            min_T = 0.01
            cooling = 0.99995
            
            # Simulated Annealing
            while time.time() - start_time < max_time:
                if T < min_T:
                    break
                    
                u, v = edge_ends[random.randint(0, m - 1)]
                
                # Find quads containing u and v
                # We need to iterate all pairs (x, y) from the remaining n-2 vertices.
                # Let's precompute the list of vertices excluding u and v?
                # Or just check inside the loop.
                
                old_bad = 0
                new_bad = 0
                
                # Optimization: Precompute the "state" of the 4 vertices in the quad?
                # No, just check is_good.
                
                # To avoid generating the quad list every time, we can just loop x, y.
                # This is O(n^2) per flip.
                
                # Let's try to optimize the is_good check.
                # Instead of calling is_good which has function call overhead,
                # we can inline it.
                
                # For a specific quad (a,b,c,d), the condition is:
                # (a->b & a->c & a->d) OR (b->a & b->c & b->d) OR ...
                
                # Let's just use the function for now and see if it's fast enough.
                # If not, we will inline.
                
                for x in range(n):
                    if x == u or x == v:
                        continue
                    for y in range(x + 1, n):
                        if y == u or y == v:
                            continue
                        
                        # Quad is {u, v, x, y}
                        # Sort them for is_good? No, is_good handles any order?
                        # My is_good assumes a<b<c<d? No, it takes a quad tuple.
                        # But idx_map works for any pair.
                        # However, my is_good implementation above uses specific variables a,b,c,d.
                        # It doesn't assume order, it just checks all 4.
                        
                        q = (u, v, x, y)
                        
                        # Check before
                        if not is_good(orient, q):
                            old_bad += 1
                            
                        # Flip
                        idx = idx_map[u][v]
                        orient[idx] ^= 1
                        
                        # Check after
                        if not is_good(orient, q):
                            new_bad += 1
                        
                        # Unflip for next iteration?
                        # No, we need to decide if we accept the flip.
                        # But the flip affects ALL quads containing u, v.
                        # We must evaluate the TOTAL change in bad_count.
                        
                        # So we cannot unflip inside the loop.
                        # We must accumulate the change.
                        
                        # Let's restructure:
                        # 1. Calculate delta_bad for this specific quad.
                        # 2. Sum delta_bad over all quads containing u, v.
                        # 3. Decide flip.
                        
                        # Let's restart the logic for this restart.
                        pass
                
                # The above loop structure is flawed for incremental update because
                # we flip the edge inside the loop but need the total delta.
                # We should calculate the total delta first, then flip.
                
                # Revised inner loop:
                delta_bad = 0
                for x in range(n):
                    if x == u or x == v:
                        continue
                    for y in range(x + 1, n):
                        if y == u or y == v:
                            continue
                        
                        q = (u, v, x, y)
                        
                        was_good = is_good(orient, q)
                        
                        # Simulate flip
                        idx = idx_map[u][v]
                        old_val = orient[idx]
                        orient[idx] = 1 - old_val
                        
                        is_good_now = is_good(orient, q)
                        
                        # Restore
                        orient[idx] = old_val
                        
                        if was_good and not is_good_now:
                            delta_bad += 1
                        elif not was_good and is_good_now:
                            delta_bad -= 1
                        
                # Now decide
                if delta_bad < 0:
                    # Improvement
                    orient[idx_map[u][v]] ^= 1
                    bad_count += delta_bad
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_orient = orient[:]
                        if best_bad == 0:
                            return best_orient
                else:
                    # Accept with prob exp(-delta/T)
                    if random.random() < math.exp(-delta_bad / T):
                        orient[idx_map[u][v]] ^= 1
                        bad_count += delta_bad
                
                T *= cooling
                
            # End of SA for this restart
            if best_bad < bad_count:
                # Keep best
                pass

        return None

    # Try to find n=66
    # Budget: 80 seconds
    if time.time() - start_time < 80:
        sol = try_find_n66(80)
        if sol is not None:
            # Verify solution
            n = 66
            m = n * (n - 1) // 2
            # Quick verification
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
            else:
                # Failed verification, return known
                pass

    return {'n': best_n, 'orientation': best_orientation}
