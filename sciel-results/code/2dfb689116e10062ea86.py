import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s for overhead
    
    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def get_color(i, j, colors, n):
        if i > j:
            i, j = j, i
        return colors[get_idx(i, j, n)]

    def set_color(i, j, c, colors, n):
        if i > j:
            i, j = j, i
        colors[get_idx(i, j, n)] = c

    def solve_for_n(n, max_time=50):
        """
        Attempts to find a 5-coloring of K_n with no monochromatic C4.
        Returns (n, colors) if successful, else (0, None).
        """
        n_edges = n * (n - 1) // 2
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # N[c][v] is a bitmask of neighbors of v with color c
        N = [[0] * n for _ in range(5)]
        colors = [0] * n_edges
        
        # Initialize randomly
        random.seed(42) # Fixed seed for reproducibility in debugging, but we can vary
        for idx in range(n_edges):
            colors[idx] = random.randint(0, 4)
            i, j = edge_list[idx]
            c = colors[idx]
            N[c][i] |= (1 << j)
            N[c][j] |= (1 << i)
            
        def calculate_energy():
            E = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        common = N[c][u] & N[c][v]
                        cnt = bin(common).count('1')
                        if cnt >= 2:
                            E += cnt * (cnt - 1) // 2
            return E

        current_E = calculate_energy()
        best_E = current_E
        best_colors = colors[:]
        
        if current_E == 0:
            return n, colors
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        end_time = start_time + max_time
        
        while time.time() < end_time and T > min_T:
            # Pick a random edge
            idx = random.randint(0, n_edges - 1)
            u, v = edge_list[idx]
            c_old = colors[idx]
            
            # Try a random new color
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            # Calculate Delta E
            # E = sum_c sum_{u<v} C(|N_c(u) & N_c(v)|, 2)
            
            delta_E = 0
            
            # Effect on c_old
            # Removing edge (u,v) from c_old
            # 1. The term for pair (u,v) in c_old changes.
            #    Old common count: cnt_old_uv = popcount(N[c_old][u] & N[c_old][v])
            #    Note: N[c_old][u] contains v, N[c_old][v] contains u.
            #    So the intersection includes each other?
            #    Wait, N[c][u] is neighbors of u. v is a neighbor of u.
            #    So v is in N[c_old][u].
            #    u is in N[c_old][v].
            #    Is u in N[c_old][u]? No.
            #    Is v in N[c_old][v]? No.
            #    So u is not in N[c_old][u] & N[c_old][v].
            #    v is not in N[c_old][u] & N[c_old][v].
            #    So the intersection does NOT contain u or v.
            #    Therefore, removing the edge (u,v) from color c_old does NOT change
            #    the common neighbors of u and v in c_old?
            #    Wait.
            #    N[c][u] is the set of vertices w such that color(u,w)=c.
            #    If we change color(u,v) from c to c', then v is removed from N[c][u] and u from N[c][v].
            #    Does this affect N[c][x] & N[c][y] for other pairs?
            #    Yes, for pairs (u, w) and (v, w).
            
            # Let's compute delta_E by recalculating the affected terms.
            # Affected terms in c_old:
            # 1. Term (u, v): 
            #    Old common: N[c_old][u] & N[c_old][v].
            #    Since v in N[c_old][u] and u in N[c_old][v], but u not in N[c_old][u] and v not in N[c_old][v],
            #    the intersection does not contain u or v.
            #    So removing the edge does not change the intersection set?
            #    Wait. The intersection is the set of COMMON neighbors.
            #    If we remove edge (u,v), does it remove a common neighbor?
            #    A common neighbor k must satisfy color(u,k)=c and color(v,k)=c.
            #    Changing color(u,v) does not change color(u,k) or color(v,k) for k != v,u.
            #    So the set of common neighbors for pair (u,v) does NOT change?
            #    Yes, it does not change.
            #    So the term for (u,v) in c_old remains the same value?
            #    Wait, the term is C(cnt, 2). If cnt doesn't change, the term doesn't change.
            #    
            # 2. Terms (u, w) for w != u, v.
            #    N[c_old][u] changes: v is removed.
            #    So N[c_old][u] & N[c_old][w] might lose v.
            #    v is in the intersection if color(u,v)=c_old and color(v,w)=c_old.
            #    We know color(u,v)=c_old.
            #    So if color(v,w)=c_old, then v was in the intersection. Now it is removed.
            #    So cnt(u,w) decreases by 1 if color(v,w)=c_old.
            #    Similarly for (v, w): N[c_old][v] loses u.
            #    u is in intersection if color(v,u)=c_old and color(u,w)=c_old.
            #    We know color(v,u)=c_old.
            #    So if color(u,w)=c_old, then u was in the intersection. Now it is removed.
            
            # 3. Terms involving other pairs? No, only N[c][u] and N[c][v] changed.
            
            # So for c_old:
            # - Term (u,v): Unchanged.
            # - For each w != u, v:
            #   - If color(v,w) == c_old: cnt(u,w) decreases by 1.
            #   - If color(u,w) == c_old: cnt(v,w) decreases by 1.
            
            # For c_new:
            # - Term (u,v): Unchanged?
            #   Adding edge (u,v) to c_new.
            #   Intersection of N[c_new][u] and N[c_new][v].
            #   u is not in N[c_new][u], v is not in N[c_new][v].
            #   So intersection doesn't contain u or v.
            #   So term (u,v) in c_new is unchanged.
            # - For each w != u, v:
            #   - If color(v,w) == c_new: cnt(u,w) increases by 1 (v added to N[c_new][u]).
            #   - If color(u,w) == c_new: cnt(v,w) increases by 1 (u added to N[c_new][v]).

            def get_cnt(c, a, b):
                return bin(N[c][a] & N[c][b]).count('1')
            
            def delta_term(cnt_old, cnt_new):
                return (cnt_new * (cnt_new - 1) // 2) - (cnt_old * (cnt_old - 1) // 2)

            # Calculate delta for c_old
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check (u, w)
                if get_color(v, w, colors, n) == c_old:
                    cnt_old = get_cnt(c_old, u, w)
                    cnt_new = cnt_old - 1
                    delta_E += delta_term(cnt_old, cnt_new)
                # Check (v, w)
                if get_color(u, w, colors, n) == c_old:
                    cnt_old = get_cnt(c_old, v, w)
                    cnt_new = cnt_old - 1
                    delta_E += delta_term(cnt_old, cnt_new)
            
            # Calculate delta for c_new
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check (u, w)
                if get_color(v, w, colors, n) == c_new:
                    cnt_old = get_cnt(c_new, u, w)
                    cnt_new = cnt_old + 1
                    delta_E += delta_term(cnt_old, cnt_new)
                # Check (v, w)
                if get_color(u, w, colors, n) == c_new:
                    cnt_old = get_cnt(c_new, v, w)
                    cnt_new = cnt_old + 1
                    delta_E += delta_term(cnt_old, cnt_new)
            
            # Accept or reject
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Apply change
                # Update N for c_old
                N[c_old][u] &= ~(1 << v)
                N[c_old][v] &= ~(1 << u)
                # Update N for c_new
                N[c_new][u] |= (1 << v)
                N[c_new][v] |= (1 << u)
                # Update colors
                colors[idx] = c_new
                # Update Energy
                current_E += delta_E
                if current_E < best_E:
                    best_E = current_E
                    best_colors = colors[:]
                    if best_E == 0:
                        return n, best_colors
            
            T *= cooling
            
        if best_E == 0:
            return n, best_colors
        else:
            return 0, None

    # Try to find a solution for n=26
    # If it fails, try 25, etc.
    # Given the known result, 26 should be possible.
    # We allocate most time to 26.
    
    best_n = 0
    best_colors = None
    
    # Try 26 with generous time
    n_found, cols_found = solve_for_n(26, max_time=100)
    if n_found > 0:
        best_n = n_found
        best_colors = cols_found
    else:
        # If 26 fails, try 25
        n_found, cols_found = solve_for_n(25, max_time=40)
        if n_found > 0:
            best_n = n_found
            best_colors = cols_found
        else:
            # Fallback to a small known valid one or random
            # A trivial valid coloring for n=5: all edges same color? No, K5 has C4.
            # n=4: K4 has no C4. So any coloring of K4 is valid.
            # n=5: K5 has C4.
            # Let's just return n=4 if all else fails.
            best_n = 4
            best_colors = [0] * (4 * 3 // 2)

    return {'n': best_n, 'colors': best_colors}
