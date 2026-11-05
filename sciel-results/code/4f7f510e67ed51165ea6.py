import time
import random
import numpy as np

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    for c in range(5):
        A2 = adj[c] @ adj[c]
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
    return True

def _construct_gf25():
    """
    Lazebnik-Woldar style construction for n=26 using GF(25).
    Returns a dictionary with 'n' and 'colors'.
    """
    p = 5
    elements = [(a, b) for a in range(p) for b in range(p)]
    
    def gf_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    def gf_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n % 2 == 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            n //= 2
        return result
    
    primitive = None
    for elem in elements:
        if elem == (0, 0):
            continue
        if gf_pow(elem, 24) == (1, 0):
            is_primitive = True
            for d in [1, 2, 3, 4, 6, 8, 12]:
                if gf_pow(elem, d) == (1, 0):
                    is_primitive = False
                    break
            if is_primitive:
                primitive = elem
                break
                
    if primitive is None:
        return None

    powers = []
    current = (1, 0)
    for _ in range(24):
        powers.append(current)
        current = gf_mul(current, primitive)
        
    log_map = {elem: i for i, elem in enumerate(powers)}
    
    n = 26
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == 25 or j == 25:
                other = j if i == 25 else i
                if other == 0:
                    colors.append(0)
                else:
                    # Use mod 5 for better distribution?
                    # The seed used mod 4. Let's stick to mod 4 for the base
                    # but ensure we have 5 colors available for extension.
                    # Actually, using mod 5 might be better for the CSP.
                    # Let's try mod 5.
                    c = log_map[elements[other]] % 5
                    colors.append(c)
            else:
                elem_i = elements[i]
                elem_j = elements[j]
                if elem_i == (0, 0) or elem_j == (0, 0):
                    colors.append(0)
                else:
                    log_i = log_map[elem_i]
                    log_j = log_map[elem_j]
                    c = (log_i - log_j) % 5
                    colors.append(c)
                    
    return {'n': n, 'colors': colors}

def _extend_csp(base_colors, n_base, n_target, time_budget=10.0):
    """
    Try to extend the coloring from n_base to n_target using CSP/Local Search.
    """
    if n_target <= n_base:
        return list(base_colors)
    
    colors = list(base_colors)
    n = n_base
    
    # Precompute conflict graphs H_k for the current n
    # H_k[u, v] = 1 if u, v have a common neighbor in color k
    
    def build_conflict_graphs(n, colors):
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                idx += 1
        
        H = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        for k in range(5):
            # Common neighbors: A^2
            A2 = adj[k] @ adj[k]
            np.fill_diagonal(A2, 0)
            # H_k[u, v] is 1 if they have at least one common neighbor
            H[k] = (A2 > 0).astype(np.int8)
        return H
    
    # We will add vertices one by one
    while n < n_target:
        if _time_left() < time_budget:
            break
            
        H = build_conflict_graphs(n, colors)
        
        # We need to assign colors c[0..n-1] for the new vertex (index n)
        # such that for each k, {v | c[v] == k} is independent in H[k]
        
        def solve_csp(n, H, max_iters=2000):
            # Random initialization
            c = [random.randint(0, 4) for _ in range(n)]
            
            def count_conflicts(c):
                confs = 0
                for k in range(5):
                    # Check all pairs in H[k]
                    # Efficient way: for each v, check neighbors in H[k]
                    # But H[k] is dense.
                    # Let's just iterate over edges in H[k]
                    # Since n is small (< 30), O(n^2) is fine
                    for u in range(n):
                        for v in range(u + 1, n):
                            if H[k][u, v] == 1 and c[u] == k and c[v] == k:
                                confs += 1
                return confs
            
            best_c = c[:]
            best_conf = count_conflicts(best_c)
            
            for _ in range(max_iters):
                if best_conf == 0:
                    return best_c
                
                # Pick a random vertex involved in a conflict
                # Or just pick a random vertex
                v = random.randint(0, n - 1)
                
                # Try all 5 colors
                old_c = c[v]
                min_conf = best_conf
                best_color = old_c
                
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    c[v] = new_c
                    conf = count_conflicts(c)
                    if conf < min_conf:
                        min_conf = conf
                        best_color = new_c
                c[v] = best_color
                best_conf = min_conf
                best_c = c[:]
                
                if best_conf == 0:
                    return best_c
            
            return best_c if best_conf == 0 else None

        solution = solve_csp(n, H)
        
        if solution is None:
            # Try a few restarts
            found = False
            for _ in range(5):
                solution = solve_csp(n, H, max_iters=500)
                if solution is not None:
                    found = True
                    break
            if not found:
                return None
        
        # Append the new colors to the list
        # The new vertex is index n.
        # Edges are (n, v) for v in 0..n-1.
        # In lexicographic order, these come after all (i, j) with i, j < n.
        # So we just append solution.
        colors.extend(solution)
        n += 1
        
    if n == n_target:
        return colors
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        # Fallback
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    # 1. Get base n=26
    res_26 = _construct_gf25()
    if not res_26:
        return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
    
    base_colors = res_26['colors']
    
    # 2. Try to extend to 28
    # We have ~140s left.
    
    # Try 28 first
    if _time_left() > 30:
        ext_28 = _extend_csp(base_colors, 26, 28, time_budget=15.0)
        if ext_28 is not None:
            if _validate(28, ext_28):
                return {'n': 28, 'colors': ext_28}
            
    # Try 27
    if _time_left() > 15:
        ext_27 = _extend_csp(base_colors, 26, 27, time_budget=10.0)
        if ext_27 is not None:
            if _validate(27, ext_27):
                return {'n': 27, 'colors': ext_27}
    
    # Fallback to 26
    if _validate(26, base_colors):
        return {'n': 26, 'colors': base_colors}
        
    # Fallback to 11
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
