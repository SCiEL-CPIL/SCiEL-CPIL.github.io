import time
import random
import numpy as np

def _build_gf49():
    """Build GF(49) arithmetic using polynomial x^2 + x + 1 over GF(7)."""
    def gf7_add(a, b):
        return (a + b) % 7
    def gf7_sub(a, b):
        return (a - b) % 7
    def gf7_mul(a, b):
        return (a * b) % 7
    def gf49_add(a, b):
        return (gf7_add(a[0], b[0]), gf7_add(a[1], b[1]))
    def gf49_sub(a, b):
        return (gf7_sub(a[0], b[0]), gf7_sub(a[1], b[1]))
    def gf49_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = gf7_add(gf7_mul(a0, b0), gf7_mul(gf7_mul(a1, b1), 6))
        c1 = gf7_add(gf7_mul(a0, b1), gf7_mul(a1, b0))
        return (c0, c1)
    def gf49_inv(a):
        a0, a1 = a
        for b0 in range(7):
            for b1 in range(7):
                if (b0, b1) == (0, 0):
                    continue
                prod = gf49_mul((a0, a1), (b0, b1))
                if prod == (1, 0):
                    return (b0, b1)
        return (0, 0)
    
    # Find primitive element
    elem = (0, 1)
    elements = [(1, 0)]
    cur = (1, 0)
    for _ in range(48):
        cur = gf49_mul(cur, elem)
        elements.append(cur)
    
    if len(set(elements)) != 48:
        for test in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]:
            elements = [(1, 0)]
            cur = (1, 0)
            for _ in range(48):
                cur = gf49_mul(cur, test)
                elements.append(cur)
            if len(set(elements)) == 48:
                elem = test
                break
    
    gen = (1, 0)
    for _ in range(4):
        gen = gf49_mul(gen, elem)
    
    cosets = [[], [], [], []]
    cur = (1, 0)
    for k in range(12):
        cosets[0].append(cur)
        cur = gf49_mul(cur, gen)
    
    for shift in range(1, 4):
        shift_elem = (1, 0)
        for _ in range(shift):
            shift_elem = gf49_mul(shift_elem, elem)
        for base in cosets[0]:
            cosets[shift].append(gf49_mul(shift_elem, base))
    
    all_elements = [(0, 0)] + elements
    return all_elements, cosets, gf49_mul, gf49_sub, gf49_inv

def _gf49_construction():
    """Construct a 49-vertex coloring using GF(49) with ratio-based coloring."""
    all_elements, cosets, gf49_mul, gf49_sub, gf49_inv = _build_gf49()
    
    coset_map = {}
    for c in range(4):
        for e in cosets[c]:
            coset_map[e] = c
    
    n = 49
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            ui = all_elements[i]
            uj = all_elements[j]
            if ui == (0, 0) or uj == (0, 0):
                c = 0
            else:
                ratio = gf49_mul(ui, gf49_inv(uj))
                c = coset_map.get(ratio, 0)
            colors.append((i, j, c))
    return colors

def _build_adj_matrices(colors, n):
    """Build adjacency matrices for fast triangle checking."""
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for i, j, c in colors:
        adj[c][i, j] = True
        adj[c][j, i] = True
    return adj

def _is_triangle_free_numpy(adj, n):
    """Check triangle-freeness using numpy matrix operations."""
    for c in range(4):
        A = adj[c]
        common = A @ A.T
        if np.any(common * A):
            return False
    return True

def _extend_vertex(adj, n, deadline):
    """Try to extend the coloring to n+1 vertices."""
    new_n = n + 1
    best_cost = float('inf')
    best_assign = None
    
    max_trials = 30
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break
        assign = np.random.randint(0, 4, size=n)
        total_cost = 0
        for c in range(4):
            mask = (assign == c)
            if np.sum(mask) > 1:
                sub = adj[c][np.ix_(mask, mask)]
                total_cost += np.sum(sub) // 2
        
        stuck = 0
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            if total_cost == 0:
                best_cost = 0
                best_assign = assign.copy()
                break
            u = random.randint(0, n - 1)
            old_c = assign[u]
            min_new_cost = total_cost
            best_new_c = old_c
            for new_c in range(4):
                if new_c == old_c:
                    continue
                nbrs_old = np.sum(adj[old_c][u, :] & (assign == old_c))
                nbrs_new = np.sum(adj[new_c][u, :] & (assign == new_c))
                delta = nbrs_new - nbrs_old
                new_cost = total_cost + delta
                if new_cost < min_new_cost:
                    min_new_cost = new_cost
                    best_new_c = new_c
            if best_new_c == old_c:
                stuck += 1
                if stuck > 60:
                    break
            else:
                stuck = 0
                assign[u] = best_new_c
                total_cost = min_new_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_assign = assign.copy()
        if best_cost == 0:
            break
    
    if best_cost == 0 and best_assign is not None:
        new_adj = [np.zeros((new_n, new_n), dtype=bool) for _ in range(4)]
        for c in range(4):
            new_adj[c][:n, :n] = adj[c]
            for i in range(n):
                if best_assign[i] == c:
                    new_adj[c][i, new_n - 1] = True
                    new_adj[c][new_n - 1, i] = True
        return True, new_adj, best_assign
    else:
        return False, None, None

def _base32_colors():
    """Return the 32-vertex base coloring."""
    def _build_mul_table():
        log = {}
        x = 1
        log[1] = 0
        for e in range(15):
            x2 = 0
            for i in range(4):
                if (x >> i) & 1:
                    x2 ^= (1 << (i + 1))
            if x2 & 0b10000:
                x2 ^= 0b10011
            x = x2
            log[x] = e + 1
        return log
    log_table = _build_mul_table()
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                ui = i % 16
                uj = j % 16
                c = log_table[ui ^ uj] % 3
            else:
                c = 3
            colors.append((i, j, c))
    return colors

def construct():
    deadline = time.time() + 110
    n = 32
    adj = None
    try:
        base_colors = _gf49_construction()
        n = 49
        adj = _build_adj_matrices(base_colors, n)
        if not _is_triangle_free_numpy(adj, n):
            raise ValueError("GF49 construction has triangles")
    except Exception:
        base_colors = _base32_colors()
        n = 32
        adj = _build_adj_matrices(base_colors, n)
    
    while n < 50 and time.time() < deadline - 15:
        success, new_adj, _ = _extend_vertex(adj, n, deadline)
        if success:
            n += 1
            adj = new_adj
        else:
            break
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            for c in range(4):
                if adj[c][i, j]:
                    colors.append((i, j, c))
                    break
    out_colors = [c for _, _, c in colors]
    return {'n': n, 'colors': out_colors}
