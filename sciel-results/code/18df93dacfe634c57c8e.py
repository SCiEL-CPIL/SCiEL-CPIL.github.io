"""Model-and-optimize: construct large triangle-free 4-colorings via Chung's construction + bitmask local search."""
import random
import time
import math
import sys

# EVOLVE-BLOCK-START

def _baseline32():
    """Greenwood-Gleason K16 doubled -> K32."""
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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def _build_chung_50():
    """
    Chung's 50-vertex 4-colouring of K_50 with no monochromatic triangle.
    Based on the Paley graph of order 51 (Z_51), using QR mod 17.
    """
    qr_17 = set()
    for k in range(17):
        qr_17.add((k * k) % 17)
    # Non-zero QR mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    
    def get_color(i, j):
        d = (j - i) % 51
        d3 = d % 3
        d17 = d % 17
        if d17 == 0:
            in_qr = True
        else:
            in_qr = (d17 in qr_17)
        
        if d3 == 1:
            return 0 if in_qr else 1
        else:  # d3 == 2
            return 2 if in_qr else 3
    
    n = 50
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            vi = i + 1
            vj = j + 1
            colors.append(get_color(vi, vj))
    
    return {'n': n, 'colors': colors}

def _validate_simple(colors, n):
    """O(n^3) validation that no monochromatic triangle exists."""
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    return False
    return True

def _extend_coloring(base_colors, base_n, target_n, rng):
    """
    Extend a valid coloring of K_{base_n} to K_{target_n} by greedily
    adding new vertices one at a time, choosing edge colors to minimize
    new monochromatic triangles.
    """
    if target_n <= base_n:
        return base_colors[:]
    
    m_target = target_n * (target_n - 1) // 2
    colors = [0] * m_target
    
    # Build index for target
    idx = [[0] * target_n for _ in range(target_n)]
    k = 0
    for i in range(target_n):
        for j in range(i + 1, target_n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    
    # Map base edge (i,j) with i<j<base_n to position in new array
    base_idx = [[0] * base_n for _ in range(base_n)]
    k = 0
    for i in range(base_n):
        for j in range(i + 1, base_n):
            base_idx[i][j] = k
            base_idx[j][i] = k
            k += 1
    
    # Copy base coloring
    for i in range(base_n):
        for j in range(i + 1, base_n):
            colors[idx[i][j]] = base_colors[base_idx[i][j]]
    
    # Now greedily color edges involving new vertices
    # For each new vertex, color its edges to all previous vertices
    for new_v in range(base_n, target_n):
        for other in range(new_v):
            # Choose color for edge (other, new_v)
            best_c = 0
            best_tri = float('inf')
            for c in range(4):
                tri = 0
                for kk in range(other):
                    # Triangle (kk, other, new_v)
                    # Edges: (kk, other) - set, (other, new_v) - being set to c, (kk, new_v) - may or may not be set
                    e_kk_other = colors[idx[kk][other]]
                    # (kk, new_v): kk < new_v, and if kk >= base_n, it was set earlier
                    # if kk < base_n, it's not set yet
                    if kk >= base_n:
                        e_kk_newv = colors[idx[kk][new_v]]
                        if e_kk_other == c and e_kk_newv == c:
                            tri += 1
                if tri < best_tri:
                    best_tri = tri
                    best_c = c
            colors[idx[other][new_v]] = best_c
    
    # Now color edges between new vertices
    for i in range(base_n, target_n):
        for j in range(i + 1, target_n):
            best_c = 0
            best_tri = float('inf')
            for c in range(4):
                tri = 0
                for kk in range(target_n):
                    if kk == i or kk == j:
                        continue
                    e_ik = colors[idx[i][kk]]
                    e_jk = colors[idx[j][kk]]
                    if e_ik == c and e_jk == c:
                        tri += 1
                if tri < best_tri:
                    best_tri = tri
                    best_c = c
            colors[idx[i][j]] = best_c
    
    return colors

def _count_triangles(colors, n, idx):
    """Count monochromatic triangles."""
    tri_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    tri_count += 1
    return tri_count

def _solve_n(n, rng, deadline, init_colors=None):
    """
    Attempt to find a valid 4-coloring of K_n with no monochromatic triangle.
    Uses bitmask local search (simulated annealing) to minimize triangle count.
    """
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    # Precompute edge indices
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1

    # Initialize coloring
    if init_colors is not None:
        colors = init_colors[:]
    else:
        # Random initialization
        colors = [rng.randint(0, 3) for _ in range(m)]
    
    # Initialize masks
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)

    # Calculate initial triangle count
    tri_count = _count_triangles(colors, n, idx)

    if tri_count == 0:
        return colors

    # Simulated Annealing with adaptive parameters
    T = max(2.0, tri_count / 30.0)
    cooling = 0.9995
    min_T = 0.001
    best_tri = tri_count
    best_colors = colors[:]
    
    steps_since_improve = 0
    improve_threshold = 3000
    
    while time.time() < deadline - 0.3:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        i_idx = idx[u][v]
        old_c = colors[i_idx]
        new_c = rng.randint(0, 3)
        if old_c == new_c:
            continue
        
        lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
        gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
        delta = gained - lost
        
        accept = False
        if delta < 0:
            accept = True
        else:
            # Accept bad moves with probability exp(-delta/T), but cap delta
            if delta <= 20:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            tri_count += delta
            
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            if tri_count == 0:
                return colors
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
            
            T *= cooling
            
            # Restart if stuck
            if steps_since_improve > improve_threshold and T > min_T * 5:
                T = max(T * 2.5, 2.0)
                steps_since_improve = 0
                improve_threshold = int(improve_threshold * 1.3)

    # Return best found
    if best_tri == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # First: try the deterministic Chung 50-vertex construction
    try:
        chung = _build_chung_50()
        if _validate_simple(chung['colors'], chung['n']):
            best = chung
            best_n = 50
        else:
            best = _baseline32()
            best_n = 32
    except Exception:
        best = _baseline32()
        best_n = 32
    
    # Now try to extend beyond best_n using local search
    rng = random.Random(42)
    current_n = best_n + 1
    
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
        
        remaining = deadline - time.time() - 3.0
        if remaining < 5.0:
            break
        
        # Use extended coloring as initial guess
        init = _extend_coloring(best['colors'], best_n, current_n, rng)
        
        # For the first attempt, give it most of the remaining time
        # Use multiple shorter restarts to explore different basins
        if remaining > 30:
            # Multiple restarts with greedy init
            sub_deadline = time.time() + min(remaining - 3.0, 20.0)
            res = _solve_n(current_n, rng, sub_deadline, init_colors=init)
            
            if res is None and time.time() < deadline - 5.0:
                # Try with slightly perturbed init
                init2 = init[:]
                m = current_n * (current_n - 1) // 2
                for _ in range(min(50, m // 10)):
                    pos = rng.randint(0, m - 1)
                    init2[pos] = rng.randint(0, 3)
                sub_deadline2 = time.time() + min(deadline - time.time() - 4.0, 20.0)
                res2 = _solve_n(current_n, rng, sub_deadline2, init_colors=init2)
                if res2 is not None:
                    res = res2
        else:
            sub_deadline = time.time() + min(remaining - 2.0, 15.0)
            res = _solve_n(current_n, rng, sub_deadline, init_colors=init)
        
        if res is not None:
            if _validate_simple(res, current_n):
                best = {'n': current_n, 'colors': res}
                best_n = current_n
                current_n += 1
            else:
                break
        else:
            # Try one more time with full remaining budget
            remaining2 = deadline - time.time() - 3.0
            if remaining2 > 10.0:
                sub_deadline2 = time.time() + min(remaining2 - 2.0, 30.0)
                init2 = _extend_coloring(best['colors'], best_n, current_n, rng)
                res2 = _solve_n(current_n, rng, sub_deadline2, init_colors=init2)
                if res2 is not None and _validate_simple(res2, current_n):
                    best = {'n': current_n, 'colors': res2}
                    best_n = current_n
                    current_n += 1
                else:
                    break
            else:
                break
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if len(colors) != n * (n - 1) // 2:
            best = _baseline32()
        else:
            if not _validate_simple(colors, n):
                best = _baseline32()
    
    return best

# EVOLVE-BLOCK-END
