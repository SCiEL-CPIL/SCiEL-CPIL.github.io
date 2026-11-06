"""Model-and-optimize: construct large triangle-free 4-colorings via Paley base + bitmask SA."""
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

def _paley25():
    """Paley graph coloring of K_25 using quadratic residues mod 25.
    This gives a 3-coloring of K_25 with no monochromatic triangles (using colors 0,1,2),
    and we can add a 4th color for inter-copy edges if we double it."""
    n = 25
    # Quadratic residues mod 25
    QR = set()
    for x in range(1, 25):
        QR.add((x * x) % 25)
    QR.discard(0)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % 25
            if d == 0:
                c = 0  # shouldn't happen since i < j
            elif d in QR:
                c = 0
            elif (-d) % 25 in QR:
                c = 1
            else:
                c = 2
            colors.append(c)
    return {'n': n, 'colors': colors}

def _build_masks(colors, n):
    """Build bitmask adjacency for each vertex and color."""
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
    return idx, masks

def _count_triangles(colors, n, idx):
    """Count monochromatic triangles."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            c_uv = colors[idx[u][v]]
            for w in range(v + 1, n):
                if colors[idx[v][w]] == c_uv and colors[idx[w][u]] == c_uv:
                    count += 1
    return count

def _sa_solve(n, rng, deadline, init_colors=None):
    """Simulated annealing to find triangle-free 4-coloring of K_n."""
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    idx, masks = _build_masks(init_colors if init_colors else [rng.randint(0, 3) for _ in range(m)], n)
    colors = [init_colors[i] if init_colors else rng.randint(0, 3) for i in range(m)]
    
    # Count initial triangles
    tri_count = _count_triangles(colors, n, idx)
    if tri_count == 0:
        return colors
    
    T = 1.0
    cooling = 0.9997
    best_tri = tri_count
    best_colors = colors[:]
    no_improve = 0
    max_no_improve = 5000
    
    while T > 0.001 and time.time() < deadline - 1.0:
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
        elif delta == 0:
            accept = rng.random() < 0.5
        else:
            if delta > 50 * T:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            tri_count += delta
            
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if tri_count == 0:
                return colors
            
            if no_improve > max_no_improve:
                # Restart with best
                colors = best_colors[:]
                idx, masks = _build_masks(colors, n)
                tri_count = best_tri
                no_improve = 0
                T = max(T, 0.5)
            
            T *= cooling
    
    return best_colors if best_tri == 0 else None

def _try_grow(base_colors, base_n, rng, deadline):
    """Try to grow a valid coloring by adding vertices via SA on larger n."""
    best_colors = base_colors[:]
    best_n = base_n
    
    for target_n in range(base_n + 1, min(base_n + 25, 51)):
        if time.time() > deadline - 3.0:
            break
        
        # Try multiple restarts
        found = None
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts and time.time() < deadline - 3.0:
            # Initialize with base coloring extended randomly
            m = target_n * (target_n - 1) // 2
            init = [rng.randint(0, 3) for _ in range(m)]
            # Copy base coloring into init
            idx_base = 0
            for i in range(base_n):
                for j in range(i + 1, base_n):
                    init[idx_base] = base_colors[idx_base]
                    idx_base += 1
            
            found = _sa_solve(target_n, rng, deadline - 2.0, init_colors=init)
            if found is not None:
                break
            attempts += 1
        
        if found is not None:
            best_colors = found
            best_n = target_n
        else:
            break
    
    return best_colors, best_n

def _validate(colors, n):
    """Validate that no monochromatic triangle exists."""
    if len(colors) != n * (n - 1) // 2:
        return False
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            for w in range(v + 1, n):
                if colors[idx[v][w]] == c and colors[idx[w][u]] == c:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 115.0  # 5s margin before 120s kill
    rng = random.Random(42)
    
    # Start with the best known baseline
    best_n = 32
    best_colors = _baseline32()['colors'][:]
    
    # Try to improve using SA from random starts for various n
    # Focus on n=33 to n=50
    
    # First, try to find a valid coloring for n=33-40 quickly
    for target_n in range(33, 41):
        if time.time() > deadline - 10.0:
            break
        
        found = None
        # Multiple restarts
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            found = _sa_solve(target_n, rng, deadline - 10.0)
            if found is not None:
                break
        
        if found is not None:
            best_n = target_n
            best_colors = found
        else:
            # If we can't find n, try n+1 might be harder, so break
            # But maybe a different random seed helps? Try one more with different seed
            if time.time() < deadline - 20.0:
                rng2 = random.Random(attempt * 1000 + target_n)
                found = _sa_solve(target_n, rng2, deadline - 10.0)
                if found is not None:
                    best_n = target_n
                    best_colors = found
                else:
                    break
            else:
                break
    
    # If we have a valid coloring, try to grow it
    if best_n < 50 and time.time() < deadline - 15.0:
        grown_colors, grown_n = _try_grow(best_colors, best_n, rng, deadline)
        if grown_n > best_n:
            best_n = grown_n
            best_colors = grown_colors
    
    # Try direct SA for larger n if we have time
    if time.time() < deadline - 10.0:
        for target_n in range(max(best_n + 1, 40), 51):
            if time.time() > deadline - 5.0:
                break
            found = _sa_solve(target_n, rng, deadline - 5.0)
            if found is not None:
                best_n = target_n
                best_colors = found
                break
    
    # Final validation
    if time.time() < deadline - 1.0:
        if not _validate(best_colors, best_n):
            # Fallback to baseline
            best_n = 32
            best_colors = _baseline32()['colors'][:]
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
