"""Four-colouring of K_50 with no monochromatic triangle (Chung's construction)."""
import time
import random
import math

# EVOLVE-BLOCK-START

# --- Utility to validate a 4-colouring for triangle-free colour classes ---
def _validate_simple(coloring, n):
    """Simple O(n^3) validation."""
    # Build edge colour lookup
    edge_color = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = coloring[idx]
            idx += 1
            edge_color[i][j] = c
            edge_color[j][i] = c
    for a in range(n):
        for b in range(a + 1, n):
            c = edge_color[a][b]
            for d in range(b + 1, n):
                if edge_color[a][d] == c and edge_color[b][d] == c:
                    return False
    return True

# --- Chung's 50-vertex construction using Z_51 Paley graph ---
def _build_chung_50():
    """
    Chung's construction: 50-vertex 4-colouring of K_50 with no monochromatic triangle.
    Based on the Paley graph of order 51 (Z_51).
    """
    # Precompute QR mod 17
    qr_17 = set()
    for k in range(17):
        qr_17.add((k * k) % 17)
    # QR mod 17: {0, 1, 2, 4, 8, 9, 13, 15, 16}
    # Non-zero QR mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    
    def get_color(i, j):
        d = (j - i) % 51
        d3 = d % 3
        d17 = d % 17
        if d17 == 0:
            # d is divisible by 17 but not 51, so d mod 3 == 1 or 2
            # d17 == 0 means d ∈ {17, 34} mod 51
            # 17 mod 3 = 2, 34 mod 3 = 1
            # Treat d17=0 as QR (since 0 is a square)
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
            # Vertices are 1..50, so vertex i (0-indexed) = i+1
            vi = i + 1
            vj = j + 1
            colors.append(get_color(vi, vj))
    
    # Validate
    if not _validate_simple(colors, n):
        # Fall back to a known valid construction
        return _build_32()
    
    return {'n': n, 'colors': colors}

def _build_32():
    """Greenwood-Gleason colouring of K16 doubled."""
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

def _solve_n(n, rng, deadline):
    """
    Attempt to find a valid 4-coloring of K_n such that no monochromatic triangle exists.
    Returns the colors list if successful, None otherwise.
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

    # Initialize random coloring
    colors = [rng.randint(0, 3) for _ in range(m)]
    
    # Initialize masks
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)

    # Calculate initial triangle count
    tri_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                c1 = colors[idx[u][v]]
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    tri_count += 1

    if tri_count == 0:
        return colors

    # Simulated Annealing / Local Search
    T = 1.0
    cooling = 0.9995
    steps = 0
    max_steps = 500000 # Safety cap, time is the real limit
    
    # We want to minimize tri_count
    while T > 0.001 and time.time() < deadline - 2.0:
        # Pick random edge
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
        
        # Calculate delta in triangle count
        lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
        gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
        
        delta = gained - lost
        
        # Accept if improvement or with probability exp(-delta/T)
        accept = False
        if delta < 0:
            accept = True
        else:
            # Avoid overflow in exp
            if delta > 50 * T:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            tri_count += delta
            
            # Update masks
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            if tri_count == 0:
                return colors
            
            T *= cooling
            steps += 1

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # First try the deterministic Chung construction
    try:
        result = _build_chung_50()
        if result['n'] >= 50:
            return result
    except Exception as e:
        pass
    
    # Fallback to search-based approach starting from 33
    rng = random.Random(42)
    best = _build_32()
    best_n = 32
    
    current_n = 33
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for this n
        sub_deadline = time.time() + min(10.0, deadline - time.time() - 2.0)
        
        # Multiple restarts within sub_deadline
        res = None
        while time.time() < sub_deadline:
            res = _solve_n(current_n, rng, sub_deadline)
            if res is not None:
                break
        
        if res is not None:
            best = {'n': current_n, 'colors': res}
            best_n = current_n
            current_n += 1
        else:
            break
    
    return best

# EVOLVE-BLOCK-END
