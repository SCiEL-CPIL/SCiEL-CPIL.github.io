import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- GF(16) construction (Greenwood-Gleason): triangle-free 3-colouring of K16 ----
def _gf16():
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            d = (i ^ j) % 16
            if d == 0:
                continue
            e = log[d]
            colors.append(0 if e in (0, 1, 2, 3) else 1)
    return colors

def _base16():
    """Triangle-free 3-colouring of K16, colours in {0,1,2}."""
    c = _gf16()
    return [x % 3 for x in c]

# ---- Product construction: 4-colouring of K48 = 3 groups x 16 ----
def _construct_48():
    base = _base16()
    n = 48
    m = n * (n - 1) // 2
    colors = [0] * m
    idx = 0
    for i in range(n):
        gi, li = i // 16, i % 16
        for j in range(i + 1, n):
            gj, lj = j // 16, j % 16
            if gi == gj:
                colors[idx] = base[li * 15 - li * (li + 1) // 2 + (lj - li - 1)]
            else:
                colors[idx] = 3
            idx += 1
    return {'n': n, 'colors': colors}

def _count_mono_triangles(colors, n):
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _compute_index(i, j, n):
    """Lexicographic index of edge (i,j) with i<j in a K_n edge list."""
    # Sum of (n-1) + (n-2) + ... + (n-i) for pairs (0,j), (1,j), ..., (i-1,j)
    # = i*n - i*(i+1)//2
    # Then within row i: edges are (i,i+1), (i,i+2), ..., (i,j)
    # index within row = j - i - 1
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _find_one_mono_triangle(colors, n):
    """Return (i,j,k,c) of one monochromatic triangle, or None."""
    adj = _build_adj(colors, n)
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

def _greedy_init(n):
    """Greedy random initialization: assign each edge a random color that doesn't create a mono triangle."""
    colors = [0] * (n * (n - 1) // 2)
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # For each color, check if assigning it creates a mono triangle
            # A mono triangle with edge (i,j) in color c exists if there's a vertex k
            # such that (i,k) and (j,k) are both already color c
            # k < i: check adj[c][i] & adj[c][j] restricted to k < i
            # i < k < j: check adj[c][i] & adj[c][j] restricted to (i, j)
            # k > j: not yet assigned, so no issue
            forbidden = set()
            for c in range(4):
                # Check k < i
                if adj[c][i] & adj[c][j] & ((1 << i) - 1):
                    forbidden.add(c)
                # Check i < k < j
                if adj[c][i] & adj[c][j] & (((1 << j) - 1) ^ ((1 << (i + 1)) - 1)):
                    forbidden.add(c)
            available = [c for c in range(4) if c not in forbidden]
            if not available:
                # Shouldn't happen with random, but fallback
                c = random.randint(0, 3)
            else:
                c = random.choice(available)
            colors[idx] = c
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return colors

def _local_search(colors, n, time_limit):
    """Simulated annealing / local search to eliminate mono triangles."""
    deadline = time.time() + time_limit
    best_colors = colors[:]
    best_count = _count_mono_triangles(colors, n)
    
    while time.time() < deadline:
        if best_count == 0:
            break
        
        # Find a random mono triangle
        tri = _find_one_mono_triangle(best_colors, n)
        if tri is None:
            break
        i, j, k, c = tri
        
        # Try to change one of the three edges to a different color
        edges = [(i, j), (i, k), (j, k)]
        random.shuffle(edges)
        
        improved = False
        for (a, b) in edges:
            idx = _compute_index(a, b, n)
            old_c = best_colors[idx]
            new_c = random.choice([cc for cc in range(4) if cc != old_c])
            best_colors[idx] = new_c
            
            new_count = _count_mono_triangles(best_colors, n)
            if new_count < best_count:
                best_count = new_count
                improved = True
                break
            else:
                best_colors[idx] = old_c
        
        if not improved:
            # Try changing a random edge to reduce triangles
            idx = random.randint(0, n * (n - 1) // 2 - 1)
            old_c = best_colors[idx]
            new_c = random.choice([cc for cc in range(4) if cc != old_c])
            best_colors[idx] = new_c
            new_count = _count_mono_triangles(best_colors, n)
            if new_count >= best_count:
                best_colors[idx] = old_c
    
    return best_colors, best_count

def _extend_to_n(base_colors, base_n, target_n):
    """Extend a coloring from base_n to target_n by adding vertices with greedy assignment."""
    if target_n <= base_n:
        return base_colors[:target_n * (target_n - 1) // 2]
    
    colors = base_colors[:]
    adj = [[0] * target_n for _ in range(4)]
    
    # Build adjacency from existing colors
    idx = 0
    for i in range(base_n):
        for j in range(i + 1, base_n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    # Add new vertices one by one
    for new_v in range(base_n, target_n):
        # For each existing vertex i, assign edge (i, new_v)
        # Choose color that doesn't create a mono triangle
        for i in range(new_v):
            idx = _compute_index(i, new_v, target_n)
            forbidden = set()
            for c in range(4):
                # Check if there's a vertex k < i such that (i,k) and (k,new_v) are both color c
                # (k, new_v) not yet assigned, so only check k between i and new_v? No...
                # Actually, a mono triangle with edge (i, new_v) in color c requires a vertex k
                # such that both (i,k) and (k,new_v) are color c.
                # Since new_v is new, (k,new_v) for k < new_v may or may not be assigned.
                # We're assigning edges in order i=0,1,...,new_v-1, so (k,new_v) for k<i is assigned.
                # (k,new_v) for i<k<new_v is not yet assigned.
                # So only k < i can form a triangle with already-assigned edges.
                if adj[c][i] & adj[c][new_v] & ((1 << i) - 1):
                    forbidden.add(c)
            available = [c for c in range(4) if c not in forbidden]
            if not available:
                c = random.randint(0, 3)
            else:
                c = random.choice(available)
            colors[idx] = c
            adj[c][i] |= (1 << new_v)
            adj[c][new_v] |= (1 << i)
    
    return colors

def construct():
    try:
        # Start with deterministic 48-vertex coloring
        res48 = _construct_48()
        best_n = res48['n']
        best_colors = res48['colors']
        
        if _count_mono_triangles(best_colors, best_n) != 0:
            return _seed_32()
        
        # Try to extend to 49, 50, 51...
        for target_n in range(49, 63):
            if _time_left() < 5.0:
                break
            
            # Try multiple random extensions
            for attempt in range(20):
                if _time_left() < 5.0:
                    break
                
                # Greedy extend
                ext_colors = _extend_to_n(res48['colors'], 48, target_n)
                
                # Local search to fix any mono triangles
                ext_colors, count = _local_search(ext_colors, target_n, _time_left() * 0.3)
                
                if count == 0:
                    best_n = target_n
                    best_colors = ext_colors
                    break  # Success, try next target_n
            
            if best_n > 48:
                # Try to extend further from the new best
                res48 = {'n': best_n, 'colors': best_colors}
        
        return {'n': best_n, 'colors': best_colors}
    except Exception:
        try:
            return _seed_32()
        except Exception:
            return {'n': 1, 'colors': []}

def _seed_32():
    """Greenwood-Gleason colouring of K32 (fallback)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
