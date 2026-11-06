# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE = None

def _now():
    return time.time()

def _edge_index(n, i, j):
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def _build_matrix(colors, n):
    m = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            m[i][j] = c
            m[j][i] = c
            idx += 1
    return m

# Precompute the 60 undirected 6-cycles on vertices 0-5
_CYCLES_6 = None
def _get_cycles():
    global _CYCLES_6
    if _CYCLES_6 is None:
        cycles = set()
        for perm in itertools.permutations([1,2,3,4,5]):
            c = (0,) + perm
            rev = (0,) + tuple(reversed(perm))
            key = min(c, rev)
            cycles.add(key)
        _CYCLES_6 = list(cycles)
    return _CYCLES_6

def _count_c6s(colors, n):
    """Count monochromatic 6-cycles."""
    m = _build_matrix(colors, n)
    cycles = _get_cycles()
    total = 0
    for combo in itertools.combinations(range(n), 6):
        for cyc in cycles:
            u0, u1, u2, u3, u4, u5 = combo
            c0 = m[u0][u1]
            if c0 < 0:
                continue
            if (m[u1][u2] == c0 and m[u2][u3] == c0 and
                m[u3][u4] == c0 and m[u4][u5] == c0 and m[u5][u0] == c0):
                total += 1
                break
    return total

def _is_valid(colors, n):
    m = _build_matrix(colors, n)
    cycles = _get_cycles()
    for combo in itertools.combinations(range(n), 6):
        for cyc in cycles:
            u0, u1, u2, u3, u4, u5 = combo
            c0 = m[u0][u1]
            if c0 < 0:
                continue
            if (m[u1][u2] == c0 and m[u2][u3] == c0 and
                m[u3][u4] == c0 and m[u4][u5] == c0 and m[u5][u0] == c0):
                return False
    return True

def _try_n16():
    """Try cyclic construction for n=16."""
    n = 16
    # Try colouring by (j-i) mod 4
    for shift in range(4):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i + shift) % 4
                colors.append(d)
        if _is_valid(colors, n):
            return {'n': n, 'colors': colors}
    
    # Try other patterns
    # Colour by min((j-i)%n, n-(j-i)%n) mapped to 4 colours
    for mod_val in [4, 5, 6, 7, 8]:
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                d = min(d, n - d)
                colors.append(d % mod_val % 4)
        if _is_valid(colors, n):
            return {'n': n, 'colors': colors}
    
    return None

def _local_search_n17(max_time):
    """Local search for n=17."""
    n = 17
    start = _now()
    cycles = _get_cycles()
    
    for attempt in range(200):
        if _now() - start > max_time:
            break
        colors = [random.randint(0, 3) for _ in range(n * (n-1) // 2)]
        m = _build_matrix(colors, n)
        c6_count = _count_c6s(colors, n)
        
        # Local search
        no_improve = 0
        while _now() - start < max_time:
            if c6_count == 0:
                return {'n': n, 'colors': list(colors)}
            
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(i+1, n-1)
            old_c = m[i][j]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Apply
            m[i][j] = new_c
            m[j][i] = new_c
            colors[_edge_index(n, i, j)] = new_c
            
            # Recompute (expensive but correct)
            new_count = _count_c6s(colors, n)
            
            if new_count < c6_count:
                c6_count = new_count
                no_improve = 0
            else:
                # Revert
                m[i][j] = old_c
                m[j][i] = old_c
                colors[_edge_index(n, i, j)] = old_c
                no_improve += 1
                if no_improve > 200:
                    break
    
    return None

def construct():
    global _DEADLINE
    _DEADLINE = _now() + 108  # Safety margin
    
    # Try n=17 with local search
    result = _local_search_n17(80)
    if result is not None:
        return result
    
    # Try n=16
    result = _try_n16()
    if result is not None:
        return result
    
    # Fallback: random search for n=16
    n = 16
    start = _now()
    for _ in range(500):
        if _now() - start > 20:
            break
        colors = [random.randint(0, 3) for _ in range(n * (n-1) // 2)]
        if _is_valid(colors, n):
            return {'n': n, 'colors': colors}
    
    # Last resort
    n = 5
    return {'n': n, 'colors': [0] * (n * (n-1) // 2)}
# EVOLVE-BLOCK-END
