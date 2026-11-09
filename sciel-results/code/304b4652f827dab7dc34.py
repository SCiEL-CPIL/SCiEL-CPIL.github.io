"""Four-colourings of K_n without monochromatic triangles.

Strategy:
- Primary: Construct 49-vertex coloring using GF(7^2) with multiplicative
  group partitioned into cosets of cubes (order 48, divisible by 3).
- Fallback: 32-vertex Clebsch-based construction if GF(7^2) fails.
- Deadline guard: return best valid result before 120 s.
"""

import time
import random
import sys

# ---------------------------------------------------------------------------
# GF(7^2) construction for 49 vertices
# ---------------------------------------------------------------------------

def _gf49_construction():
    """
    Construct a 4-coloring of K_49 using GF(7^2).
    
    GF(7^2) has 49 elements. The multiplicative group has order 48.
    Since 48 is divisible by 3, we can partition the nonzero elements into
    3 cosets of the subgroup of cubes (order 16).
    
    For vertices x, y in GF(49):
    - If x = y: no edge (same vertex)
    - If x - y = 0: not possible for distinct vertices
    - Otherwise: color based on which coset of cubes (x-y) belongs to
    
    This should avoid monochromatic triangles because if (a-b) and (b-c)
    are in the same coset, then (a-c) = (a-b) + (b-c) is unlikely to be
    in that coset (additive combinatorics argument).
    
    We use 3 colors for the 3 cosets, and color 3 for... actually we only
    need 3 colors for this construction. But the problem requires colors
    in {0,1,2,3}. We can use colors 0,1,2 for the 3 cosets.
    
    Wait - we need to verify this actually works. Let me implement and test.
    """
    # Build GF(7^2) = GF(7)[x] / (x^2 + 1) since x^2+1 is irreducible over GF(7)
    # Check: x^2+1 has no roots in GF(7): 0^2+1=1, 1^2+1=2, 2^2+1=5, 3^2+1=10=3, 4^2+1=17=3, 5^2+1=26=5, 6^2+1=37=2
    # Yes, irreducible.
    
    # Represent elements as (a, b) meaning a + b*alpha where alpha^2 = -1 = 6 mod 7
    # So alpha^2 = 6
    
    def gf49_add(x, y):
        return ((x[0] + y[0]) % 7, (x[1] + y[1]) % 7)
    
    def gf49_sub(x, y):
        return ((x[0] - y[0]) % 7, (x[1] - y[1]) % 7)
    
    def gf49_mul(x, y):
        # (a + b*alpha) * (c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = ac + 6*bd + (ad + bc)*alpha
        a, b = x
        c, d = y
        real = (a * c + 6 * b * d) % 7
        imag = (a * d + b * c) % 7
        return (real, imag)
    
    # Build all 49 elements
    elements = [(a, b) for a in range(7) for b in range(7)]
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Find a primitive element (generator of multiplicative group of order 48)
    # Try elements and check if they have order 48
    def gf49_pow(x, n):
        result = (1, 0)  # multiplicative identity
        base = x
        while n > 0:
            if n & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            n >>= 1
        return result
    
    # Find primitive element
    primitive = None
    for e in elements:
        if e == (0, 0):
            continue
        # Check if e has order 48
        # 48 = 2^4 * 3, so check e^(48/2) != 1 and e^(48/3) != 1
        if gf49_pow(e, 24) != (1, 0) and gf49_pow(e, 16) != (1, 0):
            primitive = e
            break
    
    if primitive is None:
        return None  # Should not happen
    
    # Generate all nonzero elements as powers of primitive
    # powers[i] = primitive^i for i = 0..47
    powers = []
    current = (1, 0)
    for i in range(48):
        powers.append(current)
        current = gf49_mul(current, primitive)
    
    # powers[48] should be (1,0) again
    
    # The subgroup of cubes has order 48/3 = 16
    # cubes[i] = primitive^(3*i) for i = 0..15
    cubes = [powers[3 * i] for i in range(16)]
    cube_set = set(cubes)
    
    # Partition nonzero elements into 3 cosets of the cube subgroup
    # coset_0 = cubes
    # coset_1 = primitive * cubes
    # coset_2 = primitive^2 * cubes
    
    def get_coset(elem):
        """Return which coset of cubes elem belongs to (0, 1, or 2)."""
        if elem in cube_set:
            return 0
        # Check if elem * primitive^(-1) is in cubes
        # primitive^(-1) = primitive^47 = powers[47]
        inv_prim = powers[47]
        e1 = gf49_mul(elem, inv_prim)
        if e1 in cube_set:
            return 1
        # Otherwise should be in coset 2
        return 2
    
    # Verify partition
    coset_counts = [0, 0, 0]
    for p in powers:
        c = get_coset(p)
        coset_counts[c] += 1
    # Each coset should have 16 elements
    
    # Now assign colors
    # For edge (i, j) where i < j, compute elements[i] - elements[j]
    # and assign color based on coset
    colors = []
    for i in range(49):
        for j in range(i + 1, 49):
            diff = gf49_sub(elements[i], elements[j])
            if diff == (0, 0):
                # Should not happen for i != j
                c = 0
            else:
                c = get_coset(diff)
            colors.append((i, j, c))
    
    return colors

# ---------------------------------------------------------------------------
# Fast triangle-free verification using bitsets
# ---------------------------------------------------------------------------

def _is_triangle_free_fast(colors, n):
    """
    colors: list of (i, j, c) in lex order, i < j.
    Returns True iff no monochromatic triangle.
    Uses bitset adjacency for speed.
    """
    # adj[c][v] = bitmask of neighbors of v in color c
    adj = [[0] * n for _ in range(4)]
    
    for i, j, c in colors:
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    
    # For each color, check each vertex
    for c in range(4):
        for v in range(n):
            nbrs = adj[c][v]
            # For each pair of neighbors, check if they're connected
            # Iterate through set bits
            n1 = nbrs
            while n1:
                # Get lowest set bit
                b1 = n1 & (-n1)
                n1 -= b1
                u = b1.bit_length() - 1
                # Check if u's neighbors intersect with nbrs (excluding v)
                nbrs_u = adj[c][u] & nbrs
                # If nbrs_u has any bits, there's a triangle
                # But we need to exclude v itself
                if nbrs_u & ~(1 << v):
                    return False
    return True

# ---------------------------------------------------------------------------
# GF(2,4) multiplication table for the Clebsch-graph base construction
# ---------------------------------------------------------------------------

def _build_mul_table():
    """Build multiplication in GF(2^4) with irreducible x^4+x+1 (0b10011)."""
    log = {}
    mul = {}
    x = 1
    log[1] = 0
    for e in range(15):
        x2 = 0
        for i in range(4):
            if (x >> i) & 1:
                x2 ^= (1 << (i + 1))
        # reduce
        if x2 & 0b10000:
            x2 ^= 0b10011
        x = x2
        log[x] = e + 1
    return log

# Precompute once
_log = _build_mul_table()

def _base32_colors():
    """Return the 32-vertex base coloring as a list of (i,j,color) in lex order."""
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                # same block: Clebsch
                c = _log[i % 16 ^ j % 16] % 3
            else:
                c = 3
            colors.append((i, j, c))
    return colors

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 110  # 110s budget, safety margin for 120s limit
    
    # Try GF(49) construction first
    best_colors = None
    best_n = 0
    
    gf49_colors = _gf49_construction()
    if gf49_colors is not None:
        # Verify it's triangle-free
        if _is_triangle_free_fast(gf49_colors, 49):
            best_colors = gf49_colors
            best_n = 49
    
    # If GF(49) failed or we want to try to extend, fall back to 32-vertex base
    if best_n < 32:
        best_colors = _base32_colors()
        best_n = 32
    
    # Try to extend beyond current best_n using local search
    if best_n < 50 and time.time() < deadline - 10:
        def _try_extend(colors, current_n, steps):
            """Try to extend coloring to current_n+1 vertices via local search."""
            deadline_local = min(time.time() + 20, deadline - 5)
            for trial in range(steps):
                if time.time() > deadline_local:
                    break
                # Create new vertex
                new_colors = list(colors)
                new_n = current_n + 1
                # Randomly assign colours for edges from new vertex to existing
                assign = []
                for i in range(current_n):
                    c = random.randint(0, 3)
                    new_colors.append((i, new_n - 1, c))
                    assign.append(c)
                
                if not _is_triangle_free_fast(new_colors, new_n):
                    continue
                
                # Try to fix via local search: flip edge colours
                found = False
                for it in range(3000):
                    if time.time() > deadline_local:
                        break
                    if _is_triangle_free_fast(new_colors, new_n):
                        found = True
                        break
                    # Pick a random edge involving the new vertex and flip
                    pos = random.randint(0, current_n - 1)
                    edge_pos = current_n + pos  # (i, new_n-1) for i in 0..current_n-1
                    # Flip
                    new_colors[edge_pos] = (new_colors[edge_pos][0], new_colors[edge_pos][1], (new_colors[edge_pos][2] + 1) % 4)
                
                if found:
                    return new_colors, new_n
            
            return None, None
        
        # Try to extend step by step
        max_attempts = 100000
        attempts = 0
        while time.time() < deadline - 5:
            if attempts >= max_attempts:
                break
            colors, new_n = _try_extend(best_colors, best_n, steps=3)
            if colors is not None:
                best_colors = colors
                best_n = new_n
                attempts = 0  # reset on success
            else:
                attempts += 1
    
    # Format output
    out_colors = []
    for i, j, c in best_colors:
        out_colors.append(c)
    
    return {'n': best_n, 'colors': out_colors}
