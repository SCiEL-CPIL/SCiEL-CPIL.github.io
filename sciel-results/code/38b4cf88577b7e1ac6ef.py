"""Four-colourings of K_n without monochromatic triangles.

Primary strategy: GF(49) construction.
- Vertices identified with elements of GF(7^2).
- Colours assigned based on cosets of the cubic-residue subgroup of GF(49)*.
- The multiplicative group has order 42 = 3 * 14, so it splits into 3 cosets of size 14.
- For a = x - y != 0: colour = index of coset containing a.
- This yields a 4-colouring of K_49 with no monochromatic triangles (if valid).
- Fallback: 32-vertex Clebsch-based construction + local search extension.
"""

import time
import random
import sys

# ---------------------------------------------------------------------------
# Fast triangle-free check using bitmasks
# ---------------------------------------------------------------------------

def _is_valid_fast(colors, n):
    """
    colors: list of ints, length n*(n-1)//2, in lex order (i,j) for i<j.
    Returns True iff no monochromatic triangle.
    Uses per-colour adjacency bitmasks.
    """
    adj = [0] * (4 * n)  # adj[c*n + v] = bitmask of neighbours of v in colour c
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
    
    for c in range(4):
        base = c * n
        for v in range(n):
            nbrs = adj[base + v]
            # Check if any neighbour of v has another common neighbour
            # i.e., if adj[base + u] & nbrs != 0 for any u in nbrs
            # Faster: for each u in nbrs, check if adj[base+u] & nbrs has bits other than v
            temp = nbrs
            while temp:
                low = temp & (-temp)
                u = low.bit_length() - 1
                temp ^= low
                if adj[base + u] & nbrs:
                    # u is adjacent to v in colour c, and u has some neighbour in nbrs
                    # that neighbour could be v itself; we need to check if there's
                    # another vertex w != v in nbrs that is also adjacent to u
                    # Actually: if adj[base+u] & nbrs != 0, then there exists w in nbrs
                    # such that u-w is an edge. If w == v, that's just the edge u-v.
                    # We need w != v. So check adj[base+u] & (nbrs ^ (1<<v))
                    if adj[base + u] & (nbrs ^ (1 << v)):
                        return False
    return True

# ---------------------------------------------------------------------------
# GF(49) construction
# ---------------------------------------------------------------------------

def _gf49_construction():
    """
    Construct a 4-colouring of K_49 using GF(7^2).
    Returns (n, colors_list) where colors_list is in lex order.
    Returns (0, []) if the construction is invalid.
    """
    # GF(7^2) with polynomial x^2 + 1 (since -1 is not a square mod 7)
    # Elements are (a, b) representing a + b*i where i^2 = -1 = 6 mod 7
    # Multiplication: (a,b)*(c,d) = (ac - bd, ad + bc) mod 7
    
    # Build multiplication table
    mul = [[0] * 49 for _ in range(49)]
    for a in range(7):
        for b in range(7):
            for c in range(7):
                for d in range(7):
                    idx1 = a * 7 + b
                    idx2 = c * 7 + d
                    # (a + b*i) * (c + d*i) = ac + ad*i + bc*i + bd*i^2 = ac + bd*(-1) + (ad+bc)*i
                    real = (a * c - b * d) % 7
                    imag = (a * d + b * c) % 7
                    mul[idx1][idx2] = real * 7 + imag
    
    # Find a generator of GF(49)* (order 48)
    # GF(49)* has order 48, so we need an element of order 48
    # The subgroup of cubes has order 48/gcd(3,48) = 48/3 = 16... wait
    # Actually, the cubic residues: {g^(3k) : k=0..15} has size 16
    # But we want to partition GF(49)* into 3 cosets of the subgroup of cubes
    # The subgroup of cubes has index 3, so size 48/3 = 16
    # Wait, that gives only 3 cosets of size 16, but we need 4 colours.
    # 
    # Let me reconsider. We have 48 non-zero elements. We want to assign 3 colours
    # to them such that no monochromatic triangle exists.
    # 
    # The standard construction for R(3,3,3,3) > 49 uses GF(49) where:
    # - The multiplicative group GF(49)* has order 48.
    # - We partition GF(49)* into 3 sets based on whether x is a cube, 
    #   or 2*cube, or 4*cube (where 2 and 4 are non-cube elements).
    # 
    # Actually, let me think about this more carefully.
    # 
    # For the 4-colouring to be triangle-free, we need:
    # For any three vertices a, b, c in GF(49), the three differences
    # (a-b), (b-c), (c-a) must not all have the same colour.
    # Note that (a-b) + (b-c) + (c-a) = 0.
    # 
    # So we need: there do not exist x, y, z all in the same colour class
    # such that x + y + z = 0.
    # 
    # Equivalently: for each colour class S, there are no x, y in S with
    # x + y in S (since z = -(x+y) must also be in S).
    # 
    # So each colour class must be sum-free (no x + y = z for distinct x,y,z in S,
    # and also no 2x = z, i.e., x+x=z).
    # 
    # Let's find a generator g of GF(49)*.
    
    def order_of(el):
        """Find the multiplicative order of el in GF(49)*."""
        if el == 0:
            return 0
        ord = 1
        cur = el
        while cur != 1:
            cur = mul[cur][el]
            ord += 1
            if ord > 48:
                return -1
        return ord
    
    # Find generator of order 48
    generator = None
    for el in range(1, 49):
        if order_of(el) == 48:
            generator = el
            break
    
    if generator is None:
        return 0, []
    
    # Build the cyclic group: g^0, g^1, ..., g^47
    group = [1]
    for i in range(1, 48):
        group.append(mul[group[-1]][generator])
    
    # Verify group has 48 distinct elements
    assert len(set(group)) == 48, "Generator does not have order 48"
    
    # The subgroup of cubes: {g^(3k) : k = 0, ..., 15}
    # This has 16 elements.
    # Cosets: C0 = {g^(3k)}, C1 = {g * g^(3k)}, C2 = {g^2 * g^(3k)}
    # where g and g^2 are non-cubes (since 1 and 2 are not multiples of 3).
    
    # Assign colours:
    # Colour 0: elements in C0 (cubes)
    # Colour 1: elements in C1 (g * cubes)
    # Colour 2: elements in C2 (g^2 * cubes)
    # Colour 3: unused (we only have 3 colours for 48 elements)
    # 
    # Wait, but we need 4 colours. Let me reconsider.
    # 
    # Actually, the construction gives a 3-colouring of the 48 non-zero elements.
    # But we need a 4-colouring of K_49. The 4th colour might not be used,
    # or we might need a different partition.
    # 
    # Let me check if the 3-colouring of the 48 non-zero differences
    # gives a valid 3-colouring of K_49 (which would also be a valid 4-colouring).
    # 
    # For the 3-colouring to be triangle-free:
    # For each colour class S (subset of GF(49)*), we need:
    # No x, y, z in S with x + y + z = 0.
    
    # Build colour assignment for non-zero elements
    color_of = {}
    for k in range(16):
        el = group[3 * k]  # g^(3k)
        color_of[el] = 0
        el1 = mul[group[3 * k]][generator]  # g * g^(3k) = g^(3k+1)
        color_of[el1] = 1
        el2 = mul[group[3 * k]][group[2]]  # g^2 * g^(3k) = g^(3k+2)
        color_of[el2] = 2
    
    # Verify all 48 non-zero elements are assigned
    assert len(color_of) == 48, f"Expected 48, got {len(color_of)}"
    
    # Now build the colouring of K_49
    n = 49
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (i - j) % 49
            # Wait, we need to compute i - j in GF(49), not mod 49.
            # Let me fix this.
            pass
    
    # Recompute properly using GF(49) arithmetic
    # Vertex i corresponds to element i in GF(49) (we identify 0..48 with field elements)
    # But we need proper field subtraction.
    
    # Element i in GF(49) is represented as (i // 7, i % 7)
    # Subtraction: (a,b) - (c,d) = (a-c, b-d) mod 7
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # i and j are indices 0..48, representing field elements
            ai, bi = i // 7, i % 7
            aj, bj = j // 7, j % 7
            # diff = (ai - aj, bi - bj) mod 7
            di = (ai - aj) % 7
            dj = (bi - bj) % 7
            diff_idx = di * 7 + dj
            if diff_idx == 0:
                # Should not happen since i != j
                c = 3
            else:
                c = color_of[diff_idx]
            colors.append(c)
    
    # Verify
    if _is_valid_fast(colors, n):
        return n, colors
    else:
        # The construction is invalid; try other generators or partitions
        # For now, return invalid
        return 0, []

# ---------------------------------------------------------------------------
# 32-vertex Clebsch base (fallback)
# ---------------------------------------------------------------------------

def _build_mul_table():
    """Build multiplication in GF(2^4) with irreducible x^4+x+1 (0b10011)."""
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

_log = _build_mul_table()

def _base32_colors():
    """Return the 32-vertex base coloring as a list of (i,j,color) in lex order."""
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                c = _log[i % 16 ^ j % 16] % 3
            else:
                c = 3
            colors.append(c)
    return colors

# ---------------------------------------------------------------------------
# Local search extension
# ---------------------------------------------------------------------------

def _try_extend(colors, current_n, deadline):
    """Try to extend coloring to current_n+1 vertices via local search."""
    deadline_local = min(deadline, time.time() + 20)
    for trial in range(200):
        if time.time() > deadline_local:
            break
        new_n = current_n + 1
        new_colors = list(colors)
        for i in range(current_n):
            new_colors.append(random.randint(0, 3))
        
        if _is_valid_fast(new_colors, new_n):
            return new_colors, new_n
        
        # Local search: flip edges involving new vertex
        for it in range(2000):
            if time.time() > deadline_local:
                break
            if _is_valid_fast(new_colors, new_n):
                return new_colors, new_n
            pos = current_n + random.randint(0, current_n - 1)
            new_colors[pos] = (new_colors[pos] + 1) % 4
    
    return None, None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 110  # 110s budget, safety margin for 120s limit
    
    # Try GF(49) construction first
    n_gf49, colors_gf49 = _gf49_construction()
    if n_gf49 >= 49 and _is_valid_fast(colors_gf49, n_gf49):
        return {'n': n_gf49, 'colors': colors_gf49}
    
    # Fallback: 32-vertex base + extension
    best_colors = _base32_colors()
    best_n = 32
    
    attempts = 0
    while time.time() < deadline:
        if attempts >= 50000:
            break
        colors, new_n = _try_extend(best_colors, best_n, deadline)
        if colors is not None:
            best_colors = colors
            best_n = new_n
            attempts = 0
        else:
            attempts += 1
    
    return {'n': best_n, 'colors': best_colors}
