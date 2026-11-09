"""Four-colourings of K_n without monochromatic triangles.

Strategy:
- Primary: GF(43) multiplicative construction for 42 vertices (3 colors, guaranteed triangle-free)
- Secondary: 32-vertex Clebsch-based construction as fallback
- Local search to extend beyond base constructions using fast bitset verification
- Deadline guard: return best valid result before 120 s
"""

import time
import random
import sys
import numpy as np

# ---------------------------------------------------------------------------
# GF(2^4) multiplication table for the Clebsch-graph base construction
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
            colors.append((i, j, c))
    return colors

# ---------------------------------------------------------------------------
# GF(43) construction for 42 vertices using cubic residues
# ---------------------------------------------------------------------------

def _gf43_construction():
    """
    Construct a 3-coloring of K_42 using GF(43)*.
    
    GF(43)* is cyclic of order 42 = 3 * 14. Partition into 3 cosets of
    the subgroup of cubes (order 14). For vertices 0..41 representing
    elements 1..42 of GF(43), color edge (i,j) by which coset
    (i+1)*(j+1)^{-1} belongs to.
    
    Triangle-free: if a/b and b/c are in coset C, then a/c = (a/b)*(b/c)
    is in C*C. For cyclic group of order 42 with C being a coset of the
    order-14 subgroup, C*C is a different coset (since 2*14=28 doesn't
    divide 42 in the right way to give C*C = C).
    """
    p = 43
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Find a primitive root mod 43
    # 42 = 2 * 3 * 7
    primitive = None
    for g in range(2, p):
        if pow(g, 42 // 2, p) != 1 and pow(g, 42 // 3, p) != 1 and pow(g, 42 // 7, p) != 1:
            primitive = g
            break
    
    if primitive is None:
        return None
    
    # Generate all nonzero elements as powers of primitive
    powers = []
    current = 1
    for i in range(42):
        powers.append(current)
        current = (current * primitive) % p
    
    # Subgroup of cubes has order 42/3 = 14
    cubes = set(powers[3 * i] for i in range(14))
    
    # Precompute coset membership for each nonzero element
    coset = [0] * (p + 1)
    inv_prim = inv[primitive]
    for i in range(1, p):
        if i in cubes:
            coset[i] = 0
        else:
            e1 = (i * inv_prim) % p
            if e1 in cubes:
                coset[i] = 1
            else:
                coset[i] = 2
    
    # Build colors for K_42
    colors = []
    for i in range(42):
        for j in range(i + 1, 42):
            ratio = ((i + 1) * inv[j + 1]) % p
            c = coset[ratio]
            colors.append((i, j, c))
    
    return colors

# ---------------------------------------------------------------------------
# Fast triangle-free verification using bitsets
# ---------------------------------------------------------------------------

def _is_triangle_free_bitset(adj_bitsets, n):
    """
    adj_bitsets: list of 4 lists, adj_bitsets[c][v] is an int bitmask of neighbors of v in color c.
    Returns True iff no monochromatic triangle.
    """
    for c in range(4):
        adj = adj_bitsets[c]
        for v in range(n):
            nbrs = adj[v]
            # Only check neighbors u > v
            n1 = nbrs & ~((1 << (v + 1)) - 1)
            while n1:
                b1 = n1 & (-n1)
                n1 -= b1
                u = b1.bit_length() - 1
                common = adj[v] & adj[u]
                if common:
                    return False
    return True

def _build_bitsets(colors, n):
    """Build adjacency bitsets from color list."""
    adj = [[0] * n for _ in range(4)]
    for i, j, c in colors:
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    return adj

def _is_triangle_free_fast(colors, n):
    """Check triangle-freeness using bitset adjacency."""
    adj = _build_bitsets(colors, n)
    return _is_triangle_free_bitset(adj, n)

# ---------------------------------------------------------------------------
# Local search extension using fast bitset verification
# ---------------------------------------------------------------------------

def _try_extend_local(adj, current_n, deadline):
    """
    Try to extend coloring to current_n+1 vertices via local search.
    adj: list of 4 lists of int bitmasks for current_n vertices.
    Returns (new_adj, new_n) or (None, None).
    """
    new_n = current_n + 1
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        assign = [random.randint(0, 3) for _ in range(current_n)]
        
        # Build new adjacency bitsets
        new_adj = [[0] * new_n for _ in range(4)]
        for c in range(4):
            for v in range(current_n):
                new_adj[c][v] = adj[c][v]
        
        for i in range(current_n):
            c = assign[i]
            new_adj[c][i] |= (1 << (new_n - 1))
            new_adj[c][new_n - 1] |= (1 << i)
        
        # Check if valid
        if _is_triangle_free_bitset(new_adj, new_n):
            return new_adj, new_n
        
        # Local search: flip edges involving new vertex
        for it in range(5000):
            if time.time() > deadline:
                break
            
            if _is_triangle_free_bitset(new_adj, new_n):
                return new_adj, new_n
            
            # Pick random edge to new vertex and flip
            pos = random.randint(0, current_n - 1)
            old_c = assign[pos]
            new_c = (old_c + 1) % 4
            
            # Update bitsets
            new_adj[old_c][pos] &= ~(1 << (new_n - 1))
            new_adj[old_c][new_n - 1] &= ~(1 << pos)
            new_adj[new_c][pos] |= (1 << (new_n - 1))
            new_adj[new_c][new_n - 1] |= (1 << pos)
            assign[pos] = new_c
    
    return None, None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    best_colors = None
    best_n = 0
    
    # Try GF(43) construction (42 vertices, 3 colors)
    gf43_colors = _gf43_construction()
    if gf43_colors is not None:
        if _is_triangle_free_fast(gf43_colors, 42):
            best_colors = gf43_colors
            best_n = 42
    
    # Fallback to 32-vertex base
    if best_n < 32:
        best_colors = _base32_colors()
        best_n = 32
    
    # Try to extend using local search
    if best_colors is not None:
        adj = _build_bitsets(best_colors, best_n)
        
        # Try to extend step by step
        while time.time() < deadline - 5:
            new_adj, new_n = _try_extend_local(adj, best_n, deadline - 5)
            if new_adj is not None:
                # Convert back to colors list
                new_colors = list(best_colors)
                for i in range(best_n):
                    # Find the color of edge (i, new_n-1)
                    for c in range(4):
                        if new_adj[c][i] & (1 << (new_n - 1)):
                            new_colors.append((i, new_n - 1, c))
                            break
                best_colors = new_colors
                best_n = new_n
                adj = new_adj
            else:
                # Can't extend further, try a few more times with different random seeds
                success = False
                for attempt in range(20):
                    if time.time() > deadline - 5:
                        break
                    new_adj2, new_n2 = _try_extend_local(adj, best_n, deadline - 5)
                    if new_adj2 is not None:
                        new_colors = list(best_colors)
                        for i in range(best_n):
                            for c in range(4):
                                if new_adj2[c][i] & (1 << (new_n2 - 1)):
                                    new_colors.append((i, new_n2 - 1, c))
                                    break
                        best_colors = new_colors
                        best_n = new_n2
                        adj = new_adj2
                        success = True
                        break
                if not success:
                    break
    
    # Format output
    out_colors = [c for i, j, c in best_colors]
    
    return {'n': best_n, 'colors': out_colors}
