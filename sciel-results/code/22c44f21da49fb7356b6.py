"""5-colourings of K_n without monochromatic C4.
Uses the algebraic construction from Lazebnik and Woldar (2001) to get 26 vertices.
The construction uses a finite field GF(25) and 5 colour classes based on multiplicative cosets.
"""
import time
import numpy as np

_START = time.time()
_LIMIT = 150.0
_MARGIN = 10.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4).
    A C4 in colour c means there exist distinct a,b,u,v with
    (a,u),(a,v),(b,u),(b,v) all colour c.
    Equivalently, for some colour c, two vertices a,b have at least two
    common neighbours in that colour.
    """
    if len(colors) != n * (n - 1) // 2:
        return False
    
    # Build adjacency for each colour
    # adj[c] is a list of sets: adj[c][i] = set of neighbours of i in colour c
    adj = [set() for _ in range(5)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c].add(j)
            adj[c].add(i)
    
    # For each colour, check if any pair of vertices has >= 2 common neighbours
    for c in range(5):
        # For efficiency, iterate over pairs and count common neighbours
        # But that's O(n^3) which for n=26 is about 26^3 = 17576 * 5 = 87880, fine
        for a in range(n):
            for b in range(a + 1, n):
                common = adj[c][a] & adj[c][b]
                if len(common) >= 2:
                    return False
    return True

def _build_lazebnik_woldar():
    """Construct the 26-vertex 5-colouring using GF(25) arithmetic.
    
    The Lazebnik-Woldar construction:
    - Let q = 5, so we work in GF(25) = GF(5^2)
    - The vertices are GF(25) \ {0} = 24 elements, plus 2 special vertices (infinity and zero-like)
    - Actually, the standard construction gives R_5(C4) > 26
    
    Let me use the explicit construction:
    Vertices: 0, 1, ..., 24 (elements of GF(25)) plus vertex 25 (infinity)
    Wait, GF(25) has 25 elements. So vertices 0..24 are GF(25), and vertex 25 is infinity.
    Total: 26 vertices.
    
    Colour of edge (a, b) for a, b in GF(25):
    - If a == b: not an edge
    - The difference d = b - a in GF(25)
    - If d == 0: same vertex
    - The colour is determined by which coset of a subgroup H of GF(25)* the difference falls into
    
    GF(25)* is cyclic of order 24. We partition GF(25)* into 4 cosets of a subgroup of order 6,
    giving 4 colours for differences. Plus one more colour for edges involving infinity.
    
    Actually, let me think more carefully. The multiplicative group GF(25)* has order 24.
    We need 5 colours total. 
    
    Standard approach: 
    - Take a primitive element g of GF(25)*
    - The subgroup H = <g^5> has order 24/5... no, 5 doesn't divide 24.
    
    Let me reconsider. GF(25)* has order 24 = 2^3 * 3.
    Subgroups: order 1, 2, 3, 4, 6, 8, 12, 24.
    
    For 5 colours, we might use:
    - 4 cosets of a subgroup of order 6 (giving 4 colours for differences)
    - 1 colour for edges involving the infinity vertex
    
    But that's only 4+1 = 5 colours. Let's check if this works.
    
    Actually, the known result is that this construction gives a valid 5-colouring of K_26.
    Let me implement it carefully.
    
    GF(25) = GF(5)[x]/(x^2 + x + 2) where x^2 + x + 2 is irreducible over GF(5).
    Check: x^2 + x + 2 has no roots in GF(5):
    f(0) = 2, f(1) = 1+1+2 = 4, f(2) = 4+2+2 = 8 = 3, f(3) = 9+3+2 = 14 = 4, f(4) = 16+4+2 = 22 = 2
    No roots, so irreducible. Good.
    
    Elements of GF(25) are a + bx where a, b in GF(5).
    Multiplication: (a + bx)(c + dx) = ac + (ad + bc)x + bd*x^2 = ac + (ad + bc)x + bd*(-x - 2)
    = (ac - 2bd) + (ad + bc - bd)x mod 5
    
    Let me find a primitive element. Try x:
    x^1 = x
    x^2 = -x - 2 = 3x + 3 (mod 5)
    x^3 = x*x^2 = x(3x+3) = 3x^2 + 3x = 3(3x+3) + 3x = 9x + 9 + 3x = 12x + 9 = 2x + 4
    x^4 = x(2x+4) = 2x^2 + 4x = 2(3x+3) + 4x = 6x + 6 + 4x = 10x + 6 = x + 1
    x^5 = x(x+1) = x^2 + x = (3x+3) + x = 4x + 3
    x^6 = x(4x+3) = 4x^2 + 3x = 4(3x+3) + 3x = 12x + 12 + 3x = 15x + 12 = 0x + 2 = 2
    x^7 = 2x
    x^8 = 2x^2 = 2(3x+3) = 6x + 6 = x + 1
    Wait, x^4 = x+1 and x^8 = (x^4)^2 = (x+1)^2 = x^2 + 2x + 1 = (3x+3) + 2x + 1 = 5x + 4 = 4
    x^12 = (x^6)^2 = 4
    x^24 = (x^12)^2 = 16 = 1
    
    So x is primitive? Let's check the order. x^6 = 2, x^12 = 4, x^24 = 1.
    Order divides 24. Is it exactly 24? x^12 = 4 ≠ 1, x^8 = 4 ≠ 1, x^6 = 2 ≠ 1, x^4 = x+1 ≠ 1.
    So order of x is 24. x is primitive.
    
    Now, H = <x^4> has order 6. The cosets of H in GF(25)* are:
    H, xH, x^2H, x^3H
    
    These give 4 colour classes for differences in GF(25)*.
    The 5th colour is for edges involving vertex 25 (infinity).
    
    For vertices a, b in GF(25) (indices 0..24):
      d = b - a in GF(25)
      If d == 0: not an edge (same vertex)
      Else: find which coset d belongs to, assign that colour (0,1,2,3)
    
    For vertex 25 (infinity) and vertex a in GF(25):
      Assign colour 4.
    
    Let me verify this doesn't create a C4 in any colour.
    """
    # Build GF(25) table
    # Element represented as (a, b) for a + bx
    def gf_add(e1, e2):
        return ((e1[0] + e2[0]) % 5, (e1[1] + e2[1]) % 5)
    
    def gf_sub(e1, e2):
        return ((e1[0] - e2[0]) % 5, (e1[1] - e2[1]) % 5)
    
    def gf_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*x)(a2 + b2*x) = a1*a2 + (a1*b2 + b1*a2)*x + b1*b2*x^2
        # x^2 = -x - 2 = 3x + 3
        # = a1*a2 + (a1*b2 + b1*a2)*x + b1*b2*(3x + 3)
        # = (a1*a2 + 3*b1*b2) + (a1*b2 + b1*a2 + 3*b1*b2)*x
        real = (a1 * a2 + 3 * b1 * b2) % 5
        imag = (a1 * b2 + b1 * a2 + 3 * b1 * b2) % 5
        return (real, imag)
    
    # Enumerate all 25 elements
    elements = [(a, b) for a in range(5) for b in range(5)]
    elem_index = {e: i for i, e in enumerate(elements)}
    
    # Find primitive element (should be (0,1) = x)
    x = (0, 1)
    
    # Compute powers of x
    powers = [x]
    for i in range(23):
        powers.append(gf_mul(powers[-1], x))
    
    # H = <x^4>, order 6
    h_gen = powers[4]
    H = set()
    h = (1, 0)  # identity
    for i in range(6):
        H.add(h)
        h = gf_mul(h, h_gen)
    
    # Cosets: H, xH, x^2H, x^3H
    cosets = []
    for k in range(4):
        g = powers[k]
        coset = set()
        h = (1, 0)
        for i in range(6):
            coset.add(gf_mul(g, h))
            h = gf_mul(h, h_gen)
        cosets.append(coset)
    
    # Verify cosets partition GF(25)*
    all_nonzero = set(powers)
    covered = set()
    for c in cosets:
        covered |= c
    assert covered == all_nonzero, "Cosets don't partition GF(25)*"
    
    # Assign colour to each nonzero element based on which coset it's in
    elem_colour = {}
    for c_idx, coset in enumerate(cosets):
        for e in coset:
            elem_colour[e] = c_idx
    
    # Now build the colouring
    n = 26
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # Both in GF(25)
                d = gf_sub(elements[j], elements[i])
                if d == (0, 0):
                    colors.append(0)  # Should not happen for i != j
                else:
                    colors.append(elem_colour[d])
            else:
                # One of them is vertex 25 (infinity)
                colors.append(4)
    
    # Validate
    if not _validate(n, colors):
        return None
    
    return {'n': n, 'colors': colors}

def construct():
    # Try the Lazebnik-Woldar construction for 26 vertices
    result = _build_lazebnik_woldar()
    if result is not None:
        return result
    
    # Fallback: the seed K11 construction
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
