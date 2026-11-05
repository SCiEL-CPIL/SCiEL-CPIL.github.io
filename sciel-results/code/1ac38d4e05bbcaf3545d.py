"""5-colourings of K_n without monochromatic C4.
Combines the cyclic seed idea with algebraic constructions and a deadline guard.
"""
import time
import itertools

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4).
    Equivalent: for some colour c, two vertices have >=2 common neighbours in c.
    """
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    # Build neighbour sets per colour
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _try_lazebnik_woldar():
    """
    Attempt the Lazebnik-Woldar algebraic construction for n=26.
    This uses a finite field / group construction. The known construction
    gives a 5-colouring of K_26 with no monochromatic C4.
    
    The construction uses the group Z_2 x Z_13 or similar. Let's use
    a known parametrization: vertices are elements of Z_26 (or a product group).
    
    A known explicit construction for R_5(C4) >= 26 uses the following:
    Vertices: (x, y) where x in Z_2, y in Z_13.
    Colour of edge ((x1,y1),(x2,y2)):
      Let dx = (x2-x1) mod 2, dy = (y2-y1) mod 13.
      If dx == 0: colour based on dy using a 5-colouring of Z_13's differences.
      If dx == 1: colour based on dy using another mapping.
    
    Actually, the standard Lazebnik-Woldar construction for R_5(C4) >= 26
    is based on the following:
    
    Use the group G = Z_2 x Z_13. The 5 colour classes are defined via
    quadratic residues or similar in Z_13.
    
    In Z_13, the quadratic residues are {1, 3, 4, 9, 10, 12} (6 elements)
    and non-residues are {2, 5, 6, 7, 8, 11} (6 elements).
    
    A simpler approach: use a known valid 5-colouring of K_26.
    Let me try a direct construction using Z_26 with a specific colouring rule.
    
    Alternative: use the fact that 26 = 2*13, and use a "blow-up" or product construction.
    
    Let me try a different approach: use a computer-searchable parametrization.
    For now, let's try a few algebraic constructions and validate them.
    """
    # Try construction based on Z_2 x Z_13
    n = 26
    colors = []
    
    # Quadratic residues mod 13
    qr = {1, 3, 4, 9, 10, 12}
    
    for i in range(n):
        x1 = i % 2
        y1 = i // 2
        for j in range(i + 1, n):
            x2 = j % 2
            y2 = j // 2
            dx = (x2 - x1) % 2
            dy = (y2 - y1) % 13
            if dx == 0:
                # Same "layer": colour based on dy
                if dy == 0:
                    continue  # shouldn't happen since i < j
                if dy in qr:
                    c = 0
                else:
                    c = 1
                # Need to split into 5 colours, not 2
                # This simple split won't work. Let me use a different approach.
            else:
                # Different layers
                pass
    
    # The above is too simplistic. Let me use a known valid construction.
    # Based on literature, R_5(C4) >= 26 is known via a specific construction.
    # Let me try a direct search for a valid colouring of K_26 using a greedy approach
    # with backtracking, but that's expensive.
    
    # Instead, let me try a few parametrized constructions and validate.
    
    # Construction 1: Use Z_26 with colour = f((j-i) mod 26)
    # where f maps 1..12 to 0..4 (and 13..25 mirror 12..1)
    # Try different mappings f.
    
    # For now, return None if we can't find a valid one quickly.
    return None

def _greedy_with_repair(n, max_iter=10000):
    """
    Greedy construction with local search to find a valid 5-colouring of K_n.
    Start with a random or structured initial colouring, then repair C4 violations.
    """
    import random
    random.seed(42)
    
    m = n * (n - 1) // 2
    # Start with cyclic-like initialisation
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)
    
    # Local search: find C4 violations and fix them
    for _ in range(max_iter):
        if _time_left() < 10:
            break
        # Find a violation
        nbrs = [[set() for _ in range(n)] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                nbrs[c][i].add(j)
                nbrs[c][j].add(i)
                idx += 1
        
        violation = None
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = nbrs[c][i] & nbrs[c][j]
                    if len(common) >= 2:
                        violation = (c, i, j, list(common)[:2])
                        break
                if violation:
                    break
            if violation:
                break
        
        if violation is None:
            return {'n': n, 'colors': colors}
        
        c, i, j, common = violation
        # Try to recolor one of the edges to break the C4
        # Pick a random edge from the C4 and recolor it
        edges_in_c4 = [(i, common[0]), (i, common[1]), (j, common[0]), (j, common[1])]
        random.shuffle(edges_in_c4)
        for a, b in edges_in_c4:
            if a > b:
                a, b = b, a
            idx = a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
            # Recolor to a random different colour
            new_c = (colors[idx] + 1 + random.randint(0, 3)) % 5
            colors[idx] = new_c
            # Check if this fixed the violation (quick check)
            # Rebuild nbrs for this colour
            # For simplicity, just continue and recheck in next iteration
            break
    
    # Check if valid
    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try cyclic constructions for various n
    for n in [25, 20, 15, 10, 5]:
        if _time_left() < 5:
            break
        res = _cyclic_construction(n)
        if _validate(n, res['colors']):
            if n > best['n']:
                best = res
    
    # Try to find a larger construction
    if best['n'] < 26 and _time_left() > 20:
        # Try greedy with repair for n=26
        res = _greedy_with_repair(26, max_iter=50000)
        if res and res['n'] > best['n']:
            best = res
    
    if best['n'] < 25 and _time_left() > 30:
        # Try n=25 with more iterations
        res = _greedy_with_repair(25, max_iter=100000)
        if res and res['n'] > best['n']:
            best = res
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
    print(res['colors'][:20])
