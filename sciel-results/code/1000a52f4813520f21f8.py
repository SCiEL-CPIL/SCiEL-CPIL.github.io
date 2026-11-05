import time
import random

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
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

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

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

def _find_lazebnik_partition():
    """
    Search for a valid partition of Z_13* into 5 sets for the Lazebnik-Woldar construction.
    Returns a list of 5 sets (as lists of integers) or None.
    """
    elements = list(range(1, 13))
    
    # We want to assign each element to a color 0..4
    # We will try random assignments
    for _ in range(20000):
        if _time_left() < 10:
            return None
        assignment = [random.randint(0, 4) for _ in range(12)]
        # Build the sets
        sets = [[] for _ in range(5)]
        for i, e in enumerate(elements):
            sets[assignment[i]].append(e)
        
        # Build the coloring for K_26
        n = 26
        colors = [0] * (n * (n - 1) // 2)
        
        # Map element to color
        elem_to_color = {}
        for c in range(5):
            for e in sets[c]:
                elem_to_color[e] = c
        
        idx = 0
        for i in range(n):
            x1 = i % 2
            y1 = i // 2
            for j in range(i + 1, n):
                x2 = j % 2
                y2 = j // 2
                dx = (x2 - x1) % 2
                dy = (y2 - y1) % 13
                
                if dy == 0:
                    # This case only happens if x1 != x2 and y1 == y2
                    # i.e., (0, y) and (1, y)
                    # In this case, dy=0. We need a color for this.
                    # The Lazebnik-Woldar construction usually handles dy=0 separately or
                    # the sets don't cover 0.
                    # Let's assign a fixed color, say 0, to these edges.
                    # Or better, check if this creates C4.
                    # Actually, if we use the same partition for dx=0 and dx=1,
                    # we need to define the color for dy=0.
                    # Let's just assign color 0 for now and see if it validates.
                    c = 0
                else:
                    c = elem_to_color[dy]
                
                colors[idx] = c
                idx += 1
        
        if _validate(n, colors):
            return sets
            
    return None

def _construct_lazebnik(partition):
    """Construct the K_26 coloring given a valid partition."""
    n = 26
    colors = []
    elem_to_color = {}
    for c in range(5):
        for e in partition[c]:
            elem_to_color[e] = c
            
    for i in range(n):
        x1 = i % 2
        y1 = i // 2
        for j in range(i + 1, n):
            x2 = j % 2
            y2 = j // 2
            dx = (x2 - x1) % 2
            dy = (y2 - y1) % 13
            
            if dy == 0:
                c = 0 # Fixed color for dy=0
            else:
                c = elem_to_color[dy]
                
            colors.append(c)
            
    return {'n': n, 'colors': colors}

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
    
    # Try to find a valid Lazebnik-Woldar partition for n=26
    if best['n'] < 26 and _time_left() > 20:
        partition = _find_lazebnik_partition()
        if partition:
            res = _construct_lazebnik(partition)
            if _validate(res['n'], res['colors']):
                best = res
    
    # If we still don't have n=26, try n=25 with a similar approach?
    # Or just fall back to greedy?
    # The cyclic construction for n=25 is often invalid.
    # Let's try a greedy search for n=26 if the algebraic one failed.
    if best['n'] < 26 and _time_left() > 30:
        res = _greedy_with_repair(26, max_iter=50000)
        if res and res['n'] > best['n']:
            best = res

    return best

def _greedy_with_repair(n, max_iter=50000, seed=42):
    """
    Greedy construction with local search to find a valid 5-colouring of K_n.
    """
    random.seed(seed)
    
    m = n * (n - 1) // 2
    # Start with cyclic-like initialisation
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)
    
    def _edge_idx(a, b):
        if a > b:
            a, b = b, a
        return a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
    
    for _ in range(max_iter):
        if _time_left() < 15:
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
        # Try to recolor one of the edges in the C4
        edges_in_c4 = [(i, common[0]), (i, common[1]), (j, common[0]), (j, common[1])]
        random.shuffle(edges_in_c4)
        for a, b in edges_in_c4:
            if a > b:
                a, b = b, a
            idx = _edge_idx(a, b)
            new_c = (colors[idx] + 1 + random.randint(0, 3)) % 5
            colors[idx] = new_c
            break
    
    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
