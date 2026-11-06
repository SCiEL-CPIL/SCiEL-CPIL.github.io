"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _idx(i, j):
    """Index for pair (i,j) with 0<=i<j<16 in lexicographic order."""
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _validate_list(colors_list):
    """Fast validation using the flat list of 120 colours in lexicographic order."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            return False
    return True

def _count_bad_k4(colors_list):
    """Count K4s with fewer than 5 distinct colours."""
    bad = 0
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            bad += 1
    return bad

def _make_1factorization_cyclic():
    """Standard cyclic 1-factorization of K16. Returns dict (i,j)->color."""
    color_map = {}
    for i in range(15):
        color_map[(15, i)] = i
        color_map[(i, 15)] = i
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            color_map[(a, b)] = i
    return color_map

def _color_map_to_list(color_map):
    """Convert color_map dict to flat list of 120 colours in lexicographic order."""
    result = []
    for i in range(16):
        for j in range(i + 1, 16):
            result.append(color_map[(i, j)])
    return result

def _list_to_color_map(colors_list):
    """Convert flat list of 120 colours back to dict (i,j)->color."""
    color_map = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            color_map[(i, j)] = colors_list[idx]
            idx += 1
    return color_map

def _local_search_1factor(color_map, max_iterations=5000, deadline=None):
    """
    Local search on a 1-factorization: swap edges between two factors.
    A swap means: take an edge from factor A and an edge from factor B,
    and exchange their factor assignments. This preserves the 1-factorization
    structure (each factor remains a perfect matching) only if we swap entire
    edges. But actually, to maintain 1-factorization, we need to be careful.
    
    Alternative: just reassign individual edges' colours randomly, then check.
    This breaks the 1-factorization structure but may still give valid colourings.
    
    Better approach: Start from the 1-factorization, and do random reassignments
    of edge colours from the set of 15 colours, accepting if it doesn't increase
    the number of bad K4s.
    """
    t0 = time.time()
    colors_list = _color_map_to_list(color_map)
    bad_count = _count_bad_k4(colors_list)
    
    # Precompute all K4s and their edge indices for fast evaluation
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in all_k4s:
        a, b, c, d = quad
        edges = [
            _idx(a, b), _idx(a, c), _idx(a, d),
            _idx(b, c), _idx(b, d), _idx(c, d)
        ]
        k4_edges.append(edges)
    
    def count_bad(cl):
        bad = 0
        for edges in k4_edges:
            e1, e2, e3, e4, e5, e6 = edges
            cols = {cl[e1], cl[e2], cl[e3], cl[e4], cl[e5], cl[e6]}
            if len(cols) < 5:
                bad += 1
        return bad
    
    best = colors_list[:]
    best_bad = bad_count
    num_colors = 15
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to reassign
        edge_idx = random.randint(0, 119)
        old_color = best[edge_idx]
        new_color = random.randint(0, num_colors - 1)
        if new_color == old_color:
            continue
        
        best[edge_idx] = new_color
        new_bad = count_bad(best)
        
        if new_bad <= best_bad:
            best_bad = new_bad
        else:
            # Revert
            best[edge_idx] = old_color
        
        if best_bad == 0:
            break
    
    return best, best_bad

def _try_random_colorings(deadline):
    """Try random colourings with local search."""
    t0 = time.time()
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in all_k4s:
        a, b, c, d = quad
        edges = [
            _idx(a, b), _idx(a, c), _idx(a, d),
            _idx(b, c), _idx(b, d), _idx(c, d)
        ]
        k4_edges.append(edges)
    
    def count_bad(cl):
        bad = 0
        for edges in k4_edges:
            e1, e2, e3, e4, e5, e6 = edges
            cols = {cl[e1], cl[e2], cl[e3], cl[e4], cl[e5], cl[e6]}
            if len(cols) < 5:
                bad += 1
        return bad
    
    best_result = None
    best_bad = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start with random 15-colouring
        colors = [random.randint(0, 14) for _ in range(120)]
        bad = count_bad(colors)
        
        # Local search
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            
            edge_idx = random.randint(0, 119)
            old_color = colors[edge_idx]
            new_color = random.randint(0, 14)
            if new_color == old_color:
                continue
            
            colors[edge_idx] = new_color
            new_bad = count_bad(colors)
            
            if new_bad <= bad:
                bad = new_bad
            else:
                colors[edge_idx] = old_color
            
            if bad == 0:
                break
        
        if bad < best_bad:
            best_bad = bad
            best_result = colors[:]
        
        if best_bad == 0:
            return best_result
    
    return best_result

def _make_17coloring():
    """The (i+j)%17 colouring, guaranteed valid with 17 colours."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def construct():
    t0 = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Precompute K4 edge indices for fast validation
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in all_k4s:
        a, b, c, d = quad
        edges = [
            _idx(a, b), _idx(a, c), _idx(a, d),
            _idx(b, c), _idx(b, d), _idx(c, d)
        ]
        k4_edges.append(edges)
    
    def count_bad(cl):
        bad = 0
        for edges in k4_edges:
            e1, e2, e3, e4, e5, e6 = edges
            cols = {cl[e1], cl[e2], cl[e3], cl[e4], cl[e5], cl[e6]}
            if len(cols) < 5:
                bad += 1
        return bad
    
    # 1. Try cyclic 1-factorization (15 colours)
    color_map = _make_1factorization_cyclic()
    colors_list = _color_map_to_list(color_map)
    bad = count_bad(colors_list)
    
    if bad == 0:
        return colors_list
    
    # 2. Local search on the 1-factorization starting point
    searched, searched_bad = _local_search_1factor(color_map, max_iterations=10000, deadline=deadline - 30)
    if searched_bad == 0:
        return searched
    
    # 3. Try random colourings with local search
    result = _try_random_colorings(deadline - 10)
    if result is not None:
        if count_bad(result) == 0:
            return result
    
    # 4. Fallback: 17-colouring (guaranteed valid)
    return _make_17coloring()

# EVOLVE-BLOCK-END
