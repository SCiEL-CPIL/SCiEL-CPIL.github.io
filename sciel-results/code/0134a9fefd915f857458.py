"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import Counter

N = 16
NUM_COLORS = 120

def _get_idx(i, j):
    """Get the index for edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i*(i+1)//2 + j - i - 1

def _precompute_constraints():
    """
    Precompute for each edge e, the list of triples of other edges that form 4-cycles with e.
    A 4-cycle involving edge (u,v) and vertices x,y is:
    1. u-v-x-y-u : edges (u,v), (v,x), (x,y), (y,u)
    2. u-v-y-x-u : edges (u,v), (v,y), (y,x), (x,u)
    """
    constraints = [[] for _ in range(N*(N-1)//2)]
    for u in range(N):
        for v in range(u+1, N):
            e_idx = _get_idx(u, v)
            for x in range(N):
                if x == u or x == v: continue
                for y in range(N):
                    if y == u or y == v or y == x: continue
                    # Cycle 1: u-v-x-y-u
                    # Edges: (u,v), (v,x), (x,y), (y,u)
                    e_vx = _get_idx(v, x)
                    e_xy = _get_idx(x, y)
                    e_yu = _get_idx(y, u)
                    constraints[e_idx].append((e_vx, e_xy, e_yu))
                    
                    # Cycle 2: u-v-y-x-u
                    # Edges: (u,v), (v,y), (y,x), (x,u)
                    e_vy = _get_idx(v, y)
                    e_yx = _get_idx(y, x)
                    e_xu = _get_idx(x, u)
                    constraints[e_idx].append((e_vy, e_yx, e_xu))
    return constraints

def _greedy_construct(order, constraints):
    """
    Greedy coloring in the given edge order.
    Picks the valid color with the lowest frequency to maximize diversity.
    """
    colors = [-1] * (N*(N-1)//2)
    color_counts = Counter()
    
    for e in order:
        # Determine valid colors
        valid_colors = []
        for c in range(NUM_COLORS):
            is_valid = True
            for e1, e2, e3 in constraints[e]:
                # Check if all other edges are colored
                if colors[e1] != -1 and colors[e2] != -1 and colors[e3] != -1:
                    # Check if adding c creates 4 distinct colors
                    if len({c, colors[e1], colors[e2], colors[e3]}) == 4:
                        is_valid = False
                        break
            if is_valid:
                valid_colors.append(c)
        
        if not valid_colors:
            # Fallback: should not happen if we start with a valid partial coloring
            # But if it does, pick the least frequent color anyway (risk of invalidity)
            # Or return None to indicate failure
            return None
        
        # Pick color with lowest frequency
        best_c = min(valid_colors, key=lambda c: (color_counts[c], c))
        colors[e] = best_c
        color_counts[best_c] += 1
        
    return colors

def _check_validity(colors):
    """Check that no 4-cycle has four distinct colors."""
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Cycle 1: a-b, b-c, c-d, d-a
        c1 = {colors[_get_idx(a,b)], colors[_get_idx(b,c)], colors[_get_idx(c,d)], colors[_get_idx(d,a)]}
        if len(c1) == 4:
            return False
        # Cycle 2: a-b, b-d, d-c, c-a
        c2 = {colors[_get_idx(a,b)], colors[_get_idx(b,d)], colors[_get_idx(d,c)], colors[_get_idx(c,a)]}
        if len(c2) == 4:
            return False
        # Cycle 3: a-c, c-b, b-d, d-a
        c3 = {colors[_get_idx(a,c)], colors[_get_idx(c,b)], colors[_get_idx(b,d)], colors[_get_idx(d,a)]}
        if len(c3) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline to be safe
    
    random.seed(42)
    
    constraints = _precompute_constraints()
    
    best_colors = None
    best_count = 0
    
    # Try several different orderings
    # 1. Lexicographic
    order1 = list(range(N*(N-1)//2))
    
    # 2. Reverse Lexicographic
    order2 = list(reversed(order1))
    
    # 3. Random orders
    for _ in range(20):
        order3 = order1[:]
        random.shuffle(order3)
        
        if time.time() > deadline - 5:
            break
            
        res = _greedy_construct(order3, constraints)
        if res is not None:
            if _check_validity(res):
                cnt = _count_colors(res)
                if cnt > best_count:
                    best_count = cnt
                    best_colors = res[:]
        
        # Also try lexicographic and reverse once
        if _ == 0:
            res = _greedy_construct(order1, constraints)
            if res is not None and _check_validity(res):
                cnt = _count_colors(res)
                if cnt > best_count:
                    best_count = cnt
                    best_colors = res[:]
            
            res = _greedy_construct(order2, constraints)
            if res is not None and _check_validity(res):
                cnt = _count_colors(res)
                if cnt > best_count:
                    best_count = cnt
                    best_colors = res[:]

    if best_colors is None:
        # Fallback to a simple valid coloring if greedy failed (unlikely)
        return [0] * (N*(N-1)//2)
        
    return best_colors
# EVOLVE-BLOCK-END
