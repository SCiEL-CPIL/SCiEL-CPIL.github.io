"""Edge-colourings of K16: every K4 sees >=5 colours. Minimize colours used."""
import time
import itertools
import random
import numpy as np

def _idx(i, j):
    """Index for pair (i,j) with 0<=i<j<16 in lexicographic flat list."""
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _validate_list(colors_list):
    """Check every K4 spans at least 5 distinct colours."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        if len({e1, e2, e3, e4, e5, e6}) < 5:
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
        if len({e1, e2, e3, e4, e5, e6}) < 5:
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

def _make_1factorization_cyclic2():
    """Alternative cyclic 1-factorization using different offset pattern."""
    color_map = {}
    # Use a different pairing scheme
    for i in range(15):
        color_map[(15, i)] = i
        color_map[(i, 15)] = i
        for j in range(1, 8):
            a = (i + 2*j) % 15
            b = (i - 2*j) % 15
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) not in color_map:
                color_map[(a, b)] = i
    # Fill any missing edges with a default (shouldn't happen)
    for i in range(16):
        for j in range(i+1, 16):
            if (i, j) not in color_map:
                color_map[(i, j)] = 14  # fallback
    return color_map

def _map_to_list(color_map):
    """Convert color dict to flat list of 120 colors."""
    result = []
    for i in range(16):
        for j in range(i + 1, 16):
            result.append(color_map[(i, j)])
    return result

def _local_search_1factor(colors_list, max_iters=50000, seed=42):
    """Local search: swap colors of two disjoint edges to reduce bad K4s."""
    colors = list(colors_list)
    rng = random.Random(seed)
    
    # Precompute all edges
    edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_to_idx[(i, j)] = idx
    
    # Precompute which K4s contain which edges
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        k4_edges.append([
            edge_to_idx[(a,b)], edge_to_idx[(a,c)], edge_to_idx[(a,d)],
            edge_to_idx[(b,c)], edge_to_idx[(b,d)], edge_to_idx[(c,d)]
        ])
    
    def count_bad_fast(colors):
        bad = 0
        for edge_indices in k4_edges:
            cols = {colors[i] for i in edge_indices}
            if len(cols) < 5:
                bad += 1
        return bad
    
    current_bad = count_bad_fast(colors)
    if current_bad == 0:
        return colors
    
    best_colors = list(colors)
    best_bad = current_bad
    
    for iteration in range(max_iters):
        # Pick two random disjoint edges
        e1_idx = rng.randint(0, 119)
        e2_idx = rng.randint(0, 119)
        if e1_idx == e2_idx:
            continue
        
        e1 = edges[e1_idx]
        e2 = edges[e2_idx]
        
        # Check if disjoint
        if len(set(e1) & set(e2)) > 0:
            continue
        
        # Swap colors
        c1, c2 = colors[e1_idx], colors[e2_idx]
        if c1 == c2:
            continue
        
        colors[e1_idx] = c2
        colors[e2_idx] = c1
        
        new_bad = count_bad_fast(colors)
        
        if new_bad < best_bad:
            best_bad = new_bad
            best_colors = list(colors)
            if best_bad == 0:
                break
        else:
            # Revert
            colors[e1_idx] = c1
            colors[e2_idx] = c2
    
    return best_colors

def _try_random_1factorizations(num_tries=10, seed=0):
    """Try multiple random 1-factorizations and pick the best."""
    best_colors = None
    best_bad = float('inf')
    
    rng = random.Random(seed)
    
    for trial in range(num_tries):
        # Generate a random 1-factorization
        color_map = {}
        used_edges = set()
        
        for factor in range(15):
            # Generate a random perfect matching
            remaining = list(range(16))
            rng.shuffle(remaining)
            for k in range(8):
                a, b = remaining[2*k], remaining[2*k+1]
                if a > b:
                    a, b = b, a
                color_map[(a, b)] = factor
                used_edges.add((a, b))
        
        # Verify it's a valid 1-factorization (each vertex appears once per factor)
        valid = True
        for factor in range(15):
            vertices_used = set()
            for (a, b), c in color_map.items():
                if c == factor:
                    if a in vertices_used or b in vertices_used:
                        valid = False
                        break
                    vertices_used.add(a)
                    vertices_used.add(b)
            if not valid:
                break
            if len(vertices_used) != 16:
                valid = False
                break
        
        if not valid:
            continue
        
        colors_list = _map_to_list(color_map)
        bad = _count_bad_k4(colors_list)
        
        if bad < best_bad:
            best_bad = bad
            best_colors = colors_list
        
        if best_bad == 0:
            break
    
    return best_colors, best_bad

def _make_17coloring():
    """The guaranteed valid 17-coloring."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def construct():
    t0 = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Strategy:
    # 1. Try several 1-factorizations (cyclic variants)
    # 2. Apply local search to fix violations
    # 3. If we get 0 bad K4s with 15 colors, return that
    # 4. Otherwise, fall back to 17-coloring
    
    best_result = None
    best_num_colors = 17
    best_bad = float('inf')
    
    # Try cyclic 1-factorization
    if time.time() - t0 < deadline:
        cm = _make_1factorization_cyclic()
        cl = _map_to_list(cm)
        bad = _count_bad_k4(cl)
        if bad < best_bad:
            best_bad = bad
            best_result = cl
            best_num_colors = max(cl) + 1
        if bad == 0:
            return cl
    
    # Try cyclic2
    if time.time() - t0 < deadline:
        cm = _make_1factorization_cyclic2()
        cl = _map_to_list(cm)
        bad = _count_bad_k4(cl)
        if bad < best_bad:
            best_bad = bad
            best_result = cl
            best_num_colors = max(cl) + 1
        if bad == 0:
            return cl
    
    # Try random 1-factorizations
    if time.time() - t0 < deadline:
        rc, rb = _try_random_1factorizations(num_tries=5, seed=123)
        if rc is not None and rb < best_bad:
            best_bad = rb
            best_result = rc
            best_num_colors = max(rc) + 1
        if rb == 0:
            return rc
    
    # Apply local search to the best 1-factorization found
    if best_result is not None and best_bad > 0 and time.time() - t0 < deadline:
        improved = _local_search_1factor(best_result, max_iters=100000, seed=456)
        imp_bad = _count_bad_k4(improved)
        if imp_bad < best_bad:
            best_bad = imp_bad
            best_result = improved
            best_num_colors = max(improved) + 1
        if imp_bad == 0:
            return improved
    
    # If we still have bad K4s, try more local search iterations
    if best_result is not None and best_bad > 0 and time.time() - t0 < deadline:
        improved = _local_search_1factor(best_result, max_iters=200000, seed=789)
        imp_bad = _count_bad_k4(improved)
        if imp_bad < best_bad:
            best_bad = imp_bad
            best_result = improved
            best_num_colors = max(improved) + 1
        if imp_bad == 0:
            return improved
    
    # Check if we have a valid 15-coloring
    if best_result is not None and best_bad == 0:
        return best_result
    
    # Fallback: try to reduce colors in the 17-coloring
    # The 17-coloring uses colors 0..16. Can we merge some?
    # For now, return the 17-coloring as guaranteed valid
    result_17 = _make_17coloring()
    
    # Verify it's valid (should always be)
    if _validate_list(result_17):
        return result_17
    
    # Last resort: return the best we have (might be invalid, but better than nothing)
    if best_result is not None:
        return best_result
    
    return result_17
