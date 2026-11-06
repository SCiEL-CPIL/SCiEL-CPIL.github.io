"""Improved program for edge-colourings of K16 with >=5 colours per K4."""
import time
import itertools
import random

def _validate(coloring, n=16):
    """Check that every K4 spans at least 5 distinct colours."""
    col = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = col[j][i] = coloring[idx]
            idx += 1
    for a, b, c, d in itertools.combinations(range(n), 4):
        colours = set()
        colours.add(col[a][b])
        colours.add(col[a][c])
        colours.add(col[a][d])
        colours.add(col[b][c])
        colours.add(col[b][d])
        colours.add(col[c][d])
        if len(colours) < 5:
            return False
    return True

def _count_colours(coloring):
    """Count distinct colours used."""
    return len(set(coloring))

def _edge_index(i, j):
    """Get the index in the flat array for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i*(i-1)//2 + (j - i - 1)

def _build_1_factorization(n=16):
    """Build a 1-factorization of K_n (n must be even).
    Returns list of n-1 matchings, each a list of (i,j) edges.
    Uses the standard circle method."""
    matchings = []
    # Fix vertex 0, rotate vertices 1..n-1
    # For round r = 0..n-2:
    #   matching r pairs vertex 0 with vertex (1+r) % (n-1) + 1
    #   and pairs other vertices in symmetric positions
    # Actually, standard circle method:
    # Place vertices 0, 1, 2, ..., n-2 in a circle, vertex n-1 in center.
    # For round r: pair center (n-1) with vertex r.
    #   Then pair (r+1, r-1), (r+2, r-2), ... mod (n-1)
    # Let's use a simpler approach for n=16:
    # Vertices 0..15. Fix 15. Rotate 0..14.
    # Round r: 15 paired with r.
    #   Then for k=1..7: (r+k) mod 15 paired with (r-k) mod 15
    
    matchings = []
    for r in range(n - 1):
        matching = []
        # Pair vertex (n-1) with vertex r
        matching.append((r, n - 1))
        # Pair remaining vertices
        for k in range(1, (n - 1) // 2 + 1):
            a = (r + k) % (n - 1)
            b = (r - k) % (n - 1)
            matching.append((a, b))
        matchings.append(matching)
    
    # Verify: every edge appears exactly once
    all_edges = set()
    for m in matchings:
        for (i, j) in m:
            key = (min(i,j), max(i,j))
            if key in all_edges:
                raise ValueError(f"Edge {key} appears twice")
            all_edges.add(key)
    if len(all_edges) != n*(n-1)//2:
        raise ValueError(f"Expected {n*(n-1)//2} edges, got {len(all_edges)}")
    
    return matchings

def _factorization_coloring(n=16):
    """Color each edge by the index of the matching it belongs to."""
    matchings = _build_1_factorization(n)
    coloring = [0] * (n*(n-1)//2)
    for m_idx, matching in enumerate(matchings):
        for (i, j) in matching:
            coloring[_edge_index(i, j)] = m_idx
    return coloring

def _try_linear(a, b, k, n=16):
    """Try (a*i + b*j) % k for i<j."""
    coloring = [(a*i + b*j) % k for i in range(n) for j in range(i+1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _local_search(initial_coloring, n=16, max_iters=200, deadline=None):
    """Try to reduce number of colours via local search."""
    coloring = list(initial_coloring)
    best_count = _count_colours(coloring)
    
    # Precompute all K4s and their edge indices
    k4s = list(itertools.combinations(range(n), 4))
    k4_edge_indices = []
    for a,b,c,d in k4s:
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        eidx = [_edge_index(x,y) for x,y in edges]
        k4_edge_indices.append(eidx)
    
    # For each edge, precompute which K4s contain it
    edge_to_k4s = [[] for _ in range(len(coloring))]
    for k4_idx, eidx_list in enumerate(k4_edge_indices):
        for ei in eidx_list:
            edge_to_k4s[ei].append(k4_idx)
    
    current_count = best_count
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Try to recolor each edge to a colour already in use
        for eidx in range(len(coloring)):
            if deadline is not None and time.time() > deadline:
                break
            old_c = coloring[eidx]
            
            # Try all colours currently in use
            used_colours = sorted(set(coloring))
            for new_c in used_colours:
                if new_c == old_c:
                    continue
                # Check if recoloring this edge keeps all K4s valid
                valid = True
                for k4_idx in edge_to_k4s[eidx]:
                    eidx_list = k4_edge_indices[k4_idx]
                    colours = set()
                    for ei in eidx_list:
                        if ei == eidx:
                            colours.add(new_c)
                        else:
                            colours.add(coloring[ei])
                    if len(colours) < 5:
                        valid = False
                        break
                if valid:
                    coloring[eidx] = new_c
                    new_count = _count_colours(coloring)
                    if new_count < current_count:
                        current_count = new_count
                        best_count = new_count
                        improved = True
                    break  # take the first valid recoloring
    
    return coloring, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_coloring = None
    best_count = 100
    
    # 1. Try 1-factorization: gives 15 colours
    coloring = _factorization_coloring()
    if _validate(coloring):
        best_coloring = coloring
        best_count = _count_colours(coloring)
    
    # 2. Try (i+j) % k for small k
    for k in range(13, 16):
        if time.time() > deadline - 30:
            break
        baseline = [(i + j) % k for i in range(16) for j in range(i + 1, 16)]
        if _validate(baseline) and _count_colours(baseline) < best_count:
            best_coloring = baseline
            best_count = _count_colours(baseline)
    
    # 3. Try linear forms
    for k in range(13, 16):
        if time.time() > deadline - 25:
            break
        for a in range(1, 6):
            for b in range(1, 6):
                if time.time() > deadline - 20:
                    break
                coloring, count = _try_linear(a, b, k)
                if coloring is not None and count < best_count:
                    best_coloring = coloring
                    best_count = count
    
    # 4. Local search to reduce colour count
    if best_coloring is not None and best_count > 13 and time.time() < deadline - 10:
        coloring, count = _local_search(best_coloring, deadline=deadline - 5)
        if count < best_count:
            best_coloring = coloring
            best_count = count
    
    if best_coloring is None:
        # Fallback
        best_coloring = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_coloring
