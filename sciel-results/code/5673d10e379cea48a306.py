"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

# Precompute the 60 undirected Hamiltonian cycles on vertices 0..5
# A cycle is a permutation (v0, v1, v2, v3, v4, v5) where edges are
# (v0,v1), (v1,v2), ..., (v5,v0). Two permutations define the same
# undirected cycle if one is a rotation or reversal of the other.
# We fix v0=0 to remove rotational symmetry, then deduplicate.
_CYCLES = []
_seen = set()
for _perm in permutations(range(1, 6)):
    _cycle = (0,) + _perm
    _edges = frozenset(frozenset((_cycle[k], _cycle[(k+1) % 6])) for k in range(6))
    if _edges not in _seen:
        _seen.add(_edges)
        _CYCLES.append(_cycle)
# len(_CYCLES) should be 60

def _has_c6_fast(adj, n):
    """Check if boolean adjacency matrix adj (n x n) contains a C6.
    Returns True if a C6 is found, False otherwise."""
    if n < 6:
        return False
    # For each 6-subset, check if any of the 60 cycle patterns matches
    for subset in combinations(range(n), 6):
        for cycle in _CYCLES:
            ok = True
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k + 1) % 6]]
                if not adj[u, w]:
                    ok = False
                    break
            if ok:
                return True
    return False

def _verify(colors, n):
    """Verify that the coloring has no C6 in any color class."""
    if n < 6:
        return True
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = True
            adj[c][j, i] = True
            idx += 1
    for c in range(4):
        if _has_c6_fast(adj[c], n):
            return False
    return True

def _edge_index(i, j, n):
    """Index in lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_adj_matrices(colors, n):
    """Build 4 boolean adjacency matrices from color list."""
    adjs = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjs[c][i, j] = True
            adjs[c][j, i] = True
            idx += 1
    return adjs

def _random_coloring(n):
    m = n * (n - 1) // 2
    return np.random.randint(0, 4, size=m).tolist()

def _local_search(colors, n, deadline):
    """Local search: iteratively find C6 violations and flip edges to break them."""
    for _ in range(50000):
        if time.time() > deadline:
            break
        adjs = _build_adj_matrices(colors, n)
        found = False
        for c in range(4):
            if _has_c6_fast(adjs[c], n):
                # Find a specific C6 in color class c
                for subset in combinations(range(n), 6):
                    for cycle in _CYCLES:
                        ok = True
                        for k in range(6):
                            u = subset[cycle[k]]
                            w = subset[cycle[(k + 1) % 6]]
                            if not adjs[c][u, w]:
                                ok = False
                                break
                        if ok:
                            # Found a C6; flip a random edge in it
                            k = np.random.randint(6)
                            v = subset[cycle[k]]
                            w = subset[cycle[(k + 1) % 6]]
                            if v > w:
                                v, w = w, v
                            idx = _edge_index(v, w, n)
                            old_color = colors[idx]
                            new_color = np.random.randint(0, 4)
                            while new_color == old_color:
                                new_color = np.random.randint(0, 4)
                            colors[idx] = new_color
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        if not found:
            # No C6 found; verify
            if _verify(colors, n):
                return colors
            break
    return colors

def _insert_vertex(colors, n_current, n_target, deadline):
    """
    Try to extend a valid coloring of K_{n_current} to K_{n_target}
    by inserting one vertex at a time, using a constraint-based assignment
    for the new vertex's edges.
    
    Returns a list of colors for K_{n_target} if successful, else None.
    """
    # Work with a mutable copy
    cur_colors = list(colors)
    cur_n = n_current
    
    while cur_n < n_target:
        if time.time() > deadline:
            return None
        
        new_n = cur_n + 1
        # The new vertex is vertex new_n-1 (index cur_n)
        # We need to assign colors for edges (i, cur_n) for i in 0..cur_n-1
        
        # Build current adjacency matrices
        adjs = _build_adj_matrices(cur_colors, cur_n)
        
        # For each color c, we need to check: if we assign color c to edge (i, new_v),
        # does it create a C6 in color class c?
        # A C6 involving new_v and edge (i, new_v) of color c would be:
        # new_v - i - a - b - d - e - new_v  (but that's 6 edges, 7 vertices? No)
        # Actually C6 has 6 vertices. If new_v is one vertex, the other 5 are from 0..cur_n-1.
        # The cycle is new_v - v1 - v2 - v3 - v4 - v5 - new_v.
        # Edges: (new_v, v1) has color c, (v1,v2) has color c, ..., (v5, new_v) has color c.
        # So we need: v1 and v5 are neighbors of new_v in color c (i.e., edges (v1,new_v) and (v5,new_v) are color c),
        # and there is a path of length 3 from v1 to v5 in color c within vertices 0..cur_n-1,
        # using vertices distinct from new_v (which they are, since they're in 0..cur_n-1).
        
        # So: for a candidate assignment where edge (i, new_v) gets color c,
        # we need to check that there is NO other vertex j (j != i) such that:
        #   edge (j, new_v) also gets color c, AND
        #   there exists a path of length 3 from i to j in color class c within 0..cur_n-1.
        
        # Since we're assigning all edges (i, new_v) for i=0..cur_n-1, we need to consider
        # all pairs (i, j) both assigned color c, and check for a length-3 path between them.
        
        # Strategy: try all 4^cur_n assignments? That's too many for cur_n > 10.
        # Better: use backtracking on the new vertex's edges.
        
        # Precompute: for each color c, for each pair (i, j) in 0..cur_n-1,
        # is there a path of length 3 from i to j in color class c?
        # Path of length 3: i - a - b - j, all distinct, edges (i,a), (a,b), (b,j) in color c.
        
        # Precompute path3[c][i][j]
        path3 = [[[False] * cur_n for _ in range(cur_n)] for _ in range(4)]
        for c in range(4):
            adj_c = adjs[c]
            # path3[i][j] = exists a, b: adj_c[i][a] and adj_c[a][b] and adj_c[b][j] and all distinct
            # Use matrix multiplication: (A^3)[i][j] > 0 but that counts walks, not simple paths.
            # For cur_n <= 19, we can do it directly.
            for i in range(cur_n):
                for j in range(cur_n):
                    if i == j:
                        continue
                    found = False
                    for a in range(cur_n):
                        if a == i or a == j:
                            continue
                        if not adj_c[i, a]:
                            continue
                        for b in range(cur_n):
                            if b == i or b == j or b == a:
                                continue
                            if adj_c[a, b] and adj_c[b, j]:
                                found = True
                                break
                        if found:
                            break
                    path3[c][i][j] = found
        
        # Now backtrack on the new vertex's edges
        # Assignment: assign[i] = color for edge (i, new_v), for i in 0..cur_n-1
        # Constraint: for any i != j, if assign[i] == assign[j] == c, then not path3[c][i][j]
        
        assign = [-1] * cur_n
        
        def _bt(k):
            if time.time() > deadline:
                return False
            if k == cur_n:
                return True
            for c in range(4):
                valid = True
                for j in range(k):
                    if assign[j] == c and path3[c][j][k]:
                        valid = False
                        break
                if not valid:
                    continue
                assign[k] = c
                if _bt(k + 1):
                    return True
                assign[k] = -1
            return False
        
        if _bt(0):
            # Build new color list
            new_colors = list(cur_colors)
            for i in range(cur_n):
                new_colors.append(assign[i])
            cur_colors = new_colors
            cur_n = new_n
        else:
            return None
    
    return cur_colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with a small valid coloring and try to grow it
    # K_1 is trivially valid
    best_n = 1
    best_colors = []
    
    # Try to grow from n=1 up to 20
    for target_n in range(2, 21):
        if time.time() > deadline - 10:
            break
        
        # Try multiple attempts to grow to target_n
        success = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start from best known coloring of size min(target_n - 1, best_n)
            if best_n >= target_n - 1:
                base_colors = list(best_colors)
                base_n = best_n
            else:
                # Use a random coloring of size target_n - 1
                base_n = target_n - 1
                base_colors = _random_coloring(base_n)
                # Try to make it valid via local search
                base_colors = _local_search(base_colors, base_n, deadline)
                if not _verify(base_colors, base_n):
                    continue
            
            # Try to insert one vertex
            new_colors = _insert_vertex(base_colors, base_n, target_n, deadline)
            if new_colors is not None and _verify(new_colors, target_n):
                best_n = target_n
                best_colors = new_colors
                success = True
                break
        
        if not success:
            # Try random restarts for this target_n
            for restart in range(30):
                if time.time() > deadline - 5:
                    break
                colors = _random_coloring(target_n)
                colors = _local_search(colors, target_n, deadline)
                if _verify(colors, target_n):
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
