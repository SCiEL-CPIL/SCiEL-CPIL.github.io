"""Improved program for R_4(C6): maximize n with 4-colouring of K_n with no monochromatic C6."""
import itertools
import time
from collections import defaultdict

def has_c6(edges, n):
    """Check if the graph (edges set of frozensets or similar) on n vertices contains a C6."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check for C6: enumerate all 6-tuples of distinct vertices and check if they form a cycle
    # To be efficient, we can try to find any 6-cycle
    # A 6-cycle is a sequence v0, v1, v2, v3, v4, v5, v0 where all are distinct
    # We can fix v0 = 0 to reduce symmetry, then try all permutations of 5 other vertices
    # But that's 5! = 120 per starting vertex, times n starting vertices = manageable for n<=19
    
    # Better approach: use backtracking to find a 6-cycle
    # For each pair of vertices (a, b) that are not adjacent, try to find a path of length 5 between them
    # Actually, let's just enumerate cycles directly
    
    # For n up to 19, we can check all 6-element subsets and all 60 possible cycle orderings
    # C(19,6) = 27132, times 60 = ~1.6M checks, each check is O(6) = manageable
    
    from itertools import combinations, permutations
    
    # Precompute adjacency matrix for fast lookup
    adjmat = [[False]*n for _ in range(n)]
    for (u, v) in edges:
        adjmat[u][v] = True
        adjmat[v][u] = True
    
    # For each 6-subset, check if any permutation forms a cycle
    for subset in combinations(range(n), 6):
        # A cycle on 6 vertices: fix the first vertex to reduce symmetry
        # There are (6-1)!/2 = 60 distinct cycles
        # We can just check all 720 permutations but that's wasteful
        # Better: fix v0 = subset[0], then permute the other 5
        v0 = subset[0]
        rest = list(subset[1:])
        for perm in itertools.permutations(rest):
            v = [v0] + list(perm)
            # Check if v0-v1-v2-v3-v4-v5-v0 is a cycle
            ok = True
            for i in range(6):
                if not adjmat[v[i]][v[(i+1)%6]]:
                    ok = False
                    break
            if ok:
                return True
    return False

def construct():
    # We want to find the largest n such that there exists a 4-colouring of K_n
    # with no monochromatic C6.
    # Known: n=17 is possible. We'll try to construct one.
    # 
    # Approach: Use a known construction or search.
    # Since we have 120 seconds, let's try a smart search or use a known result.
    #
    # One known construction for R_4(C6) >= 17 uses algebraic or probabilistic methods.
    # Let's try to use a greedy/randomized search with local improvement.
    
    # Start with a known good seed: try n=17 with a structured colouring
    # Use a finite field or cyclic group construction
    
    # Let's try a cyclic construction on Z_17
    # Colour edge (i,j) based on (j-i) mod 17 mapped to {0,1,2,3}
    # We need to partition the 16 non-zero differences into 4 sets of 4,
    # such that no set contains a 6-cycle pattern.
    
    # Actually, let's just try a randomized search with backtracking for small n,
    # and for larger n use a heuristic.
    
    # Given time constraints, let's try to return a known valid colouring for n=17
    # if we can construct one, otherwise fall back to smaller n.
    
    # Try n from 17 down to 1
    for n in range(17, 0, -1):
        # Try a few random colourings with local search
        import random
        random.seed(42 + n)
        
        best_colors = None
        for attempt in range(50):
            # Random initial colouring
            colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
            
            # Check if valid
            if _is_valid(colors, n):
                best_colors = colors
                break
        
        if best_colors is not None:
            return {'n': n, 'colors': best_colors}
    
    # Fallback: n=1 is always valid
    return {'n': 1, 'colors': []}

def _is_valid(colors, n):
    """Check if a 4-colouring of K_n has no monochromatic C6."""
    # Build edge list for each colour
    color_edges = {0: [], 1: [], 2: [], 3: []}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            color_edges[c].append((i, j))
            idx += 1
    
    for c in range(4):
        if _has_c6(color_edges[c], n):
            return False
    return True

def _has_c6(edges, n):
    """Check if graph with given edges on n vertices contains a C6."""
    if len(edges) < 6:
        return False
    
    adjmat = [[False]*n for _ in range(n)]
    for (u, v) in edges:
        adjmat[u][v] = True
        adjmat[v][u] = True
    
    from itertools import combinations, permutations
    
    for subset in combinations(range(n), 6):
        v0 = subset[0]
        rest = list(subset[1:])
        for perm in permutations(rest):
            v = [v0] + list(perm)
            ok = True
            for i in range(6):
                if not adjmat[v[i]][v[(i+1)%6]]:
                    ok = False
                    break
            if ok:
                return True
    return False
