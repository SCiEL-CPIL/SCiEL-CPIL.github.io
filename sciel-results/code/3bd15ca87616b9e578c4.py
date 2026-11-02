"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

D = 11
TIME_LIMIT = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

D = 11
TIME_LIMIT = 140

def construct():
    start_time = time.time()
    
    # Generate all candidate nonzero integer vectors with norm <= 2
    candidates = []
    
    # Type A: one coordinate is ±1 (norm 1)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Type B: two coordinates are ±1 (norm 2)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type C: one coordinate is ±2 (norm 2)
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    n = len(candidates)
    
    # Build conflict graph: edge between i,j if distance < 2 (i.e., distance^2 < 4)
    # distance^2 = ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v
    # We need distance >= 2, so conflict if distance^2 < 4
    
    # Compute norms squared
    norms_sq = np.array([sum(x*x for x in v) for v in candidates])
    
    # Build adjacency as bitsets for efficiency
    # adj[i] is a set of j that conflict with i
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            # Compute dot product
            dot = sum(candidates[i][k] * candidates[j][k] for k in range(D))
            dist_sq = norms_sq[i] + norms_sq[j] - 2 * dot
            if dist_sq < 4:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum Independent Set via branch-and-bound with coloring
    # Convert to complement: we want max set with no edges between them
    # In the conflict graph, independent set = set with no conflicts
    
    # Use greedy coloring bound (Tomita's algorithm)
    # We'll work on the conflict graph; independent set in G = clique in complement
    
    # Actually, let's do MIS directly on the conflict graph using branch and bound
    # with a greedy coloring upper bound
    
    # For MIS: at each step, pick a vertex, branch on include/exclude
    # Upper bound: greedy coloring of remaining candidates
    
    # Let's use a simpler but effective approach: greedy + local search
    # Or: use the fact that n=240 is small enough for a good B&B
    
    # Greedy coloring for upper bound on independent set size
    def greedy_color_upper_bound(vertices, adj):
        """Greedy coloring of the subgraph induced by vertices.
        The number of colors is an upper bound on the independent set size."""
        if not vertices:
            return 0
        
        # Order vertices by degree (descending) for better coloring
        verts = list(vertices)
        verts.sort(key=lambda v: -len(adj[v] & set(verts)))
        
        colors = {}
        next_color = 0
        for v in verts:
            # Find smallest available color
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            next_color = max(next_color, c + 1)
        
        return next_color
    
    best_set = set()
    
    def mis_branch(remaining, current_set, depth=0):
        nonlocal best_set
        nonlocal start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if not remaining:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return
        
        # Upper bound
        ub = greedy_color_upper_bound(remaining, adj)
        if len(current_set) + ub <= len(best_set):
            return
        
        # Pick a vertex with maximum degree in remaining
        rem_list = list(remaining)
        v = max(rem_list, key=lambda x: len(adj[x] & set(remaining)))
        
        # Branch 1: include v
        new_current = current_set | {v}
        new_remaining = remaining - {v}
        new_remaining = new_remaining - adj[v]  # remove neighbors
        mis_branch(new_remaining, new_current, depth + 1)
        
        # Branch 2: exclude v
        new_remaining = remaining - {v}
        mis_branch(new_remaining, current_set, depth + 1)
    
    all_vertices = set(range(n))
    mis_branch(all_vertices, set())
    
    # Convert back to vectors
    result = [candidates[i] for i in sorted(best_set)]
    
    if not result:
        # Fallback: return original
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
