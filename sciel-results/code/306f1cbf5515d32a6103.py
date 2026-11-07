import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14],
                        [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
                        [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE_EDGES = set(tuple(sorted(e)) for e in P['edges'])
D_TARGET = 4

# Precompute base adjacency
BASE_ADJ = [set() for _ in range(N)]
for u, v in BASE_EDGES:
    BASE_ADJ[u].add(v)
    BASE_ADJ[v].add(u)

# Precompute base distance matrix (path graph, so dist = |u-v|)
BASE_DIST = [[abs(i - j) for j in range(N)] for i in range(N)]

# Candidate edges: not in base, and |u-v| >= 3 (to avoid triangles with base edges)
ALL_CANDIDATES = []
for u in range(N):
    for v in range(u + 1, N):
        if (u, v) not in BASE_EDGES and abs(u - v) >= 3:
            ALL_CANDIDATES.append((u, v))

NUM_CANDIDATES = len(ALL_CANDIDATES)

# For each candidate edge, precompute which pairs it helps cover
# A pair (i,j) is "bad" if BASE_DIST[i][j] > D_TARGET
# Adding edge (u,v) can reduce dist(i,j) to min(BASE_DIST[i][j], 
#   BASE_DIST[i][u]+1+BASE_DIST[v][j], BASE_DIST[i][v]+1+BASE_DIST[u][j])
# But this is only valid if no triangle is formed. We'll handle triangle constraints dynamically.

# Precompute coverage: for each candidate edge e_idx, which bad pairs does it potentially cover?
BAD_PAIRS = []
for i in range(N):
    for j in range(i + 1, N):
        if BASE_DIST[i][j] > D_TARGET:
            BAD_PAIRS.append((i, j))

NUM_BAD = len(BAD_PAIRS)

# coverage[e_idx] = set of bad_pair indices that this edge could potentially cover
# (i.e., the new distance via this edge alone would be <= D_TARGET)
coverage = [set() for _ in range(NUM_CANDIDATES)]
for e_idx, (u, v) in enumerate(ALL_CANDIDATES):
    for bp_idx, (i, j) in enumerate(BAD_PAIRS):
        # Check if adding (u,v) makes dist(i,j) <= D_TARGET
        # New path: i->u->v->j or i->v->u->j
        d1 = BASE_DIST[i][u] + 1 + BASE_DIST[v][j]
        d2 = BASE_DIST[i][v] + 1 + BASE_DIST[u][j]
        if min(d1, d2) <= D_TARGET:
            coverage[e_idx].add(bp_idx)

# Precompute triangle conflicts between candidate edges
# Two candidate edges (u1,v1) and (u2,v2) conflict if they share a vertex or form a triangle
# Actually, they conflict if adding both creates a triangle.
# A triangle is formed if there exist three vertices a,b,c such that all three edges (a,b),(b,c),(c,a) exist.
# With base edges (path), a triangle can only be formed if we add an edge (u,v) with |u-v|=2, which we've excluded.
# But two added edges plus a base edge could form a triangle.
# E.g., add (u,v) and (v,w) where |u-w|=1 (base edge). Then u-v, v-w, w-u form triangle.
# So (u,v) and (v,w) conflict if |u-w| <= 1, i.e., if u and w are adjacent or same.
# More generally, edges (a,b) and (c,d) conflict if:
#   - They share a vertex AND the other two endpoints are adjacent or same
#   - Or they don't share a vertex but together with a base edge form a triangle
# Let's be more careful.

# Two edges e1=(a,b) and e2=(c,d) form a triangle with a base edge if:
# Case 1: They share a vertex, say b=c. Then we need |a-d| <= 1 for triangle a-b-d.
# Case 2: They don't share a vertex. Then we need a base edge connecting one endpoint of e1 to one of e2.
#   E.g., a-c is base edge, then a-b, b-? no. We need three edges forming a cycle of length 3.
#   The three edges would be e1, e2, and one base edge. For a triangle, we need three vertices.
#   If e1=(a,b) and e2=(c,d) with all distinct, then a triangle would need a base edge between two of {a,b,c,d}
#   such that the three edges form a 3-cycle. E.g., if a-c is base, then we'd need a-b, b-?, ?-c. 
#   This is getting complex. Let me just check dynamically.

# For efficiency, let's precompute which pairs of candidate edges are compatible (no triangle).
# We'll check: does adding both edges create any triangle?
# A triangle is three vertices x,y,z with edges (x,y),(y,z),(z,x) all present.
# The edges come from base + added. Since base is a path, the only non-base edges are the added ones.
# A triangle can involve 0, 1, or 2 added edges (3 added edges would be a triangle of added edges, but we check that too).

# Let's precompute a compatibility matrix for candidate edges.
# compatible[e1_idx][e2_idx] = True if adding both doesn't create a triangle.

def check_triangle_free(added_set):
    """Check if adding edges in added_set (indices into ALL_CANDIDATES) to base graph is triangle-free."""
    adj = [set(BASE_ADJ[i]) for i in range(N)]
    for e_idx in added_set:
        u, v = ALL_CANDIDATES[e_idx]
        adj[u].add(v)
        adj[v].add(u)
    for u in range(N):
        for v in adj[u]:
            if v > u:
                if adj[u] & adj[v]:
                    return False
    return True

# Precompute pairwise compatibility
compatible = [[True] * NUM_CANDIDATES for _ in range(NUM_CANDIDATES)]
for i in range(NUM_CANDIDATES):
    for j in range(i + 1, NUM_CANDIDATES):
        if not check_triangle_free({i, j}):
            compatible[i][j] = False
            compatible[j][i] = False

# Also check single edge compatibility (should all be true by construction, but verify)
# Single edges with |u-v|>=3 don't form triangles with base path edges.

def compute_diameter_with_edges(added_indices):
    """Compute diameter of base graph + edges at given indices. Returns inf if disconnected."""
    adj = [set(BASE_ADJ[i]) for i in range(N)]
    for e_idx in added_indices:
        u, v = ALL_CANDIDATES[e_idx]
        adj[u].add(v)
        adj[v].add(u)
    max_d = 0
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        queue = [s]
        while queue:
            nq = []
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        nq.append(v)
                        if dist[v] > max_d:
                            max_d = dist[v]
            queue = nq
        if -1 in dist:
            return float('inf')
    return max_d

def is_valid_solution(added_indices):
    """Check if the set of added edges is valid: triangle-free, connected, diameter <= 4."""
    if not check_triangle_free(added_indices):
        return False
    d = compute_diameter_with_edges(added_indices)
    return d <= D_TARGET

# Branch and bound for minimum set cover with triangle constraints
# Universe: BAD_PAIRS (indices 0..NUM_BAD-1)
# Sets: candidate edges, each covers some bad pairs
# Constraint: selected edges must be mutually compatible (no triangles)
# Objective: minimize number of selected edges

def branch_and_bound(remaining_bad, selected, available, best_sol, best_size, deadline):
    """
    remaining_bad: set of bad pair indices not yet covered
    selected: set of selected edge indices
    available: set of candidate edge indices that are compatible with all selected
    best_sol: current best solution (list of edge indices)
    best_size: current best size
    deadline: time limit
    """
    if time.time() > deadline:
        return best_sol, best_size
    
    if not remaining_bad:
        if len(selected) < best_size:
            best_sol = list(selected)
            best_size = len(selected)
        return best_sol, best_size
    
    # Lower bound: at least ceil(|remaining_bad| / max_coverage) more edges needed
    if available:
        max_cov = max(len(coverage[e] & remaining_bad) for e in available)
    else:
        max_cov = 0
    lb = len(selected) + (len(remaining_bad) + max_cov - 1) // max_cov if max_cov > 0 else float('inf')
    if lb >= best_size:
        return best_sol, best_size
    
    # Choose the bad pair with fewest covering edges (MRV heuristic)
    # Find bad pair in remaining_bad that is covered by fewest available edges
    min_count = float('inf')
    chosen_bad = None
    covering_edges = None
    for bp in remaining_bad:
        count = 0
        cov = []
        for e in available:
            if bp in coverage[e]:
                count += 1
                cov.append(e)
        if count < min_count:
            min_count = count
            chosen_bad = bp
            covering_edges = cov
        if count == 0:
            break
    
    if min_count == 0:
        # No available edge covers this bad pair, dead end
        return best_sol, best_size
    
    # Branch: try each edge that covers chosen_bad
    # Order edges by coverage (most covering first)
    covering_edges.sort(key=lambda e: -len(coverage[e] & remaining_bad))
    
    for e in covering_edges:
        if e not in available:
            continue
        # Add edge e
        new_selected = selected | {e}
        new_remaining = remaining_bad - coverage[e]
        
        # Find new available edges: those compatible with all in new_selected
        new_available = set()
        for e2 in available:
            if e2 == e:
                continue
            # Check if e2 is compatible with all in new_selected
            ok = True
            for e3 in new_selected:
                if e3 != e2 and not compatible[e3][e2]:
                    ok = False
                    break
            if ok:
                new_available.add(e2)
        
        best_sol, best_size = branch_and_bound(
            new_remaining, new_selected, new_available, best_sol, best_size, deadline
        )
    
    return best_sol, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial upper bound: use a greedy solution
    # Greedy: repeatedly pick the edge that covers the most uncovered bad pairs, 
    # subject to triangle constraints
    def greedy_solution():
        selected = set()
        remaining = set(range(NUM_BAD))
        available = set(range(NUM_CANDIDATES))
        while remaining:
            if time.time() > deadline:
                break
            best_e = None
            best_cov = -1
            for e in available:
                cov = len(coverage[e] & remaining)
                if cov > best_cov:
                    best_cov = cov
                    best_e = e
            if best_e is None or best_cov <= 0:
                break
            selected.add(best_e)
            remaining -= coverage[best_e]
            # Update available: remove edges incompatible with best_e
            new_available = set()
            for e in available:
                if e == best_e:
                    continue
                ok = True
                for e2 in selected:
                    if e2 != e and not compatible[e2][e]:
                        ok = False
                        break
                if ok:
                    new_available.add(e)
            available = new_available
        return selected
    
    greedy_sol = greedy_solution()
    if greedy_sol and is_valid_solution(greedy_sol):
        best_sol = list(greedy_sol)
        best_size = len(best_sol)
    else:
        # Fallback: try all edges in order
        best_sol = []
        best_size = NUM_CANDIDATES + 1
    
    # Branch and bound
    initial_available = set(range(NUM_CANDIDATES))
    initial_remaining = set(range(NUM_BAD))
    
    best_sol, best_size = branch_and_bound(
        initial_remaining, set(), initial_available, best_sol, best_size, deadline
    )
    
    # Convert to output format
    out = [[int(ALL_CANDIDATES[e][0]), int(ALL_CANDIDATES[e][1])] for e in sorted(best_sol)]
    
    # Final validation
    if not is_valid_solution(set(best_sol)):
        # Fallback to greedy
        out = [[int(ALL_CANDIDATES[e][0]), int(ALL_CANDIDATES[e][1])] for e in sorted(greedy_sol)]
    
    return out
