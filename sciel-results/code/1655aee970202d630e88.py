"""
Pentagon-free graph on 32 vertices (8 disjoint K4 blocks) with a bounded,
time-budgeted local-search attempt to reach n=33-35.

Guaranteed valid fallback: n=32 (8 x K4).
If the search succeeds within the time budget, a larger valid graph is returned.
"""
import time
import random
from itertools import combinations

# Time budget: 150s wall-clock limit; use 130s to be safe
TIME_BUDGET = 130.0

def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains a C5."""
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if not adj[c][d] or not adj[b][d] or adj[a][d]:
                        continue
                    # Need e adjacent to a and d, not to b and c
                    for e in range(d + 1, n):
                        if adj[a][e] and adj[d][e] and not adj[b][e] and not adj[c][e]:
                            return True
                    # Also check e < a (can't since e > d > c > b > a, so e is always > a)
                    # But we need to also check if e could be between a and b, etc.
                    # Actually with ordering a<b<c<d<e, all pairs are determined.
                    # But C5 can have different vertex orderings. Let me be more careful.
    return False


def _has_c5_fast(adj, n):
    """More thorough C5 check: for each 5-tuple of distinct vertices, check all 12 cycle orderings.
    Too slow for large n. Instead, use a smarter approach."""
    # A C5 exists iff there are vertices v0,v1,v2,v3,v4 with edges
    # (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0) and no other constraint (subgraph, not induced)
    # We just need any 5-cycle as a subgraph.
    # Efficient check: for each pair (a,b) with edge, look for paths of length 3 from a to b
    # that don't use edge (a,b) directly, and where the intermediate vertices are distinct.
    # Path: a - x - y - b, where x != y, x != b, y != a, and (a,x),(x,y),(y,b) are edges.
    # Then a-b-x-y-a is a 4-cycle, not 5.
    # For C5: a - x - y - z - b - a, where all distinct and edges (a,x),(x,y),(y,z),(z,b),(b,a).
    # So for edge (a,b), find two vertex-disjoint paths of length 2 from a to b avoiding each other.
    # a-x-y-b and a-z-w-b where {x,y} ∩ {z,w} = ∅, and all of a,b,x,y,z,w distinct.
    
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            # Find all length-2 paths from a to b: a-x-b where x adjacent to both a and b
            # No wait, we need length-3 paths: a-x-y-z-b where a-x, x-y, y-z, z-b are edges
            # and a,b,x,y,z all distinct, and x != z (otherwise a-x-z-b is length 3 with middle vertex z=x, giving 4 vertices a,x,z,b -> 4-cycle)
            # Actually: a - x - y - z - b is 4 edges, 5 vertices. We need a-x, x-y, y-z, z-b, b-a (the last is our fixed edge).
            # So we need: x adjacent to a, y adjacent to x, z adjacent to y and b, all of a,b,x,y,z distinct, x≠z.
            
            # For efficiency, precompute for each vertex its neighbors
            # Let's just do a direct check with some pruning
            
            neighbors_a = [v for v in range(n) if v != b and adj[a][v]]
            neighbors_b = [v for v in range(n) if v != a and adj[b][v]]
            
            for x in neighbors_a:
                for z in neighbors_b:
                    if x == z:
                        continue
                    # Need y adjacent to both x and z
                    for y in range(n):
                        if y == a or y == b or y == x or y == z:
                            continue
                        if adj[x][y] and adj[y][z]:
                            return True
    return False


def _independence_number_upper_bound(adj, n):
    """Upper bound on independence number using greedy coloring of the complement.
    Returns an upper bound (may be loose)."""
    # Complement adjacency
    comp_adj = [[1 - adj[i][j] if i != j else 0 for j in range(n)] for i in range(n)]
    
    # Greedy coloring: color vertices in order, assign smallest available color
    color = [-1] * n
    colors_used = 0
    for v in range(n):
        used = set()
        for u in range(n):
            if comp_adj[v][u] and color[u] != -1:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
        colors_used = max(colors_used, c + 1)
    # The number of colors in a coloring of the complement = size of a partition into independent sets of the complement = cliques in original... 
    # Actually, coloring the complement gives a partition into independent sets of the complement, which are cliques in the original graph.
    # The independence number of the original graph = clique number of the complement.
    # A coloring of the complement gives an upper bound on the clique number of the complement = upper bound on alpha(original).
    # Wait: coloring G' (complement) uses k colors means V(G') partitioned into k independent sets of G'. Independent set in G' = clique in G. So we've partitioned V(G) into k cliques. This doesn't directly bound alpha(G).
    
    # Let me think again. alpha(G) = omega(G'). A proper coloring of G' with k colors means each color class is an independent set in G', i.e., a clique in G. So V is partitioned into k cliques in G. This gives that G is a union of k cliques. The independence number of G is at most k (pick at most one from each clique). So k is an upper bound on alpha(G).
    
    return colors_used


def _greedy_independence_number(adj, n):
    """Greedy heuristic for independence number. Returns a lower bound."""
    # Greedy: repeatedly pick a vertex of minimum degree, remove it and its neighbors
    remaining = set(range(n))
    count = 0
    while remaining:
        # Pick vertex with minimum degree in remaining
        min_v = min(remaining, key=lambda v: sum(adj[v][u] for u in remaining if u != v))
        count += 1
        # Remove min_v and all its neighbors
        to_remove = {min_v}
        for u in remaining:
            if adj[min_v][u]:
                to_remove.add(u)
        remaining -= to_remove
    return count


def _build_adj(edges, n):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj


def _check_valid(adj, n):
    """Check C5-free and alpha < 9."""
    if _has_c5_fast(adj, n):
        return False
    # Check alpha < 9: try to find independent set of size 9
    # Use a simple backtracking or greedy check
    # Greedy upper bound: if coloring of complement gives <= 8, then alpha <= 8 < 9
    ub = _independence_number_upper_bound(adj, n)
    if ub >= 9:
        # Need exact check: try to find independent set of size 9
        # Use greedy to find max independent set
        lb = _greedy_independence_number(adj, n)
        if lb >= 9:
            return False
        # If greedy found < 9 but upper bound >= 9, need more careful check
        # For safety, do a more thorough check
        # Try all subsets? Too expensive for n=35.
        # Use a branch-and-bound for independent set of size 9
        if _exists_independent_set_size(adj, n, 9):
            return False
    return True


def _exists_independent_set_size(adj, n, k):
    """Check if there's an independent set of size k using backtracking with pruning."""
    # Greedy upper bound on remaining independent set
    def _greedy_ub(vertices):
        if not vertices:
            return 0
        remaining = set(vertices)
        count = 0
        while remaining:
            min_v = min(remaining, key=lambda v: sum(adj[v][u] for u in remaining if u != v))
            count += 1
            to_remove = {min_v}
            for u in remaining:
                if adj[min_v][u]:
                    to_remove.add(u)
            remaining -= to_remove
        return count
    
    def _backtrack(available, needed):
        if needed == 0:
            return True
        if len(available) < needed:
            return False
        if _greedy_ub(available) < needed:
            return False
        v = available[0]
        # Branch 1: include v
        new_avail_incl = [u for u in available[1:] if not adj[v][u]]
        if _backtrack(new_avail_incl, needed - 1):
            return True
        # Branch 2: exclude v
        if _backtrack(available[1:], needed):
            return True
        return False
    
    return _backtrack(list(range(n)), k)


def _make_8xk4():
    """Construct 8 disjoint K4 blocks, n=32."""
    num_blocks = 8
    block_size = 4
    n = num_blocks * block_size
    edges = []
    for b in range(num_blocks):
        start = b * block_size
        for i in range(block_size):
            for j in range(i + 1, block_size):
                edges.append([start + i, start + j])
    return n, edges


def _try_search_for_larger(start_n, start_edges, deadline):
    """Try to extend the graph to n+1, n+2, etc. using local search.
    Returns (n, edges) if a valid larger graph is found, else None."""
    n, edges = start_n, start_edges
    adj = _build_adj(edges, n)
    
    # Verify starting graph is valid
    if not _check_valid(adj, n):
        return None
    
    current_n = n
    current_edges = list(edges)
    current_adj = adj
    
    for target_n in range(current_n + 1, min(36, current_n + 4)):
        if time.time() > deadline - 5.0:
            break
        # Try to add a vertex
        success = False
        for _attempt in range(200):
            if time.time() > deadline - 5.0:
                break
            new_n = target_n
            # Start from current graph, add one vertex with some edges
            new_adj = [[False] * new_n for _ in range(new_n)]
            for i in range(current_n):
                for j in range(i + 1, new_n):
                    if i < current_n and j < current_n:
                        new_adj[i][j] = current_adj[i][j]
                        new_adj[j][i] = current_adj[i][j]
            
            # Decide edges from new vertex (index current_n) to existing vertices
            # Try random subsets
            num_existing = current_n
            candidate_edges_from_new = []
            for v in range(num_existing):
                if random.random() < 0.4:  # 40% chance of edge
                    new_adj[current_n][v] = True
                    new_adj[v][current_n] = True
                    candidate_edges_from_new.append(v)
            
            # Check C5-free
            if _has_c5_fast(new_adj, new_n):
                continue
            
            # Check alpha < 9
            ub = _independence_number_upper_bound(new_adj, new_n)
            if ub >= 9:
                if _exists_independent_set_size(new_adj, new_n, 9):
                    continue
            
            # Valid!
            new_edges = list(current_edges)
            for v in candidate_edges_from_new:
                new_edges.append([current_n, v])
            current_n = new_n
            current_edges = new_edges
            current_adj = new_adj
            success = True
            break
        
        if not success:
            break
    
    if current_n > start_n:
        return current_n, current_edges
    return None


def construct():
    start_time = time.time()
    deadline = start_time + TIME_BUDGET
    
    # Guaranteed valid fallback
    fallback_n, fallback_edges = _make_8xk4()
    
    # Check if we have time to search
    if time.time() > deadline - 30.0:
        return {'n': fallback_n, 'edges': fallback_edges}
    
    # Try to find a larger graph
    try:
        result = _try_search_for_larger(fallback_n, fallback_edges, deadline)
        if result is not None:
            n, edges = result
            # Final verification
            adj = _build_adj(edges, n)
            if _check_valid(adj, n):
                return {'n': n, 'edges': edges}
    except Exception:
        pass
    
    return {'n': fallback_n, 'edges': fallback_edges}
