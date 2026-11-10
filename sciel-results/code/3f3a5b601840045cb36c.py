"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85.0

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Start with K_{7,8} as baseline (56 edges, proven valid)
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    best_edges = base_edges
    best_count = len(base_edges)

    # Try to add extra edges (within A or within B) while maintaining validity
    # Extra edges can only be within A (7 vertices) or within B (8 vertices)
    # Max possible: C(7,2) + C(8,2) = 21 + 28 = 49 extra edges
    # But we need to check validity constraint.
    
    # Let's think about what happens when we add an edge within A, say (a1, a2).
    # For edge (a1, a2):
    #   neighbors of a1 (excl a2) = B ∪ (A\{a1,a2} ∩ adj[a1])
    #   neighbors of a2 (excl a1) = B ∪ (A\{a1,a2} ∩ adj[a2])
    #   For s in neighbors of a1 (excl a2), t in neighbors of a2 (excl a1), s≠t:
    #   We need deg(s) ≠ deg(t).
    #
    # In base K_{7,8}: deg(a) = 8 for all a in A, deg(b) = 7 for all b in B.
    # If we add edges within A, degrees of A vertices increase.
    # If we add edges within B, degrees of B vertices increase.
    #
    # Key insight: if all vertices in A have the same degree, and all in B have the same degree,
    # then for any edge (a1,a2) within A:
    #   s can be in B (deg 7+extra) or in A (deg 8+extra)
    #   t can be in B (deg 7+extra) or in A (deg 8+extra)
    #   If deg_A ≠ deg_B, then s in B and t in A is fine, but s in B and t in B could be bad
    #   if they have the same degree.
    #
    # Actually, if all B vertices have the same degree, then for s,t both in B: deg(s)=deg(t), violation!
    # So we cannot have two vertices s,t both in B that are both neighbors of a1 and a2 (excl each other).
    # In K_{7,8}, every a in A connects to every b in B. So for edge (a1,a2) within A:
    #   neighbors of a1 (excl a2) includes all of B, neighbors of a2 (excl a1) includes all of B.
    #   So s=b1, t=b2 for any distinct b1,b2 in B: deg(b1)=deg(b2)=7 (or whatever), VIOLATION.
    #
    # So we CANNOT add any edge within A if the graph remains complete bipartite!
    # Similarly for B.
    #
    # This means we need a different approach: remove some bipartite edges to break symmetry,
    # then add intra-part edges.
    #
    # Alternative: use a more clever construction.
    #
    # Let's try: make degrees all distinct or nearly distinct.
    # With 15 vertices, max edges with distinct degrees: as analyzed, impossible (sum must be even).
    # Best with one repeat: 14,14,13,12,11,10,9,8,7,6,5,4,3,2,1 -> sum=108, edges=54.
    # But K_{7,8} has 56 edges. So 54 < 56.
    #
    # Can we get more than 56? Max edges is C(15,2)=105.
    # The constraint is quite restrictive. Let me think more carefully.
    #
    # For a graph to be valid: for every edge (u,v), and every s in N(u)\{v}, t in N(v)\{u}, s≠t:
    #   deg(s) ≠ deg(t).
    #
    # This means: for edge (u,v), the set of degrees {deg(s) : s in N(u)\{v}} must be disjoint from
    # {deg(t) : t in N(v)\{u}} \ {deg(s) for s that equals t}... actually more precisely,
    # no degree value can appear in both sets (unless it's the same vertex, but s≠t).
    #
    # Let's try a greedy search: start from K_{7,8}, try adding intra-A or intra-B edges,
    # and if invalid, try removing some bipartite edges to fix it.
    #
    # Actually, let me try a completely different approach: use local search.
    # Start from K_{7,8}, try to add edges one at a time, checking validity.
    # If an edge can't be added, try removing a different edge and adding this one (swap).
    
    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Current edge set
    current = set(base_edges)
    
    # Try to add each possible missing edge
    missing = [e for e in all_possible if e not in current]
    random.shuffle(missing)
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        for e in missing:
            if time.time() > deadline:
                break
            if e in current:
                continue
            trial = current | {e}
            if is_valid(trial, n):
                current = trial
                missing.remove(e)
                added_any = True
                if len(current) > best_count:
                    best_edges = current
                    best_count = len(current)
    
    # Try swap moves: remove one edge, add another
    for _ in range(100):
        if time.time() > deadline:
            break
        current_list = list(current)
        random.shuffle(current_list)
        removed = None
        for rem in current_list:
            if time.time() > deadline:
                break
            trial = current - {rem}
            # Try adding each missing edge
            for add in all_possible:
                if add in trial:
                    continue
                trial2 = trial | {add}
                if len(trial2) <= len(current):
                    continue
                if is_valid(trial2, n):
                    current = trial2
                    removed = rem
                    break
            if removed:
                break
    
    if len(current) > best_count:
        best_edges = current
        best_count = len(current)

    # Convert to result format
    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END
