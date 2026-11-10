"""Rewritten program: clique partition of split graphs via greedy set-cover on independent sets."""
import time
import random
from itertools import combinations

# Keep the exact generator under the name 'instance'
def instance(n, k, p, seed):
    rng = random.Random(seed)
    edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
    edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
    return sorted(edges)


def _solve_split(n, k, p, seed, deadline):
    """
    Split graph: clique C = {0..k-1}, independent set I = {k..n-1}.
    Cross edges: for each (u in C, v in I) present with prob p.

    Edges to cover:
      - All clique edges within C (k choose 2)
      - All cross edges present

    A clique in a split graph is either:
      - A subset of C (any size >=2)
      - A subset of C plus some vertices from I that are all adjacent to every vertex in that subset.

    Strategy:
      For each subset S of C with |S|>=2, the vertices in I that can be added are those
      adjacent to every vertex in S. Let N(S) = {v in I : for all u in S, (u,v) in E}.
      Then S ∪ N(S) is a clique. Also, S alone is a clique (covers edges within S).

    However, we need to cover all edges with minimum number of cliques.

    Key insight: The clique edges within C must be covered. If we use cliques that include
    vertices from I, they still cover the same clique edges as their C-subset. So we can
    think of it as: choose a collection of subsets of C (each of size >=2) such that every
    edge in C is covered, and for each chosen subset S, we may optionally include N(S) vertices
    from I to also cover cross edges incident to those I-vertices.

    Actually, a cleaner formulation:
      - Partition the edges into cliques.
      - Each clique is determined by its C-part S (|S|>=2) and its I-part T ⊆ N(S).
      - The clique covers: all edges within S, and all edges (u,v) for u in S, v in T.

    We want to minimize the number of cliques.

    Greedy approach:
      1. Build the set of all cross edges.
      2. For each vertex v in I, let A(v) = {u in C : (u,v) in E}. Then v is in N(S) iff S ⊆ A(v).
      3. We need to cover all cross edges. A clique with C-part S covers cross edges (u,v) where u in S and v in N(S).
      
    Alternative greedy:
      - First, handle the clique C edges. We need to cover all (u,w) for u<w in C.
      - A clique with C-part S covers all edges within S.
      - So we need a family of subsets of C (each size >=2) whose union of "edge sets" covers all C-edges.
      - Additionally, each such subset S, when used as a clique, can absorb up to N(S) vertices from I,
        covering their cross edges to S.
      
    This is complex. Let's use a simpler greedy:
      1. Start with all edges (clique edges + cross edges) uncovered.
      2. Repeatedly pick the clique that covers the most uncovered edges.
      3. Mark those edges as covered.
      4. Continue until all edges are covered.

    But the space of possible cliques is large. However, in a split graph, every maximal clique is of the form S ∪ N(S) for some S ⊆ C. And S must be such that S is a clique (which it always is since C is a clique) and |S| >= 2.

    We can enumerate all maximal cliques: for each subset S of C with |S|>=2, compute N(S). But 2^k is too large for k=30.

    Better: Note that N(S) = intersection of A(v) for... no, N(S) = {v in I : S ⊆ A(v)}.

    Let's think differently. The cross edges form a bipartite graph between C and I. We need to cover these edges with "stars" centered at subsets of C, but the star center S covers edges from S to N(S).

    Actually, a clique with C-part S covers exactly the cross edges between S and N(S). So if we choose a collection of pairs (S_i, T_i) where T_i ⊆ N(S_i), the cross edges covered are ∪_i (S_i × T_i). And the clique edges covered are ∪_i (edges within S_i).

    To minimize the number of cliques, we want each clique to cover as many edges as possible.

    Greedy:
      - Maintain uncovered_edges set.
      - For each candidate clique (defined by S ⊆ C, |S|>=2, and T = N(S) or a subset), compute how many uncovered edges it covers.
      - Pick the best, add it, update uncovered_edges.

    Candidate cliques: We can restrict to maximal cliques S ∪ N(S) for S being a "maximal" subset in some sense. But to keep it tractable:

    Observation: If S1 ⊆ S2, then N(S2) ⊆ N(S1). So larger S gives smaller N(S). There's a trade-off.

    For k up to 30, we can't enumerate all 2^30 subsets. But we can use a heuristic:
      - Consider only S of size 2, 3, or "full" C.
      - Or use a greedy that builds S.

    Let me try a different angle: Since C is a clique, we can cover all C-edges with just one clique: C itself. But then N(C) = {v in I : v adjacent to all of C}. If N(C) is large, using C ∪ N(C) covers many cross edges. The remaining cross edges are those from I \ N(C) to C.

    Then for the remaining I vertices, we need to cover their cross edges. Each remaining vertex v has A(v) ⊂ C (not all of C). We can group them.

    Greedy algorithm:
      1. Compute all cross edges.
      2. Let remaining_I = set of all I vertices.
      3. Let remaining_edges = set of all edges (clique + cross).
      4. While remaining_edges is not empty:
           a. For each candidate S (subset of C, |S|>=2), compute T = N(S) ∩ remaining_I (I vertices still needing coverage, but actually we need to cover edges, not vertices).
           
    This is getting complicated. Let me just implement a greedy that considers candidate cliques defined by S being a pair, a triple, or the full set, and picks the one covering the most uncovered edges.

    Actually, for the given sizes (n=30,60,90; k=10,20,30), k is small enough that we might be able to do something smarter.

    Let me try: 
      - The clique edges within C: we can cover them all with one clique C. But maybe it's better to split C into smaller cliques to absorb more I vertices.
      
    New plan:
      1. Build adjacency: for each v in I, A(v) ⊆ C.
      2. We need to cover all cross edges (u,v) and all C-edges (u,w).
      3. A clique is (S, T) where S ⊆ C, |S|>=2, T ⊆ ∩_{u∈S} A(u) (i.e., T ⊆ N(S)).
         It covers: all (u,w) with u,w∈S, and all (u,v) with u∈S, v∈T.
      4. Greedy: start with all edges uncovered. Repeatedly pick (S,T) maximizing uncovered edges covered.

    To limit candidates: for each S, the best T is N(S) (all possible I vertices). So candidate cliques are (S, N(S)) for various S.
    
    Candidate S's: all pairs in C, all triples in C, and C itself. For k=30, pairs: 435, triples: 4060, total ~4500 candidates. That's manageable.

    Let's implement this greedy.

    Steps:
      a. Precompute for each candidate S: N(S) = {v in I : S ⊆ A(v)}.
      b. Precompute the set of edges covered by (S, N(S)): C-edges within S + cross edges (u,v) for u in S, v in N(S).
      c. Greedy: 
           uncovered = set(all_edges)
           cliques = []
           while uncovered:
               best = None
               best_count = 0
               for S in candidates:
                   covered = edges_of(S) & uncovered
                   if len(covered) > best_count:
                       best_count = len(covered)
                       best = S
               if best_count == 0: break  # shouldn't happen
               cliques.append((best, N(best)))
               uncovered -= edges_of(best)
      d. Convert to list of vertex lists.

    But wait: the clique (S, N(S)) covers C-edges within S and cross edges from S to N(S). But what about cross edges from S to I \ N(S)? Those are not covered by this clique. They need to be covered by other cliques.

    Also, C-edges not in any S are not covered. So we need ∪_i (edges within S_i) = all C-edges.

    The greedy will handle this since it picks cliques that cover uncovered edges.

    Let me code this up.
"""


def _solve_split(n, k, p, seed, deadline):
    # Generate edges
    edges = instance(n, k, p, seed)
    
    # Build adjacency sets
    # C = {0..k-1}, I = {k..n-1}
    cross_edges = set()
    clique_edges = set()
    for (u, v) in edges:
        if u < k and v < k:
            clique_edges.add((u, v))
        elif u < k and v >= k:
            cross_edges.add((u, v))
        elif u >= k and v < k:
            cross_edges.add((v, u))
        # No edges within I
    
    all_edges = clique_edges | cross_edges
    
    # For each v in I, compute A(v) = {u in C : (u,v) in cross_edges}
    A = {}
    for v in range(k, n):
        A[v] = set()
    for (u, v) in cross_edges:
        # u in C, v in I
        A[v].add(u)
    
    # Candidate S's: all pairs, all triples, and C itself
    candidates = set()
    # Pairs
    for i in range(k):
        for j in range(i+1, k):
            candidates.add(frozenset([i, j]))
    # Triples
    for i in range(k):
        for j in range(i+1, k):
            for l in range(j+1, k):
                candidates.add(frozenset([i, j, l]))
    # Full C
    candidates.add(frozenset(range(k)))
    
    # For each candidate S, compute N(S) and the set of edges it covers
    # N(S) = {v in I : S ⊆ A(v)}
    # Edges covered: C-edges within S + cross edges (u,v) for u in S, v in N(S)
    
    S_list = list(candidates)
    N_map = {}
    edges_map = {}
    
    for S in S_list:
        S_set = set(S)
        # N(S)
        N_S = set()
        for v in range(k, n):
            if S_set.issubset(A[v]):
                N_S.add(v)
        N_map[S] = N_S
        
        # Edges covered
        cov = set()
        # C-edges within S
        for a, b in combinations(sorted(S), 2):
            cov.add((a, b))
        # Cross edges from S to N_S
        for u in S:
            for v in N_S:
                cov.add((u, v))
        edges_map[S] = cov
    
    # Greedy set cover
    uncovered = set(all_edges)
    cliques = []
    
    while uncovered:
        best_S = None
        best_count = 0
        for S in S_list:
            cov = edges_map[S] & uncovered
            cnt = len(cov)
            if cnt > best_count:
                best_count = cnt
                best_S = S
        if best_count == 0:
            # Shouldn't happen, but break to avoid infinite loop
            break
        cliques.append((best_S, N_map[best_S]))
        uncovered -= edges_map[best_S]
    
    # Convert to list of vertex lists
    result = []
    for (S, N_S) in cliques:
        vertices = list(S) + list(N_S)
        # Ensure at least 2 vertices (S has at least 2)
        result.append(sorted(vertices))
    
    return result


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    result = []
    for spec in specs:
        n, k, p, seed = spec
        if time.time() > deadline:
            # Fallback: return trivial valid solution (each edge as its own clique)
            edges = instance(n, k, p, seed)
            result.append([list(e) for e in edges])
            continue
        sol = _solve_split(n, k, p, seed, deadline)
        result.append(sol)
    
    return result
