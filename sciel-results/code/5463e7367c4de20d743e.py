import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        C_edges = set()
        cross_edges = set()

        for u, v in edges:
            if u < k and v < k:
                C_edges.add((u, v))
            else:
                cross_edges.add((u, v))

        # Build bipartite graph: side A = clique vertices [0,k), side B = independent vertices [k,n)
        # adjA[a] = set of b in B adjacent to a
        # adjB[b] = set of a in A adjacent to b
        adjA = defaultdict(set)
        adjB = defaultdict(set)
        for (u, v) in cross_edges:
            if u < k and v >= k:
                a, b = u, v
            elif v < k and u >= k:
                a, b = v, u
            else:
                continue  # shouldn't happen for cross
            adjA[a].add(b)
            adjB[b].add(a)

        # Edge-color the bipartite graph via greedy matching decomposition.
        # We repeatedly extract a maximal matching from the remaining edges,
        # assign it a color, and remove those edges.
        # Number of colors = number of matchings needed = at most max degree.
        #
        # For efficiency, use a simple greedy: for each color, find a maximal
        # matching by iterating over A vertices and greedily matching to available B.
        remaining = set(cross_edges)
        color_classes = []  # list of sets of (a, b) pairs

        while remaining:
            # Greedy maximal matching
            matched = set()
            used_B = set()
            for a in sorted(adjA.keys()):
                if a in used_B:  # a already matched
                    continue
                # Try to find an unmatched b
                found = False
                for b in sorted(adjA[a]):
                    if (a, b) in remaining and b not in used_B:
                        matched.add((a, b))
                        used_B.add(b)
                        found = True
                        break
            if not matched:
                # Fallback: take one edge
                e = next(iter(remaining))
                matched = {e}
            color_classes.append(matched)
            remaining -= matched

        # Now build cliques:
        # 1. The clique-part: all edges in C_edges. Since C is a clique on [0,k),
        #    the entire set [0,k) is a clique. But we need to cover C_edges.
        #    If C_edges is non-empty, we can use one clique {0,...,k-1} (all k vertices)
        #    to cover all C_edges. But wait - if k==1, no C_edges exist.
        #    Actually, if there are C_edges, the clique {0,...,k-1} covers them all.
        #    But we must ensure this clique has >= 2 vertices.
        #
        # 2. For each color class (matching), assign to a clique vertex a.
        #    The clique is {a} ∪ {b : (a,b) in matching}.
        #    Since the matching is a set of vertex-disjoint edges, all b's are distinct
        #    and all adjacent to a (since they're in adjA[a]).
        #    So {a} ∪ {b's} is a valid clique (a connects to all b's, and b's are
        #    independent among themselves since they're in the independent set).
        #
        #    Wait - but different color classes may use the same 'a' vertex.
        #    That's fine; each clique is separate.
        #
        #    However, we need each cross-edge to be in EXACTLY ONE clique.
        #    Each cross-edge (a,b) appears in exactly one matching (color class),
        #    and that matching is assigned to vertex a (or we can assign to any
        #    endpoint). Let's assign each matching to its "first" vertex a.
        #
        #    Actually, a matching can contain edges with different a's.
        #    E.g., matching = {(0,30), (1,31)}. This is NOT a star, so we can't
        #    form a single clique {a} ∪ {b's} because 0 and 1 are different.
        #
        #    CORRECTION: A matching in the bipartite graph is a set of edges no two
        #    sharing a vertex. So (0,30) and (1,31) share no vertex - that's a valid
        #    matching. But {0, 30, 1, 31} is NOT a clique because 0 and 1 are both
        #    in the clique part (they ARE adjacent), and 30 and 31 are in the
        #    independent set (they are NOT adjacent). So {0,1,30,31} is not a clique
        #    unless 30-31 is an edge, which it isn't (independent set).
        #
        #    So a matching does NOT directly form a clique unless it's a star!
        #
        #    REVISED APPROACH: We need each color class to be a STAR (all edges
        #    share one center). So we should decompose the bipartite graph into
        #    edge-disjoint stars (each star centered at one vertex).
        #
        #    Alternative: decompose into stars centered at A-side vertices.
        #    For each a in A, the edges incident to a form a star. We can assign
        #    each such star to clique {a} ∪ N_B(a). But that's what the original
        #    program does (one clique per independent vertex, which is equivalent
        #    to one star per B-vertex... no, per A-vertex).
        #
        #    Let me reconsider. The key insight from the directive:
        #    "assign each color class to a distinct independent vertex i to form a
        #    clique {i} ∪ matched_C_neighbors"
        #
        #    So each color class should be a matching, and we assign it to ONE
        #    vertex i (in the independent set, i >= k). The clique is {i} ∪
        #    {a : (a,i) in matching}. For this to be a clique, all a's in the
        #    matching must be adjacent to i (true by construction) and the a's
        #    must be mutually adjacent (true since they're all in the clique part).
        #    And i is adjacent to all a's. So {i} ∪ {a's} IS a valid clique!
        #
        #    So: decompose cross-edges into matchings. Each matching M is a set of
        #    edges with no shared vertices. Pick any vertex i (in B) that is an
        #    endpoint of some edge in M. Then {i} ∪ {the other endpoints} is a clique.
        #    Wait, but M might have edges (a1,i), (a2,j), (a3,l) where i,j,l are
        #    different B-vertices. Then we can't use a single i.
        #
        #    CORRECT INSIGHT: A matching in the bipartite graph has edges with
        #    distinct endpoints on BOTH sides. So if M = {(a1,b1), (a2,b2), ...},
        #    all b's are distinct. We CANNOT form {i} ∪ {a's} for a single i.
        #
        #    UNLESS we restrict: each "color class" is a set of edges all sharing
        #    the SAME b-vertex (a star centered at b). That's not a matching in
        #    general (it's a star).
        #
        #    OK let me re-read the directive more carefully:
        #    "each color class corresponds to a matching (a set of vertex-disjoint
        #    cross-edges), and assign each color class to a distinct independent
        #    vertex i to form a clique {i} ∪ matched_C_neighbors"
        #
        #    I think the idea is: if the matching is {(a1,i), (a2,i), ...} - i.e.,
        #    all edges share vertex i - then it's a star centered at i, and
        #    {i} ∪ {a1,a2,...} is a clique. But that's not a "matching" in the
        #    graph theory sense (matching = no shared vertices).
        #
        #    I think the directive means: decompose the bipartite graph into
        #    edge-disjoint STARS (each star = all edges from one vertex). The
        #    number of stars needed = max degree (by a simple argument: each star
        #    covers all edges of one vertex, and we need at least Δ stars for
        #    the vertex of max degree, and Δ stars suffice by a greedy argument
        #    for bipartite graphs... actually this is just edge-coloring where
        #    each color class is a matching, and we then group edges by their
        #    B-endpoint).
        #
        #    SIMPLEST CORRECT APPROACH:
        #    For each independent vertex b in [k,n), the star centered at b
        #    (all edges (a,b)) forms a clique {b} ∪ {a : (a,b) in cross_edges}.
        #    This is exactly what the original program does! One clique per
        #    independent vertex that has any cross-edges.
        #
        #    To do BETTER, we need to MERGE stars. Two stars centered at b1 and
        #    b2 can be merged into one clique IF... {b1, b2, a's} is a clique.
        #    But b1 and b2 are both in the independent set, so they're NOT
        #    adjacent. Thus we CANNOT merge two independent-set vertices into
        #    one clique.
        #
        #    So the minimum number of cross-cliques is at least the number of
        #    independent vertices with degree > 0, and at most... the same.
        #    Wait, that means the original program is already optimal for the
        #    cross-edges? Each cross-edge (a,b) must be in a clique containing b
        #    (since b must be in the clique for edge (a,b) to be covered, and the
        #    only way to have b in a clique with a is {b} ∪ {subset of N(b)}).
        #    And since b's are mutually non-adjacent, no clique can contain two
        #    different b's. So each b with degree > 0 needs its own clique.
        #
        #    Hmm, but that contradicts the inspiration that got 1.3357.
        #    Let me re-examine: maybe a cross-edge (a,b) can be covered by a
        #    clique {a, b, c} where c is another clique-vertex? Yes! {a,b,c}
        #    is a clique if a-c and b-c are edges. b-c is a cross-edge (c in
        #    clique part, b in independent part). So {a,b,c} is a valid clique
        #    that covers edges (a,b), (a,c), and (b,c).
        #
        #    So a single clique can cover cross-edges from MULTIPLE independent
        #    vertices, as long as all the independent vertices in the clique are
        #    adjacent to all the clique-part vertices in the clique, AND the
        #    independent vertices... wait, they're in the independent set, so
        #    they're NOT adjacent to each other. So a clique can contain at most
        #    ONE independent vertex.
        #
        #    CONFIRMED: Each clique contains at most one vertex from the
        #    independent set. Therefore, each cross-edge (a,b) must be in a
        #    clique whose unique independent vertex is b. So each independent
        #    vertex b with deg(b)>0 requires at least one clique. The original
        #    program uses exactly one such clique per b, which is OPTIMAL.
        #
        #    So the only way to improve is to reduce the number of cliques in
        #    the clique part. The original uses 1 clique for all C_edges.
        #    That's already minimal (1 if C_edges non-empty, 0 otherwise).
        #
        #    Wait, then the original should be near-optimal. Let me check the
        #    scores: current = 1.0, inspiration 1 = 1.3357. The score is
        #    seed/value, so higher score = lower value = fewer cliques.
        #    Inspiration 1 has FEWER cliques. How?
        #
        #    Let me recount. For n=30, k=10, p=0.7:
        #    - C_edges: C(10,2) = 45 edges
        #    - Cross edges: ~0.7 * 10 * 20 = 140 edges
        #    - Original: 1 (clique part) + 20 (one per independent vertex with
        #      deg>0, likely all 20) = 21 cliques
        #    - But wait, some independent vertices might have degree 0.
        #      With p=0.7, P(deg(b)=0) = (1-0.7)^10 = 0.3^10 ≈ 0.00006.
        #      So essentially all 20 have deg>0. Total ≈ 21.
        #
        #    For n=60, k=20: 1 + 40 = 41
        #    For n=90, k=30: 1 + 60 = 61
        #    Total ≈ 123.
        #
        #    Seed value = 123 (approx). Score 1.0 means value = seed.
        #    Inspiration 1 score 1.3357 means value ≈ 123/1.3357 ≈ 92.
        #    So inspiration 1 uses ~31 fewer cliques. How?
        #
        #    AH WAIT. I misread the problem. Let me re-read:
        #    "every edge must lie in exactly one listed clique"
        #
        #    The cross-edge (a,b) where a is in clique part and b is in
        #    independent set. A clique containing this edge must contain both a
        #    and b. Since b is in the independent set, the clique can contain
        #    at most one independent vertex (b). So the clique is {b} ∪ S where
        #    S ⊆ N(b) ∩ [0,k). This covers edges (b,s) for s in S, AND edges
        #    (s1,s2) for s1,s2 in S (since S is in the clique part, all
        #    mutually adjacent).
        #
        #    So one clique {b} ∪ S covers:
        #    - Cross edges: (b,s) for s in S
        #    - Clique-part edges: (s1,s2) for s1,s2 in S
        #
        #    The key: clique-part edges can be covered by MULTIPLE cliques (one
        #    per independent vertex), not just the one big clique. So we DON'T
        #    need a separate clique for C_edges if the cross-cliques already
        #    cover all C_edges!
        #
        #    The original program adds a big clique {0,...,k-1} for remaining
        #    C_edges not covered by cross-cliques. If we make the cross-cliques
        #    large enough to cover all C_edges, we save that 1 clique.
        #
        #    But that's only 1 clique saved per instance, ~3 total. Not enough
        #    for a 1.33x improvement.
        #
        #    Let me reconsider. Maybe the improvement is that some independent
        #    vertices can be grouped... no, they can't.
        #
        #    Actually wait - I think I need to re-examine whether a clique can
        #    contain TWO independent vertices. The independent set means no edges
        #    BETWEEN independent vertices. So if b1, b2 are both in the
        #    independent set, (b1,b2) is NOT an edge. A clique requires ALL
        #    pairs to be edges. So {b1, b2, ...} cannot be a clique. Confirmed:
        #    at most one independent vertex per clique.
        #
        #    So the minimum number of cliques covering all cross-edges is at
        #    least the number of independent vertices with degree > 0. And the
        #    original achieves this. So the only savings are from the C_edges
        #    clique (at most 1 per instance).
        #
        #    Unless... I'm wrong about the graph structure. Let me re-read the
        #    instance generator:
        #    "a clique on the first k vertices, an independent set on the rest
        #    and each cross pair joined with probability p"
        #
        #    Yes, my understanding is correct.
        #
        #    So the theoretical minimum is:
        #    - For each instance: (number of independent vertices with deg > 0)
        #      + (0 or 1 for remaining C_edges)
        #    - Original achieves this.
        #
        #    The inspiration 1 must be doing something I'm missing. Let me just
        #    implement the "greedy max-clique extraction" approach and see.
        #
        #    Actually, you know what, let me just implement a clean version that:
        #    1. For each independent vertex b with deg>0, creates clique {b} ∪ N(b)
        #    2. Checks if all C_edges are covered; if not, adds the big clique
        #    3. Returns the result
        #
        #    This should match or beat the original. Let me also try to be
        #    smarter: maybe I can skip the big clique if the union of all
        #    N(b) covers all of [0,k) (which it does with p=0.7 and k=10+).

        # For each independent vertex b, form clique {b} ∪ N(b)
        result_cliques = []

        # Track which C_edges are covered
        covered_C_edges = set()

        for b in range(k, n):
            neighbors = adjB.get(b, set())
            if not neighbors:
                continue
            clique = [b] + sorted(neighbors)
            result_cliques.append(clique)
            # Mark C_edges covered by this clique
            neighbors_list = sorted(neighbors)
            for idx in range(len(neighbors_list)):
                for jdx in range(idx + 1, len(neighbors_list)):
                    c1, c2 = neighbors_list[idx], neighbors_list[jdx]
                    if c1 < c2:
                        covered_C_edges.add((c1, c2))
                    else:
                        covered_C_edges.add((c2, c1))

        # Check if any C_edges remain uncovered
        remaining_C = C_edges - covered_C_edges
        if remaining_C:
            v_set = set()
            for u, v in remaining_C:
                v_set.add(u)
                v_set.add(v)
            if v_set:
                result_cliques.append(sorted(v_set))

        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)

        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)

    return results
