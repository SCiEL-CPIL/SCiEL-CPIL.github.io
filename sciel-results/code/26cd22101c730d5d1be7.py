import time
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))
        
        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs
        
        # Group I vertices by their exact neighborhood in C
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)
        
        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))
        
        best_partition = None
        best_count = float('inf')
        
        # We will run a few variations of the heuristic to find the best partition.
        # The key improvement here is a more robust clique-finding heuristic for the C-C edges
        # and a better strategy for packing I vertices with C vertices.
        
        # Pre-compute a "maximal clique" finder for the C subgraph.
        # Since C is a clique in the original split graph definition? No, C is a clique in the
        # "clique partition" sense? Wait, the task says "a clique on the first k vertices".
        # So G[C] is a complete graph.
        # Therefore, ANY subset of C is a clique.
        # The constraint is that each EDGE must be in EXACTLY ONE clique.
        # So we are partitioning the set of edges of G into cliques.
        
        # Since G[C] is a complete graph, the edges among C form a complete graph K_k.
        # The edges between I and C are a bipartite graph.
        # There are no edges within I (independent set).
        
        # So the problem decomposes?
        # 1. Partition edges of K_k into cliques.
        # 2. Partition edges of the bipartite graph (I, C) into cliques.
        # BUT, a clique can contain vertices from both I and C.
        # If a clique contains vertices from I and C, it covers some bipartite edges.
        # Can a single clique cover edges from both the "C-C" part and the "I-C" part?
        # Yes. A clique {c1, c2, i1} covers edge (c1,c2) [C-C] and edges (c1,i1), (c2,i1) [I-C].
        # So we cannot simply solve them separately.
        
        # However, note that I vertices are independent. So a clique can contain AT MOST ONE I vertex?
        # No! Wait. "Independent set on the rest". So no edges between I vertices.
        # Therefore, a clique cannot contain two I vertices, because they would not be connected.
        # CORRECTION: A clique is a set of vertices where every pair is connected by an edge.
        # If i1, i2 in I, there is no edge (i1, i2). So {i1, i2} is not a clique.
        # Thus, every clique contains at most one vertex from I.
        
        # This is a HUGE simplification!
        # Each clique is either:
        # 1. A subset of C (covers only C-C edges).
        # 2. A subset of C plus exactly one vertex from I (covers C-C edges among the C subset,
        #    and C-I edges from that specific I vertex to the C subset).
        
        # Let's verify:
        # If a clique has 2 I vertices, they must be connected. But I is independent. Contradiction.
        # So max 1 I vertex per clique.
        
        # Strategy:
        # For each I vertex v, let S_v be its neighborhood in C.
        # We must cover all edges (v, u) for u in S_v.
        # We can group these edges into cliques of the form {v} ∪ K, where K ⊆ S_v is a clique in G[C] (which is always true since G[C] is complete).
        # However, the edges (u, w) for u, w in K are also covered by this clique.
        # So we are "spending" C-C edges to help cover I-C edges.
        
        # We want to minimize the total number of cliques.
        # Let's think about the cost.
        # If we use a clique {v} ∪ K to cover I-C edges, it uses |K| vertices from C.
        # It covers |K| I-C edges (from v to each u in K).
        # It also covers |K|*(|K|-1)/2 C-C edges.
        
        # Alternatively, we could cover I-C edges with cliques of size 2: {v, u}.
        # This uses 1 C-C edge? No, it uses 0 C-C edges (since |K|=1, 1*0/2 = 0).
        # It covers 1 I-C edge.
        # Cost: 1 clique per edge.
        
        # If we group multiple C vertices into K, we cover multiple I-C edges in 1 clique.
        # But we "consume" the C-C edges among K.
        
        # The C-C edges must be covered anyway.
        # So the question is: does grouping I-C edges into larger cliques save the total count?
        # Let's compare:
        # Option A: Cover I-C edges individually.
        #   Number of cliques for I-C: sum(|S_v|) for all v.
        #   C-C edges covered: 0.
        #   Remaining C-C edges: All E_C.
        #   Cliques for C-C: Partition E_C into cliques.
        
        # Option B: Group I-C edges.
        #   Suppose we form a clique {v, c1, c2}.
        #   Covers I-C: (v,c1), (v,c2). (2 edges, 1 clique).
        #   Covers C-C: (c1,c2). (1 edge, 0 extra cliques).
        #   Compared to Option A:
        #     Option A would use {v,c1} and {v,c2} (2 cliques) and {c1,c2} (1 clique) -> 3 cliques.
        #     Option B uses {v,c1,c2} (1 clique) -> 1 clique.
        #     Saving: 2 cliques!
        
        # So it is ALWAYS beneficial to include as many C vertices as possible in a clique with an I vertex,
        # PROVIDED that the C-C edges among those C vertices are not "wasted" (i.e., they would have been covered
        # by other cliques anyway, and we are just merging them).
        
        # Actually, since we must partition ALL edges, any C-C edge is covered exactly once.
        # If we include c1, c2 in a clique with v, we cover (c1,c2) there.
        # If we didn't, we'd cover (c1,c2) in a C-only clique.
        # The "cost" of including c1, c2 in the v-clique is that we can't use them in other v'-cliques?
        # No, vertices can be in multiple cliques. Only EDGES are partitioned.
        # So c1 can be in {v, c1, c2} AND {v', c1, c3}.
        # The edge (c1, c2) is covered in the first. The edge (c1, c3) is covered in the second.
        # This is fine.
        
        # So the strategy is:
        # For each I vertex v, we want to cover its incident edges to C.
        # We can do this by partitioning S_v into subsets K_1, ..., K_m.
        # Each subset K_j forms a clique {v} ∪ K_j.
        # This covers all I-C edges from v.
        # It also covers the C-C edges within each K_j.
        
        # The remaining C-C edges (those not covered by any I-clique) must be covered by C-only cliques.
        
        # To minimize total cliques:
        # 1. Minimize the number of I-cliques: sum_v m_v.
        #    To minimize m_v, we should make K_j as large as possible.
        #    Since any subset of C is a clique, we can take K_1 = S_v.
        #    Then m_v = 1 for all v.
        #    This uses 1 clique per I vertex (that has edges).
        #    This covers ALL C-C edges within S_v.
        
        # 2. The remaining C-C edges are those (u, w) such that there is NO I vertex v
        #    with u, w ∈ S_v.
        #    Wait, if we take K_1 = S_v, we cover ALL edges within S_v.
        #    So the remaining edges are those (u, w) that are NOT in S_v for ANY v.
        #    i.e., there is no I vertex connected to both u and w.
        
        # Is it always optimal to take K_1 = S_v?
        # Let's check the cost.
        # If we take K_1 = S_v, we use 1 clique for v.
        # We cover |S_v| I-C edges.
        # We cover |S_v|*(|S_v|-1)/2 C-C edges.
        
        # If we split S_v into two sets A, B.
        # We use 2 cliques: {v}∪A, {v}∪B.
        # We cover |A| + |B| I-C edges.
        # We cover |A|(|A|-1)/2 + |B|(|B|-1)/2 C-C edges.
        # The edges between A and B are NOT covered by these cliques.
        # They must be covered by C-only cliques.
        # Number of C-only cliques needed to cover edges between A and B?
        # At least 1 if there are edges.
        
        # Compare:
        # Option 1 (Merge): 1 clique. Covers A-B edges.
        # Option 2 (Split): 2 cliques + (cliques for A-B edges).
        # Clearly Option 1 is better or equal in terms of clique count for the I part.
        # And it covers MORE C-C edges, leaving FEWER for C-only cliques.
        
        # So, intuitively, for each I vertex v, we should form ONE clique {v} ∪ S_v.
        # This covers all I-C edges from v in 1 clique.
        # And it covers as many C-C edges as possible.
        
        # Let's verify this logic.
        # Total cliques = (number of I vertices with non-empty S_v) + (cliques to cover remaining C-C edges).
        
        # Let's test this on a small example.
        # C = {1, 2, 3}. I = {4}.
        # Edges: (1,2), (1,3), (2,3) in C.
        # I-C edges: (4,1), (4,2). (4,3) missing.
        # S_4 = {1, 2}.
        # Option 1: Clique {4, 1, 2}.
        #   Covers I-C: (4,1), (4,2).
        #   Covers C-C: (1,2).
        #   Remaining C-C: (1,3), (2,3).
        #   Cover (1,3) and (2,3).
        #   Can we cover them in 1 clique? {1, 2, 3} is a clique.
        #   But (1,2) is already used.
        #   We need a partition of the remaining edges {(1,3), (2,3)}.
        #   These two edges share vertex 3? No, (1,3) and (2,3) share 3.
        #   Are they in a clique? {1, 2, 3} is a clique, but we can't reuse edge (1,2).
        #   We can use clique {1, 3} for (1,3) and {2, 3} for (2,3).
        #   Total cliques: {4,1,2}, {1,3}, {2,3}. Count = 3.
        
        # Option 2: Split S_4 into {1} and {2}.
        #   Cliques: {4,1}, {4,2}.
        #   Covers I-C: (4,1), (4,2).
        #   Covers C-C: None.
        #   Remaining C-C: (1,2), (1,3), (2,3).
        #   Cover these with C-only cliques.
        #   {1,2,3} covers all 3 edges.
        #   Total cliques: {4,1}, {4,2}, {1,2,3}. Count = 3.
        
        # Tie.
        
        # What if S_v is larger?
        # C = {1,2,3,4}. I = {5}.
        # S_5 = {1,2,3,4}.
        # Option 1: {5,1,2,3,4}. 1 clique.
        #   Covers all I-C.
        #   Covers all C-C.
        #   Total: 1 clique.
        
        # Option 2: Split.
        #   More cliques.
        
        # So Option 1 is strictly better here.
        
        # Why did the tie happen in the first example?
        # Because the remaining edges (1,3) and (2,3) could not be merged into the main clique
        # because 3 was not in S_4.
        # But in Option 2, the C-only clique {1,2,3} was very efficient.
        
        # Is there a case where Splitting is better?
        # Suppose S_v = {1, 2}.
        # Remaining C-C edges: (1,3), (2,3), (1,2) is covered by Option 1.
        # In Option 1, we needed 2 cliques for remaining edges.
        # In Option 2, we needed 1 clique for remaining edges.
        # But Option 2 used 2 cliques for I, Option 1 used 1.
        # Net: Option 1: 1 + 2 = 3. Option 2: 2 + 1 = 3.
        
        # What if we have another I vertex w?
        # S_w = {3, 4}.
        # Option 1 for v: {v, 1, 2}. Covers (1,2).
        # Option 1 for w: {w, 3, 4}. Covers (3,4).
        # Remaining C-C: (1,3), (1,4), (2,3), (2,4).
        # These form a complete bipartite graph K_{2,2} between {1,2} and {3,4}.
        # To cover K_{2,2} edges with cliques:
        #   Each clique can contain at most 1 from {1,2} and 1 from {3,4}?
        #   No, {1, 3, 4} is a clique? Yes, if C is complete.
        #   But {1,3,4} covers (1,3), (1,4), (3,4).
        #   (3,4) is already covered by w-clique!
        #   So we cannot use {1,3,4} as a clique to cover remaining edges because (3,4) is used.
        #   We can use {1,3} and {1,4} and {2,3} and {2,4}.
        #   Or {1,3, ...}?
        #   Since C is complete, any subset is a clique.
        #   But we can only use UNUSED edges.
        #   So we need to partition the edges of K_{2,2} into cliques such that no clique contains an edge already used.
        #   Edges (3,4) is used. So no clique can contain both 3 and 4.
        #   So cliques must be of form {u, v} where u in {1,2}, v in {3,4}.
        #   Each such clique covers 1 edge.
        #   So we need 4 cliques.
        #   Total: 1 (v) + 1 (w) + 4 = 6.
        
        # Option 2 (Split v):
        #   v cliques: {v,1}, {v,2}.
        #   w cliques: {w,3}, {w,4}.
        #   Remaining C-C: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
        #   Wait, (1,2) is NOT covered by v cliques (size 2).
        #   (3,4) is NOT covered by w cliques.
        #   So we have the full K_4 edges to cover?
        #   No, (1,3), (1,4), (2,3), (2,4) are also there.
        #   Basically all C-C edges are remaining.
        #   We can cover all C-C edges with 1 clique: {1,2,3,4}.
        #   Total: 2 (v) + 2 (w) + 1 (C) = 5.
        
        # 5 < 6!
        # So Splitting IS better in this case!
        
        # The key is that merging I-C edges into large cliques "consumes" C-C edges that might
        # otherwise be covered efficiently by a single large C-only clique.
        # Specifically, if the C-C edges consumed are "bridges" between parts of C that are
        # also connected to other I vertices, it might be better to leave them for the C-only clique.
        
        # So the optimal strategy is not trivial.
        # However, note that the "Cost" of a clique {v} ∪ K is 1.
        # It "saves" |K| I-C edges (compared to individual cliques) but "costs" C-C edges.
        
        # Let's formulate this as a search problem.
        # Since n is small (90), k is 30.
        # Number of I vertices is 60.
        # We can try a greedy approach with local search.
        
        # But wait, the current program already does something similar.
        # It groups I vertices by shadow.
        # My new insight: We should consider splitting the shadow S_v into multiple cliques
        # to preserve C-C edges for a global C-only clique.
        
        # Heuristic:
        # 1. Start with all I vertices forming cliques {v} ∪ S_v.
        # 2. Identify the set of "unused" C-C edges.
        # 3. Try to "unmerge" some I-cliques to free up C-C edges, if it reduces the total count.
        #    Specifically, if splitting a clique {v} ∪ K into {v} ∪ A and {v} ∪ B
        #    frees up edges in K\A and K\B that allow the remaining C-C graph to be covered
        #    with fewer cliques, do it.
        
        # This is complex. Let's stick to a simpler heuristic that might work well.
        # Observation: The "waste" in Option 1 was that we couldn't cover the K_{2,2} edges efficiently.
        # If we leave the "core" C-C edges uncovered by I-cliques, we can cover them with 1 clique.
        
        # New Strategy:
        # 1. Identify a "Core" set of C vertices that forms a clique and whose internal edges
        #    are NOT covered by any I-clique.
        #    Ideally, the Core is large.
        # 2. For each I vertex v, form cliques {v} ∪ K where K ⊆ S_v \ Core.
        #    This ensures no I-clique uses edges within Core.
        # 3. Cover the Core edges with 1 clique (the Core itself).
        # 4. Cover any remaining edges (outside Core and not in I-cliques) with additional cliques.
        
        # How to choose Core?
        # We want Core to be large, but also we want S_v \ Core to be small enough that
        # covering I-C edges is efficient.
        
        # Actually, if we exclude Core from I-cliques, we might need more cliques for I-C edges.
        # In the example:
        # Core = {1, 2, 3, 4}.
        # S_v \ Core = empty.
        # So v has no I-cliques?
        # But v has edges to 1, 2.
        # We must cover them.
        # If Core is excluded, we must cover (v,1) and (v,2) with cliques that don't use Core-Core edges?
        # No, the rule is: Core edges are covered by the Core clique.
        # I-cliques cannot use Core-Core edges.
        # But they CAN use vertices from Core?
        # If an I-clique contains v and 1 (from Core), it covers (v,1).
        # Does it cover a Core-Core edge? No, only if it contains 2 Core vertices.
        # So {v, 1} is fine. It uses 1 Core vertex, 0 Core-Core edges.
        # So we can include Core vertices in I-cliques, as long as we don't include TWO Core vertices
        # in the same I-clique (which would cover a Core-Core edge).
        
        # Refined Strategy:
        # 1. Choose a set Core ⊆ C.
        # 2. Cover all edges within Core with 1 clique: Core.
        # 3. For each I vertex v:
        #    We need to cover edges (v, u) for u in S_v.
        #    We can form cliques {v} ∪ K where K ⊆ S_v.
        #    Constraint: K cannot contain two vertices from Core.
        #    So K can contain at most 1 vertex from Core.
        #    To minimize cliques, we want to maximize |K|.
        #    We can group S_v into cliques where each clique has at most 1 Core vertex.
        #    This is equivalent to:
        #    - Take all non-Core vertices in S_v. They can all be in one clique with v.
        #      {v} ∪ (S_v \ Core).
        #    - The remaining edges are (v, c) for c in S_v ∩ Core.
        #      Each such edge must be covered by a clique {v, c} (since we can't put two Core vertices).
        #      Or we can group multiple Core vertices? No, at most 1 per clique.
        #      So we need |S_v ∩ Core| cliques of size 2: {v, c}.
        
        # Total cliques for v: 1 (for non-Core) + |S_v ∩ Core| (for Core).
        # Wait, if S_v \ Core is empty, we don't need the first clique.
        
        # Let's evaluate this on the previous example.
        # C = {1,2,3,4}. I = {v, w}.
        # S_v = {1,2}. S_w = {3,4}.
        # Choose Core = {1,2,3,4}.
        # Core clique: {1,2,3,4}. (1 clique).
        # v: S_v \ Core = empty. S_v ∩ Core = {1,2}.
        #   Cliques: {v,1}, {v,2}. (2 cliques).
        # w: S_w \ Core = empty. S_w ∩ Core = {3,4}.
        #   Cliques: {w,3}, {w,4}. (2 cliques).
        # Total: 1 + 2 + 2 = 5.
        
        # This matches the "Split" strategy which was better!
        
        # What if Core = empty?
        # Core clique: None.
        # v: S_v \ empty = {1,2}. S_v ∩ empty = empty.
        #   Clique: {v,1,2}. (1 clique).
        # w: S_w \ empty = {3,4}. S_w ∩ empty = empty.
        #   Clique: {w,3,4}. (1 clique).
        # Remaining C-C edges: (1,3), (1,4), (2,3), (2,4).
        #   Need to cover these.
        #   As before, 4 cliques.
        # Total: 1 + 1 + 4 = 6.
        
        # So choosing a large Core is better.
        
        # What if Core = {1,2}?
        # Core clique: {1,2}.
        # v: S_v \ Core = empty. S_v ∩ Core = {1,2}.
        #   Cliques: {v,1}, {v,2}.
        # w: S_w \ Core = {3,4}. S_w ∩ Core = empty.
        #   Clique: {w,3,4}.
        # Remaining C-C edges:
        #   Edges within Core: (1,2) covered.
        #   Edges in S_w \ Core: (3,4) covered by w-clique.
        #   Cross edges: (1,3), (1,4), (2,3), (2,4).
        #   Also (1,2) is covered.
        #   Are there other edges? (1,2) is the only Core-Core edge.
        #   (3,4) is covered.
        #   So remaining: (1,3), (1,4), (2,3), (2,4).
        #   Same as before, 4 cliques.
        # Total: 1 (Core) + 2 (v) + 1 (w) + 4 (rem) = 8. Worse.
        
        # So the choice of Core matters.
        # Ideally, Core should include vertices that are heavily connected to I vertices?
        # No, in the example, Core = All C was best.
        
        # Let's try Core = All C.
        # This means NO I-clique can contain 2 C vertices.
        # So every I-clique is of the form {v, c} or {v} (if no edges).
        # This maximizes the number of I-cliques (sum |S_v|).
        # But it minimizes the remaining C-C edges (0).
        # Total: sum |S_v| + 1 (for Core).
        
        # Let's compare with Core = empty.
        # Total: (# I vertices) + (cliques for remaining C-C).
        
        # Which is smaller?
        # sum |S_v| + 1 vs #I + C_rem.
        
        # In the example:
        # sum |S_v| = 2 + 2 = 4. Total = 5.
        # #I = 2. C_rem = 4. Total = 6.
        
        # So Core = All C was better.
        
        # Is Core = All C always optimal?
        # Let's check the first example.
        # C = {1,2,3}. I = {4}. S_4 = {1,2}.
        # Core = All C = {1,2,3}.
        # Core clique: {1,2,3}.
        # v=4: S_v ∩ Core = {1,2}. S_v \ Core = empty.
        #   Cliques: {4,1}, {4,2}.
        # Total: 1 + 2 = 3.
        
        # Core = empty.
        # v=4: {4,1,2}.
        # Remaining: (1,3), (2,3).
        #   Cliques: {1,3}, {2,3}.
        # Total: 1 + 2 = 3.
        
        # Tie.
        
        # It seems Core = All C is a strong candidate.
        # It reduces the problem to:
        # 1. One clique for all C.
        # 2. For each I vertex v, |S_v| cliques of size 2: {v, u} for u in S_v.
        
        # Total = 1 + sum(|S_v|).
        
        # Can we do better?
        # What if we allow some I-cliques to have 2 C vertices, but choose WHICH ones carefully?
        
        # Given the time constraints and the complexity, I will implement the "Core = All C" strategy
        # and compare it with the current greedy strategy.
        
        # Actually, let's implement a search over "Core size".
        # Or just try Core = All C and Core = Empty and take the best.
        
        # Let's refine the "Core = All C" strategy.
        # It's very fast to compute.
        
        # Let's also try a mixed strategy:
        # For each I vertex, decide whether to use "Large" cliques (Core=empty style) or "Small" cliques (Core=All style).
        
        # But the interactions between I vertices (via C-C edges) make this hard.
        
        # I will implement the "Core = All C" strategy as a baseline and see how it scores.
        # Then I can add a heuristic to merge some small cliques if it helps.
        
        # Wait, the current program's score is 1.68.
        # If "Core = All C" is simple, let's calculate its expected score.
        # For n=30, k=10.
        # sum |S_v| is roughly (n-k)*k*p = 20*10*0.7 = 140.
        # Total = 141.
        
        # Current program:
        # It groups I vertices.
        # If it uses large cliques, it might get close to #I + C_rem.
        
        # Let's just implement the "Core = All C" strategy and return it.
        # It's a valid partition.
        
        def strategy_all_core(n, k, p, seed):
            # 1. Clique for C
            partition = [list(range(k))]
            
            # 2. Cliques for I-C edges
            for v in range(k, n):
                nbrs = adj[v] & C_set
                for u in nbrs:
                    partition.append(sorted([v, u]))
            
            return partition

        def strategy_empty_core(n, k, p, seed):
            # This is basically the "greedy large clique" for I, then greedy for C.
            # I'll reuse the logic from the current program but simplified.
            partition = []
            used_c_edges = set()
            
            # Process I vertices
            # To minimize cliques, group I vertices by shadow?
            # Or just process each v?
            # If we process each v independently with {v} ∪ S_v, we get #I cliques.
            # Let's try that.
            for v in range(k, n):
                nbrs = adj[v] & C_set
                if nbrs:
                    clique = sorted([v] + list(nbrs))
                    partition.append(clique)
                    # Mark C-C edges as used
                    nbrs_list = list(nbrs)
                    for i in range(len(nbrs_list)):
                        for j in range(i+1, len(nbrs_list)):
                            used_c_edges.add((min(nbrs_list[i], nbrs_list[j]), max(nbrs_list[i], nbrs_list[j])))
            
            # Cover remaining C-C edges
            rem_c_edges = internal_edges - used_c_edges
            rem_set = set(rem_c_edges)
            
            while rem_set:
                if time.time() > deadline - 5.0:
                    break
                # Greedy: pick an edge, expand to maximal clique in remaining graph
                # Simple heuristic: pick first edge
                a, b = next(iter(rem_set))
                clique = [a, b]
                
                # Expand
                candidates = []
                for c in range(k):
                    if c in clique:
                        continue
                    ok = True
                    for u in clique:
                        if (min(u,c), max(u,c)) not in rem_set:
                            ok = False
                            break
                    if ok:
                        candidates.append(c)
                
                clique.extend(candidates)
                
                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
            
            return partition

        # Run both strategies and pick the best
        part1 = strategy_all_core(n, k, p, seed)
        part2 = strategy_empty_core(n, k, p, seed)
        
        if len(part1) <= len(part2):
            return part1
        else:
            return part2

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        # Clean types
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results

