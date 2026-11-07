import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # We need to partition all edges into cliques.
        # Edges are of two types:
        # 1. Edges within K (clique K_k).
        # 2. Edges between K and I.
        
        # Strategy:
        # For each v in I, let N(v) be its neighbors in K.
        # The edges incident to v are exactly {(u, v) : u in N(v)}.
        # These edges must be covered by cliques containing v.
        # Since v is in I, it has no edges to other vertices in I.
        # So any clique containing v can only contain vertices from K.
        # Let C be a clique containing v. Then C \ {v} must be a clique in K.
        # Since K is a clique, any subset of K is a clique.
        # However, we must cover the edges (u, v) for u in N(v) exactly once.
        # This means we need to partition N(v) into subsets S_1, ..., S_m such that
        # each S_i forms a clique with v (which is always true since K is a clique).
        # BUT, we must also consider edges within K.
        # Edges within K are (u, w) for u, w in K.
        # If we form a clique {u, w, v}, it covers edges (u, v), (w, v), and (u, w).
        # If we form a clique {u, v}, it covers edge (u, v).
        # If we form a clique {u, w}, it covers edge (u, w).
        
        # The key constraint is that every edge must be covered exactly once.
        # This is equivalent to finding a set of cliques that partition the edge set.
        
        # Let's think about the edges within K.
        # Total edges in K: k*(k-1)/2.
        # These edges must be covered by cliques.
        # A clique in the graph that is a subset of K covers all edges within that subset.
        # A clique that includes vertices from I (say v) and vertices from K (say S) covers:
        #   - Edges (u, v) for u in S
        #   - Edges within S (i.e., edges in K[S])
        
        # So, if we assign a subset S_v of N(v) to be in a clique with v, we "use up" the edges within S_v.
        # The remaining edges in K must be covered by cliques entirely within K.
        
        # To minimize the number of cliques, we want to maximize the size of S_v.
        # Specifically, for each v, we want to partition N(v) into as few subsets as possible,
        # but the subsets for different v's must be disjoint with respect to the edges they cover in K?
        # No, the edges in K are fixed. If S_v and S_w (for v != w) share an edge (u, w'), that edge is covered twice.
        # So, the sets of edges in K covered by cliques involving I-vertices must be disjoint.
        # Let E_K be the set of all edges in K.
        # For each v in I, let P_v be a partition of N(v) into subsets S_{v,1}, ..., S_{v,m_v}.
        # The edges in K covered by cliques involving v are the union of E_K[S_{v,j}] for all j.
        # These sets of edges must be pairwise disjoint for different v.
        # The remaining edges in E_K are covered by cliques within K.
        
        # To minimize total cliques:
        # 1. Minimize sum over v of m_v (number of cliques involving v).
        # 2. Minimize the number of cliques needed to cover the remaining edges in K.
        
        # Note that if S_{v,j} has size s, it covers s-1 edges in K (if s>=2).
        # Actually, a clique of size s covers s*(s-1)/2 edges.
        # Wait, the clique is {v} U S_{v,j}. The edges covered are:
        #   - (v, u) for u in S_{v,j}  [s edges]
        #   - (u, w) for u, w in S_{v,j} [s*(s-1)/2 edges]
        
        # So, by including a subset S in a clique with v, we cover s cross-edges and s*(s-1)/2 K-edges.
        # If we instead split S into singletons, we cover s cross-edges and 0 K-edges.
        # Covering K-edges with I-cliques is "free" in terms of cross-edges, but it uses up K-edges that might be better covered by large K-cliques?
        # Actually, covering K-edges with I-cliques reduces the number of K-cliques needed.
        # A K-clique of size t covers t*(t-1)/2 edges and costs 1 clique.
        # An I-clique with subset S of size s covers s*(s-1)/2 K-edges and s cross-edges, and costs 1 clique.
        # But the cross-edges MUST be covered by I-cliques.
        # So we MUST form I-cliques.
        # The question is: how to partition N(v) to minimize total cliques.
        
        # Let's assume we partition N(v) into singletons first.
        # Then we have |N(v)| cliques of size 2 for each v.
        # Total I-cliques = sum |N(v)|.
        # Remaining K-edges = all K-edges.
        # We need to cover K-edges with K-cliques.
        # The minimum number of cliques to cover a complete graph K_k is 1 (the whole clique).
        # So total cliques = sum |N(v)| + 1.
        
        # Can we do better?
        # If we merge two singletons {u}, {w} into one clique {u, w, v}, we reduce the number of I-cliques by 1 (from 2 to 1).
        # But we also cover the edge (u, w) in K.
        # This edge (u, w) is no longer available to be covered by a K-clique.
        # Does this increase the number of K-cliques?
        # If we cover (u, w) with the I-clique, we remove one edge from K.
        # The remaining K-edges might still be coverable by 1 clique? No, because K is no longer a complete graph.
        # The remaining graph on K is K_k minus the edges covered by I-cliques.
        # We need to cover this remaining graph with cliques.
        
        # So the trade-off is:
        # - Merging two neighbors u, w in v's clique saves 1 I-clique.
        # - But it removes edge (u, w) from the K-remaining graph.
        # - This might increase the number of K-cliques needed.
        
        # However, note that if we merge many pairs, we might significantly reduce the density of the remaining K-graph,
        # potentially allowing it to be covered by fewer cliques? Or maybe more?
        # Actually, removing edges makes it easier to cover with cliques? No, it makes it harder to form large cliques.
        # But we are counting the number of cliques.
        # Covering a graph with cliques: the minimum number of cliques to cover all edges is the "clique cover number".
        # For a complete graph, it is 1.
        # For an empty graph, it is 0.
        # For a graph with some edges removed, it is at least 1 (if any edge remains).
        
        # Let's test with small k.
        # k=2. K has 1 edge.
        # If we don't merge, we have 1 K-clique.
        # If we merge u, w in v, we cover the edge (u, w). No K-edges left. 0 K-cliques.
        # Total: 1 I-clique (size 3) vs 2 I-cliques (size 2) + 1 K-clique.
        # 1 < 3. So merging is better.
        
        # k=3. K has 3 edges.
        # If we don't merge, 1 K-clique.
        # If we merge u, w in v, edge (u, w) covered. Remaining edges: (u, z), (w, z).
        # These two edges share z. They can be covered by 2 cliques: {u, z} and {w, z}.
        # Total: 1 I-clique + 2 K-cliques = 3.
        # Without merging: 2 I-cliques + 1 K-clique = 3.
        # Same.
        
        # What if we merge u, z in v? Same.
        # What if we don't merge? 3.
        
        # It seems that for small k, the benefit of saving I-cliques is offset by the cost of more K-cliques.
        # But for large k, maybe it's different?
        
        # Let's look at the structure.
        # The remaining K-graph is a subgraph of K_k.
        # The edges removed are exactly those covered by I-cliques.
        # Let E_removed be the set of K-edges covered by I-cliques.
        # We need to cover K_k \ E_removed with cliques.
        
        # If we choose the I-cliques to be "triangles" (v, u, w), we remove edges (u, w).
        # If we choose them to be "stars" (v, u), we remove no K-edges.
        
        # Hypothesis: It is optimal to use only 2-cliques for I (i.e., no merging), and cover K with 1 clique.
        # Why? Because merging saves 1 I-clique but breaks the completeness of K, potentially requiring more than 1 K-clique.
        # If K remains complete, 1 K-clique suffices.
        # If we remove any edge, K is no longer complete.
        # Can a non-complete graph on k vertices be covered by 1 clique? No, unless it has no edges.
        # So if we remove any edge, we need at least 2 K-cliques (one for the edge, one for the rest? No, cliques can overlap in vertices but not edges).
        # Actually, if we remove one edge (u, w), the remaining graph has k(k-1)/2 - 1 edges.
        # Can we cover it with 1 clique? No, because (u, w) is missing, so no clique can contain both u and w and be a clique in the remaining graph?
        # Wait, a clique in the remaining graph must be a set of vertices where all pairs are edges in the remaining graph.
        # If (u, w) is missing, no clique can contain both u and w.
        # So the maximum clique size is k-1 (excluding either u or w).
        # But we need to cover all edges.
        # The edge (u, w) is gone, so we don't need to cover it.
        # But we need to cover edges like (u, z) and (w, z).
        # A clique containing u and z cannot contain w.
        # A clique containing w and z cannot contain u.
        # So we might need multiple cliques.
        
        # Let's try to cover K_3 minus edge (0, 1). Edges: (0, 2), (1, 2).
        # Clique {0, 2} covers (0, 2).
        # Clique {1, 2} covers (1, 2).
        # Total 2 cliques.
        # Versus 1 clique for K_3.
        # So removing 1 edge increased K-cliques from 1 to 2.
        # And we saved 1 I-clique.
        # Net change: 0.
        
        # What if we remove more edges?
        # If we remove a matching of size m in K.
        # Then we save m I-cliques.
        # How many K-cliques are needed to cover K_k minus a matching of size m?
        # This is a known problem?
        
        # Let's assume the optimal strategy is to NOT merge, i.e., use only 2-cliques for I.
        # Then the number of I-cliques is sum |N(v)|.
        # The number of K-cliques is 1.
        # Total = sum |N(v)| + 1.
        
        # Can we do better by merging?
        # If merging saves 1 I-clique but adds 1 K-clique, it's a wash.
        # If merging saves 1 I-clique and adds 0 K-cliques, it's better.
        # When does merging add 0 K-cliques?
        # Only if the remaining K-graph can still be covered by 1 clique.
        # This is only possible if the remaining K-graph is empty.
        # i.e., we cover all K-edges with I-cliques.
        # This requires that the union of all S_{v,j} covers all pairs in K.
        # i.e., for every pair (u, w) in K, there exists some v and some j such that {u, w} subset of S_{v,j}.
        # This is possible if the neighborhoods N(v) are large enough.
        
        # In our case, p=0.7, so N(v) is large.
        # k=10, 20, 30.
        # n=30, 60, 90.
        # |I| = 20, 40, 60.
        # Each v has ~0.7*k neighbors.
        
        # If we can cover all K-edges with I-cliques, then we need 0 K-cliques.
        # And we minimize the number of I-cliques by maximizing the size of S_{v,j}.
        # Ideally, for each v, we take S_v = N(v) as one clique.
        # But S_v must be such that the edges covered by different v's are disjoint.
        # If we take S_v = N(v), then the edges covered in K are all edges within N(v).
        # These sets of edges might overlap.
        # If they overlap, we have covered an edge twice, which is invalid.
        
        # So we need a set of subsets S_{v,j} such that:
        # 1. For each v, {S_{v,1}, ..., S_{v,m_v}} is a partition of N(v).
        # 2. The sets of edges E_K[S_{v,j}] are pairwise disjoint for all v, j.
        # 3. We want to minimize sum m_v + (cliques needed for remaining K-edges).
        
        # This is a complex optimization problem.
        # Given the time limit, a simple heuristic might be sufficient.
        
        # Let's try the "no merging" strategy first.
        # Total cliques = sum |N(v)| + 1.
        
        # Let's estimate the score for this.
        # For (30, 10, 0.7, 301):
        # k=10, n=30, |I|=20.
        # Expected |N(v)| = 7.
        # Sum |N(v)| ~ 140.
        # Total ~ 141.
        
        # Current score is 2.0167.
        # Seed value (from prompt) is not explicitly given, but score = seed/value.
        # If seed is, say, 285 (2 * 141 + 3?), then value 141 gives score 2.02.
        # This matches the current score!
        # So the current program is likely doing exactly this: no merging, 1 K-clique.
        # Wait, the current program does greedy matching.
        # Let's check the current program's logic.
        # It finds a maximal matching in the subgraph of available_k_edges induced by N_v.
        # This means it merges pairs (u, w) if (u, w) is an edge in K.
        # Since K is a clique, (u, w) is always an edge.
        # So it finds a maximal matching in the complete graph on N_v.
        # The size of a maximal matching in a complete graph on m vertices is floor(m/2).
        # So it merges floor(|N(v)|/2) pairs.
        # The number of cliques for v is floor(|N(v)|/2) (for pairs) + (|N(v)| % 2) (for singles).
        # This is ceil(|N(v)|/2).
        # So it reduces the number of I-cliques by roughly half.
        # But it removes a matching of edges from K.
        # The remaining K-graph is K_k minus a matching (for each v, a matching in N(v)).
        # The matchings for different v's are disjoint?
        # The program removes edges from available_k_edges.
        # So yes, the edges removed are disjoint.
        # So the remaining K-graph is K_k minus a set of disjoint edges (a matching, possibly not maximal in K_k, but disjoint per v).
        # Actually, the matchings for different v's are disjoint because edges are removed from a global pool.
        # So the total set of removed edges is a matching in K_k.
        # Let M be the total matching removed.
        # Size of M is sum floor(|N(v)|/2).
        # The remaining K-graph is K_k \ M.
        # How many cliques are needed to cover K_k \ M?
        # K_k \ M is a complete graph minus a matching.
        # If M is a perfect matching (k even), the remaining graph has no perfect matching?
        # No, it has many matchings.
        # But we need a clique cover.
        # In K_k \ M, any two vertices u, w are adjacent unless (u, w) in M.
        # So a clique can contain at most one vertex from each pair in M.
        # So the maximum clique size is k/2 (if k even) or (k+1)/2 (if k odd).
        # To cover all edges, we might need multiple cliques.
        
        # Let's calculate the number of cliques needed to cover K_k \ M.
        # If M is a matching of size m.
        # The graph is (k-2m) isolated vertices + (k/2 - m) K_2's? No.
        # The vertices are partitioned into m pairs (from M) and k-2m singletons.
        # Edges exist between all vertices except within the m pairs.
        # So it's a complete k-partite graph where m parts have size 2 and k-2m parts have size 1?
        # No, it's the complement of a matching.
        # Clique cover of complement of matching.
        # This is equivalent to coloring the matching?
        # No.
        
        # Let's just implement the current strategy (greedy matching) and see if we can improve the K-clique cover.
        # The current program covers the remaining K-edges greedily.
        # Maybe a better cover for K_k \ M is possible.
        
        # However, given the time, I will stick to a robust implementation of the current strategy,
        # but ensure the K-clique cover is optimal for the remaining graph.
        # For a graph that is K_k minus a matching, the minimum clique cover number is known?
        # For K_4 minus a matching of size 2 (i.e., 2 disjoint edges removed, 4 edges remaining).
        # Remaining edges: (0,1) removed, (2,3) removed. Edges: (0,2), (0,3), (1,2), (1,3).
        # This is C_4.
        # Clique cover of C_4: 2 cliques of size 2.
        # Versus K_4: 1 clique.
        # So we saved 2 I-cliques (from 4 singles to 2 pairs) but added 1 K-clique (from 1 to 2).
        # Net: 4+1=5 vs 2+2=4. Improvement!
        
        # So the current strategy is likely good.
        # I will rewrite it cleanly.

        # 1. Generate neighbors.
        # 2. For each v in I, find maximal matching in K[N(v)].
        #    Since K is a clique, this is just matching in a complete graph.
        #    Greedy matching is optimal for size.
        # 3. Remove matched edges from K.
        # 4. Cover remaining K-edges.
        #    The remaining graph is K_k minus a matching.
        #    We can cover it by:
        #    - For each unmatched edge (u, w), add clique [u, w].
        #    - But we can do better by finding large cliques.
        #    - In K_k \ M, a clique is a set of vertices with no pair in M.
        #    - This is an independent set in M.
        #    - Since M is a matching, the maximum independent set is k - |M| (pick one from each pair, and all singletons).
        #    - Wait, if we pick one from each pair, we get a clique of size k - |M|?
        #    - Yes, because no two picked vertices are in M.
        #    - So we can cover the graph with cliques?
        #    - No, one clique of size k-|M| covers all edges incident to those vertices?
        #    - No, it covers edges between picked vertices.
        #    - What about edges between a picked vertex u and an unpicked vertex w (where w is in a pair with u')?
        #    - (u, w) is an edge in K_k \ M.
        #    - So we need to cover those too.
        
        #    Actually, the graph K_k \ M is the union of:
        #    - A clique on the set of "unmatched" vertices? No.
        #    - Let's just use a greedy algorithm for the remaining K-edges.
        #    - Or, since the structure is simple, we can derive a formula.
        
        #    Let's just use the greedy clique cover for the remaining edges.
        #    It should be fast enough for k=30.

        available_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        cliques = []
        
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Find maximal matching in complete graph on N_v
            # Since it's a complete graph, just pair them up.
            # To be safe, we check if edge is in available_k_edges.
            # But since we only remove edges, and initially all are there,
            # and we only remove edges within N_v for this v,
            # any pair in N_v is an edge in available_k_edges UNLESS it was removed by a previous v.
            # Wait, if (u, w) was removed by a previous v', then (u, w) is not in available_k_edges.
            # So we must check.
            
            # Greedy matching
            matched = set()
            pairs = []
            # Sort N_v to make it deterministic?
            # Or just iterate.
            # To maximize matching, we should be careful.
            # In a complete graph minus some edges, greedy is not always optimal.
            # But the removed edges form a matching? No, they form a union of matchings.
            # Actually, the set of removed edges is a matching in K_k.
            # So the graph on N_v is a complete graph minus a matching (subset of the global matching).
            # In such a graph, a maximal matching is optimal?
            # Yes, for a graph that is a complete graph minus a matching, any maximal matching is maximum.
            # Proof: The graph is the complement of a matching.
            # The matching number of the complement of a matching is ...
            # Anyway, greedy should be fine.
            
            N_v_list = list(N_v)
            for i in range(len(N_v_list)):
                if N_v_list[i] in matched:
                    continue
                for j in range(i + 1, len(N_v_list)):
                    if N_v_list[j] in matched:
                        continue
                    u, w = N_v_list[i], N_v_list[j]
                    edge = (min(u, w), max(u, w))
                    if edge in available_k_edges:
                        pairs.append((u, w))
                        matched.add(u)
                        matched.add(w)
                        available_k_edges.remove(edge)
                        break # Move to next i
            
            for u, w in pairs:
                cliques.append([u, w, v])
            
            for u in N_v:
                if u not in matched:
                    cliques.append([u, v])
        
        # Cover remaining K-edges
        remaining_edges = list(available_k_edges)
        while remaining_edges:
            # Greedy: pick an edge, extend to maximal clique
            u, w = remaining_edges[0]
            clique = {u, w}
            # Find all x such that (u, x) and (w, x) in remaining_edges
            # And x not in clique
            candidates = set()
            for e in remaining_edges:
                a, b = e
                if a == u and b != w:
                    candidates.add(b)
                elif b == u and a != w:
                    candidates.add(a)
                if a == w and b != u:
                    candidates.add(b)
                elif b == w and a != u:
                    candidates.add(a)
            
            # Filter candidates
            valid_c = []
            for c in candidates:
                if (min(u, c), max(u, c)) in remaining_edges and (min(w, c), max(w, c)) in remaining_edges:
                    valid_c.append(c)
            
            if valid_c:
                # Greedily add valid candidates
                # To be safe, just add one?
                # Or add all that form a clique?
                # Since K is a clique, any subset of valid_c forms a clique with u, w.
                # So we can add all of them.
                for c in valid_c:
                    clique.add(c)
            
            clique_list = list(clique)
            cliques.append(clique_list)
            
            # Remove edges covered by this clique
            edges_to_remove = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edges_to_remove.add((min(a, b), max(a, b)))
            
            remaining_edges = [e for e in remaining_edges if e not in edges_to_remove]
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return empty or minimal valid?
            # Must return valid result.
            # If we run out of time, we can't compute.
            # But the algorithm is fast.
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
