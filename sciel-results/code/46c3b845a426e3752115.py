import time
import random
from collections import deque

def construct():
    start = time.time()
    deadline = start + 110.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(n, k, p, seed):
        if time.time() > deadline:
            # Fallback: just return all edges as cliques of size 2
            rng = random.Random(seed)
            edges = []
            for u in range(k):
                for v in range(u + 1, k):
                    edges.append([u, v])
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        edges.append([u, v])
            return edges

        rng = random.Random(seed)
        
        # Build the graph
        # K = {0, ..., k-1} is a clique.
        # I = {k, ..., n-1} is an independent set.
        # Edges between K and I with prob p.
        
        cross = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            cross[v] = nbrs

        # We need to partition the edge set E into cliques.
        # A clique in this graph is either:
        # 1. A subset of K (since K is a clique).
        # 2. A subset of K union {v} for v in I, such that {v} is connected to all vertices in the subset of K.
        # Note: A clique cannot contain two vertices from I because I is independent.
        
        # Strategy:
        # We can view this as covering edges.
        # Edges within K (K-K edges) must be covered by cliques contained in K.
        # Edges between K and I (K-I edges) must be covered by cliques that contain the I-vertex and a subset of K.
        
        # Let's separate the problem.
        # Let E_KK be the set of edges within K.
        # Let E_KI be the set of edges between K and I.
        
        # Any clique covering an edge in E_KI must be of the form C = S ∪ {v} where v ∈ I and S ⊆ N(v) ∩ K.
        # This clique covers all edges (u, v) for u ∈ S, and all edges (u, w) for u, w ∈ S (which are in E_KK).
        
        # So, choosing a clique S ∪ {v} covers:
        # - The K-I edges (u, v) for u ∈ S.
        # - The K-K edges (u, w) for u, w ∈ S.
        
        # We want to minimize the number of cliques.
        # This looks like a set cover or packing problem.
        
        # Observation:
        # If we pick a clique C = S ∪ {v}, it "uses up" the edges in S × {v} and the edges within S.
        # The edges within S are part of the complete graph K_k.
        
        # Let's think about the K-K edges. They must be partitioned into cliques.
        # Let the cliques covering K-K edges be K_1, K_2, ..., K_m.
        # Note that a clique covering K-K edges might also cover K-I edges if it includes an I-vertex.
        # But wait, a clique containing an I-vertex v is S ∪ {v}. The K-K edges it covers are exactly those within S.
        # So every clique in the partition corresponds to a subset S ⊆ K.
        # If the clique is S ∪ {v}, it covers K-K edges within S and K-I edges (u,v) for u∈S.
        # If the clique is S (subset of K), it covers K-K edges within S and no K-I edges.
        
        # Let's refine the structure.
        # For each v ∈ I, let the cliques containing v be C_{v,1}, ..., C_{v,d_v}.
        # Each C_{v,i} is of the form S_{v,i} ∪ {v}.
        # The sets S_{v,i} must be disjoint? No.
        # Wait, the edges (u, v) for u ∈ K are distinct for each u.
        # So for a fixed v, the sets S_{v,i} must partition N(v).
        # Because every edge (u, v) must be covered by exactly one clique containing v.
        # And a clique containing v covers (u, v) iff u ∈ S_{v,i}.
        # So, for each v, {S_{v,i}} is a partition of N(v).
        
        # Now consider the K-K edges.
        # An edge (u, w) in K is covered by a clique S ∪ {v} if u, w ∈ S.
        # Or by a clique S ⊆ K if u, w ∈ S.
        # Actually, any clique in the partition is defined by its intersection with K.
        # Let's call the "core" of a clique its intersection with K.
        # Let the cores be Q_1, ..., Q_M.
        # Each Q_j is a subset of K.
        # If Q_j comes from a clique with an I-vertex v, then Q_j ⊆ N(v).
        # If Q_j comes from a clique in K, then Q_j ⊆ K.
        
        # The condition "every edge is in exactly one clique" translates to:
        # 1. For each v ∈ I, the sets {Q_j | Q_j is core of a clique containing v} partition N(v).
        # 2. For each edge (u, w) in K, there is exactly one core Q_j such that {u, w} ⊆ Q_j.
        
        # This is a very specific structure.
        # Let's simplify.
        # For each v ∈ I, we choose a partition of N(v) into subsets.
        # Let P_v be the collection of subsets in the partition of N(v).
        # The cores contributed by v are the sets in P_v.
        # Additionally, we can have "pure" K-cliques with cores R_1, ..., R_L.
        # The total collection of cores is Q = (∪_{v∈I} P_v) ∪ {R_1, ..., R_L}.
        # The condition is that the edges of K_k are partitioned by these cores.
        # i.e., for each edge {u, w} ⊆ K, exactly one Q ∈ Q contains {u, w}.
        
        # This is equivalent to saying that the family Q is a partition of the edge set of K_k into cliques.
        # Wait, not exactly. The sets in Q are cliques in K.
        # The condition is that every edge of K is contained in exactly one set from Q.
        # This is a clique partition of K_k using the sets in Q.
        
        # So the problem reduces to:
        # 1. For each v ∈ I, choose a partition P_v of N(v).
        # 2. Let Q_v be the sets in P_v.
        # 3. Add extra sets R_1, ..., R_L (subsets of K) such that Q = (∪ Q_v) ∪ {R_j} partitions the edges of K_k.
        # 4. Minimize |Q| = (∑ |P_v|) + L.
        
        # Note: A set Q ∈ Q must be a clique in the graph.
        # If Q ∈ Q_v, then Q ⊆ N(v), so it's a clique (since Q ⊆ K and K is a clique).
        # If Q ∈ {R_j}, then Q ⊆ K, so it's a clique.
        # So all sets in Q are valid cliques in K.
        
        # The cost is the number of sets in Q.
        
        # How to solve?
        # This is still complex. Let's try a greedy heuristic.
        
        # Greedy approach:
        # 1. For each v ∈ I, we must partition N(v).
        #    To minimize the number of sets, we want large sets.
        #    But the sets in P_v also contribute to covering edges in K.
        #    If we make sets in P_v large, they cover more edges in K, potentially reducing the need for R_j.
        
        # Let's try:
        # For each v, let P_v be just {N(v)}. i.e., one set per v.
        # Then Q_v = {N(v)}.
        # The total sets from I are {N(v) | v ∈ I}.
        # Now we need to cover the remaining edges of K_k using R_j.
        # The edges covered by {N(v)} are the edges within each N(v).
        # Let E_covered = ∪_{v} E(N(v)).
        # Let E_remaining = E(K_k) \ E_covered.
        # We need to partition E_remaining into cliques R_j.
        # To minimize L, we should find a clique partition of the graph (K, E_remaining) with minimum number of cliques.
        
        # Is {N(v)} a good choice for P_v?
        # If N(v) is large, it covers many edges.
        # But what if N(v) is not a clique? It is a subset of K, so it is a clique.
        # So this is always valid.
        
        # Let's test this strategy.
        # Step 1: For each v, set P_v = {N(v)}.
        # Step 2: Compute E_covered = union of edges within each N(v).
        # Step 3: Compute E_remaining = E(K_k) - E_covered.
        # Step 4: Find a clique partition of (K, E_remaining).
        
        # How to find a clique partition of (K, E_remaining)?
        # This is the "clique partition number" of the graph G_rem = (K, E_remaining).
        # We want to cover E_rem with minimum number of cliques.
        # Greedy: Repeatedly find a maximal clique in G_rem, remove its edges, repeat.
        
        # Let's implement this.
        
        # 1. Build N(v) for all v.
        N_v = []
        for v in range(k, n):
            N_v.append(cross[v])
        
        # 2. Determine edges covered by N(v).
        # An edge (u, w) is covered if there exists v such that u ∈ N(v) and w ∈ N(v).
        # i.e., u and w are in the intersection of some N(v)? No.
        # u and w must both be in N(v) for some v.
        # So (u, w) is covered if N(u) ∩ N(w) is not empty?
        # Wait, N(v) is the set of neighbors of v in K.
        # So u ∈ N(v) means (u, v) is an edge.
        # So (u, w) is covered by the clique N(v) ∪ {v} if u, w ∈ N(v).
        # This is true if there exists v such that (u, v) and (w, v) are edges.
        # i.e., u and w have a common neighbor in I.
        
        # Let's compute the set of covered edges.
        covered = set()
        for v in range(k, n):
            nv = cross[v]
            for u in nv:
                for w in nv:
                    if u < w:
                        covered.add((u, w))
        
        # 3. Determine remaining edges in K.
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered:
                    remaining_edges.add((u, w))
        
        # 4. Clique partition of (K, remaining_edges).
        # We need to cover remaining_edges with cliques in K.
        # Greedy:
        rem_adj = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
        
        extra_cliques = []
        rem_edges = set(remaining_edges)
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: single edges
                for e in rem_edges:
                    extra_cliques.append([e[0], e[1]])
                rem_edges.clear()
                break
            
            # Find a maximal clique in the graph defined by rem_edges.
            # We can use Bron-Kerbosch or just a greedy heuristic.
            # Given k is small (10, 20, 30), we can afford a decent search.
            
            # Let's find the largest clique in the remaining graph.
            # We'll use a simple recursive backtracking with pruning.
            # Or just a greedy approach: pick a vertex, add neighbors, etc.
            
            # Let's try a greedy maximal clique.
            # Pick a vertex with max degree in rem_edges.
            if not rem_edges:
                break
            
            # Build adjacency from rem_edges
            # We already have rem_adj, but it needs to be updated as we remove edges.
            # Let's rebuild or update.
            
            # To keep it simple, let's just find a clique by checking subsets? No, too slow.
            # Let's use a heuristic:
            # 1. Pick vertex u with max degree in current rem_edges.
            # 2. Let C = {u}.
            # 3. While possible, add a vertex v in N(u) ∩ ... that is connected to all in C.
            
            # Actually, let's just find the maximum clique using a simple algorithm since k is small.
            # Max clique in a graph of size 30 is fast.
            
            def max_clique(V, adj):
                V = set(V)
                best = []
                def bk(R, P, X):
                    nonlocal best
                    if not P and not X:
                        if len(R) > len(best):
                            best = list(R)
                        return
                    pool = P | X
                    if not pool:
                        return
                    # Pivot selection
                    # Choose pivot u in P ∪ X that maximizes |N(u) ∩ P|
                    u = max(pool, key=lambda x: len(adj.get(x, set()) & P))
                    # Candidates are P \ N(u)
                    for v in list(P - adj.get(u, set())):
                        R.add(v)
                        bk(R, P & adj.get(v, set()), X & adj.get(v, set()))
                        R.remove(v)
                        P.remove(v)
                        X.add(v)
                bk(set(), V, set())
                return best
            
            # Current vertices with edges
            active_vertices = set()
            for u, w in rem_edges:
                active_vertices.add(u)
                active_vertices.add(w)
                
            mc = max_clique(active_vertices, rem_adj)
            if not mc:
                # No edges?
                break
                
            # Add this clique
            extra_cliques.append(sorted(mc))
            
            # Remove edges covered by this clique
            mc_set = set(mc)
            new_rem_edges = set()
            for u, w in rem_edges:
                if u in mc_set and w in mc_set:
                    continue
                new_rem_edges.add((u, w))
            
            # Update rem_adj? 
            # It's easier to just rebuild rem_adj from new_rem_edges or check membership.
            # For the next iteration, we need the graph defined by new_rem_edges.
            # Let's update rem_adj by removing edges in the clique.
            # Actually, max_clique uses rem_adj. We should update rem_adj to reflect removed edges.
            # Or, simpler: just pass the new edge set to the next iteration's logic.
            # But max_clique takes adj.
            # Let's rebuild rem_adj from new_rem_edges.
            rem_edges = new_rem_edges
            rem_adj = {u: set() for u in range(k)}
            for u, w in rem_edges:
                rem_adj[u].add(w)
                rem_adj[w].add(u)

        # Construct the final list of cliques
        cliques = []
        
        # Cliques from I vertices
        for v in range(k, n):
            nv = cross[v]
            if nv:
                cliques.append(sorted(list(nv)) + [v])
            # If nv is empty, no clique needed for v (no edges to cover).
            # But wait, if nv is empty, v is isolated in the cross edges.
            # So no clique containing v is needed.
            
        # Cliques from K partition
        for c in extra_cliques:
            if len(c) >= 2:
                cliques.append(c)
            elif len(c) == 1:
                # Should not happen if we only add cliques from edges
                pass
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve(*spec)
            results.append(res)

    cleaned = [[[int(x) for x in c] for c in clist] for clist in results]
    return cleaned

# Test run
if __name__ == "__main__":
    res = construct()
    total = sum(len(c) for c in res)
    print(f"Total cliques: {total}")
