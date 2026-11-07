"""Clique partition of split graphs using structured greedy + local improvement."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """
        Split graph: K = {0..k-1} (clique), I = {k..n-1} (independent set).
        Edges: all K-K edges, and each (u,v) for u in K, v in I with prob p.
        
        Strategy:
        1. For each i in I, let N(i) = {u in K : (u,i) is an edge}.
        2. We need to cover all edges. K-K edges must be covered by cliques containing >=2 K vertices.
           I-vertex edges: edge (u,i) must be in a clique containing u and i.
        
        Key insight: A clique can contain at most one I-vertex (since I is independent).
        So each clique either:
          - Contains one I-vertex i and some subset S of N(i) (|S|>=1 for the edge, but to be a clique with >=2 vertices, |S|>=1 and we need at least 2 total vertices, so |S|>=1 and the clique is S ∪ {i}, need |S|>=1; if |S|==1 then clique has 2 vertices which is fine)
          - Or contains only K vertices (a clique on a subset of K with >=2 vertices)
        
        To minimize total cliques:
        - Each I-vertex i needs at least ceil(|N(i)| / (something)) cliques? No.
          Actually, one clique containing i can cover multiple edges incident to i: it can include multiple neighbors u1, u2, ... from N(i). 
          So for vertex i, we can cover all its edges with ONE clique: N(i) ∪ {i}. That's a valid clique since N(i) ⊆ K and K is a clique, and i connects to all in N(i).
          So each i can be "handled" with one clique: N(i) ∪ {i}, provided |N(i)| >= 1. If |N(i)| == 0, i has no edges, no clique needed for it.
        
        - But we also need to cover K-K edges. If we use cliques N(i) ∪ {i} for various i, these cliques contain subsets of K. The K-K edges are covered if for every pair (u,v) in K, there exists some clique containing both u and v.
        
        So the problem reduces to: choose a set of subsets S_i ⊆ N(i) (one per i with N(i) nonempty, where S_i is the K-part of the clique for i), such that:
          - Every K-K edge (u,v) is covered by at least one S_i that contains both u and v, OR by a dedicated K-only clique.
          - Minimize total number of cliques = (number of i with nonempty S_i) + (number of K-only cliques needed for uncovered K-K edges).
        
        Actually, we can also have K-only cliques. Let's think differently.
        
        Alternative approach: 
        - Start by assigning each i to a clique C_i = N(i) ∪ {i}. This covers all I-edges.
        - Now check which K-K edges are covered. A K-K edge (u,v) is covered if there exists some i such that u,v ∈ N(i).
        - For uncovered K-K edges, we need additional cliques on K.
        
        To minimize total cliques, we want to maximize the number of K-K edges covered by the I-cliques, and minimize the number of additional K-only cliques.
        
        But we can also choose to NOT use a clique for some i, and instead cover i's edges via other means? No, because i only connects to K, and any clique containing i must be {i} ∪ S where S ⊆ N(i). We can't cover i's edges with a K-only clique. So each i with N(i) nonempty MUST have at least one clique containing i.
        
        So the number of I-cliques is fixed: it's the count of i with N(i) nonempty. Let's call this m.
        
        Wait, no. We could potentially merge... no, each clique contains at most one I-vertex. So yes, exactly one clique per i with N(i) nonempty.
        
        So total cliques = m + (number of K-only cliques needed).
        
        To minimize K-only cliques, we want the union of all S_i (where S_i = N(i)) to cover as many K-K edges as possible. But S_i is fixed as N(i) — we can't change it. The clique for i is N(i) ∪ {i}, so the K-part is exactly N(i).
        
        So the K-K edges covered by I-cliques are: {(u,v) : ∃i, u,v ∈ N(i)}.
        The uncovered K-K edges must be covered by K-only cliques.
        
        So the problem is: given the set of uncovered K-K edges, find a minimum clique cover of the graph on K induced by those uncovered edges. Since K is a clique, any subset of K is a clique. So we need to partition the uncovered edges into as few cliques as possible. But wait, a clique on K covers all edges within it. So we need to cover the uncovered edges with cliques on K. This is equivalent to: find the minimum number of cliques (subsets of K) such that every uncovered edge is contained in at least one clique.
        
        This is the edge clique cover problem on the subgraph of K-K uncovered edges. Since the underlying graph is a complete graph on k vertices (K is a clique), and we have a subset of edges to cover, this is: given a graph G' on k vertices (the uncovered edges), find the minimum number of cliques in the complete graph K_k such that every edge of G' is in at least one clique.
        
        This is equivalent to: the minimum number of cliques needed to cover the edges of G'. Since any subset of vertices forms a clique in K_k, this is the same as the edge clique cover number of G', which equals the minimum number of cliques to partition the edges of G' (since we can always assume cliques are edge-disjoint without loss of generality for minimization... actually no, but for covering, we can assume disjoint).
        
        For small k (up to 30), we can use a greedy or exact method.
        
        Let me implement:
        1. Compute N(i) for each i.
        2. Compute covered K-K edges.
        3. Find uncovered K-K edges.
        4. Solve minimum clique cover on the uncovered edges (on the k-vertex graph).
        
        For step 4, since k ≤ 30, we can use a simple greedy: repeatedly pick the largest clique (in terms of edges covered) that covers the most uncovered edges.
        """
        # Build adjacency for I vertices
        N = [set() for _ in range(n)]
        # K-K edges are all present
        # I-K edges
        rng = Random(seed)
        # We need to regenerate the same random sequence
        # The instance function does:
        # edges = K-K edges first, then for v in range(k, n): for u in range(k): if rng.random() < p
        # So the random calls are in order: for each v from k to n-1, for each u from 0 to k-1
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        
        # Determine which K-K edges are covered by I-cliques
        # A K-K edge (u,v) is covered if ∃ i ≥ k such that u,v ∈ N[i]
        covered_kk = set()
        for i in range(k, n):
            if len(N[i]) >= 2:
                ni = N[i]
                for u in ni:
                    for v in ni:
                        if u < v:
                            covered_kk.add((u, v))
        
        # All K-K edges
        all_kk = set()
        for u in range(k):
            for v in range(u+1, k):
                all_kk.add((u, v))
        
        uncovered_kk = all_kk - covered_kk
        
        # Build I-cliques
        cliques = []
        for i in range(k, n):
            if N[i]:
                clique = list(N[i]) + [i]
                cliques.append(clique)
        
        # Now cover uncovered_kk with K-only cliques
        if uncovered_kk:
            # Greedy: repeatedly find the largest set of vertices whose induced subgraph 
            # contains the most uncovered edges
            remaining = set(uncovered_kk)
            while remaining:
                # For each subset of K, compute how many edges in remaining it covers
                # Greedy: pick the vertex subset that covers the most edges
                # Since k is small, we can try all pairs, triples, etc.
                # Simple greedy: pick the edge with the most co-edges, then expand
                # Even simpler: just pick individual edges? No, that's the seed.
                # Let's do: find the maximum clique in the graph of remaining edges
                # Actually, since the underlying graph is complete, any subset is a clique.
                # So we want to find a subset S of K such that the number of edges in remaining 
                # with both endpoints in S is maximized.
                
                # Greedy heuristic: start with the vertex of highest degree in remaining,
                # then add vertices that add the most edges
                deg = {}
                for u, v in remaining:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                
                if not deg:
                    break
                
                # Start with the highest degree vertex
                start = max(deg.keys(), key=lambda x: deg[x])
                S = {start}
                
                # Greedily add vertices
                while True:
                    best_v = None
                    best_gain = 0
                    for v in range(k):
                        if v in S:
                            continue
                        gain = 0
                        for u in S:
                            if (min(u,v), max(u,v)) in remaining:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    if best_v is None or best_gain == 0:
                        break
                    S.add(best_v)
                
                # Add this clique
                clique = sorted(S)
                cliques.append(clique)
                
                # Remove covered edges
                for u in S:
                    for v in S:
                        if u < v:
                            remaining.discard((u, v))
        
        return cliques

    # Solve all three instances
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            # Return what we have with fallback
            break
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    # If we ran out of time, fill remaining with seed solution
    while len(results) < 3:
        idx = len(results)
        specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
        n, k, p, seed = specs[idx]
        edges = instance(n, k, p, seed)
        results.append([list(e) for e in edges])
    
    # Ensure types are correct
    final = []
    for cliques in results:
        final.append([list(c) for c in cliques])
    
    return final

# EVOLVE-BLOCK-END
