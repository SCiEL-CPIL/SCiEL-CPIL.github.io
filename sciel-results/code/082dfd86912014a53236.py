import random
from collections import defaultdict

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P union X to minimize branching
        # We want u with maximum neighbors in P
        if P:
            # Find vertex in P with max degree within P
            # Or just pick any vertex in P union X
            # Standard pivot: u in P ∪ X
            u_candidates = list(P) + list(X)
            if not u_candidates:
                return
            # Pick u that maximizes |N(u) ∩ P|
            max_deg = -1
            pivot = None
            for u in u_candidates:
                deg = len(adj[u].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # Iterate over P \ N(pivot)
            for v in list(P):
                if v in adj[pivot]:
                    continue
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)
        else:
             # If P is empty, check X
             if not X:
                 if len(R) >= 2:
                     cliques.append(frozenset(R))
                 return
             # If P is empty but X is not, we are in a non-maximal branch, do nothing?
             # Actually, Bron-Kerbosch with pivot:
             # If P is empty, we found a maximal clique if X is also empty.
             # If P is empty and X is not empty, R is not maximal (it's subset of a clique in X).
             # So we only add if P and X are empty.
             pass

    def find_maximal_cliques(n, adj):
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        return cliques

    def greedy_transversal(cliques):
        if not cliques:
            return []
        
        remaining = list(cliques)
        T = set()
        
        while remaining:
            counts = defaultdict(int)
            for c in remaining:
                for v in c:
                    counts[v] += 1
            
            # Pick vertex with max count
            best_v = max(counts, key=counts.get)
            T.add(best_v)
            
            # Remove all cliques containing best_v
            remaining = [c for c in remaining if best_v not in c]
        
        # Post-processing: Remove redundant vertices
        # A vertex v in T is redundant if removing it still hits all original cliques
        # We need to check against the original list of cliques
        final_T = set(T)
        for v in list(final_T):
            candidate_T = final_T - {v}
            # Check if candidate_T hits all original cliques
            if all(any(x in candidate_T for x in c) for c in cliques):
                final_T = candidate_T
        
        return sorted(final_T)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges, adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, adj)
        # Filter cliques with size >= 2 (Bron-Kerbosch might return singletons if isolated vertices? 
        # No, Bron-Kerbosch finds maximal cliques. An isolated vertex is a maximal clique of size 1.
        # The task requires meeting every maximal clique with at least two vertices.
        # So we filter out cliques of size 1.
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        transversal = greedy_transversal(relevant_cliques)
        results.append(transversal)
        
    return results
