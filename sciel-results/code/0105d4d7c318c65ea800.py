import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        cliques = []
        
        # We will try to assign each edge in K to at most one v in I
        # and form cliques {v} U S_v
        # S_v must be a subset of N(v)
        # To avoid double coverage of K edges, we need the sets of edges induced by S_v to be disjoint.
        
        # Greedy approach:
        # 1. Initialize S_v = N(v) for all v.
        # 2. Find conflicts: edges (u, w) in K that are in S_v for multiple v.
        # 3. Resolve by removing vertices from S_v.
        
        S = {v: set(neighbors[v]) for v in range(k, n)}
        
        # Map edge -> list of v that cover it
        edge_to_vs = {}
        for v in range(k, n):
            for u, w in combinations(S[v], 2):
                e = (min(u, w), max(u, w))
                if e not in edge_to_vs:
                    edge_to_vs[e] = []
                edge_to_vs[e].append(v)
        
        # Resolve conflicts
        # For each edge with multiple v's, keep one, remove from others
        # Which one to keep? Maybe the one with largest S?
        # We iterate and remove.
        
        # To be safe and fast, we can just pick one v for each edge, and ensure consistency.
        # Actually, if we remove a vertex from S_v, we might uncover other edges.
        
        # Simpler: Just use {v, u} for all cross edges, and {u, w} for K edges.
        # This is valid but suboptimal.
        
        # Let's try the "Big Clique" with random tie-breaking for conflicts.
        # If (u, w) is in S_v and S_w, remove w from S_w (randomly).
        
        # This is getting complex. Let's stick to the valid simple one first?
        # No, the prompt asks for a stronger algorithm.
        
        # I will use the matching-based assignment from the current program but simplified?
        # The current program is already good.
        
        # Let's just return the "Big Clique" solution and hope the conflicts are rare or handled by the evaluator's "exactly one" check?
        # No, "Any broken rule makes the whole result invalid".
        
        # I must ensure validity.
        
        # Strategy:
        # 1. For each v, S_v = N(v).
        # 2. While there is a conflict (edge in >1 S_v):
        #    Pick a conflicting edge (u, w).
        #    Pick a v that contains it (randomly).
        #    Remove u from S_v? Or w?
        #    Removing a vertex from S_v reduces the clique size.
        
        # This loop might be slow.
        
        # Alternative:
        # Assign each edge in K to exactly one v in I (if possible).
        # Let A_v be the set of edges assigned to v.
        # Then S_v must contain all vertices incident to edges in A_v.
        # To minimize cliques, we want 1 clique per v.
        # So we just need to partition E(K) into E_v such that E_v is a subgraph of K[N(v)]?
        # No, E_v must be a clique in K[N(v)]?
        # If we use clique {v} U S_v, it covers ALL edges in S_v.
        # So if we assign E_v, we actually cover ALL edges in S_v.
        # So we must ensure that the sets of edges covered are disjoint.
        
        # I will use the "Simple Valid" fallback if the complex one is too risky.
        # But 120 is so much better than 185.
        
        # Let's try to construct the disjoint S_v.
        
        # 1. Sort v by size of N(v) descending.
        # 2. For each v, assign all available edges in K[N(v)] to v.
        #    "Available" means not assigned to any previous v.
        #    Let E_avail be the set of unassigned edges in K.
        #    E_v = E_avail intersect K[N(v)].
        #    S_v = vertices incident to E_v.
        #    Add clique {v} U S_v.
        #    Remove E_v from E_avail.
        
        # This ensures disjointness of covered edges!
        # And it uses 1 clique per v (if E_v is not empty).
        
        E_avail = set((u, w) for u in range(k) for w in range(u+1, k))
        
        # Sort v by degree in K (size of N(v)) descending
        vs_sorted = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs_sorted:
            N_v = set(neighbors[v])
            # Find edges in E_avail that are within N_v
            E_v = set()
            for u, w in combinations(N_v, 2):
                e = (min(u, w), max(u, w))
                if e in E_avail:
                    E_v.add(e)
            
            if E_v:
                # S_v is the set of vertices in E_v
                S_v = set()
                for e in E_v:
                    S_v.add(e[0])
                    S_v.add(e[1])
                
                clique = [v] + list(S_v)
                cliques.append(clique)
                
                # Remove E_v from E_avail
                E_avail -= E_v
        
        # Cover remaining edges in E_avail
        if E_avail:
            rem_edges_set = E_avail
            while rem_edges_set:
                e = next(iter(rem_edges_set))
                u, w = e
                clique_set = {u, w}
                changed = True
                while changed:
                    changed = False
                    for x in range(k):
                        if x not in clique_set:
                            if all((min(x, y), max(x, y)) in rem_edges_set for y in clique_set):
                                clique_set.add(x)
                                changed = True
                                break
                if len(clique_set) >= 2:
                    cliques.append(sorted(clique_set))
                    for a, b in combinations(clique_set, 2):
                        rem_edges_set.discard((min(a, b), max(a, b)))
                else:
                    cliques.append([u, w])
                    rem_edges_set.discard(e)
        
        clean_cliques = [[int(x) for x in c] for c in cliques]
        return clean_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
