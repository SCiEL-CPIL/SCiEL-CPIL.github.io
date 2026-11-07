import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        # Build adjacency
        # K: 0..k-1, I: k..n-1
        adj_K = [set() for _ in range(k)]
        adj_I = [set() for _ in range(n - k)] # adj_I[i] is set of K neighbors of vertex k+i
        
        # Generate cross edges
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)
                    adj_K[u].add(v) # Not strictly needed for K-K edges, but good for consistency if we needed it.
        
        # Note: K is a clique, so all K-K edges exist.
        
        used_K_edges = set()
        cliques = []
        
        # Process I vertices
        # Randomize order to avoid bias?
        i_indices = list(range(n - k))
        rng.shuffle(i_indices)
        
        for i in i_indices:
            v = k + i
            U = set(adj_I[i])
            while U:
                # Find largest subset S of U such that all pairs in S are not in used_K_edges
                # Greedy: try to add vertices one by one to S
                # Or try all subsets? 2^|U| is too big if |U| is large.
                # |U| <= k = 30.
                # Greedy construction of S:
                S = set()
                # Sort U by some heuristic? 
                # We want to maximize |S|.
                # Try to add each u in U if it doesn't conflict with current S
                # This is not guaranteed to be max, but fast.
                # Let's try a few restarts or just one greedy pass.
                
                # Better: 
                # Start with S = empty.
                # Candidate vertices = U.
                # While Candidate:
                #   Pick u in Candidate
                #   Check if all (u, w) for w in S are not in used_K_edges
                #   If yes, add u to S, remove u from Candidate
                #   If no, remove u from Candidate (try next)
                # This finds an independent set in the "conflict graph" restricted to U?
                # Conflict graph on U: edge (u, w) exists if (u, w) in used_K_edges.
                # We want Max Independent Set in this conflict graph?
                # No, we want a clique in the "available" graph.
                # Available graph on U: edge (u, w) exists if (u, w) NOT in used_K_edges.
                # We want Max Clique in Available[U].
                
                # Heuristic for Max Clique:
                # Greedy: pick vertex with max degree in Available[U], add to S, restrict to neighbors.
                
                S = greedy_max_clique(U, used_K_edges)
                
                if S:
                    # Add clique {v} U S
                    clique = [v] + list(S)
                    cliques.append(clique)
                    # Update used_K_edges
                    for u, w in combinations(S, 2):
                        used_K_edges.add((min(u, w), max(u, w)))
                    U -= S
                else:
                    # Should not happen, as single vertex is always valid
                    # But if S is empty, we must pick at least one
                    # This implies the greedy failed to find even 1?
                    # If U is not empty, we can always pick 1.
                    u = next(iter(U))
                    clique = [v, u]
                    cliques.append(clique)
                    U.remove(u)
        
        # Cover remaining K edges
        # Build list of remaining edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_K_edges:
                    rem_edges.append((u, w))
        
        if rem_edges:
            # Graph G_rem on K
            # Find min edge clique cover
            # Greedy: repeatedly find max clique in G_rem, add it, remove edges.
            # Represent G_rem by adjacency set
            adj_rem = [set() for _ in range(k)]
            for u, w in rem_edges:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
            
            remaining = set(rem_edges)
            
            while remaining:
                # Find max clique in G_rem
                # Vertices involved in remaining edges
                verts = set()
                for u, w in remaining:
                    verts.add(u)
                    verts.add(w)
                
                C = greedy_max_clique(verts, set() - remaining) # Wait, greedy_max_clique uses used_K_edges logic.
                # Let's write a specific max clique finder for the remaining graph.
                
                C = find_max_clique_in_graph(verts, adj_rem)
                
                if not C:
                     # Should not happen
                     break
                     
                clique = list(C)
                cliques.append(clique)
                
                # Remove edges in C from remaining
                for u, w in combinations(C, 2):
                    if (min(u, w), max(u, w)) in remaining:
                        remaining.remove((min(u, w), max(u, w)))
                
                # Update adj_rem? Not strictly needed if we just filter remaining.
                # But find_max_clique_in_graph uses adj_rem.
                # We should update adj_rem to remove edges in C.
                for u in C:
                    for w in C:
                        if u != w:
                            if w in adj_rem[u]:
                                adj_rem[u].remove(w)
                                adj_rem[w].remove(u)

        return cliques

    def greedy_max_clique(verts, used_edges):
        # verts: set of candidate vertices
        # used_edges: set of edges that are ALREADY USED (conflict)
        # We want a clique C in verts such that no edge in C is in used_edges.
        # This is Max Independent Set in the graph formed by used_edges restricted to verts.
        # Or Max Clique in the graph formed by (complete - used_edges) restricted to verts.
        
        # Heuristic:
        # 1. Filter verts to those that have no conflict with current C?
        # Let's just do a simple greedy:
        C = set()
        candidates = set(verts)
        
        # Order candidates by degree in the "available" graph?
        # Available graph: u-v edge if (u,v) NOT in used_edges.
        
        while candidates:
            # Pick vertex with max available degree in candidates
            best_v = None
            max_deg = -1
            for v in candidates:
                deg = 0
                for u in candidates:
                    if u == v: continue
                    if (min(u, v), max(u, v)) not in used_edges:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            # Check if best_v can be added to C
            # It must be compatible with all in C
            can_add = True
            for u in C:
                if (min(u, best_v), max(u, best_v)) in used_edges:
                    can_add = False
                    break
            
            if can_add:
                C.add(best_v)
                candidates.remove(best_v)
            else:
                # If we can't add it, maybe we should skip it?
                # This greedy is flawed.
                # If best_v conflicts with C, we can't add it.
                # But we picked it because it has high degree in candidates.
                # If it conflicts with C, it's useless for the current C.
                # So we should remove it from candidates?
                candidates.remove(best_v)
                
        return C

    def find_max_clique_in_graph(verts, adj):
        # adj: list of sets
        # verts: set of active vertices
        # Greedy Max Clique
        C = set()
        candidates = set(verts)
        
        while candidates:
            best_v = None
            max_deg = -1
            for v in candidates:
                deg = len(adj[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v is None:
                break
            
            # Check compatibility with C
            can_add = True
            for u in C:
                if u not in adj[best_v]:
                    can_add = False
                    break
            
            if can_add:
                C.add(best_v)
                # Restrict candidates to neighbors of best_v
                candidates = candidates & adj[best_v]
            else:
                candidates.remove(best_v)
                
        return C

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
