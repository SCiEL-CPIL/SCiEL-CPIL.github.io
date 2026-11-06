import time
from itertools import chain
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds margin
    
    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(R, P, X, graph, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(tuple(sorted(R)))
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        pivot = None
        max_deg = -1
        for u in chain(P, X):
            deg = sum(1 for v in P if v in graph[u])
            if deg > max_deg:
                max_deg = deg
                pivot = u
        
        if pivot is None:
            # P and X are empty, handled above, but just in case
            return
            
        # Iterate over candidates in P that are not neighbors of pivot
        candidates = [v for v in P if v not in graph[pivot]]
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & graph[v]
            new_X = X & graph[v]
            bron_kerbosch(new_R, new_P, new_X, graph, cliques)
            P.remove(v)
            X.add(v)

    def find_maximal_cliques(n, edges):
        # Build adjacency list
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        cliques = []
        bron_kerbosch(set(), set(range(n)), set(), graph, cliques)
        return cliques

    def min_hitting_set(n, cliques):
        if not cliques:
            return set()
        
        # Greedy Set Cover
        # State: set of uncovered cliques, set of chosen vertices
        uncovered = set(map(tuple, cliques))
        chosen = set()
        
        # Precompute for each vertex, which cliques it hits
        # To be efficient, we can just check membership, but n is small.
        # Let's keep a list of uncovered cliques for faster removal?
        # Since n is small (<=40) and number of cliques is likely small in random graphs,
        # simple set operations are fine.
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Find vertex that hits the most uncovered cliques
            best_v = -1
            max_hits = 0
            
            for v in range(n):
                if v in chosen:
                    continue
                hits = 0
                for c in uncovered:
                    if v in c:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_v = v
            
            if best_v == -1:
                # Should not happen if uncovered is not empty and cliques have vertices
                break
                
            chosen.add(best_v)
            
            # Remove all cliques containing best_v from uncovered
            uncovered = {c for c in uncovered if best_v not in c}
            
        # Local Search: Remove redundant vertices
        # A vertex v is redundant if for every clique c containing v, there is another vertex u in chosen (u != v) such that u in c.
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for v in list(chosen):
                # Check if v is redundant
                redundant = True
                for c in cliques:
                    if v in c:
                        # Is there another chosen vertex in this clique?
                        other_in_c = False
                        for u in chosen:
                            if u != v and u in c:
                                other_in_c = True
                                break
                        if not other_in_c:
                            redundant = False
                            break
                if redundant:
                    chosen.remove(v)
                    changed = True
                    
        return chosen

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices to ensure validity if we run out of time
            out.append(list(range(n)))
            continue
            
        edges = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        
        # Filter cliques of size >= 2 (Bron-Kerbosch already does this in the check, but double check)
        cliques = [c for c in cliques if len(c) >= 2]
        
        hit_set = min_hitting_set(n, cliques)
        
        # Ensure validity: if hit_set is empty but there are cliques, something went wrong.
        # Fallback to greedy or all vertices if it fails.
        if cliques and not hit_set:
             hit_set = set(range(n))
             
        # Final verification and cleanup
        # If any clique is not hit, add a vertex from it
        for c in cliques:
            if not (set(c) & hit_set):
                hit_set.add(c[0])
                
        out.append(sorted(hit_set))
        
    return out
