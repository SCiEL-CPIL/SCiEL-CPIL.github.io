"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate maximal cliques using Bron-Kerbosch with pivot.
        adj: list of sets of neighbors.
        Returns list of cliques (frozensets of vertices) with size >= 2.
        """
        cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u from P U X with max |P \cap N(u)|
            pivot_candidates = P.union(X)
            if not pivot_candidates:
                return
            
            pivot = -1
            max_inter = -1
            for u in pivot_candidates:
                inter = len(P.intersection(adj[u]))
                if inter > max_inter:
                    max_inter = inter
                    pivot = u
            
            # Iterate over P \ N(pivot)
            P_minus_Np = P.difference(adj[pivot])
            
            for v in list(P_minus_Np):
                new_R = R.union([v])
                new_P = P.intersection(adj[v])
                new_X = X.intersection(adj[v])
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        expand(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def solve_hitting_set(n, cliques, time_limit):
        """
        Find minimum set S of vertices hitting all cliques in 'cliques'.
        Uses greedy initialization + local search (remove, swap).
        """
        if not cliques:
            return []
        
        # Precompute: for each vertex, which cliques does it hit?
        vertex_to_cliques = {i: [] for i in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Greedy Initialization
        uncovered = set(range(len(cliques)))
        cover = []
        
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                count = 0
                for idx in vertex_to_cliques[v]:
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            cover.append(best_v)
            for idx in vertex_to_cliques[best_v]:
                if idx in uncovered:
                    uncovered.discard(idx)
        
        best_cover = list(cover)
        best_size = len(cover)
        
        # Local Search
        cover_set = set(best_cover)
        
        def update_counts(cover_set):
            counts = [0] * len(cliques)
            for v in cover_set:
                for idx in vertex_to_cliques[v]:
                    counts[idx] += 1
            return counts

        counts = update_counts(cover_set)
        
        improved = True
        while improved:
            improved = False
            
            # Try to remove
            for v in list(cover_set):
                can_remove = True
                for idx in vertex_to_cliques[v]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    cover_set.remove(v)
                    for idx in vertex_to_cliques[v]:
                        counts[idx] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps
            found_swap = False
            for v in list(cover_set):
                for u in range(n):
                    if u in cover_set:
                        continue
                    
                    # Check validity of swap
                    valid = True
                    for idx in vertex_to_cliques[v]:
                        if counts[idx] == 1:
                            if u not in cliques[idx]:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # Simulate swap
                    new_counts = counts[:]
                    for idx in vertex_to_cliques[v]:
                        new_counts[idx] -= 1
                    for idx in vertex_to_cliques[u]:
                        new_counts[idx] += 1
                    
                    # Check if any other vertex w (w != v) can be removed
                    potential_improvement = False
                    for w in cover_set:
                        if w == v:
                            continue
                        can_remove_w = True
                        for idx in vertex_to_cliques[w]:
                            if new_counts[idx] == 1:
                                can_remove_w = False
                                break
                        if can_remove_w:
                            potential_improvement = True
                            break
                    
                    if potential_improvement:
                        # Execute swap
                        cover_set.remove(v)
                        cover_set.add(u)
                        for idx in vertex_to_cliques[v]:
                            counts[idx] -= 1
                        for idx in vertex_to_cliques[u]:
                            counts[idx] += 1
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if found_swap:
                improved = True

        return list(cover_set)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append([])
            continue
            
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = bron_kerbosch(adj, n)
        
        # If no cliques, return empty
        if not cliques:
            out.append([])
            continue
        
        # Solve hitting set
        result = solve_hitting_set(n, cliques, deadline - time.time())
        out.append(sorted(result))
    
    return out
# EVOLVE-BLOCK-END
