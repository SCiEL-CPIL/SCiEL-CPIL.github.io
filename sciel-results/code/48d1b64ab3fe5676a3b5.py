import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        # Adjacency matrix/list for fast access
        adj = [[False] * n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append((u, v))
        return adj, edges

    def solve_instance(n, p, seed):
        adj, edges = get_graph(n, p, seed)
        uncovered = set(edges)
        cliques = []
        
        if not uncovered:
            return []

        # Precompute neighbors for each vertex for faster intersection
        neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        while uncovered:
            if time.time() - start_time > timeout:
                # Fallback: return remaining edges as size-2 cliques
                for u, v in uncovered:
                    cliques.append([u, v])
                break

            best_clique = None
            best_uncovered_count = 0
            
            # Sample some edges to find a good clique
            # If uncovered is small, check all. Otherwise, sample.
            if len(uncovered) < 50:
                candidates = list(uncovered)
            else:
                candidates = random.sample(list(uncovered), 50)
            
            for u, v in candidates:
                # Greedily expand clique starting with {u, v}
                C = {u, v}
                # Candidates must be neighbors of both u and v
                cand_set = neighbors[u].intersection(neighbors[v])
                
                while cand_set:
                    # Pick w in cand_set that maximizes uncovered edges to C
                    best_w = None
                    max_gain = 0
                    for w in cand_set:
                        # Count uncovered edges between w and C
                        gain = 0
                        for c in C:
                            if (min(w, c), max(w, c)) in uncovered:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_w = w
                    
                    if best_w is None or max_gain == 0:
                        break
                    
                    C.add(best_w)
                    # Update candidates: must be neighbors of best_w too
                    cand_set = cand_set.intersection(neighbors[best_w])
                
                # Calculate total uncovered edges in C
                count = 0
                for e in combinations(sorted(C), 2):
                    if e in uncovered:
                        count += 1
                
                if count > best_uncovered_count:
                    best_uncovered_count = count
                    best_clique = C
            
            if best_clique is None or best_uncovered_count == 0:
                # No expansion found or no gain, just take one edge
                u, v = next(iter(uncovered))
                best_clique = {u, v}
                best_uncovered_count = 1
            
            # Add clique to result
            cliques.append(sorted(best_clique))
            
            # Remove covered edges
            for e in combinations(sorted(best_clique), 2):
                uncovered.discard(e)
        
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
