import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 115  # Leave 5s margin for 120s limit

    def get_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_graph(n, p, seed)
        if not edges:
            return []
        
        # Initialize adjacency structures
        neighbors = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
            
        cliques = []
        
        while edge_set:
            # Check time limit
            if time.time() > start_time + DEADLINE:
                # Fallback: add all remaining edges as size-2 cliques
                for u, v in edge_set:
                    cliques.append([u, v])
                break
            
            # Find vertex with highest degree in remaining graph
            max_deg = -1
            best_v = -1
            for v in range(n):
                deg = len(neighbors[v])
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1 or max_deg == 0:
                # No edges left or isolated vertices (should be handled by edge_set check)
                break
            
            # Candidates for the clique are neighbors of best_v
            # We want to form a clique K in N(best_v)
            # Sort candidates by their degree in the remaining graph (descending)
            # This heuristic helps find larger cliques
            candidates = sorted(neighbors[best_v], key=lambda x: len(neighbors[x]), reverse=True)
            
            K = []
            for u in candidates:
                # Check if u is connected to all vertices in K
                # Since K is a clique, we just check against all members of K
                is_connected = True
                for k in K:
                    if k not in neighbors[u]:
                        is_connected = False
                        break
                if is_connected:
                    K.append(u)
            
            # The clique is {best_v} U K
            C = [best_v] + K
            
            if len(C) >= 2:
                cliques.append(C)
                
                # Remove all edges within C from the remaining graph
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        if a > b:
                            a, b = b, a
                        edge_to_remove = (a, b)
                        if edge_to_remove in edge_set:
                            edge_set.remove(edge_to_remove)
                            neighbors[a].remove(b)
                            neighbors[b].remove(a)
            else:
                # Fallback: if K is empty, we only have best_v, which is not a clique of edges.
                # This shouldn't happen if max_deg > 0, as neighbors[best_v] is not empty.
                # But if it does, we pick an arbitrary neighbor to form a size-2 clique.
                if neighbors[best_v]:
                    u = next(iter(neighbors[best_v]))
                    cliques.append([best_v, u])
                    a, b = best_v, u
                    if a > b:
                        a, b = b, a
                    edge_to_remove = (a, b)
                    if edge_to_remove in edge_set:
                        edge_set.remove(edge_to_remove)
                        neighbors[a].remove(b)
                        neighbors[b].remove(a)
                else:
                    # Isolated vertex, remove from consideration (no edges to remove)
                    break

        return cliques

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
