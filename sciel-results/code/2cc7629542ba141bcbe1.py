import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = build_adj(edges, n)
        
        if not edges:
            return []
        
        # Work with a mutable copy of edges
        remaining = set(edges)
        remaining_adj = [set(s) for s in adj]
        
        selected_cliques = []
        
        # Precompute initial "clique potential" for each edge:
        # number of common neighbors (potential to grow into larger clique)
        # We'll process edges greedily, growing each into a clique
        
        def grow_clique(seed_edge, remaining, remaining_adj):
            """Grow a seed edge into a clique in the remaining graph."""
            u, v = seed_edge
            clique = [u, v]
            
            # Find all vertices connected to both u and v in remaining graph
            candidates = remaining_adj[u] & remaining_adj[v]
            
            # Sort candidates by degree in remaining graph (higher degree = more potential)
            candidates = sorted(candidates, key=lambda x: -len(remaining_adj[x]))
            
            for w in candidates:
                # Check if w connects to all vertices in current clique
                ok = True
                for c in clique:
                    if w not in remaining_adj[c]:
                        ok = False
                        break
                if ok:
                    clique.append(w)
            
            return clique

        def edge_potential(e, remaining_adj):
            """Score an edge by how well it can be grown into a clique."""
            u, v = e
            # Number of common neighbors = potential clique size - 2
            common = len(remaining_adj[u] & remaining_adj[v])
            # Also factor in degrees of u and v
            return common * 10 + len(remaining_adj[u]) + len(remaining_adj[v])
        
        # Main loop: repeatedly pick the best edge, grow it into a clique
        while remaining:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find the edge with highest potential
            # For efficiency, we can sample or use a heap, but n is small enough
            best_edge = None
            best_score = -1
            for e in remaining:
                score = edge_potential(e, remaining_adj)
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break
            
            # Grow this edge into a clique
            clique = grow_clique(best_edge, remaining, remaining_adj)
            
            if len(clique) < 2:
                # Shouldn't happen, but safety
                clique = list(best_edge)
            
            selected_cliques.append(clique)
            
            # Remove all edges within this clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    ce = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    remaining.discard(ce)
                    remaining_adj[clique[i]].discard(clique[j])
                    remaining_adj[clique[j]].discard(clique[i])
        
        return selected_cliques

    # Solve both instances
    t1 = time.time()
    budget1 = min(65, DEADLINE - (t1 - start_time) - 5)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(65, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
