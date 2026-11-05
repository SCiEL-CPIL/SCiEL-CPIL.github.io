import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        """Generate edges of a random graph with given parameters."""
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_adjacency(n, edges_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def expand_clique(start_vertices, adj, n):
        """Greedy expansion of a clique."""
        C = set(start_vertices)
        P = set(range(n))
        for v in C:
            P &= adj[v]
        P -= C
        
        while P:
            # Pick vertex with max degree in P
            v = max(P, key=lambda x: len(adj[x] & P))
            C.add(v)
            P &= adj[v]
            P -= C
            
        return C

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        if not edges_set:
            return []
        
        adj = get_adjacency(n, edges_set)
        remaining_edges = set(edges_set)
        partition = []
        
        # Use a specific RNG for the heuristic to ensure reproducibility of the heuristic itself
        heuristic_rng = random.Random(seed + 1000)
        
        while remaining_edges:
            if time.time() - start_time > DEADLINE:
                # Fallback
                for e in remaining_edges:
                    partition.append(list(e))
                break
            
            best_clique = None
            best_covered_edges = -1
            best_clique_edge_set = None
            
            # Determine number of samples
            num_remaining = len(remaining_edges)
            if num_remaining > 500:
                samples = 50
            elif num_remaining > 100:
                samples = 20
            else:
                samples = num_remaining
            
            if samples > 0:
                sample_edges = heuristic_rng.sample(list(remaining_edges), samples)
                
                for e in sample_edges:
                    u, v = e
                    # Expand (u, v) to a maximal clique
                    clique = expand_clique({u, v}, adj, n)
                    
                    # Count how many edges of this clique are in remaining_edges
                    covered = 0
                    clique_verts = sorted(clique)
                    clique_edges = set()
                    for i in range(len(clique_verts)):
                        for j in range(i+1, len(clique_verts)):
                            ce = (clique_verts[i], clique_verts[j])
                            if ce in remaining_edges:
                                covered += 1
                                clique_edges.add(ce)
                    
                    if covered > best_covered_edges:
                        best_covered_edges = covered
                        best_clique = clique
                        best_clique_edge_set = clique_edges
            
            # If we didn't find a clique with > 1 edge
            if best_clique is None or best_covered_edges <= 1:
                # Pick a random edge and make it a 2-clique
                e = heuristic_rng.choice(list(remaining_edges))
                partition.append(list(e))
                remaining_edges.remove(e)
            else:
                partition.append(sorted(best_clique))
                remaining_edges -= best_clique_edge_set
        
        return partition

    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > DEADLINE:
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]
    
    result2 = solve_instance(64, 0.55, 602)
    return [result1, result2]
