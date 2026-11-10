import random
import networkx as nx

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    G.add_edge(u, v)
        return G

    def solve_instance(G):
        remaining_edges = set(G.edges())
        cliques = []
        
        # Pre-compute adjacency sets for quick lookup
        # We need to dynamically update degrees or just use the set of remaining edges
        
        while remaining_edges:
            # Find a vertex with max degree in the remaining graph
            # To do this efficiently, we might just pick a vertex from an edge
            # Let's pick the first edge (u, v) and check degrees of u and v in remaining graph
            
            # Actually, iterating all vertices to find max degree is O(n^2) per step.
            # Number of steps is at most E/2 (if triangles) or E (if edges).
            # E ~ 1000. n=64. 64*1000 = 64000 operations. Very fast.
            
            max_deg = -1
            best_v = -1
            for v in G.nodes():
                # Count neighbors in remaining edges
                deg = 0
                for u in G.neighbors(v):
                    if (min(v, u), max(v, u)) in remaining_edges:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            # Find neighbors of best_v in remaining graph
            neighbors = []
            for u in G.neighbors(best_v):
                if (min(best_v, u), max(best_v, u)) in remaining_edges:
                    neighbors.append(u)
            
            if not neighbors:
                # Isolated vertex in remaining graph? Should not happen if we pick from edges
                # But if max_deg is 0, we break.
                break
            
            # Find a large clique in the subgraph induced by neighbors
            # Greedy: sort by degree in the induced subgraph
            # Induced subgraph edges: (u, w) in remaining_edges and u, w in neighbors
            
            # Calculate degrees in induced subgraph
            degs = {u: 0 for u in neighbors}
            for u in neighbors:
                for w in neighbors:
                    if u < w and (u, w) in remaining_edges:
                        degs[u] += 1
                        degs[w] += 1
            
            # Sort neighbors by degree descending
            sorted_neighbors = sorted(neighbors, key=lambda x: degs[x], reverse=True)
            
            clique = [best_v]
            for u in sorted_neighbors:
                # Check if u connects to all in clique
                is_clique = True
                for v in clique:
                    if (min(u, v), max(u, v)) not in remaining_edges:
                        is_clique = False
                        break
                if is_clique:
                    clique.append(u)
            
            # Remove edges of this clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    remaining_edges.discard((min(u, v), max(u, v)))
            
            cliques.append(clique)
            
        return cliques

    G1 = generate_graph(32, 0.6, 601)
    G2 = generate_graph(64, 0.55, 602)
    
    C1 = solve_instance(G1)
    C2 = solve_instance(G2)
    
    return [C1, C2]
