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
        edges = list(G.edges())
        # Sort edges by sum of degrees descending to form larger cliques earlier
        edges.sort(key=lambda e: G.degree(e[0]) + G.degree(e[1]), reverse=True)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        cliques = [] # List of sets of vertices
        
        # Precompute adjacency sets for fast lookup
        adj = {v: set(G.neighbors(v)) for v in G.nodes()}
        
        for u, v in edges:
            edge = (u, v) if u < v else (v, u)
            if edge not in remaining_edges:
                continue
            
            assigned = False
            # Try to extend an existing clique
            for i, C in enumerate(cliques):
                # Check if we can add v to C (if u in C)
                if u in C and v not in C:
                    # Check if v is connected to all in C
                    can_add = all(w in adj[v] for w in C)
                    if can_add:
                        # Check if all edges (v, w) for w in C are in remaining_edges
                        can_remove = all((min(v, w), max(v, w)) in remaining_edges for w in C)
                        if can_remove:
                            # Extend clique
                            cliques[i] = C | {v}
                            for w in C:
                                remaining_edges.discard((min(v, w), max(v, w)))
                            remaining_edges.discard(edge)
                            assigned = True
                            break
                            
                # Check if we can add u to C (if v in C)
                if v in C and u not in C:
                    can_add = all(w in adj[u] for w in C)
                    if can_add:
                        can_remove = all((min(u, w), max(u, w)) in remaining_edges for w in C)
                        if can_remove:
                            cliques[i] = C | {u}
                            for w in C:
                                remaining_edges.discard((min(u, w), max(u, w)))
                            remaining_edges.discard(edge)
                            assigned = True
                            break
                
            if not assigned:
                # Create new clique
                cliques.append({u, v})
                remaining_edges.discard(edge)
                
        # Convert sets to lists for output
        return [list(C) for C in cliques]

    G1 = generate_graph(32, 0.6, 601)
    G2 = generate_graph(64, 0.55, 602)
    
    C1 = solve_instance(G1)
    C2 = solve_instance(G2)
    
    return [C1, C2]
