import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Adjacency list for fast neighbor lookup
        # We will update this as we remove edges
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        while edges:
            if time.time() > deadline:
                # Fallback: return current cliques and remaining edges as singletons
                # This ensures validity but might be suboptimal
                for e in edges:
                    cliques.append(list(e))
                break
            
            # Find vertex with max degree in remaining graph
            max_deg = -1
            u = -1
            for node in list(adj.keys()):
                deg = len(adj[node])
                if deg > max_deg:
                    max_deg = deg
                    u = node
            
            if u == -1 or max_deg == 0:
                break

            # Build a maximal clique starting from u
            clique = [u]
            # Candidates must be neighbors of u
            candidates = set(adj[u])
            
            # We want to add vertices to clique that are connected to all existing clique members
            # and have high connectivity.
            while candidates:
                # Pick candidate v with max degree in the remaining graph
                best_v = -1
                best_deg = -1
                for v in candidates:
                    deg = len(adj[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                # Check if best_v is connected to all in clique
                # Since best_v is in candidates, it is connected to u.
                # We need to check connection to other clique members.
                # Note: We haven't removed edges yet, so adj reflects the state before this clique extraction.
                # However, as we add vertices to clique, we must ensure the new vertex connects to ALL previous ones.
                # The `candidates` set is maintained such that all vertices in it are connected to all vertices in `clique`.
                # Initially candidates = neighbors of u. All connected to u.
                # When we add v to clique, new candidates must be neighbors of v AND in current candidates.
                
                clique.append(best_v)
                
                # Update candidates: must be neighbors of best_v
                # Intersection of current candidates and neighbors of best_v
                candidates = candidates.intersection(adj[best_v])
            
            # Remove edges of the clique from the graph
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u_c = clique[i]
                    v_c = clique[j]
                    if u_c > v_c:
                        u_c, v_c = v_c, u_c
                    edge = (u_c, v_c)
                    if edge in edges:
                        edges.remove(edge)
                        adj[u_c].remove(v_c)
                        adj[v_c].remove(u_c)
            
            cliques.append(clique)
        
        return cliques

    # Instance 1
    cliques1 = solve_instance(32, 0.6, 601)
    
    # Instance 2
    cliques2 = solve_instance(64, 0.55, 602)
    
    return [cliques1, cliques2]
