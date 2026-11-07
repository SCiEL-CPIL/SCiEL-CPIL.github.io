import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # C vertices are 0..k-1, I vertices are k..n-1
        C = list(range(k))
        I = list(range(k, n))
        
        # Uncovered edges in C
        uncovered_C_edges = set()
        for u in C:
            for v in C:
                if u < v and (u, v) in edge_set:
                    uncovered_C_edges.add((u, v))
        
        # Uncovered edges between C and I
        uncovered_CI_edges = set()
        for u in C:
            for v in I:
                if (u, v) in edge_set:
                    uncovered_CI_edges.add((u, v))
                    
        # Neighbors of each v in I within C
        N_in_C = {v: set() for v in I}
        for u in C:
            for v in I:
                if (u, v) in edge_set:
                    N_in_C[v].add(u)
        
        # Sort I vertices by degree in C (descending)
        I_sorted = sorted(I, key=lambda v: len(N_in_C[v]), reverse=True)
        
        cliques = []
        
        # Helper to find max clique in a graph defined by a set of vertices and a set of edges
        # Graph is small (subset of C)
        def find_max_clique(vertices, edges_available):
            # vertices: list of int
            # edges_available: set of tuples (u, v) with u < v
            if not vertices:
                return []
            
            # Build adjacency for this subgraph
            adj = {v: set() for v in vertices}
            for u, v in edges_available:
                if u in adj and v in adj:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Greedy max clique: repeatedly pick vertex with max degree in remaining graph
            # This is an approximation, but good for small graphs
            remaining = set(vertices)
            clique = []
            
            while remaining:
                # Find vertex with max degree in remaining
                best_v = None
                max_deg = -1
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                
                clique.append(best_v)
                remaining.remove(best_v)
                # Restrict remaining to neighbors of best_v
                remaining = remaining & adj[best_v]
                
            return clique

        # Process each v in I
        for v in I_sorted:
            if time.time() > deadline:
                break
                
            if not N_in_C[v]:
                continue
                
            # Find max clique in N_in_C[v] using uncovered_C_edges
            # We need to check which edges in N_in_C[v] are still uncovered
            # The candidate vertices are N_in_C[v]
            candidates = list(N_in_C[v])
            
            # Filter candidates: only those where edges to other candidates are uncovered?
            # No, find_max_clique will handle this by checking edges_available.
            # But we must ensure we only use edges that are currently uncovered.
            # So we pass uncovered_C_edges to find_max_clique.
            
            K = find_max_clique(candidates, uncovered_C_edges)
            
            if len(K) >= 2:
                # Form clique K U {v}
                new_clique = K + [v]
                cliques.append(new_clique)
                
                # Remove edges within K from uncovered_C_edges
                for i in range(len(K)):
                    for j in range(i + 1, len(K)):
                        u, w = K[i], K[j]
                        if u > w:
                            u, w = w, u
                        uncovered_C_edges.discard((u, w))
                        
                # Remove edges between K and v from uncovered_CI_edges
                for u in K:
                    uncovered_CI_edges.discard((u, v))
                    # Also check (v, u) if stored that way? 
                    # We stored (u, v) with u in C, v in I. So (u, v) is the key.
        
        # Cover remaining uncovered edges in C
        # Greedy: while there are uncovered edges, pick one, find max clique containing it
        while uncovered_C_edges:
            if time.time() > deadline:
                break
                
            # Pick an arbitrary uncovered edge
            u, v = next(iter(uncovered_C_edges))
            
            # Find max clique in the subgraph of uncovered edges containing u and v
            # Candidates: neighbors of u and v in the uncovered graph
            # Actually, just find max clique in the whole uncovered graph, 
            # but it must contain u and v? 
            # If we just find any max clique, it might not cover (u, v).
            # So we must force (u, v) to be in the clique.
            
            # Intersection of neighbors of u and v in the uncovered graph
            # Neighbors of u in uncovered graph:
            neigh_u = set()
            neigh_v = set()
            for a, b in uncovered_C_edges:
                if a == u:
                    neigh_u.add(b)
                elif b == u:
                    neigh_u.add(a)
                if a == v:
                    neigh_v.add(b)
                elif b == v:
                    neigh_v.add(a)
            
            common_neigh = neigh_u & neigh_v
            
            # Candidates for the clique: {u, v} U common_neigh
            candidates = [u, v] + list(common_neigh)
            
            # Find max clique in this candidate set, restricted to uncovered edges
            # And it MUST contain u and v.
            # We can modify find_max_clique to force vertices, or just 
            # find max clique in the subgraph induced by candidates and 
            # check if it contains u and v. If not, we might have to 
            # fall back to just {u, v}.
            
            # Let's just find max clique in candidates.
            K = find_max_clique(candidates, uncovered_C_edges)
            
            # Ensure u and v are in K. If not, something went wrong or 
            # the max clique found didn't include them.
            # Since u and v are adjacent in uncovered graph, they can be in a clique.
            # But greedy might pick a different large clique.
            # To be safe, if u not in K or v not in K, just use {u, v}.
            if u not in K or v not in K:
                K = [u, v]
            
            cliques.append(K)
            
            # Remove edges within K from uncovered_C_edges
            for i in range(len(K)):
                for j in range(i + 1, len(K)):
                    a, b = K[i], K[j]
                    if a > b:
                        a, b = b, a
                    uncovered_C_edges.discard((a, b))
                    
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return empty lists? No, must be valid.
            # If we run out of time, we can't guarantee validity of partial result.
            # But solve_split is fast.
            pass
        
        cliques = solve_split(*spec)
        result.append(cliques)
        
    return result

# EVOLVE-BLOCK-END
