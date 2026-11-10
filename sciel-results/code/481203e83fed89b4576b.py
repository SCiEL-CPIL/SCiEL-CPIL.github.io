import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edge_in_clique(clique, adj):
        """Check all pairs in clique are edges."""
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def max_clique_tomita(n, adj):
        """
        Find a maximum clique in the graph defined by adj (list of bitmasks).
        Returns the list of vertices in the clique.
        """
        best_clique = []
        
        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return
            
            # Pivot selection: choose u in P with max degree in P
            u = -1
            max_deg = -1
            for v in range(n):
                if (P >> v) & 1:
                    deg = bin(P & adj[v]).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
            
            if u == -1:
                # Fallback if P is not empty but no pivot found (should not happen)
                for v in range(n):
                    if (P >> v) & 1:
                        u = v
                        break
                if u == -1:
                    return

            # Q = P \ N(u)
            Q = P & ~adj[u]
            
            while Q:
                # Pick a vertex v from Q
                # Heuristic: pick v with highest degree in P
                v = -1
                max_v_deg = -1
                for i in range(n):
                    if (Q >> i) & 1:
                        deg = bin(P & adj[i]).count('1')
                        if deg > max_v_deg:
                            max_v_deg = deg
                            v = i
                            
                if v == -1: break
                
                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)
                
                P &= ~(1 << v)
                Q = Q & P # Update Q to reflect P changes
                
        expand([], (1 << n) - 1)
        return best_clique

    def tomita_partition(n, edges, adj, max_time):
        """Partition edges using Max Clique extraction."""
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        
        # Check if graph is empty
        has_edges = any(rem_adj[i] != 0 for i in range(n))
        
        while has_edges:
            if time.time() > max_time:
                return None  # timeout
            
            # Find max clique
            clique = max_clique_tomita(n, rem_adj)
            
            # If no clique or size 1, we might have isolated vertices or no edges
            # But has_edges check ensures edges exist.
            # If max clique size is 1, it means no edges?
            # If there are edges, max clique size >= 2.
            if len(clique) < 2:
                # Fallback: take an edge
                # Find any edge
                found = False
                for u in range(n):
                    if rem_adj[u]:
                        # Find a neighbor
                        v = (rem_adj[u] & -rem_adj[u]).bit_length() - 1
                        # Check if v is valid
                        if (rem_adj[u] >> v) & 1:
                            clique = [u, v]
                            found = True
                            break
                if not found:
                    break
                
            # Remove edges of clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
            
            # Update has_edges
            has_edges = any(rem_adj[i] != 0 for i in range(n))
        
        return cliques

    def solve_instance(n, p, seed):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        # Use Tomita Max Clique partition
        # This is deterministic, so no restarts needed for the algorithm itself.
        # But we can try to improve if it's too slow or fails?
        # For now, just run it once.
        
        result = tomita_partition(n, edges, adj, DEADLINE)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_cliques = result
        
        return best_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
