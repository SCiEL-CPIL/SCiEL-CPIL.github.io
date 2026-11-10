import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 150s kill

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

    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def greedy_clique_extraction(n, edges, adj, rng):
        """
        Greedy clique extraction:
        1. Find vertex u with max degree in remaining graph.
        2. Greedily build a clique starting from u.
        3. Remove edges of the clique.
        4. Repeat.
        """
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        remaining_edges = len(edges)
        
        while remaining_edges > 0:
            # Find vertex with max degree
            max_deg = -1
            u = -1
            for i in range(n):
                d = popcount(rem_adj[i])
                if d > max_deg:
                    max_deg = d
                    u = i
            
            if u == -1 or max_deg == 0:
                break
            
            # Build clique greedily
            clique = [u]
            P = rem_adj[u]
            
            while P:
                # Pick vertex v in P with max degree in P
                max_v_deg = -1
                v = -1
                P_copy = P
                while P_copy:
                    v_bit = P_copy & (-P_copy)
                    i = v_bit.bit_length() - 1
                    P_copy ^= v_bit
                    # Degree in P
                    deg = popcount(P & rem_adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v = i
                
                if v == -1:
                    break
                
                clique.append(v)
                P &= rem_adj[v]
            
            # Remove edges of clique
            edges_removed = 0
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u_c, v_c = clique[i], clique[j]
                    if (rem_adj[u_c] >> v_c) & 1:
                        rem_adj[u_c] &= ~(1 << v_c)
                        rem_adj[v_c] &= ~(1 << u_c)
                        edges_removed += 1
            
            if edges_removed == 0:
                break
            
            cliques.append(clique)
            remaining_edges -= edges_removed
            
            if time.time() > DEADLINE:
                return None
        
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        start = time.time()
        attempt = 0
        
        # Run many fast greedy restarts
        while time.time() < start + time_limit:
            rng = random.Random(seed + attempt)
            result = greedy_clique_extraction(n, edges, adj, rng)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
            
            attempt += 1
            
        return best_cliques

    # Solve both instances
    # Allocate time based on problem size
    # n=32 is much faster than n=64
    time_budget_1 = 35.0
    time_budget_2 = 75.0
    
    # Ensure we don't exceed total deadline
    current_time = time.time()
    remaining = DEADLINE - current_time
    if time_budget_1 + time_budget_2 > remaining:
        scale = remaining / (time_budget_1 + time_budget_2)
        time_budget_1 *= scale
        time_budget_2 *= scale

    result1 = solve_instance(32, 0.6, 601, time_budget_1)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, time_budget_2)
    
    return [result1, result2]
