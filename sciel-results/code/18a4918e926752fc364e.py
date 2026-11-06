import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    def compute_score(adj):
        """Compute max degree sum over all triangles."""
        deg = [sum(adj[u]) for u in range(n)]
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def make_initial_k88_plus1():
        """K_{8,8} + one edge inside part A."""
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    def make_random_near_turan():
        """Random graph with 65 edges, biased toward bipartite structure."""
        edges = set()
        # Start with random bipartite-ish structure
        a = list(range(8))
        b = list(range(8, 16))
        random.shuffle(a)
        random.shuffle(b)
        # Add 64 random bipartite edges (complete bipartite minus one)
        all_bipartite = [(u, v) for u in a for v in b]
        random.shuffle(all_bipartite)
        for u, v in all_bipartite[:64]:
            edges.add((min(u,v), max(u,v)))
        # Add 1 more edge (random, possibly creating triangle)
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u,v) not in edges]
        if remaining:
            u, v = random.choice(remaining)
            edges.add((u, v))
        return [[u, v] for u, v in edges]
    
    def make_balanced_random():
        """Random 65-edge graph with roughly balanced degrees."""
        edges = set()
        # Randomly pick 65 edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:65]:
            edges.add((u, v))
        return [[u, v] for u, v in edges]
    
    def local_search(adj, max_time):
        """Systematic edge-swap local search to minimize max triangle degree sum."""
        best_score = compute_score(adj)
        current_score = best_score
        edge_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edge_set.add((u, v))
        
        non_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.add((u, v))
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try all possible swaps: remove one edge, add another
            for rem_u, rem_v in list(edge_set):
                if time.time() >= max_time:
                    break
                for add_u, add_v in list(non_edges):
                    if (add_u, add_v) == (rem_u, rem_v):
                        continue
                    # Check if swap is valid (don't duplicate)
                    if (add_u, add_v) in edge_set:
                        continue
                    
                    # Perform swap
                    adj[rem_u][rem_v] = False
                    adj[rem_v][rem_u] = False
                    adj[add_u][add_v] = True
                    adj[add_v][add_u] = True
                    
                    new_score = compute_score(adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                        edge_set.remove((rem_u, rem_v))
                        edge_set.add((add_u, add_v))
                        non_edges.add((rem_u, rem_v))
                        non_edges.remove((add_u, add_v))
                        improved = True
                    else:
                        # Revert
                        adj[rem_u][rem_v] = True
                        adj[rem_v][rem_u] = True
                        adj[add_u][add_v] = False
                        adj[add_v][add_u] = False
        
        return adj, best_score
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # 1. K_{8,8} + 1 edge (baseline)
    initial_solutions.append(make_initial_k88_plus1())
    
    # 2. K_{8,8} + 1 edge with different internal edge
    edges2 = [[u, v] for u in range(8) for v in range(8, 16)]
    edges2.append([0, 2])
    initial_solutions.append(edges2)
    
    # 3. K_{8,8} + 1 edge in other part
    edges3 = [[u, v] for u in range(8) for v in range(8, 16)]
    edges3.append([8, 9])
    initial_solutions.append(edges3)
    
    # 4-7. Random near-Turán
    for _ in range(4):
        initial_solutions.append(make_random_near_turan())
    
    # 8-9. Balanced random
    for _ in range(2):
        initial_solutions.append(make_balanced_random())
    
    best_overall = None
    best_overall_score = float('inf')
    
    for i, init_edges in enumerate(initial_solutions):
        if time.time() > deadline:
            break
        
        # Verify edge count
        if len(init_edges) != target_edges:
            continue
        
        adj = build_adj(init_edges)
        initial_score = compute_score(adj)
        
        # Run local search with time budget
        time_budget = min(15, (deadline - time.time()) / max(1, len(initial_solutions) - i))
        adj, score = local_search(adj, time.time() + time_budget)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall = get_edges_from_adj(adj)
    
    # If we found something better, return it; otherwise return baseline
    if best_overall is not None and best_overall_score < 24:
        return {'n': 16, 'edges': best_overall}
    
    # Fallback: return the best we found or baseline
    if best_overall is not None:
        return {'n': 16, 'edges': best_overall}
    
    # Absolute fallback
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return {'n': 16, 'edges': edges}
# EVOLVE-BLOCK-END
