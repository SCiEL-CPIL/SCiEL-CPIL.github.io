# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import math

def construct():
    # Deadline guard: 150s wall-clock limit, keep safety margin
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time) - SAFETY_MARGIN

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def decompose_graph_sa(edges, time_limit=40.0):
        if not edges:
            return []
        
        adj = build_adjacency(edges)
        total_edges = len(edges)
        
        # Initial solution: Each edge is a path
        # Represent path as list of vertices
        # We need a fast way to check if a sequence of vertices is a valid simple path
        # and if it uses only unused edges.
        
        # State: List of paths
        # We will use a set of used edges to quickly check availability?
        # Or we can just check adjacency and simplicity.
        
        # To make SA fast, we need efficient move evaluation.
        # Let's represent the state as a list of paths.
        # Cost = len(paths)
        
        # Initial random valid decomposition:
        # Just list of single edges.
        initial_paths = [[u, v] for u, v in edges]
        
        best_paths = initial_paths
        best_cost = len(initial_paths)
        
        current_paths = [list(p) for p in initial_paths]
        current_cost = len(current_paths)
        
        # To speed up checks, maintain a set of used edges?
        # Actually, since paths are edge-disjoint and cover all edges,
        # the set of edges in current_paths is always equal to 'edges'.
        # So we just need to check if a proposed path is valid (simple, edges exist).
        
        def is_valid_path(path):
            if len(path) < 2:
                return False
            # Check simplicity
            if len(set(path)) != len(path):
                return False
            # Check edges
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if (u, v) in edges_set or (v, u) in edges_set:
                    continue
                else:
                    return False
            return True

        edges_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        # Precompute edge existence for O(1) check
        # We already have edges_set.
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.9999
        max_iters = 50000 # Adjust based on time
        
        start_sa_time = time.time()
        
        while time_left() > 5.0 and (time.time() - start_sa_time) < time_limit:
            if T < T_min:
                break
                
            # Generate a move
            move_type = random.randint(0, 2)
            
            new_paths = None
            
            if move_type == 0:
                # Try to merge two paths
                if len(current_paths) > 1:
                    i, j = random.sample(range(len(current_paths)), 2)
                    p1 = current_paths[i]
                    p2 = current_paths[j]
                    
                    # Try 4 combinations of merging
                    # 1. p1 + p2 (if p1[-1] == p2[0] or p1[-1] == p2[-1] with reverse)
                    # We need to check if the concatenated path is valid.
                    
                    candidates = []
                    # Direct
                    if p1[-1] == p2[0]:
                        candidates.append(p1 + p2[1:])
                    if p1[-1] == p2[-1]:
                        candidates.append(p1 + p2[::-1][1:])
                    if p1[0] == p2[0]:
                        candidates.append(p1[::-1] + p2[1:])
                    if p1[0] == p2[-1]:
                        candidates.append(p1[::-1] + p2[::-1][1:])
                    
                    if candidates:
                        # Pick one candidate
                        new_candidate = candidates[0]
                        if is_valid_path(new_candidate):
                            new_paths = [list(p) for p in current_paths]
                            new_paths[i] = new_candidate
                            del new_paths[j]
                            new_cost = len(new_paths)
                            delta = new_cost - current_cost
                            if delta < 0 or random.random() < math.exp(-delta / T):
                                current_paths = new_paths
                                current_cost = new_cost
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_paths = [list(p) for p in current_paths]
                                    if best_cost <= 2: # Theoretical min for connected graph with max degree 4?
                                         # Actually min is ceil(Delta/2). If Delta <= 4, min is 2.
                                         # If we reach 2, it's likely optimal for sparse graphs.
                                         # We can continue to ensure no better exists (1 is impossible for connected graph with >1 edge).
                                         pass
                    T *= alpha
                    continue
            elif move_type == 1:
                # Move an edge from one path to another
                # This is complex. Let's try "Swap" instead.
                pass
            else:
                # Swap segments or just reverse a path (no cost change, helps escaping)
                i = random.randint(0, len(current_paths) - 1)
                current_paths[i] = current_paths[i][::-1]
                # No cost change, so accept
                # But this doesn't help much.
                pass
            
            T *= alpha
        
        # Final local search: Greedy merge
        # Try to merge any pair that can be merged
        improved = True
        while improved:
            improved = False
            for i in range(len(best_paths)):
                for j in range(i + 1, len(best_paths)):
                    p1 = best_paths[i]
                    p2 = best_paths[j]
                    merged = None
                    if p1[-1] == p2[0]:
                        merged = p1 + p2[1:]
                    elif p1[-1] == p2[-1]:
                        merged = p1 + p2[::-1][1:]
                    elif p1[0] == p2[0]:
                        merged = p1[::-1] + p2[1:]
                    elif p1[0] == p2[-1]:
                        merged = p1[::-1] + p2[::-1][1:]
                    
                    if merged and is_valid_path(merged):
                        best_paths[i] = merged
                        del best_paths[j]
                        improved = True
                        break
                if improved:
                    break
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time_left() < 10.0:
            # Fallback
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        
        edges = instance(*spec)
        paths = decompose_graph_sa(edges, time_limit=40.0)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
