import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # safety margin before 150s
    
    best_result = None
    best_n = 0
    
    # Allocate time budget for each n
    # n=47 is very hard, give it a small budget. n=46 is the target.
    budgets = {
        47: 5,  # seconds
        46: 130, # seconds
        45: 5,
        44: 2,
        43: 1,
        42: 0.5,
        41: 0.2,
        40: 0.1
    }
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        time_budget = budgets.get(n, 0.5)
        start_time = time.time()
        
        # Try multiple strategies within the budget
        while time.time() < start_time + time_budget and time.time() < deadline:
            attempt_seed = int(time.time() * 1000000) % 100000
            result = _construct_c4_free(n, required_min_deg, deadline, attempt_seed)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    return result
                    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _construct_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req."""
    rng = random.Random(seed_offset)
    
    # Use sets for neighbors for fast intersection and iteration
    neighbors = [set() for _ in range(n)]
    
    def check_c4_add(u, v):
        """Check if adding (u,v) creates a C4."""
        for x in neighbors[u]:
            if x == v:
                continue
            common = neighbors[x] & neighbors[v]
            for y in common:
                if y != u and y != x:
                    return True
        return False

    def add_edge(u, v):
        neighbors[u].add(v)
        neighbors[v].add(u)

    def remove_edge(u, v):
        neighbors[u].remove(v)
        neighbors[v].remove(u)

    # Phase 1: Greedy Random Addition
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not check_c4_add(u, v):
            add_edge(u, v)
    
    # Phase 2: Local Search (Edge Swaps)
    degrees = [len(neighbors[i]) for i in range(n)]
    
    max_iters = 5000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                for v in neighbors[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find a low-degree vertex
        v = degrees.index(min_deg)
        
        # Find a candidate non-neighbor u
        non_neighbors = [u for u in range(n) if u != v and u not in neighbors[v]]
        if not non_neighbors:
            break 
        
        rng.shuffle(non_neighbors)
        success_swap = False
        
        for u in non_neighbors:
            if not check_c4_add(u, v):
                add_edge(u, v)
                degrees[u] += 1
                degrees[v] += 1
                success_swap = True
                break
            
            # Blocked. Find a blocking pair (x, y)
            # x in N(u), y in N(v), x~y
            blocking_pairs = []
            for x in neighbors[u]:
                if x == v: continue
                for y in neighbors[v]:
                    if y == u: continue
                    if x == y: continue
                    if y in neighbors[x]:
                        if degrees[x] > min_deg and degrees[y] > min_deg:
                            blocking_pairs.append((x, y))
            
            if blocking_pairs:
                x, y = rng.choice(blocking_pairs)
                remove_edge(x, y)
                if not check_c4_add(u, v):
                    add_edge(u, v)
                    degrees[u] += 1
                    degrees[v] += 1
                    degrees[x] -= 1
                    degrees[y] -= 1
                    success_swap = True
                    break
                else:
                    add_edge(x, y)
                    continue
            
            # If no valid blocking pair found for this u, try next u
        
        if not success_swap:
            break

    return None

# EVOLVE-BLOCK-END
