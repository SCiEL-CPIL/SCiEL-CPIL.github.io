"""Greedy local search for values and edges to minimize distinct sums/products."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Calculate the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u] + values[v]
        pu = values[u] * values[v]
        sums.add(su)
        prods.add(pu)
    return max(len(sums), len(prods))

def _is_connected(edges, n):
    """Check if the graph is connected (not strictly required by contract, but good practice)."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    seen = [False] * n
    seen[0] = True
    stack = [0]
    cnt = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                stack.append(v)
    return cnt == n

def _count_distinct(values, edges):
    """Efficiently count distinct sums and products for a given state."""
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return len(sums), len(prods)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    m = P['m']
    
    # Initial heuristic: use small primes to reduce product collisions
    # Primes are good because p1*p2 = p3*p4 is rare.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    initial_values = primes[:]
    
    # Initial edges: take the first m edges from a random permutation of all possible edges
    # or just a dense subgraph. Let's start with a random spanning tree + random extra edges.
    all_edges = list(itertools.combinations(range(n), 2))
    random.shuffle(all_edges)
    initial_edges = [list(e) for e in all_edges[:m]]
    
    best_values = initial_values[:]
    best_edges = [e[:] for e in initial_edges]
    best_score = _raw_score(best_values, best_edges)
    
    # Local Search
    # We will alternate between swapping values and swapping edges.
    
    # Precompute edge set for O(1) lookup
    edge_set = set(map(tuple, best_edges))
    
    while time.time() < deadline:
        improved = False
        
        # --- Move Type 1: Swap two values ---
        # Randomly pick two indices and swap their values
        i, j = random.sample(range(n), 2)
        old_vi = best_values[i]
        old_vj = best_values[j]
        
        # Apply swap
        best_values[i] = old_vj
        best_values[j] = old_vi
        
        # Check validity: distinct positive integers <= 10000
        # Since we start with distinct primes and only swap, distinctness is preserved.
        # We must ensure <= 10000. Primes are small, so this is fine.
        
        new_score = _raw_score(best_values, best_edges)
        
        if new_score < best_score:
            best_score = new_score
            improved = True
        else:
            # Revert
            best_values[i] = old_vi
            best_values[j] = old_vj
        
        # --- Move Type 2: Swap an edge with a non-edge ---
        # Pick a random existing edge
        if m > 0:
            edge_idx = random.randint(0, m - 1)
            old_edge = best_edges[edge_idx]
            old_edge_tuple = tuple(old_edge)
            
            # Find a random non-edge
            # To speed this up, we can generate random pairs until we find one not in edge_set
            # But generating random pairs might be slow if graph is dense.
            # m=35, total possible = 91. Density is ~38%.
            # So finding a non-edge should be fast.
            
            new_edge_found = False
            while not new_edge_found:
                u, v = random.sample(range(n), 2)
                candidate_edge = tuple(sorted((u, v)))
                if candidate_edge not in edge_set:
                    new_edge = list(candidate_edge)
                    new_edge_found = True
            
            # Apply edge swap
            best_edges[edge_idx] = new_edge
            edge_set.remove(old_edge_tuple)
            edge_set.add(candidate_edge)
            
            new_score = _raw_score(best_values, best_edges)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_edges[edge_idx] = list(old_edge)
                edge_set.remove(candidate_edge)
                edge_set.add(old_edge_tuple)
        
        # If no improvement in a "cycle" of moves, we might want to restart or perturb.
        # For now, simple random walk with acceptance only on improvement (Greedy).
        # To avoid getting stuck in local minima, we can allow occasional "sideways" moves 
        # or restarts, but given time constraints, strict greedy is safer for stability.
        
        # If we want to escape local minima, we can do a "kick" every X iterations.
        # Let's add a simple kick: if no improvement in 1000 steps, randomize some edges.
        # But we need to track steps.
        
        # For simplicity and robustness within time, let's just run the greedy loop.
        # If stuck, the random nature of the swaps will eventually try different combinations.
        
    # Final validation and type checking
    # Ensure values are Python ints
    final_values = [int(v) for v in best_values]
    # Ensure edges are lists of Python ints
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
