import numpy as np
import time
import random
import heapq

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Precompute neighbors and degrees
    adj_list = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=int)
    for i in range(n):
        nbrs = np.where(adjacency[i])[0]
        adj_list[i] = nbrs.tolist()
        degrees[i] = len(nbrs)

    # Simulate Greedy MIS with a specific tie-breaking strategy
    # Returns the order in which vertices are selected
    def simulate_greedy_mis(strategy='random'):
        degs = degrees.copy().astype(int)
        removed = [False] * n
        heap = []
        
        for i in range(n):
            if strategy == 'random':
                key = random.random()
            else:
                key = i
            heapq.heappush(heap, (degs[i], key, i))
            
        order = []
        
        while heap:
            d, key, v = heapq.heappop(heap)
            if removed[v]:
                continue
            if degs[v] != d:
                continue
                
            order.append(v)
            removed[v] = True
            
            for u in adj_list[v]:
                if not removed[u]:
                    degs[u] -= 1
                    if strategy == 'random':
                        new_key = random.random()
                    else:
                        new_key = u
                    heapq.heappush(heap, (degs[u], new_key, u))
                    
        return order

    # Evaluate a static order using the FIXED DECODER logic
    def evaluate_order(order):
        is_picked = [False] * n
        is_size = 0
        for v in order:
            valid = True
            for u in adj_list[v]:
                if is_picked[u]:
                    valid = False
                    break
            if valid:
                is_picked[v] = True
                is_size += 1
        return is_size

    # Main Search Loop
    # Use 110s to be safe against the 120s limit
    deadline = time.time() + 110.0
    
    best_size = -1
    best_order = None
    
    # Run initial deterministic heuristics to establish a baseline
    for strat in ['index', 'random']:
        order = simulate_greedy_mis(strat)
        size = evaluate_order(order)
        if size > best_size:
            best_size = size
            best_order = order

    # Stochastic search
    iterations = 0
    while time.time() < deadline:
        # Use random tie-breaking to explore different orderings
        order = simulate_greedy_mis('random')
        size = evaluate_order(order)
        
        if size > best_size:
            best_size = size
            best_order = order
            # If we find a perfect or near-perfect solution, we can break early?
            # Max IS is hard to know, but if size is very high, it's good.
            # For N=200, max IS is at least N/2 in sparse graphs?
            # We'll just keep searching to ensure we don't miss better ones.
            
        iterations += 1
        
        # Safety break if time is very tight
        if time.time() > deadline - 1.0:
            break

    if best_order is None:
        # Fallback
        best_order = list(range(n))
        
    # Convert best_order to scores
    # Decoder picks descending score.
    # Order[0] is the most preferred -> Highest Score.
    scores = np.zeros(n, dtype=float)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores
