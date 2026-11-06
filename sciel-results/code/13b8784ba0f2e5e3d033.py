import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflicts
    # conflict[i] is a set of j such that i and j conflict
    # Using 0-based indexing internally for speed, but values are 1-based
    conflicts = [set() for _ in range(N)]
    
    # We only need to check pairs (a, b) with a < b
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Greedy function: given an order of indices, build a set
    def greedy(order):
        selected = []
        selected_set = set()
        for idx in order:
            # Check if idx conflicts with any in selected_set
            # We can optimize by checking intersection of conflicts[idx] and selected_set
            if not (conflicts[idx] & selected_set):
                selected.append(idx)
                selected_set.add(idx)
        return selected

    best_size = 0
    best_set = []
    
    # 1. Try sorted by degree (ascending)
    degrees = [len(conflicts[i]) for i in range(N)]
    order_by_degree = sorted(range(N), key=lambda x: degrees[x])
    current_set = greedy(order_by_degree)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set
        
    # 2. Try random shuffles
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random_order = list(range(N))
        random.shuffle(random_order)
        current_set = greedy(random_order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # Early exit if we hit a theoretical upper bound? 
            # Max possible is 256. If we get close to 187+, it's good.
            # We can also try local search if we have time.
            
    # Convert 0-based indices to 1-based values
    return [i + 1 for i in best_set]
