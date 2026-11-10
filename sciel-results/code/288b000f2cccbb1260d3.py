import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add((x * x) % m)
    forbidden.discard(0)
    
    # 2. Build conflict graph
    # conflicts[i] is a set of j such that i and j cannot be in the same set
    conflicts = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Helper to check if a node can be added to a set
    def can_add(node, current_set):
        # current_set is a list or set
        # Using set for O(1) lookup if we maintain one, but here we pass list and check conflicts
        # conflicts[node] contains all nodes conflicting with node
        # So we just check if any element in current_set is in conflicts[node]
        conflict_set = conflicts[node]
        for n in current_set:
            if n in conflict_set:
                return False
        return True

    # Greedy construction
    def greedy_construct(order):
        current = []
        current_set = set()
        for node in order:
            if node in current_set:
                continue
            if can_add(node, current):
                current.append(node)
                current_set.add(node)
        return current

    # Initial best solution
    best_set = []
    # Try a few random orders to find a decent starting point
    for _ in range(50):
        order = list(range(m))
        random.shuffle(order)
        s = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
            
    start_time = time.time()
    deadline = 600.0
    best_size = len(best_set)
    
    # Local Search: Kick and Polish
    while time.time() - start_time < deadline - 5:
        # Create a working copy
        current = list(best_set)
        current_set = set(current)
        
        # Kick: remove k random elements
        # k can vary to escape local minima
        k = random.randint(3, 10)
        if len(current) > k:
            # Remove k random elements
            to_remove = random.sample(current, k)
            to_remove_set = set(to_remove)
            current = [x for x in current if x not in to_remove_set]
            current_set = set(current)
            
        # Polish: add nodes back
        # Candidates are nodes not in current
        candidates = [i for i in range(m) if i not in current_set]
        
        # Strategy: Sort by degree (number of conflicts) to prioritize "easy" nodes
        # Or random. Let's try a mix: 50% random, 50% sorted
        if random.random() < 0.5:
            random.shuffle(candidates)
        else:
            candidates.sort(key=lambda x: len(conflicts[x]))
            
        for node in candidates:
            if can_add(node, current):
                current.append(node)
                current_set.add(node)
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            # Optional: if we hit a new record, we might want to reduce kick size
            # to refine, but for now keep it simple.
            
    return np.array(best_set, dtype=np.int64)
