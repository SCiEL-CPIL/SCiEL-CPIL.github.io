import numpy as np
import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    n = 16
    # Generate all triples
    triples = []
    triple_indices = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_indices[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    num_triples = len(triples)
    
    # Generate constraints
    constraint_to_triples = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    def get_idx(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_indices[(a, b, c)]

    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            for v3 in range(v2 + 1, n):
                for v4 in range(v3 + 1, n):
                    for v5 in range(v4 + 1, n):
                        for v6 in range(v5 + 1, n):
                            verts = [v1, v2, v3, v4, v5, v6]
                            
                            for j in range(1, 6):
                                pair1 = (verts[0], verts[j])
                                remaining = [v for i, v in enumerate(verts) if i != 0 and i != j]
                                
                                pairings = [
                                    [(remaining[0], remaining[1]), (remaining[2], remaining[3])],
                                    [(remaining[0], remaining[2]), (remaining[1], remaining[3])],
                                    [(remaining[0], remaining[3]), (remaining[1], remaining[2])]
                                ]
                                for p in pairings:
                                    pair2 = p[0]
                                    pair3 = p[1]
                                    
                                    t_indices = []
                                    for a in pair1:
                                        for b in pair2:
                                            for c in pair3:
                                                t_idx = get_idx(a, b, c)
                                                t_indices.append(t_idx)
                                                triple_to_constraints[t_idx].append(len(constraint_to_triples))
                                    
                                    constraint_to_triples.append(t_indices)
                                    
    num_constraints = len(constraint_to_triples)
    
    # State
    present = np.zeros(num_triples, dtype=bool)
    counts = np.zeros(num_constraints, dtype=np.int32)
    current_size = 0
    
    best_size = 0
    best_present = present.copy()
    
    def is_addable(t):
        for c in triple_to_constraints[t]:
            if counts[c] >= 7:
                return False
        return True

    def add_triple(t):
        nonlocal current_size
        present[t] = True
        current_size += 1
        for c in triple_to_constraints[t]:
            counts[c] += 1

    def remove_triple(t):
        nonlocal current_size
        present[t] = False
        current_size -= 1
        for c in triple_to_constraints[t]:
            counts[c] -= 1

    def greedy_fill(order):
        nonlocal current_size
        for t in order:
            if not present[t] and is_addable(t):
                add_triple(t)

    def local_search():
        nonlocal best_size, best_present
        # Create a random order for greedy fill
        order = list(range(num_triples))
        random.shuffle(order)
        
        # Reset state
        present.fill(False)
        counts.fill(0)
        current_size = 0
        
        # Initial Greedy
        greedy_fill(order)
        
        if current_size > best_size:
            best_size = current_size
            best_present = present.copy()
        
        # Iterated Local Search
        # We try to remove a random triple and refill
        # If the refill results in a larger set, we keep it.
        # This effectively allows us to escape local optima where a "bad" triple blocks many "good" ones.
        
        while time.time() < deadline - 5:
            # Pick a random present triple to remove
            present_indices = np.where(present)[0]
            if len(present_indices) == 0:
                break
                
            t_out = present_indices[random.randint(0, len(present_indices) - 1)]
            
            # Save state
            present_backup = present.copy()
            counts_backup = counts.copy()
            size_backup = current_size
            
            # Remove t_out
            remove_triple(t_out)
            
            # Greedy fill again
            # Use a new random order to encourage different fills
            order2 = list(range(num_triples))
            random.shuffle(order2)
            greedy_fill(order2)
            
            if current_size > size_backup:
                # Improvement! Keep the new state
                if current_size > best_size:
                    best_size = current_size
                    best_present = present.copy()
            else:
                # Revert
                present[:] = present_backup
                counts[:] = counts_backup
                current_size = size_backup
                # Optional: If we revert too often, we might be stuck.
                # But since we shuffle order2, we might find a different fill next time.
                # If we want to be more aggressive, we can do multiple reverts.
                
    # Run multiple restarts or a long local search
    # Given the time, a single long local search with random restarts inside might be best.
    # Or just loop the local_search function which resets each time?
    # The local_search function above resets to empty each time.
    # This is "Iterated Greedy".
    
    # Let's do Iterated Greedy with Local Search (Remove-Refill)
    
    restarts = 0
    while time.time() < deadline - 2:
        restarts += 1
        local_search()
        
    result = []
    for i in range(num_triples):
        if best_present[i]:
            result.append(list(triples[i]))
            
    return result
