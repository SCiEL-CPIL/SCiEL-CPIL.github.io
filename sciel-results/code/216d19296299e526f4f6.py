import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 58.0  # Leave 2s buffer before 60s kill

    # Precompute for each x the list of triples (a,b,c) such that {a,b,c,x} forms a 4-AP
    # We'll use numpy arrays for fast membership testing
    conflict_triples = [None] * (n + 1)
    for x in range(1, n + 1):
        triples = []
        # Case 1: x is first term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            triples.append((x + d, x + 2 * d, x + 3 * d))
        
        # Case 2: x is second term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            triples.append((x - d, x + d, x + 2 * d))
        
        # Case 3: x is third term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            triples.append((x - 2 * d, x - d, x + d))
        
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            triples.append((x - 3 * d, x - 2 * d, x - d))
        
        conflict_triples[x] = np.array(triples, dtype=np.int32) if triples else np.empty((0, 3), dtype=np.int32)

    def creates_4ap_bitset(S, x):
        """Check if adding x to set S creates a 4-AP. S is a numpy boolean array."""
        triples = conflict_triples[x]
        if len(triples) == 0:
            return False
        # Check if any triple has all 3 elements in S
        in_S = S[triples]
        return np.any(np.all(in_S, axis=1))

    def greedy_fill_from(S, order=None):
        """Greedily fill set S given an order. S is a numpy boolean array (copy)."""
        if order is None:
            order = list(range(1, n + 1))
        S = S.copy()
        for x in order:
            if not S[x]:
                if not creates_4ap_bitset(S, x):
                    S[x] = True
        return S

    def greedy_fill_empty(order=None):
        """Greedily fill from empty set given an order."""
        if order is None:
            order = list(range(1, n + 1))
        S = np.zeros(n + 1, dtype=bool)
        for x in order:
            if not creates_4ap_bitset(S, x):
                S[x] = True
        return S

    random.seed(42)
    
    best_set = np.zeros(n + 1, dtype=bool)
    best_size = 0
    
    # Phase 1: Diverse random restarts to find a good starting point
    num_random_tries = 800
    for i in range(num_random_tries):
        if time.time() - start_time > time_limit * 0.35:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = greedy_fill_empty(order)
        current_size = int(np.sum(current))
        if current_size > best_size:
            best_set = current.copy()
            best_size = current_size
    
    # Also try some structured orders
    for order in [list(range(1, n + 1)), list(range(n, 0, -1))]:
        if time.time() - start_time > time_limit * 0.40:
            break
        current = greedy_fill_empty(order)
        current_size = int(np.sum(current))
        if current_size > best_size:
            best_set = current.copy()
            best_size = current_size

    # Phase 2: Local search with kicks and perturbation restarts
    current_set = best_set.copy()
    current_size = best_size
    
    # Track best so far
    global_best_set = best_set.copy()
    global_best_size = best_size
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # Try to add any available element
        added = False
        candidates = [i for i in range(1, n + 1) if not current_set[i]]
        
        if not candidates:
            break
        
        random.shuffle(candidates)
        for x in candidates:
            if not creates_4ap_bitset(current_set, x):
                current_set[x] = True
                current_size += 1
                added = True
                if current_size > global_best_size:
                    global_best_set = current_set.copy()
                    global_best_size = current_size
                    no_improve_count = 0
                break
        
        if not added:
            no_improve_count += 1
            
            # If stuck for too long, do a perturbation restart
            if no_improve_count >= max_no_improve or random.random() < 0.1:
                # Perturbation: start from global best, remove k random elements, greedy refill
                k = random.randint(2, 6)
                if global_best_size > k:
                    base = global_best_set.copy()
                    to_remove = random.sample([i for i in range(1, n + 1) if base[i]], k)
                    for x in to_remove:
                        base[x] = False
                    
                    # Greedy refill from perturbed state
                    order = list(range(1, n + 1))
                    random.shuffle(order)
                    current_set = greedy_fill_from(base, order)
                    current_size = int(np.sum(current_set))
                    
                    if current_size > global_best_size:
                        global_best_set = current_set.copy()
                        global_best_size = current_size
                        no_improve_count = 0
                else:
                    # Small set, just do a fresh random restart
                    order = list(range(1, n + 1))
                    random.shuffle(order)
                    current_set = greedy_fill_empty(order)
                    current_size = int(np.sum(current_set))
                    if current_size > global_best_size:
                        global_best_set = current_set.copy()
                        global_best_size = current_size
                        no_improve_count = 0
            else:
                # Standard kick: remove 1-4 random elements from current, greedy refill
                k = random.randint(1, 4)
                if current_size > k:
                    to_remove = random.sample([i for i in range(1, n + 1) if current_set[i]], k)
                    for x in to_remove:
                        current_set[x] = False
                        current_size -= 1
                    
                    # Greedy refill
                    order = list(range(1, n + 1))
                    random.shuffle(order)
                    current_set = greedy_fill_from(current_set, order)
                    current_size = int(np.sum(current_set))
                    
                    if current_size > global_best_size:
                        global_best_set = current_set.copy()
                        global_best_size = current_size
                        no_improve_count = 0

    # Phase 3: Final polish - try to add any remaining elements systematically
    final_set = global_best_set.copy()
    final_size = int(np.sum(final_set))
    
    # Try adding elements in increasing order (structured)
    for x in range(1, n + 1):
        if not final_set[x]:
            if not creates_4ap_bitset(final_set, x):
                final_set[x] = True
                final_size += 1
    
    # Try adding elements in decreasing order
    for x in range(n, 0, -1):
        if not final_set[x]:
            if not creates_4ap_bitset(final_set, x):
                final_set[x] = True
                final_size += 1
    
    # One more random pass
    order = list(range(1, n + 1))
    random.shuffle(order)
    for x in order:
        if not final_set[x]:
            if not creates_4ap_bitset(final_set, x):
                final_set[x] = True
                final_size += 1

    if final_size > global_best_size:
        global_best_set = final_set
        global_best_size = final_size

    return sorted([i for i in range(1, n + 1) if global_best_set[i]])
