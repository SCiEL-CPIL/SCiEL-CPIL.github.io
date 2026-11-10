import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    def can_add(chosen, counts, x):
        """Check if x can be added without violating B2[2] constraint."""
        if counts[2 * x] >= 2:
            return False
        for y in chosen:
            if counts[x + y] == 2:
                return False
        return True
        
    def add_element(chosen, counts, x):
        """Add x to chosen set and update counts."""
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        
    def remove_element(chosen, counts, x):
        """Remove x from chosen set and update counts."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
        
    def greedy_with_order(order='random', seed=None):
        """Run greedy construction with specified ordering."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
            
        if order == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order == 'squares':
            squares = [i*i for i in range(1, int(np.sqrt(N)) + 1) if i*i <= N]
            others = [x for x in range(1, N + 1) if x not in squares]
            rng.shuffle(others)
            candidates = squares + others
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'weighted_small':
            candidates = list(range(1, N + 1))
            # Weight towards smaller numbers
            weights = [1.0 / (1 + i * 0.01) for i in range(N)]
            indexed = list(zip(weights, candidates))
            rng.shuffle(indexed)
            candidates = [x[1] for x in indexed]
        elif order == 'sparse_fill':
            # Start with sparse (every 3rd), then fill in
            sparse = list(range(1, N + 1, 3))
            fill = list(range(2, N + 1, 3))
            rng.shuffle(fill)
            candidates = sparse + fill
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add(chosen, counts, x):
                add_element(chosen, counts, x)
                
        return chosen, counts
        
    def local_search(chosen, counts, max_rounds=100, time_limit=None, start_time=None, rng=None):
        """Improved local search with insertion before swaps."""
        if rng is None:
            rng = random
        chosen_set = set(chosen)
        improved = True
        rounds = 0
        no_improve_count = 0
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            no_improve_count += 1
            
            if time_limit and start_time:
                if time.time() - start_time > time_limit:
                    break
                    
            # 1. Try to add any element
            added_any = False
            # Try in random order for diversity
            candidates_to_try = [x for x in range(1, N + 1) if x not in chosen_set]
            rng.shuffle(candidates_to_try)
            for x in candidates_to_try:
                if time_limit and start_time and time.time() - start_time > time_limit:
                    break
                if can_add(chosen, counts, x):
                    add_element(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    added_any = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
                
            # 2. If no add, try swaps to escape local max
            if len(chosen) > 0:
                # Try multiple swap attempts in this round
                swap_attempts = 0
                max_swap_attempts = 5
                swap_found = False
                
                while swap_attempts < max_swap_attempts and not swap_found:
                    if time_limit and start_time and time.time() - start_time > time_limit:
                        break
                    swap_attempts += 1
                    
                    x_to_remove = rng.choice(chosen)
                    remove_element(chosen, counts, x_to_remove)
                    chosen_set.discard(x_to_remove)
                    
                    # Try to add any element
                    added = False
                    candidates_to_try = [y for y in range(1, N + 1) if y not in chosen_set]
                    rng.shuffle(candidates_to_try)
                    for y in candidates_to_try:
                        if time_limit and start_time and time.time() - start_time > time_limit:
                            break
                        if can_add(chosen, counts, y):
                            add_element(chosen, counts, y)
                            chosen_set.add(y)
                            added = True
                            swap_found = True
                            break
                        
                    if not added:
                        # Put back
                        add_element(chosen, counts, x_to_remove)
                        chosen_set.add(x_to_remove)
                    else:
                        # We swapped. Size is same.
                        improved = True
                        # Reset no_improve_count since we made a change
                        no_improve_count = 0
                        break
        return chosen, counts
        
    start_time = time.time()
    timeout = 110
    
    # Strategy 1: Squares initialization
    current_set, current_counts = greedy_with_order('squares', 42)
    current_set, current_counts = local_search(current_set, current_counts, max_rounds=100, time_limit=timeout - (time.time() - start_time), start_time=start_time)
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
        
    # Strategy 2: Ascending greedy
    if time.time() - start_time < timeout - 5:
        current_set, current_counts = greedy_with_order('ascending', 42)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=50, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    # Strategy 3: Descending greedy
    if time.time() - start_time < timeout - 5:
        current_set, current_counts = greedy_with_order('descending', 42)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=50, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    # Strategy 4: Weighted small numbers
    if time.time() - start_time < timeout - 5:
        current_set, current_counts = greedy_with_order('weighted_small', 42)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=50, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    # Strategy 5: Sparse fill
    if time.time() - start_time < timeout - 5:
        current_set, current_counts = greedy_with_order('sparse_fill', 42)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=50, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    # Strategy 6: Random restarts with diverse orderings
    rng = random.Random(123)
    while time.time() - start_time < timeout - 2:
        # Cycle through different orderings
        order_choice = rng.choice(['random', 'random', 'random', 'weighted_small', 'sparse_fill'])
        seed = rng.randint(0, 1000000)
        current_set, current_counts = greedy_with_order(order_choice, seed)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=30, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    return best_set
