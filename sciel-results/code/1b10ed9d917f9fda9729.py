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
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            if y != x and counts[x + y] + 1 > 2:
                return False
        return True
    
    def add_element(chosen, counts, x):
        """Add x to chosen set and update counts."""
        for y in chosen:
            if y != x:
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
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'weighted':
            # Weight towards numbers that are less likely to conflict
            candidates = list(range(1, N + 1))
            weights = [1.0 / (1 + i % 10) for i in range(N)]
            rng.shuffle(candidates)
        elif order == 'sparse_first':
            # Try odd numbers first, then even
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        elif order == 'gap_based':
            # Start with numbers that have larger gaps
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add(chosen, counts, x):
                add_element(chosen, counts, x)
                
        return chosen, counts
    
    def local_search(chosen, counts, max_rounds=30, time_limit=None, start_time=None):
        """Improved local search with faster checks."""
        chosen_set = set(chosen)
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            
            if time_limit and start_time:
                if time.time() - start_time > time_limit:
                    break
            
            # First, try to add any element
            for x in range(1, N + 1):
                if x not in chosen_set and can_add(chosen, counts, x):
                    add_element(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: for each element in chosen, see if removing it allows adding another
            # Use random sampling for speed
            if len(chosen) > 0:
                sample_size = min(len(chosen), 50)
                sampled = random.sample(chosen, sample_size)
                
                for x in sampled:
                    # Temporarily remove x
                    remove_element(chosen, counts, x)
                    chosen_set.discard(x)
                    
                    # Try to find an element that can now be added
                    found = False
                    for y in range(1, N + 1):
                        if y not in chosen_set and can_add(chosen, counts, y):
                            add_element(chosen, counts, y)
                            chosen_set.add(y)
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Put x back
                        add_element(chosen, counts, x)
                        chosen_set.add(x)
                    
                    if improved:
                        break
        
        return chosen, counts
    
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Try multiple strategies
    strategies = [
        ('random', None),
        ('random', None),
        ('ascending', None),
        ('descending', None),
        ('sparse_first', None),
        ('weighted', None),
        ('gap_based', None),
    ]
    
    # First pass: quick greedy to get a baseline
    for order, seed in strategies[:3]:
        if time.time() - start_time > timeout:
            break
        current_set, current_counts = greedy_with_order(order, seed)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=10, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
    
    # Second pass: more thorough local search on best found
    if best_set and time.time() - start_time < timeout - 5:
        chosen = best_set[:]
        counts = [0] * (max_sum + 1)
        for x in chosen:
            for y in chosen:
                if y >= x:
                    counts[x + y] += 1
        chosen, counts = local_search(chosen, counts, max_rounds=50, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
    
    # Continue with more restarts if time allows
    while time.time() - start_time < timeout - 2:
        order = random.choice(['random', 'sparse_first', 'weighted', 'gap_based'])
        seed = random.randint(0, 1000000)
        current_set, current_counts = greedy_with_order(order, seed)
        current_set, current_counts = local_search(current_set, current_counts, max_rounds=20, time_limit=timeout - (time.time() - start_time), start_time=start_time)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
    
    return best_set
