import time
import random
import sys

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    def check_and_add(chosen, counts, x):
        """Check if x can be added. Return True if added."""
        for y in chosen:
            if counts[x + y] + 1 > 2:
                return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        return True
    
    def remove_from_set(chosen, counts, x):
        """Remove x from chosen set."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
    
    def can_add_fast(chosen, counts, x, chosen_set):
        """Fast check if x can be added."""
        if x in chosen_set:
            return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            if counts[x + y] + 1 > 2:
                return False
        return True
    
    def run_greedy(order='random', seed=None):
        if seed is not None:
            random.seed(seed)
        if order == 'random':
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse':
            candidates = list(range(1, N + 1, 2))
            random.shuffle(candidates)
            candidates += list(range(2, N + 1, 2))
            random.shuffle(candidates)
        elif order == 'weighted':
            # Weight towards numbers that are less likely to conflict
            weights = []
            for i in range(1, N + 1):
                # Numbers with fewer divisors are better
                div_count = sum(1 for d in range(1, int(i**0.5) + 1) if i % d == 0)
                weights.append(1.0 / (1 + div_count))
            total_w = sum(weights)
            candidates = []
            for _ in range(N):
                r = random.random() * total_w
                cum = 0
                for i, w in enumerate(weights):
                    cum += w
                    if cum >= r:
                        candidates.append(i + 1)
                        weights[i] = 0
                        break
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        chosen = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add_fast(chosen, counts, x, chosen_set):
                check_and_add(chosen, counts, x)
                chosen_set.add(x)
                
        return chosen, counts, chosen_set
    
    def local_search(chosen, counts, chosen_set, time_limit=10):
        """Try to improve by inserting or swapping elements."""
        start_time = time.time()
        improved = True
        max_iters = 200
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < time_limit:
            improved = False
            iters += 1
            
            # First, try pure insertions
            for x in range(1, N + 1):
                if can_add_fast(chosen, counts, x, chosen_set):
                    check_and_add(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one element, try to add another
            for i, x in enumerate(chosen[:]):
                # Try to find a y that can replace x
                for y in range(1, N + 1):
                    if y in chosen_set:
                        continue
                    
                    # Check if removing x and adding y would work
                    # First check if y can be added if x were not there
                    # This is complex, so we'll just try it
                    remove_from_set(chosen, counts, x)
                    chosen_set.discard(x)
                    
                    if can_add_fast(chosen, counts, y, chosen_set):
                        check_and_add(chosen, counts, y)
                        chosen_set.add(y)
                        improved = True
                        break
                    else:
                        # Put x back
                        check_and_add(chosen, counts, x)
                        chosen_set.add(x)
                
                if improved:
                    break
                
                # Time check
                if time.time() - start_time > time_limit:
                    break
        return chosen, counts, chosen_set
    
    def structured_build():
        """Build using a structured approach based on modular arithmetic."""
        # Use a base pattern that avoids many conflicts
        chosen = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)
        
        # Start with multiples of a prime to reduce conflicts
        base_primes = [3, 5, 7, 11, 13]
        for p in base_primes:
            for k in range(1, N // p + 1):
                x = p * k
                if x <= N and can_add_fast(chosen, counts, x, chosen_set):
                    check_and_add(chosen, counts, x)
                    chosen_set.add(x)
        
        # Fill in remaining numbers
        for x in range(1, N + 1):
            if can_add_fast(chosen, counts, x, chosen_set):
                check_and_add(chosen, counts, x)
                chosen_set.add(x)
        
        return chosen, counts, chosen_set
    
    start_time = time.time()
    timeout = 115  # Safety margin
    
    # First, try structured build
    current_set, current_counts, current_set_set = structured_build()
    current_set, current_counts, current_set_set = local_search(current_set, current_counts, current_set_set, time_limit=5)
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
    
    # Then try various greedy strategies with local search
    orders = ['random', 'random', 'random', 'ascending', 'descending', 'sparse', 'weighted']
    attempt = 0
    
    while time.time() - start_time < timeout:
        order = random.choice(orders)
        seed = attempt * 7919 + 12345  # Deterministic seed for reproducibility
        current_set, current_counts, current_set_set = run_greedy(order, seed=seed)
        
        # Use more time for local search on promising starts
        time_left = timeout - (time.time() - start_time)
        if time_left > 15:
            ls_time = min(8, time_left - 5)
        else:
            ls_time = max(1, time_left - 2)
        
        current_set, current_counts, current_set_set = local_search(current_set, current_counts, current_set_set, time_limit=ls_time)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
        
        attempt += 1
        
        # Periodically try structured build with different parameters
        if attempt % 5 == 0:
            current_set, current_counts, current_set_set = structured_build()
            current_set, current_counts, current_set_set = local_search(current_set, current_counts, current_set_set, time_limit=3)
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set[:]
            
    return best_set
