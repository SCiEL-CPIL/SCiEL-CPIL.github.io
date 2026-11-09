import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80
    
    def is_sidon(elements, indices):
        sums = set()
        n = len(indices)
        for i in range(n):
            for j in range(i, n):
                s = elements[indices[i]] + elements[indices[j]]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(elements):
        n = len(elements)
        # Quick check: full set
        if is_sidon(elements, tuple(range(n))):
            return n
        for size in range(n - 1, 0, -1):
            for combo in combinations(range(n), size):
                if is_sidon(elements, combo):
                    return size
        return 0
    
    best = list(range(12))
    best_val = max_sidon_size(best)
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            current = list(range(12))
        else:
            current = random.sample(range(129), 12)
        
        current_val = max_sidon_size(current)
        if current_val < best_val:
            best_val = current_val
            best = current[:]
        
        # Simulated annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.99995
        no_improve = 0
        
        while time.time() < deadline - 5 and T > T_min:
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            if new_val == old_val:
                continue
            if any(current[i] == new_val for i in range(12) if i != idx):
                continue
            
            current[idx] = new_val
            new_size = max_sidon_size(current)
            
            delta = new_size - current_val
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_val = new_size
                if current_val < best_val:
                    best_val = current_val
                    best = current[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current[idx] = old_val
                no_improve += 1
            
            T *= cooling
            
            if no_improve > 500:
                break
    
    return best
