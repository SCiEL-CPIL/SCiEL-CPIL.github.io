import random
import time
from itertools import combinations

def _is_sidon_4(subset):
    """Check if a 4-element subset is Sidon."""
    # Sums of pairs i<=j
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _count_sidon_4s(ambient_set):
    """Count the number of 4-element subsets that are Sidon."""
    count = 0
    for combo in combinations(ambient_set, 4):
        if _is_sidon_4(combo):
            count += 1
    return count

def construct():
    deadline = time.time() + 55  # 5s safety margin
    
    # Start with a dense set
    current_set = list(range(12))
    current_cost = _count_sidon_4s(current_set)
    
    best_set = list(current_set)
    best_cost = current_cost
    
    rng = random.Random(42)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline and T > T_min:
        improved = False
        # Try multiple steps per temperature
        for _ in range(100):
            if time.time() > deadline:
                break
            
            idx = rng.randint(0, 11)
            old_val = current_set[idx]
            new_val = rng.randint(0, 128)
            
            if new_val in current_set:
                continue
                
            candidate_set = list(current_set)
            candidate_set[idx] = new_val
            
            new_cost = _count_sidon_4s(candidate_set)
            
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < pow(2.71828, -delta / T):
                current_set = candidate_set
                current_cost = new_cost
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = list(current_set)
                    
                    if best_cost == 0:
                        # Found a set with no Sidon 4-subsets
                        # Max Sidon size is likely 3
                        return best_set
        
        T *= cooling
        
    return best_set
