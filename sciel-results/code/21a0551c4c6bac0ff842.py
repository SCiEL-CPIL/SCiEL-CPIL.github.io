import time
import random
from math import gcd

def construct():
    start_time = time.time()
    MAX_TIME = 115.0 # Safety margin
    
    N = 511
    # Precompute values
    values = [0.0] * (N + 1)
    for i in range(1, N + 1):
        values[i] = 1.0 / (512.0 - i)
        
    # Precompute prime factors for gcd checks? 
    # Actually, gcd is fast enough. 
    # But for conflict tracking, we need to know which numbers conflict with a given 'a'.
    # Let's precompute conflicts[a] = list of x in 1..511 such that gcd(a, x) > 1.
    # This is O(N^2) ~ 250k operations. Fine.
    
    conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for x in range(1, N + 1):
            if a != x and gcd(a, x) > 1:
                conflicts[a].append(x)
                
    # 1. Greedy Initialization
    def greedy_init():
        chosen = []
        for a in range(N, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen

    best_set = greedy_init()
    best_score = sum(values[a] for a in best_set)
    
    # Helper to calculate score
    def calc_score(s):
        return sum(values[a] for a in s)
        
    # Local Search with Simulated Annealing
    def local_search(initial_set):
        current_set = set(initial_set)
        current_score = calc_score(current_set)
        best_local_set = list(current_set)
        best_local_score = current_score
        
        # Conflict count array
        # conflict[x] = number of elements in current_set that conflict with x
        conflict = [0] * (N + 1)
        for a in current_set:
            for x in conflicts[a]:
                conflict[x] += 1
                
        T = 5.0 # Initial temperature
        min_T = 0.01
        cooling = 0.999
        
        while time.time() - start_time < MAX_TIME:
            # Try a move: Remove a, Add b
            # Choose a random element to remove
            if not current_set:
                break
            a = random.choice(list(current_set))
            
            # Remove a
            current_set.remove(a)
            for x in conflicts[a]:
                conflict[x] -= 1
                
            # Find best valid candidate b
            best_b = -1
            best_b_val = 0.0
            # We want to maximize score, so pick b with highest values[b]
            # among those with conflict[b] == 0
            for b in range(1, N + 1):
                if b not in current_set and conflict[b] == 0:
                    if values[b] > best_b_val:
                        best_b_val = values[b]
                        best_b = b
                        
            if best_b == -1:
                # Cannot add anything, revert
                current_set.add(a)
                for x in conflicts[a]:
                    conflict[x] += 1
                # Maybe try a different a or just cool down
                T *= cooling
                continue
                
            # Calculate new score
            new_score = current_score - values[a] + best_b_val
            delta = new_score - current_score
            
            # Simulated Annealing acceptance
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                # Accept
                current_set.add(best_b)
                for x in conflicts[best_b]:
                    conflict[x] += 1
                current_score = new_score
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = list(current_set)
            else:
                # Reject, revert
                current_set.add(a)
                for x in conflicts[a]:
                    conflict[x] += 1
            
            T *= cooling
            if T < min_T:
                break
                
        return best_local_set, best_local_score

    # Run multiple restarts to find better global optimum
    # The greedy start is deterministic. We can perturb it.
    for _ in range(10): # Max 10 restarts if time permits
        if time.time() - start_time > MAX_TIME:
            break
            
        # Perturb the best set found so far, or start fresh?
        # Starting fresh with greedy is safe. Perturbing best might get stuck in same basin.
        # Let's try starting from best and perturbing.
        current_best = best_set
        if len(current_best) > 1:
            # Remove a random element to allow re-optimization
            # Or just run local search on the current best.
            # To escape local optima, we should introduce randomness.
            # Let's run local search on a slightly perturbed version of the best set.
            perturbed = list(best_set)
            if perturbed:
                # Remove 1-3 random elements
                num_removals = random.randint(1, min(3, len(perturbed)))
                for _ in range(num_removals):
                    if perturbed:
                        idx = random.randint(0, len(perturbed)-1)
                        perturbed.pop(idx)
            
            s, score = local_search(perturbed)
            if score > best_score:
                best_score = score
                best_set = s
        else:
            s, score = local_search(greedy_init())
            if score > best_score:
                best_score = score
                best_set = s

    return best_set
