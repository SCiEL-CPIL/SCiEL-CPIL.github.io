import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    candidates = list(range(2, 41))
    
    # Helper to check if a set has distinct subset products
    def is_valid(S):
        prods = {1}
        for x in S:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            if len(prods | new_prods) != len(prods) + len(new_prods):
                return False
            prods |= new_prods
        return True

    # Initial solution: Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
        
    best_set = [x for x in candidates if is_prime(x)]
    best_size = len(best_set)
    
    # Local search
    def local_search(S):
        S = list(S)
        improved = True
        while improved:
            improved = False
            # Try to add
            for x in candidates:
                if x not in S:
                    if is_valid(S + [x]):
                        S.append(x)
                        improved = True
                        break # Restart loop with new S
            if improved:
                continue
                
            # Try to swap
            for i in range(len(S)):
                for j in candidates:
                    if j in S: continue
                    new_S = S[:i] + S[i+1:] + [j]
                    if is_valid(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        return S

    # Randomized restarts
    while time.time() < deadline:
        # Perturb best_set
        current = list(best_set)
        # Remove some
        for _ in range(random.randint(0, 3)):
            if current:
                current.pop(random.randint(0, len(current)-1))
        # Add some random
        for _ in range(random.randint(0, 3)):
            x = random.choice(candidates)
            if x not in current:
                current.append(x)
                
        # Local search
        candidate_set = local_search(current)
        
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
            if best_size >= 20: # Theoretical max? Unlikely, but check
                break

    return [int(x) for x in best_set]
