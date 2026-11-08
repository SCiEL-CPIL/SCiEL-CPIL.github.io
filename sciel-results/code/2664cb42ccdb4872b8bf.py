import time
import random
from math import log, floor
from itertools import product

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    # Optimization: Use list for iteration
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_set = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    best_score = score_set(best_set)
    
    # New Strategy: Product of Digit Sets
    # U = { sum_{i=0}^{k-1} d_i * b^i | d_i in D_i }
    
    def generate_product_set(D, base):
        """
        D is a list of sets of digits.
        D[i] is the set of allowed digits at position i.
        """
        U = set()
        # Recursive or iterative product
        # Iterative product of sets
        current = {0}
        for i, digit_set in enumerate(D):
            new_current = set()
            pow_b = base ** i
            for val in current:
                for d in digit_set:
                    new_current.add(val + d * pow_b)
            current = new_current
        return sorted(current)
        
    def score_and_update(D, base):
        U = generate_product_set(D, base)
        if not U:
            return -1, U
        # Limit size to keep scoring fast
        if len(U) > 6000:
            return -1, U
        s = score_set(U)
        return s, U

    # Search over bases and depths
    # We want to explore digit sets that are not just {0,1}
    # Maybe {0, 2}, {1, 2}, {0, 1, 2}, etc.
    
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]:
        if time.time() > deadline - 15:
            break
            
        # Max depth depends on base to keep |U| reasonable
        # |U| approx prod(|D_i|)
        # If |D_i|=2, |U|=2^k. 2^12=4096. OK.
        # If |D_i|=3, |U|=3^k. 3^8=6561. OK.
        
        max_depth = 12 if base == 2 else 10 if base <= 4 else 8 if base <= 8 else 6
        
        for k in range(2, max_depth + 1):
            if time.time() > deadline - 5:
                break
                
            # Initialize D with {0, 1}
            D = [set([0, 1])] * k
            # Ensure 0 is in U (it is, since 0 in all D_i)
            
            current_score, current_U = score_and_update(D, base)
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_U
            
            # Local Search on D
            # We will randomly perturb D
            for _ in range(50): # Max iterations per (base, k)
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random position
                pos = random.randint(0, k - 1)
                
                # Generate a new digit set for this position
                # Current digits: D[pos]
                # Options:
                # 1. Add a random digit
                # 2. Remove a random digit (if len > 1)
                # 3. Swap a digit
                
                new_D = [set(s) for s in D] # Deep copy
                old_set = new_D[pos]
                
                action = random.random()
                if action < 0.4:
                    # Add
                    available = [d for d in range(base) if d not in old_set]
                    if available:
                        new_D[pos].add(random.choice(available))
                elif action < 0.7:
                    # Remove
                    if len(old_set) > 1:
                        to_remove = random.choice(list(old_set))
                        new_D[pos].remove(to_remove)
                        # Ensure we don't remove 0 if we want 0 in U? 
                        # Actually, if 0 is removed from D_0, 0 might not be in U.
                        # But we can just check score.
                else:
                    # Swap
                    available = [d for d in range(base) if d not in old_set]
                    if available and len(old_set) > 0:
                        to_remove = random.choice(list(old_set))
                        to_add = random.choice(available)
                        new_D[pos].remove(to_remove)
                        new_D[pos].add(to_add)
                        
                new_score, new_U = score_and_update(new_D, base)
                
                if new_score > current_score:
                    current_score = new_score
                    current_U = new_U
                    D = new_D
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_U

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
