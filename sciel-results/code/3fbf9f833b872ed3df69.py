import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def get_score(A):
        A = list(A)
        sums = set()
        prods = set()
        for i in range(len(A)):
            for j in range(len(A)):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    # Initial candidate
    best_A = list(range(1, 21))
    best_score = get_score(best_A)
    
    # Local search
    current_A = best_A[:]
    current_score = best_score
    
    # To diversify, we can try a few different starts
    starts = [
        list(range(1, 21)),
        list(range(1, 20)) + [21],
        list(range(1, 19)) + [20, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27]
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
            
        current_A = start[:]
        current_score = get_score(current_A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        
        while time.time() < deadline and T > T_min:
            # Pick a random element to replace
            idx = random.randint(0, len(current_A) - 1)
            old_val = current_A[idx]
            
            # Pick a new value
            # Bias towards numbers that are close to existing or have factors in common
            # Simple: random 1..10000
            new_val = random.randint(1, 10000)
            
            if new_val in current_A:
                continue
                
            new_A = current_A[:]
            new_A[idx] = new_val
            
            new_score = get_score(new_A)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_A = new_A
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
            
            T *= cooling
            
    return best_A
