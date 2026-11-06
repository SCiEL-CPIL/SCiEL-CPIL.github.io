import random
import time

def construct():
    n = 20
    max_val = 10000
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial solution: Arithmetic Progression {1, ..., 20}
    best_A = list(range(1, n + 1))
    best_score = None
    
    def get_score(A):
        # Calculate sumset size
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
        
    best_score = get_score(best_A)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    current_A = best_A[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Perturb
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        new_val = random.randint(1, max_val)
        
        # Ensure distinctness
        if new_val in current_A:
            # Try a few times to find a distinct value, or just skip
            # Actually, picking random in 1..10000, collision prob is 20/10000 = 0.2%
            # So just check
            if new_val == old_val:
                continue
            # Check if new_val is in the set excluding idx
            if any(x == new_val for x in current_A if x != old_val):
                continue
                
        current_A[idx] = new_val
        new_score = get_score(current_A)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            current_A[idx] = old_val
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_A
