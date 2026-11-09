import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_cost(A):
        sums = set()
        prods = set()
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    def local_search(init_A, time_limit):
        A = init_A.copy()
        current_cost = get_cost(A)
        best_A = A.copy()
        best_cost = current_cost
        
        while time.time() < time_limit:
            idx = random.randint(0, len(A) - 1)
            old_val = A[idx]
            
            # Sample new value
            # If current cost is dominated by products, try to find collisions.
            # If dominated by sums, try to keep numbers close.
            # Simple heuristic: sample from 1..200
            new_val = random.randint(1, 200)
            
            if new_val == old_val or new_val in A:
                continue
                
            A[idx] = new_val
            new_cost = get_cost(A)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = A.copy()
            else:
                A[idx] = old_val
                
        return best_A, best_cost

    # Initial state 1: Arithmetic Progression
    A1 = list(range(1, 21))
    
    # Initial state 2: 5-Smooth numbers
    smooths = []
    for a in range(0, 15):
        for b in range(0, 10):
            for c in range(0, 8):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000:
                    smooths.append(val)
    smooths = sorted(set(smooths))
    A2 = smooths[:20]
    
    # Run search on A1
    time_per_search = (deadline - time.time()) / 2.0
    A1_best, cost1 = local_search(A1, time.time() + time_per_search)
    
    # Run search on A2
    A2_best, cost2 = local_search(A2, time.time() + time_per_search)
    
    # Return the better one
    if cost1 <= cost2:
        return A1_best
    else:
        return A2_best
