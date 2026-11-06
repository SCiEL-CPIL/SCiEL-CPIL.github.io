import random
import math
import time

def construct():
    # Generate 7-smooth numbers up to 10000
    smooth_numbers = set([1])
    max_val = 10000
    
    candidates = [1]
    for a in range(15):
        for b in range(10):
            for c in range(7):
                for d in range(5):
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val <= max_val:
                        candidates.append(val)
    
    candidates = sorted(list(set(candidates)))
    
    # Initial set: 1 to 20
    initial_set = list(range(1, 21))
    
    def evaluate(A):
        A_set = set(A)
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))
        
    current_A = initial_set[:]
    current_score = evaluate(current_A)
    
    best_A = current_A[:]
    best_score = current_score
    
    # Tabu Search
    start_time = time.time()
    timeout = 55
    
    # Candidate pool: 7-smooth numbers + small integers
    # We also add some random numbers to escape local optima
    base_pool = list(set(candidates + list(range(1, 101))))
    
    tabu_list = [] # List of (index, old_val, new_val)
    tabu_tenure = 10
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < timeout:
        # Generate neighbors
        neighbors = []
        for i in range(20):
            old_val = current_A[i]
            # Sample a subset of the pool to keep it fast, or iterate all if pool is small
            # Pool size is ~200-300. 20 * 300 = 6000 evaluations.
            # Each eval is O(400). 2.4M ops. Might be too slow for 55s?
            # 2.4M * (small constant) is fine in Python if optimized.
            # But let's sample 50 candidates per index to be safe.
            sampled_pool = random.sample(base_pool, min(50, len(base_pool)))
            for new_val in sampled_pool:
                if new_val == old_val:
                    continue
                if new_val in current_A:
                    continue
                
                # Check tabu
                is_tabu = False
                for t_move in tabu_list:
                    if t_move[0] == i and t_move[1] == old_val and t_move[2] == new_val:
                        is_tabu = True
                        break
                
                new_A = current_A[:]
                new_A[i] = new_val
                new_score = evaluate(new_A)
                
                # Aspiration: if it improves best, accept even if tabu
                if new_score < best_score:
                    is_tabu = False
                
                if not is_tabu:
                    neighbors.append((new_score, i, old_val, new_val, new_A))
        
        if not neighbors:
            # If no valid moves, restart or random jump
            break
            
        # Select best neighbor
        neighbors.sort(key=lambda x: x[0])
        best_move = neighbors[0]
        best_move_score, idx, old_val, new_val, new_A = best_move
        
        # Update tabu list
        tabu_list.append((idx, old_val, new_val))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
        current_A = new_A
        current_score = best_move_score
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Diversification: if stuck, do a random jump
        if no_improve_count > max_no_improve:
            # Replace 3 random elements with random values from pool
            for _ in range(3):
                idx = random.randint(0, 19)
                new_val = random.choice(base_pool)
                if new_val not in current_A:
                    current_A[idx] = new_val
            current_score = evaluate(current_A)
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
            no_improve_count = 0
            tabu_list = [] # Clear tabu after jump

    return best_A
