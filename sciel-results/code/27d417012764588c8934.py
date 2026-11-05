import random
import math
import time

def construct():
    def compute_sizes(A):
        sums = set()
        prods = set()
        for i in range(20):
            for j in range(20):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return len(sums), len(prods)
    
    def objective(A):
        s, p = compute_sizes(A)
        return max(s, p)

    def run_sa(init_A, max_time, seed):
        rng = random.Random(seed)
        A = init_A[:]
        current_score = objective(A)
        best_A = A[:]
        best_score = current_score
        
        start_time = time.time()
        start_temp = 50.0
        end_temp = 0.1
        
        # Adaptive step count? Or fixed?
        # Let's run until time limit.
        while time.time() - start_time < max_time:
            # Estimate remaining steps? No, just loop.
            # Temperature schedule based on steps is hard if we don't know total steps.
            # Use time-based cooling?
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                break
            # Linear cooling in time?
            temp = start_temp * (1 - elapsed / max_time) + end_temp
            
            move_type = rng.random()
            new_A = A[:]
            
            if move_type < 0.4:
                # Swap
                i, j = rng.sample(range(20), 2)
                new_A[i], new_A[j] = new_A[j], new_A[i]
            elif move_type < 0.7:
                # Replace with random
                i = rng.randint(0, 19)
                new_val = rng.randint(1, 10000)
                if new_val in new_A:
                    continue
                new_A[i] = new_val
            else:
                # Replace with derived
                i = rng.randint(0, 19)
                j = rng.randint(0, 19)
                if i == j:
                    continue
                op = rng.randint(0, 2)
                if op == 0:
                    new_val = new_A[i] + new_A[j]
                elif op == 1:
                    new_val = new_A[i] * new_A[j]
                else:
                    new_val = new_A[i] * 2
                if new_val > 10000 or new_val < 1 or new_val in new_A:
                    continue
                new_A[i] = new_val
                
            if len(set(new_A)) != 20:
                continue
                
            new_score = objective(new_A)
            
            if new_score < current_score:
                current_score = new_score
                A = new_A
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
            else:
                delta = new_score - current_score
                if delta < 0:
                    accept = True
                else:
                    # Avoid division by zero if temp is very small
                    if temp < 0.01:
                        accept = False
                    else:
                        accept_prob = math.exp(-delta / temp)
                        accept = rng.random() < accept_prob
                
                if accept:
                    current_score = new_score
                    A = new_A
                    
        return best_A, best_score

    # Generate initial sets
    inits = []
    # 1. Arithmetic
    inits.append(list(range(1, 21)))
    
    # 2. Geometric (powers of 2, then fill)
    gp = []
    val = 1
    while val <= 10000 and len(gp) < 20:
        gp.append(val)
        val *= 2
    # Fill remaining with random small numbers?
    # Actually, just use what we have and pad with 1..N?
    # Better: use a different base.
    # Let's just use random samples for other inits.
    
    # 3. Random
    rng_init = random.Random(1)
    inits.append(sorted(rng_init.sample(range(1, 10001), 20)))
    
    # 4. Multiplicative lattice
    # {2^i * 3^j}
    lattice = set()
    for i in range(15):
        for j in range(15):
            v = (2**i) * (3**j)
            if v <= 10000:
                lattice.add(v)
    if len(lattice) >= 20:
        lattice_list = sorted(list(lattice))[:20]
        inits.append(lattice_list)
        
    best_overall_A = None
    best_overall_score = float('inf')
    
    total_time = 55.0 # Leave 5s for overhead
    num_restarts = 5
    time_per_restart = total_time / num_restarts
    
    for i, init_A in enumerate(inits):
        if i >= num_restarts:
            break
        # Ensure valid
        if len(set(init_A)) != 20 or any(x < 1 or x > 10000 for x in init_A):
            continue
            
        A, sc = run_sa(init_A, time_per_restart, seed=i)
        if sc < best_overall_score:
            best_overall_score = sc
            best_overall_A = A[:]
            
    if best_overall_A is None:
        return list(range(1, 21))
        
    return sorted(best_overall_A)
