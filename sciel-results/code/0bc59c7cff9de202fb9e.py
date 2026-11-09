import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe
    
    # 1. Generate pool of candidate numbers - 5-smooth numbers
    primes = [2, 3, 5]
    max_val = 10000
    pool_set = {1}
    for p in primes:
        current = list(pool_set)
        for x in current:
            val = x
            while val * p <= max_val:
                val *= p
                pool_set.add(val)
    
    pool = sorted(pool_set)
    n_pool = len(pool)
    
    # Precompute exponent representation for each smooth number
    exp_map = {}
    for n in pool:
        a, b, c = 0, 0, 0
        temp = n
        while temp % 2 == 0:
            a += 1
            temp //= 2
        while temp % 3 == 0:
            b += 1
            temp //= 3
        while temp % 5 == 0:
            c += 1
            temp //= 5
        exp_map[n] = (a, b, c)
    
    # 2. Evaluation function
    def evaluate(A):
        s_set = set()
        p_set = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                s_set.add(ai + aj)
                p_set.add(ai * aj)
        return max(len(s_set), len(p_set))
    
    # 3. Simulated Annealing
    best_score = float('inf')
    best_A = None
    
    # Generate diverse initial states
    initial_states = []
    
    # State 1: Smallest 20
    initial_states.append(sorted(pool[:20]))
    
    # State 2: Powers of 2
    p2 = [2**i for i in range(1, 15)]
    p2 = [x for x in p2 if x <= max_val]
    if len(p2) >= 20:
        initial_states.append(sorted(p2[:20]))
    else:
        others = [x for x in pool if x not in p2]
        initial_states.append(sorted(p2 + others[:20-len(p2)]))
    
    # State 3: Geometric-like progression
    geo = [2]
    val = 2
    while len(geo) < 20 and val <= max_val:
        val = int(val * 1.4) + 1
        # Find nearest smooth number >= val
        for cand in pool:
            if cand >= val and cand not in geo:
                geo.append(cand)
                break
    if len(geo) >= 20:
        initial_states.append(sorted(geo[:20]))
    
    # State 4: Mixed small values
    small_vals = [x for x in pool if x <= 100]
    if len(small_vals) >= 20:
        initial_states.append(sorted(random.sample(small_vals, 20)))
    
    # State 5: Random
    initial_states.append(sorted(random.sample(pool, 20)))
    
    # State 6: Random
    initial_states.append(sorted(random.sample(pool, 20)))
    
    # State 7: Balanced exponent sums
    by_exp_sum = sorted(pool, key=lambda x: sum(exp_map[x]))
    initial_states.append(sorted(by_exp_sum[:20]))
    
    # State 8: High exponent diversity
    by_exp_var = sorted(pool, key=lambda x: max(exp_map[x]) - min(exp_map[x]), reverse=True)
    initial_states.append(sorted(by_exp_var[:20]))
    
    for init_A in initial_states:
        if time.time() > deadline:
            break
            
        current_A = list(init_A)
        current_score = evaluate(current_A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = list(current_A)
        
        temp = 30.0
        cooling_rate = 0.9999
        min_temp = 0.01
        
        while temp > min_temp:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            old_exp = exp_map[old_val]
            
            # Choose new value
            # 60% random from pool, 40% exponent-perturbed
            if random.random() < 0.6:
                new_val = random.choice(pool)
            else:
                # Perturb exponents to find nearby smooth number
                new_exp = list(old_exp)
                for _ in range(random.randint(1, 2)):
                    dim = random.randint(0, 2)
                    step = random.choice([-1, 1])
                    new_exp[dim] = max(0, new_exp[dim] + step)
                target = (2 ** new_exp[0]) * (3 ** new_exp[1]) * (5 ** new_exp[2])
                if target <= max_val and target in exp_map:
                    new_val = target
                else:
                    new_val = min(pool, key=lambda x: abs(x - target))
            
            if new_val == old_val or new_val in current_A:
                continue
            
            # Apply and evaluate
            current_A[idx] = new_val
            new_score = evaluate(current_A)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
            else:
                current_A[idx] = old_val
            
            temp *= cooling_rate
    
    # Final local search polish
    if best_A is not None and time.time() < deadline - 3.0:
        current_A = list(best_A)
        current_score = best_score
        for _ in range(3):  # A few passes
            if time.time() > deadline - 1.0:
                break
            improved = True
            while improved:
                improved = False
                for idx in range(20):
                    if time.time() > deadline - 1.0:
                        break
                    old_val = current_A[idx]
                    best_new = old_val
                    best_new_score = current_score
                    # Try a subset of pool (not all, for speed)
                    candidates = random.sample(pool, min(50, len(pool)))
                    for new_val in candidates:
                        if new_val == old_val or new_val in current_A:
                            continue
                        current_A[idx] = new_val
                        s = evaluate(current_A)
                        if s < best_new_score:
                            best_new_score = s
                            best_new = new_val
                    current_A[idx] = best_new
                    if best_new != old_val:
                        current_score = best_new_score
                        improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A
    
    return best_A
