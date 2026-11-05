import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    # Evaluator kills at 90s, we stop at 80s to be safe
    deadline = start_time + 80.0
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_primes(n):
        if n <= 1:
            return []
        primes = []
        while n > 1:
            p = spf[n]
            primes.append(p)
            while n % p == 0:
                n //= p
        return primes

    n_elements = 12
    max_val = 10000
    
    def random_state():
        return random.sample(range(1, max_val + 1), n_elements)
    
    def calculate_cost(state):
        prime_counts = defaultdict(int)
        total_distinct = 0
        for i in range(n_elements):
            for j in range(i+1, n_elements):
                s = state[i] + state[j]
                for p in get_primes(s):
                    if prime_counts[p] == 0:
                        total_distinct += 1
                    prime_counts[p] += 1
        return total_distinct, prime_counts

    best_state = None
    best_cost = float('inf')
    
    # Initial random restarts to find a good starting point
    for _ in range(10):
        if time.time() > deadline:
            break
        state = random_state()
        cost, _ = calculate_cost(state)
        if cost < best_cost:
            best_cost = cost
            best_state = state[:]
            
    if best_state is None:
        return list(range(1, 13))
        
    current_state = best_state[:]
    current_cost, current_counts = calculate_cost(current_state)
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5.0:
        if no_improve_count > max_no_improve:
            # Restart with a new random state mixed with best
            current_state = best_state[:]
            # Perturb
            for _ in range(3):
                idx = random.randint(0, n_elements - 1)
                current_state[idx] = random.randint(1, max_val)
                # Ensure distinct
                while len(set(current_state)) < n_elements:
                    current_state[random.randint(0, n_elements - 1)] = random.randint(1, max_val)
            current_cost, current_counts = calculate_cost(current_state)
            T = 10.0
            no_improve_count = 0
            continue

        i = random.randint(0, n_elements - 1)
        old_val = current_state[i]
        
        # Generate new value
        # Try to bias towards small numbers initially? No, random is safer for diversity.
        new_val = random.randint(1, max_val)
        
        # Ensure distinctness
        if new_val in current_state and new_val != old_val:
            continue
            
        # Calculate delta cost
        temp_counts = defaultdict(int, current_counts)
        temp_cost = current_cost
        
        # Remove old contributions
        for j in range(n_elements):
            if j == i:
                continue
            s_old = old_val + current_state[j]
            for p in get_primes(s_old):
                temp_counts[p] -= 1
                if temp_counts[p] == 0:
                    temp_cost -= 1
        
        # Add new contributions
        for j in range(n_elements):
            if j == i:
                continue
            s_new = new_val + current_state[j]
            for p in get_primes(s_new):
                if temp_counts[p] == 0:
                    temp_cost += 1
                temp_counts[p] += 1
        
        # Acceptance criterion
        if temp_cost < current_cost or (T > T_min and random.random() < pow(2.718281828, -(temp_cost - current_cost) / T)):
            current_state[i] = new_val
            current_cost = temp_cost
            current_counts = temp_counts
            no_improve_count = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = current_state[:]
                best_counts = current_counts[:]
        else:
            no_improve_count += 1
            
        T *= alpha
        if T < T_min:
            T = T_min

    return best_state
