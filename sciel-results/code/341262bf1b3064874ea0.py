import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 512
    vals = [0.0] * N
    for a in range(1, N):
        vals[a] = 1.0 / (N - a)
        
    # Precompute conflicts
    conflicts = [[] for _ in range(N)]
    for c in range(1, N):
        for s in range(1, N):
            if math.gcd(c, s) != 1:
                conflicts[c].append(s)
                
    # Helper to get score from boolean array
    def calc_score(in_set):
        return sum(vals[i] for i in range(1, N) if in_set[i])
        
    # 1. Initial States
    # State A: Greedy Largest First
    in_set_a = [False] * N
    for a in range(N-1, 0, -1):
        ok = True
        for b in range(1, N):
            if in_set_a[b] and math.gcd(a, b) != 1:
                ok = False
                break
        if ok:
            in_set_a[a] = True
    score_a = calc_score(in_set_a)
    
    # State B: All Primes
    is_prime = [True] * N
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    in_set_b = [False] * N
    for i in range(2, N):
        if is_prime[i]:
            in_set_b[i] = True
    score_b = calc_score(in_set_b)
    
    # Choose best start
    if score_a >= score_b:
        in_set = in_set_a
        current_score = score_a
    else:
        in_set = in_set_b
        current_score = score_b
        
    best_score = current_score
    best_set = in_set[:]
    
    # Simulated Annealing
    T = 0.1  # Start with moderate T
    cooling_rate = 0.999
    min_T = 0.001
    
    random.seed(42)
    steps_since_improve = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        c = random.randint(1, N-1)
        
        # Calculate Delta
        delta = vals[c]
        for s in conflicts[c]:
            if in_set[s]:
                delta -= vals[s]
                
        accepted = False
        if delta > 0:
            accepted = True
        else:
            if T > 0:
                exp_arg = delta / T
                if exp_arg > -700:
                    if random.random() < math.exp(exp_arg):
                        accepted = True
                        steps_since_improve += 1 # Count non-improving moves? No, count total steps?
        
        if accepted:
            # Apply Move
            for s in conflicts[c]:
                if in_set[s]:
                    in_set[s] = False
            in_set[c] = True
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_set = in_set[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Restart if stuck
        if steps_since_improve > max_steps_no_improve:
            # Reheat
            T = 0.5
            steps_since_improve = 0
            # Maybe randomize current state slightly?
            # For now, just reheat.
            
    result = [i for i in range(1, N) if best_set[i]]
    return result
