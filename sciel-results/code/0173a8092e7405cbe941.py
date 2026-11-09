import time
from math import gcd
from functools import reduce
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i

    N = 30
    # We need to return 30 coefficients c[0]...c[29] corresponding to 1/1 ... 1/30
    # The sum is S = sum(c[i-1] * A[i]) / L
    # We want to minimize |S| such that S != 0.
    # Since L is fixed, we minimize |sum(c[i-1] * A[i])|.
    
    # Initial state: all zeros? No, sum must be nonzero. 
    # Actually, if we start with all zeros, sum is 0. 
    # We need to ensure the final result is nonzero.
    # During SA, we can allow sum=0 but penalize it heavily or reject it if we want to stay in nonzero space.
    # However, the problem says "The signed sum ... must be nonzero".
    # It is easier to search over all {-1,0,1}^30 and just filter out 0 at the end, 
    # or guide the search away from 0.
    # Given the density of points, finding a nonzero value close to 0 is likely.
    # Let's just find the minimum absolute value. If it's 0, we need the next smallest.
    # But 0 is a "hole". The set of reachable sums is discrete.
    # If the minimum is 0, we must find the min non-zero.
    
    def compute_sum(c):
        s = 0
        for i in range(N):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s

    def simulated_annealing(init_state, max_iters, start_temp):
        c = init_state[:]
        cur_sum = compute_sum(c)
        cur_abs = abs(cur_sum)
        
        best_c = c[:]
        best_sum = cur_sum
        best_abs = cur_abs
        
        # If best is 0, we need to be careful. 
        # We track the best non-zero separately if 0 is found?
        # Actually, if we find 0, it's the best "value" but invalid.
        # The task requires nonzero.
        # So we want min |sum| subject to sum != 0.
        
        # Let's track best_nonzero
        if cur_sum != 0:
            best_nonzero_c = c[:]
            best_nonzero_sum = cur_sum
            best_nonzero_abs = cur_abs
        else:
            best_nonzero_c = None
            best_nonzero_sum = 0
            best_nonzero_abs = float('inf')

        step = 1
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random index to flip or change
            idx = random.randint(0, N - 1)
            old_val = c[idx]
            
            # Choose new value: -1, 0, 1
            # To make moves more effective, maybe bias towards flipping sign or zeroing?
            # Simple: random choice from {-1, 0, 1} excluding old_val?
            # Or just random from {-1, 0, 1}.
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
                
            delta = (new_val - old_val) * A[idx + 1]
            new_sum = cur_sum + delta
            new_abs = abs(new_sum)
            
            # Acceptance criteria
            # We want to minimize new_abs, but avoid new_sum == 0 if possible?
            # If new_sum == 0, it's a "trap" for the final answer, but good for intermediate?
            # Actually, if we hit 0, we can't return it. 
            # If we accept 0, we might get stuck there if we only accept improvements.
            # So we should treat 0 as "bad" for the objective function we are optimizing?
            # Let's define the cost function: 
            # Cost = |sum| if sum != 0, else INF.
            # This forces the search to avoid 0.
            
            cur_cost = cur_abs if cur_sum != 0 else float('inf')
            new_cost = new_abs if new_sum != 0 else float('inf')
            
            delta_cost = new_cost - cur_cost
            
            temp = start_temp * (0.999 ** it)
            
            if delta_cost < 0 or (new_cost < float('inf') and random.random() < pow(2.71828, -delta_cost / max(temp, 1e-10))):
                # Accept
                c[idx] = new_val
                cur_sum = new_sum
                cur_abs = new_abs
                
                if new_sum != 0:
                    if new_abs < best_nonzero_abs:
                        best_nonzero_abs = new_abs
                        best_nonzero_sum = new_sum
                        best_nonzero_c = c[:]
            else:
                # Reject
                pass
                
        return best_nonzero_c, best_nonzero_sum, best_nonzero_abs

    # Run multiple restarts
    best_global_c = None
    best_global_sum = 0
    best_global_abs = float('inf')
    
    # Initial states
    # 1. Random
    # 2. Greedy: try to cancel terms
    # 3. Alternating signs
    
    num_restarts = 1000 # We will break early based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Choose initial state
        if r == 0:
            # Start with alternating signs to get close to 0 quickly
            init = [1 if i % 2 == 0 else -1 for i in range(N)]
        elif r == 1:
            # Start with random sparse
            init = [random.choice([-1, 0, 1]) for _ in range(N)]
        else:
            # Perturb the previous best? Or random?
            # Let's just use random for diversity
            init = [random.choice([-1, 0, 1]) for _ in range(N)]
            
        # If init sum is 0, it's a bad start for our cost function, but SA will move away.
        # Let's run SA for a number of steps proportional to remaining time?
        # Or fixed steps per restart?
        # 30 variables. 1000 steps per restart is fast.
        steps = 2000
        temp = 1000000.0 # L is huge, so differences are huge. 
        # L for 1..30 is approx 2.3e12. 
        # So temp should be in that range.
        
        c, s, a = simulated_annealing(init, steps, temp)
        
        if c is not None and a < best_global_abs:
            best_global_abs = a
            best_global_sum = s
            best_global_c = c[:]
            
        # If we found a very small value, we can stop early?
        # The minimum non-zero integer is 1.
        if best_global_abs == 1:
            break

    if best_global_c is None:
        # Fallback: return a valid nonzero sum
        # e.g. c[0]=1, rest 0 -> sum = 1/1 = 1. Value = 1.
        best_global_c = [1] + [0]*29
        best_global_sum = A[1]
        best_global_abs = abs(best_global_sum)
        
    return best_global_c
