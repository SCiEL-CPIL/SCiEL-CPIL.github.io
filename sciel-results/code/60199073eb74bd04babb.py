import time
import random
import sys

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin from 60s limit

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums(A):
    n = len(A)
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    for i in range(n):
        x = A[i]
        # Iterate backwards to avoid using x multiple times in same step
        for j in range(i, -1, -1):
            if S[j]:
                # Use local variable for speed
                Sj = S[j]
                Sj1 = S[j+1]
                for s in Sj:
                    Sj1.add(s + x)
    return S

def is_valid(A):
    S = compute_sums(A)
    n = len(A)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if S[i] & S[j]:
                return False
    return True

def get_cost(A):
    S = compute_sums(A)
    n = len(A)
    cost = 0
    for i in range(n + 1):
        Si = S[i]
        for j in range(i + 1, n + 1):
            Sj = S[j]
            cost += len(Si & Sj)
    return cost

def try_sa(K, time_limit=5.0):
    start_time = time.time()
    
    # Multiple restarts if time permits
    best_cost = float('inf')
    best_A = None
    
    while time.time() - start_time < time_limit and not is_done():
        # Initialize
        # Bias towards smaller numbers? 
        # Random sample is fine.
        current_A = random.sample(range(1, 257), K)
        current_cost = get_cost(current_A)
        
        if current_cost == 0:
            return current_A
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = current_A[:]
            
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        no_improve_count = 0
        
        while time.time() - start_time < time_limit and not is_done():
            if T < T_min:
                break
                
            # Pick a random element in A to replace
            i = random.randint(0, K - 1)
            old_x = current_A[i]
            
            # Pick a random replacement from 1..256
            new_x = random.randint(1, 256)
            
            if new_x == old_x:
                continue
                
            if new_x in current_A:
                continue
                
            current_A[i] = new_x
            new_cost = get_cost(current_A)
            
            delta = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if current_cost == 0:
                    return current_A
            else:
                current_A[i] = old_x
                no_improve_count += 1
                
            T *= cooling
            
            # If stuck for a long time, restart inner loop?
            # Or just let it cool down.
            
    return best_A

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    best_solution = base_solution
    best_size = len(base_solution)
    
    # Try to find larger sets
    # Max possible size is limited by 256.
    # Let's try up to 14.
    for K in range(10, 15):
        if is_done():
            break
        
        remaining = DEADLINE - elapsed()
        if remaining < 2.0:
            break
            
        # Allocate time based on remaining tries
        # Give more time to smaller Ks as they are more likely to be solvable?
        # Or equal time?
        time_limit = remaining / (15 - K)
        
        res = try_sa(K, time_limit)
        if res is not None:
            if len(res) > best_size:
                best_solution = res
                best_size = len(res)
        # If res is None, we didn't find a valid set of size K in time.
        # We continue to next K? 
        # If K=10 is impossible, K=11 is also likely impossible.
        # But we don't know for sure.
        # However, if we spend all time on K=10 and fail, we have no time for K=11.
        # The loop structure handles this: if try_sa takes all time, next iteration sees is_done().
        
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
