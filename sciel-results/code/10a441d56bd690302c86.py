import time
import random
import sys

START_TIME = time.time()
DEADLINE = 58.0  # Safety margin from 90s limit

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums_bitset(A):
    """Compute subset sums using Python integers as bitsets.
    Returns list where S[k] is a bitset of all subset sums of size k."""
    n = len(A)
    S = [0] * (n + 1)
    S[0] = 1  # sum 0 is achievable with empty subset
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if S[j]:
                S[j+1] |= (S[j] << x)
    return S

def is_valid(A):
    S = compute_sums_bitset(A)
    n = len(A)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if S[i] & S[j]:
                return False
    return True

def get_cost(A):
    """Count total number of sum collisions across different cardinalities."""
    S = compute_sums_bitset(A)
    n = len(A)
    cost = 0
    for i in range(n + 1):
        Si = S[i]
        for j in range(i + 1, n + 1):
            Sj = S[j]
            # Count bits in intersection
            inter = Si & Sj
            cost += bin(inter).count('1')
    return cost

def try_sa(K, time_limit=5.0):
    """Simulated annealing to find a valid set of size K."""
    start_time = time.time()
    
    best_cost = float('inf')
    best_A = None
    
    while time.time() - start_time < time_limit and not is_done():
        # Initialize with random sample
        current_A = random.sample(range(1, 257), K)
        current_cost = get_cost(current_A)
        
        if current_cost == 0:
            return current_A
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = current_A[:]
            
        T = 100.0
        T_min = 0.5
        cooling = 0.99
        
        while time.time() - start_time < time_limit and not is_done():
            if T < T_min:
                break
            
            # Pick a random element to replace
            i = random.randint(0, K - 1)
            old_x = current_A[i]
            
            # Pick a random replacement
            new_x = random.randint(1, 256)
            
            if new_x == old_x or new_x in current_A:
                continue
            
            current_A[i] = new_x
            new_cost = get_cost(current_A)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
                if current_cost == 0:
                    return current_A
            else:
                current_A[i] = old_x
            
            T *= cooling
    
    return best_A if best_cost == 0 else None

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    best_solution = base_solution
    best_size = len(base_solution)
    
    # Try to find larger sets using bitset-based SA
    for K in range(10, 16):
        if is_done():
            break
        
        remaining = DEADLINE - elapsed()
        if remaining < 2.0:
            break
            
        # Allocate time for this K
        time_limit = min(remaining / (16 - K), 8.0)
        
        res = try_sa(K, time_limit)
        if res is not None:
            if len(res) > best_size:
                best_solution = res
                best_size = len(res)
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
