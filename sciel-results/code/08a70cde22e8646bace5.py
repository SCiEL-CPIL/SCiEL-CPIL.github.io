import math
import time
import heapq

# Precompute 12!
T12 = math.factorial(12)  # 479001600

# Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
exponents = [10, 5, 2, 1, 1]
T12_EXP = tuple(exponents)  # (10, 5, 2, 1, 1)

def generate_divisors():
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * p_power)
            p_power *= p
        divisors = new_divisors
    return divisors

ALL_DIVISORS = generate_divisors()
VALID_DIVISORS = sorted([d for d in ALL_DIVISORS if d > 12])

# Precompute factorization for each valid divisor
def factorize(n):
    factors = [0] * 5
    for i, p in enumerate(primes):
        while n % p == 0:
            factors[i] += 1
            n //= p
    return tuple(factors)

# Map divisor -> exponent tuple
DIV_FACTORS = {}
for d in VALID_DIVISORS:
    DIV_FACTORS[d] = factorize(d)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    # State: (max_factor_used, exponent_vector, path)
    # We want to go from T12_EXP to (0,0,0,0,0) by dividing by valid divisors
    # Each step: choose a divisor d > 12 that divides the current number
    # The "cost" is the maximum divisor used
    
    # Dijkstra: state = exponent_vector, cost = max factor so far
    # Start: T12_EXP, cost = 0
    # Goal: (0,0,0,0,0)
    
    # Priority queue: (max_factor, exponent_tuple, path)
    # We want to minimize max_factor
    
    # For efficiency, we'll use a visited dict to track the best (min max_factor) 
    # seen for each exponent state
    
    best_cost = {}  # exponent_tuple -> min max_factor seen
    best_path = {}  # exponent_tuple -> path that achieved it
    
    # Initial state
    init_state = T12_EXP
    init_cost = 0
    
    pq = [(init_cost, init_state, [])]
    best_cost[init_state] = init_cost
    best_path[init_state] = []
    
    # Precompute for each exponent state, which divisors can be applied
    # A divisor with factorization f can be applied to state s if s[i] >= f[i] for all i
    
    # For speed, precompute list of (divisor, factor_tuple) sorted by divisor value
    div_list = [(d, DIV_FACTORS[d]) for d in VALID_DIVISORS]
    div_list.sort()  # Sort by divisor value ascending
    
    found_solution = None
    
    while pq:
        if time.time() > deadline:
            break
            
        cost, state, path = heapq.heappop(pq)
        
        # If this state was already reached with lower cost, skip
        if state in best_cost and best_cost[state] < cost:
            continue
            
        # Check if we reached the goal
        if state == (0, 0, 0, 0, 0):
            found_solution = path
            break
        
        # Try applying each divisor
        # For pruning: only try divisors that are >= cost (since cost is the max so far,
        # and we want to minimize the max, trying smaller divisors first helps)
        # Actually, we should try all valid divisors, but the new cost will be max(cost, d)
        
        for d, f in div_list:
            # Check if we can apply this divisor to current state
            valid = True
            new_state = [0] * 5
            for i in range(5):
                if state[i] < f[i]:
                    valid = False
                    break
                new_state[i] = state[i] - f[i]
            
            if not valid:
                continue
            
            new_state = tuple(new_state)
            new_cost = max(cost, d)
            
            # Pruning: if we've already found a solution with cost <= new_cost, 
            # and new_cost >= best solution cost, we can skip
            if found_solution is not None and new_cost >= found_solution_max if 'found_solution_max' in dir() else False:
                pass
            
            # If this state was reached with a lower or equal cost, skip
            if new_state in best_cost and best_cost[new_state] <= new_cost:
                continue
            
            # Update best
            best_cost[new_state] = new_cost
            best_path[new_state] = path + [d]
            
            heapq.heappush(pq, (new_cost, new_state, path + [d]))
    
    if found_solution is not None:
        # Verify
        prod = 1
        for f in found_solution:
            prod *= f
        if prod == T12 and len(set(found_solution)) == len(found_solution) and all(f > 12 for f in found_solution) and 1 <= len(found_solution) <= 12:
            return found_solution
    
    # Fallback: try a simpler approach - just return [T12]
    return [T12]
