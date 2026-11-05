import time
import random
import math
import sys

MAX_VAL = 20000
N = 12
MAX_VAL_ARR = 10000

# Precompute smallest prime factor
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def generate_smooth_numbers(primes, limit):
    """Generate all numbers <= limit whose prime factors are in 'primes'."""
    smooth = {1}
    for p in primes:
        current = list(smooth)
        power = p
        while power <= limit:
            for v in current:
                if v * power <= limit:
                    smooth.add(v * power)
                else:
                    break
            power *= p
    return smooth

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Strategy: Find a set of 12 distinct integers such that all pairwise sums
    # have prime factors from a small set P.
    # We iterate over increasing sizes of P (starting with smallest primes) and
    # attempt to find a clique of size 12 in the compatibility graph.
    # If no clique is found, we expand P by adding the next most frequent prime
    # seen in "near-miss" sums or simply the next smallest prime.
    
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    best_arr = [3 + 6 * k for k in range(12)]
    best_cost = count_distinct_primes(best_arr)
    
    # Try different prime sets
    # Start with small sets, expand if necessary
    # The "iterative expansion" idea:
    # 1. Start with P = {2, 3}
    # 2. Build graph, try to find clique.
    # 3. If fail, identify sums that are "close" to being smooth (i.e., have few extra primes)
    #    and add those primes to P.
    #    OR simpler: just try {2,3}, then {2,3,5}, then {2,3,5,7}, etc.
    #    Given the time limit, simple expansion is safer.
    
    for num_primes in range(2, 8):  # Try up to 7 primes
        if time.time() > deadline - 10:
            break
            
        P = base_primes[:num_primes]
        
        # Generate P-smooth numbers up to 20000
        smooth_nums = generate_smooth_numbers(P, MAX_VAL)
        
        if not smooth_nums:
            continue
        
        # Build adjacency sets
        # adj[x] = set of y such that x+y is P-smooth
        adj = [set() for _ in range(MAX_VAL_ARR + 1)]
        
        # Optimization: Iterate over smooth sums S.
        # For each S, x+y=S => y=S-x.
        # We want x,y in [1, 10000].
        for S in smooth_nums:
            # x in [max(1, S-10000), min(10000, S-1)]
            low = max(1, S - MAX_VAL_ARR)
            high = min(MAX_VAL_ARR, S - 1)
            if low > high:
                continue
            # To avoid O(N^2) in worst case, we only add edges for valid pairs.
            # However, iterating all x in range is O(N) per S.
            # Number of smooth nums is small for small P.
            for x in range(low, high + 1):
                y = S - x
                if x != y:
                    adj[x].add(y)
                    adj[y].add(x)
        
        # Find clique of size 12
        # Use a randomized greedy/backtracking approach restricted to high-degree nodes.
        # Nodes: 1..10000
        # Sort by degree
        degrees = [(len(adj[i]), i) for i in range(1, MAX_VAL_ARR + 1)]
        degrees.sort(reverse=True)
        
        # Take top K nodes as candidates
        # K needs to be large enough to contain a clique, but small enough for search.
        # If the graph is dense, top 100 might suffice. If sparse, we might need more.
        # Let's try a few different K values or just a large one like 500.
        
        found_clique = None
        
        # Heuristic: The clique is likely among the highest degree nodes.
        # Let's try to find it in the top 200 nodes first.
        top_nodes = [i for _, i in degrees[:200]]
        
        # Randomized search for clique in top_nodes
        # We can use a simple recursive backtracking with pruning.
        # Since N=12 is small, and candidate pool is 200, it should be fast if a clique exists.
        
        def find_clique(candidates, current_clique, target_size):
            if len(current_clique) == target_size:
                return current_clique[:]
            if len(candidates) < target_size - len(current_clique):
                return None
            
            # Filter candidates to those connected to all in current_clique
            valid = []
            for v in candidates:
                is_connected = True
                for u in current_clique:
                    if v not in adj[u]:
                        is_connected = False
                        break
                if is_connected:
                    valid.append(v)
            
            if len(valid) < target_size - len(current_clique):
                return None
            
            # Recurse
            for i, v in enumerate(valid):
                new_clique = current_clique + [v]
                new_candidates = [u for u in valid[i+1:] if u in adj[v]]
                
                res = find_clique(new_candidates, new_clique, target_size)
                if res:
                    return res
            
            return None
        
        # Try a few random shuffles of the top nodes to avoid getting stuck in bad ordering
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            random.shuffle(top_nodes)
            res = find_clique(top_nodes, [], 12)
            if res:
                found_clique = res
                break
        
        if found_clique:
            # Verify
            cost = count_distinct_primes(found_clique)
            if cost <= num_primes:
                return found_clique
            else:
                # Should not happen if logic is correct, but update best if better
                if cost < best_cost:
                    best_cost = cost
                    best_arr = found_clique
        
    # Fallback: Simulated Annealing if no clique found with small prime sets
    # This handles cases where the optimal solution requires a larger prime set
    # or has a structure not captured by the simple smooth-number clique search.
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.999
    
    # Ensure distinctness
    def is_valid(arr):
        return len(set(arr)) == len(arr) and all(1 <= x <= 10000 for x in arr)
    
    while time.time() < deadline:
        if T < 0.01:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        
        T *= cooling_rate
    
    return best_arr

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(res)
    print(count_distinct_primes(res))
