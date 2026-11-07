"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    N = 256
    
    # Step 1: Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_set = set(primes)
    
    # Step 2: For each number 2..N, compute its set of prime factors
    # A number is "coprime-compatible" with a set S if it shares no prime factor with any element in S
    # We want to select a subset of {1, 2, ..., N} such that every pair is coprime.
    # This is equivalent to: for each prime p, at most one selected number can be divisible by p.
    
    # Key insight: If we select a composite number n, we "use up" all primes that divide n.
    # It's always better to select a prime p rather than a composite number that is a multiple of p,
    # because selecting p uses up only p, while selecting the composite uses up p AND other primes.
    # However, we might want to select a composite if it allows us to include other numbers.
    
    # Actually, the optimal strategy is:
    # 1. Always include 1 (it's coprime with everything).
    # 2. For each prime p, we can either include p itself, or include some composite number that is a multiple of p.
    #    But if we include a composite number m that is a multiple of p, we cannot include any other number that shares a factor with m.
    #    Since m >= p, and m uses up at least one prime (p) plus potentially others, it's generally better to just take the primes.
    #    
    #    Wait, let's think again. The constraint is pairwise coprime. 
    #    If I take prime p, I can't take any multiple of p.
    #    If I take composite m (which has prime factors p1, p2, ..., pk), I can't take any multiple of p1, p2, ..., pk.
    #    Since m >= p1 * p2 * ... * pk >= 2^k, and m is in 2..N, k is small.
    #    
    #    The sum of all primes <= 256 is much larger than any composite.
    #    Let's verify: is it ever beneficial to replace a prime p with a multiple of p?
    #    No, because if we replace p with m (a multiple of p), we lose p and gain m. But m is divisible by p, so we already couldn't have any other multiple of p.
    #    However, m might also be divisible by other primes q, r, etc. By choosing m, we block q, r, etc.
    #    If we just chose p, we only block multiples of p, but we could still choose q, r, etc. (if they are primes or other composites not sharing factors with p).
    #    So choosing p is strictly better or equal to choosing m, because p < m and choosing p leaves more options open.
    #    
    #    Therefore, the optimal solution is: {1} union {all primes <= N}.
    
    # Let's double check this logic.
    # Suppose we have primes p and q.
    # Option A: include p and q. Sum = p + q.
    # Option B: include pq (if pq <= N). Sum = pq.
    # Is pq > p + q? For p=2, q=3, pq=6, p+q=5. 6 > 5. So Option B is better for this pair!
    # But wait, if we include 6, we can't include 2 or 3. 
    # In Option A, we include 2 and 3. Sum = 5.
    # In Option B, we include 6. Sum = 6.
    # But we also can't include any other multiples of 2 or 3.
    # What if we have p=2, q=3, r=5?
    # Option A: 2+3+5 = 10.
    # Option B: 6 (uses 2,3) + 5 = 11. (Since 5 is coprime with 6).
    # Option C: 10 (uses 2,5) + 3 = 13.
    # Option D: 15 (uses 3,5) + 2 = 17.
    # Option E: 30 (uses 2,3,5) = 30.
    # So 30 is better than 17.
    
    # So the strategy of just taking primes is NOT optimal. We need to solve a more complex problem.
    # This is a maximum weight independent set problem in a conflict graph, or a packing problem.
    # Given N=256, the number of primes is 54. The number of possible composites is 202.
    # We can model this as a maximum weight matching or set packing.
    
    # Since N is small (256), we can use a Branch and Bound or Dynamic Programming approach.
    # However, the state space is large.
    
    # Alternative: Formulate as an Integer Linear Program? No, we only have standard library.
    # We can use a greedy algorithm with local search, or a more sophisticated combinatorial search.
    
    # Let's use a Branch and Bound approach.
    # We will consider numbers in decreasing order of value.
    # At each step, we decide to include or exclude a number.
    # We maintain a set of used primes.
    # If a number's prime factors are a subset of used primes, we can't include it.
    # We use an upper bound to prune: sum of remaining numbers that are coprime with the current set.
    
    # To speed up, we can precompute prime factors for each number.
    prime_factors = [set() for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            prime_factors[multiple].add(p)
    
    # Sort numbers 2..N in decreasing order
    candidates = list(range(N, 1, -1))
    
    best_sum = 1  # Start with 1
    best_set = {1}
    
    # For upper bound calculation, we can simply sum all remaining candidates that are coprime with the current set.
    # This is a loose bound but easy to compute.
    
    def dfs(idx, used_primes, current_sum, current_set):
        nonlocal best_sum, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if idx == len(candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Pruning: if current_sum + sum of all remaining candidates <= best_sum, prune
        # We can compute a tighter bound by only summing candidates that are coprime with used_primes
        remaining_sum = 0
        for j in range(idx, len(candidates)):
            n = candidates[j]
            if not prime_factors[n] & used_primes:
                remaining_sum += n
        
        if current_sum + remaining_sum <= best_sum:
            return
        
        n = candidates[idx]
        
        # Option 1: Include n
        if not prime_factors[n] & used_primes:
            new_used = used_primes | prime_factors[n]
            dfs(idx + 1, new_used, current_sum + n, current_set | {n})
        
        # Option 2: Exclude n
        dfs(idx + 1, used_primes, current_sum, current_set)
    
    # Start DFS
    dfs(0, set(), 1, {1})
    
    # Convert best_set to a list of ints
    result = list(best_set)
    
    # Ensure types are correct
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
