"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a set of distinct pairwise coprime integers a in 1..511 that maximizes
    the sum of 1/(512 - a).
    
    Approach: Exact Maximum Weight Independent Set via Branch and Bound.
    """
    import sys
    sys.setrecursionlimit(10000)
    
    N = 512
    # Sieve for primes up to 511
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N) if is_prime[p]]
    
    # Map primes to bit positions
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Compute prime factor bitset for each number 1..511
    factor_bitset = [0] * N
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= prime_to_bit[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= prime_to_bit[x]
        factor_bitset[a] = mask
    # factor_bitset[1] remains 0

    # Weights: 1 / (512 - a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Order vertices by decreasing weight (which is decreasing a)
    # We want to pick high weights first to get a good lower bound early and prune effectively
    order = list(range(511, 0, -1))
    
    # Precompute suffix sums of weights for the loose upper bound
    # suffix_sum[i] = sum(weights[order[i]] + ... + weights[order[-1]])
    suffix_sum = [0.0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
        
    best_score = 0.0
    best_set = []
    
    def dfs(idx, score, used_mask, chosen):
        nonlocal best_score, best_set
        
        # Loose Upper Bound: Current score + sum of all remaining weights
        if score + suffix_sum[idx] <= best_score:
            return
            
        if idx == len(order):
            if score > best_score:
                best_score = score
                best_set = chosen[:]
            return
            
        v = order[idx]
        
        # Tighter Upper Bound check (optional, but helpful if loose bound fails)
        # Sum of weights of remaining vertices that are compatible with used_mask
        # We only do this if the loose bound is close to best_score to save time
        # Actually, let's just do the branch and let the loose bound do the heavy lifting first.
        # If we want tighter pruning:
        # tight_bound_sum = 0.0
        # for j in range(idx, len(order)):
        #     if factor_bitset[order[j]] & used_mask == 0:
        #         tight_bound_sum += weights[order[j]]
        # if score + tight_bound_sum <= best_score:
        #     return
        
        # Branch 1: Include v (if compatible)
        if factor_bitset[v] & used_mask == 0:
            chosen.append(v)
            dfs(idx + 1, score + weights[v], used_mask | factor_bitset[v], chosen)
            chosen.pop()
            
        # Branch 2: Exclude v
        dfs(idx + 1, score, used_mask, chosen)
        
    # Start with a good initial solution (Greedy) to set best_score high
    # Greedy: pick highest weight compatible
    greedy_used_mask = 0
    greedy_chosen = []
    for v in order:
        if factor_bitset[v] & greedy_used_mask == 0:
            greedy_chosen.append(v)
            greedy_used_mask |= factor_bitset[v]
            
    best_score = sum(weights[v] for v in greedy_chosen)
    best_set = greedy_chosen[:]
    
    # Run DFS
    dfs(0, 0.0, 0, [])
    
    return best_set

# EVOLVE-BLOCK-END
