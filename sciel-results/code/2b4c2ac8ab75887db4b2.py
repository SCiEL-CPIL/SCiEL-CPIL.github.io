"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit
    
    N = 512
    # 1. Precompute weights and prime masks
    # Primes below 512
    primes = []
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, N):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N, i):
                is_prime[j] = False
    
    # Map prime to index
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # weight[a] = 1 / (512 - a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # prime_mask[a] = bitmask of primes dividing a
    prime_mask = [0] * N
    for a in range(1, N):
        n = a
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                prime_mask[a] |= (1 << prime_index[p])
                while n % p == 0:
                    n //= p
        if n > 1:
            prime_mask[a] |= (1 << prime_index[n])

    # 2. Sort candidates by weight descending
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    # 3. Greedy initialization to get a good initial lower bound
    def greedy_init():
        chosen = []
        used_mask = 0
        score = 0.0
        for a in candidates:
            if (prime_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_mask[a]
                score += weights[a]
        return chosen, score
        
    best_chosen, best_score = greedy_init()
    
    # 4. Branch and Bound
    # We pass the list of chosen elements to reconstruct the solution
    # To save memory/time, we can just store the best score and the best mask, 
    # but we need the actual set. 
    # Storing the set in the recursion stack is expensive.
    # Instead, we can store the best_score and best_mask, and reconstruct? 
    # No, different sets can have the same mask? No, mask is unique to the set of primes used.
    # But we need the specific integers.
    # Let's keep a global best_set and update it.
    
    # To speed up UB calculation, we can precompute suffix sums of weights?
    # But the compatibility check depends on used_mask.
    # So we must iterate.
    
    def dfs(idx, current_score, used_mask, current_set):
        nonlocal best_score, best_chosen
        
        # Time check
        if time.time() > DEADLINE:
            return
            
        # Upper Bound: Sum of weights of all remaining compatible items
        # This is O(N) per node. 
        # Optimization: If current_score + sum(weights[idx:]) <= best_score, prune.
        # sum(weights[idx:]) is precomputable.
        # But we need compatible sum.
        
        # Quick check: if current_score is already close to best, and remaining potential is low.
        # Let's calculate the compatible sum.
        ub_add = 0.0
        for i in range(idx, len(candidates)):
            a = candidates[i]
            if (prime_mask[a] & used_mask) == 0:
                ub_add += weights[a]
                # Early termination if ub is already too low? 
                # if current_score + ub_add + max_possible_rest <= best_score
                # max_possible_rest is sum of all remaining weights (ignoring compatibility)
                # We can precompute suffix_sums of weights.
        
        # Precompute suffix sums of weights for tighter early pruning
        # suffix_sum[i] = sum(weights[candidates[i]] ... weights[candidates[-1]])
        
        if current_score + ub_add <= best_score:
            return
            
        if idx == len(candidates):
            if current_score > best_score:
                best_score = current_score
                best_chosen = list(current_set)
            return
            
        a = candidates[idx]
        
        # Branch 1: Include a (if compatible)
        if (prime_mask[a] & used_mask) == 0:
            current_set.append(a)
            dfs(idx + 1, current_score + weights[a], used_mask | prime_mask[a], current_set)
            current_set.pop()
            
        # Branch 2: Exclude a
        dfs(idx + 1, current_score, used_mask, current_set)
        
    # Precompute suffix sums for a very loose but fast bound to prune early
    # Actually, the loop inside dfs to calculate ub_add is the bottleneck.
    # Let's try to optimize the UB calculation.
    # We can't easily precompute compatible sums for all masks.
    # However, N=511 is small. 511 iterations per node.
    # If we have 10^6 nodes, that's 5*10^8 operations. Might be too slow in Python.
    
    # Optimization: 
    # Use a tighter bound. 
    # The sum of weights of compatible items is an upper bound.
    # We can also use: current_score + sum(weights[idx:]) 
    # If that <= best_score, prune immediately without checking compatibility.
    
    suffix_sum = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + weights[candidates[i]]
        
    def dfs_optimized(idx, current_score, used_mask, current_set):
        nonlocal best_score, best_chosen
        
        if time.time() > DEADLINE:
            return
            
        # Loose bound
        if current_score + suffix_sum[idx] <= best_score:
            return
            
        # Tighter bound: Sum of compatible weights
        ub_add = 0.0
        for i in range(idx, len(candidates)):
            a = candidates[i]
            if (prime_mask[a] & used_mask) == 0:
                ub_add += weights[a]
                
        if current_score + ub_add <= best_score:
            return
            
        if idx == len(candidates):
            if current_score > best_score:
                best_score = current_score
                best_chosen = list(current_set)
            return
            
        a = candidates[idx]
        
        # Branch 1: Include a
        if (prime_mask[a] & used_mask) == 0:
            current_set.append(a)
            dfs_optimized(idx + 1, current_score + weights[a], used_mask | prime_mask[a], current_set)
            current_set.pop()
            
        # Branch 2: Exclude a
        dfs_optimized(idx + 1, current_score, used_mask, current_set)

    # Start B&B
    dfs_optimized(0, 0.0, 0, [])
    
    return best_chosen
# EVOLVE-BLOCK-END
