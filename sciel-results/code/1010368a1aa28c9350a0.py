"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def _evaluate(coeffs):
    """Return (numerator, denominator) of the signed sum."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1  # term is c / (i+1)
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def construct():
    deadline = time.time() + 55  # Safety margin
    n = 30
    L = _lcm_range(n)
    # weights[i] = L / (i+1)
    # We want to minimize |sum c_i * weights[i]| where c_i in {-1, 0, 1} and sum != 0
    weights = [L // (i + 1) for i in range(n)]
    
    # Initial best: simple alternating or known small sum
    # Try a few simple patterns first
    initial_candidates = [
        [1, -1] + [0]*28,
        [1, 0, -1] + [0]*27,
        [1, 1, -1, -1] + [0]*26,
        [1]*15 + [-1]*15
    ]
    
    best_coeffs = [0]*n
    best_abs_N = float('inf')
    best_N = 0
    
    def compute_N(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
        
    for cand in initial_candidates:
        if len(cand) < n:
            cand = cand + [0]*(n-len(cand))
        N = compute_N(cand)
        if N != 0 and abs(N) < best_abs_N:
            best_abs_N = abs(N)
            best_N = N
            best_coeffs = cand[:]
            
    # If we haven't found anything good, initialize with random
    if best_abs_N == float('inf'):
         rng = random.Random(42)
         cand = [rng.choice([-1, 0, 1]) for _ in range(n)]
         N = compute_N(cand)
         if N != 0:
             best_abs_N = abs(N)
             best_N = N
             best_coeffs = cand[:]
         else:
             best_coeffs = [0]*28 + [1, -1]
             best_N = compute_N(best_coeffs)
             best_abs_N = abs(best_N)

    # Karmarkar-Karp / Difficult Knapsack style heuristic
    # We maintain a list of "active" indices.
    # We try to partition the active weights into two sets A and B such that |sum(A) - sum(B)| is minimized.
    # This corresponds to c_i = 1 for i in A, c_i = -1 for i in B, c_i = 0 for inactive.
    
    # Greedy Partition Algorithm:
    # 1. Sort weights descending.
    # 2. Iterate, placing each weight in the set with the currently smaller sum.
    # This gives a good approximation for subset sum difference.
    
    # To handle 0s, we can also try removing the largest weight and solving for the rest, 
    # or just letting the greedy partition handle it (though greedy doesn't naturally output 0s 
    # unless we allow skipping). 
    # However, a standard K-K algorithm works on a multiset. 
    # Let's implement a variant that allows "skipping" (setting to 0) if it reduces the final diff.
    
    # Actually, a simpler and very effective approach for small N (30) is to use 
    # "Meet in the Middle" or exhaustive search if N was smaller, but 2^30 is too big.
    
    # Let's use a randomized local search but with a smarter move set: 
    # 1. Flip sign of one element.
    # 2. Zero out one element.
    # 3. Set one element from 0 to +/- 1.
    
    # But the prompt asks for a DIFFERENT strategy. 
    # The current one is Random Restart + Hill Climbing.
    # Let's try a **Deterministic Greedy Construction** followed by **Local Refinement**.
    
    # Strategy:
    # 1. Start with all 0s.
    # 2. Add weights one by one (sorted by magnitude descending) to minimize current absolute sum.
    #    For each weight w, decide c in {-1, 0, 1} that minimizes |current_sum + c*w|.
    # 3. This is a greedy online algorithm.
    # 4. Then, do a pass of local optimization (flipping/zeroing) to fix local minima.
    
    def greedy_construct(indices, weights):
        # indices: list of indices to consider
        # weights: corresponding weights
        coeffs = [0] * n
        current_sum = 0
        # Sort by weight magnitude descending
        sorted_indices = sorted(indices, key=lambda i: weights[i], reverse=True)
        
        for i in sorted_indices:
            w = weights[i]
            # Options: -1, 0, 1
            best_c = 0
            best_diff = abs(current_sum) # if we choose 0
            
            for c in [-1, 1]:
                new_sum = current_sum + c * w
                diff = abs(new_sum)
                if diff < best_diff:
                    best_diff = diff
                    best_c = c
            
            coeffs[i] = best_c
            current_sum += best_c * w
        return coeffs, current_sum

    # Run greedy construction with different orderings (randomized sorts)
    rng = random.Random(123)
    
    # 1. Pure Descending
    c1, s1 = greedy_construct(range(n), weights)
    if s1 != 0 and abs(s1) < best_abs_N:
        best_abs_N = abs(s1)
        best_N = s1
        best_coeffs = c1[:]

    # 2. Randomized Descending (Simulated Annealing on the construction order)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        # Randomly perturb the order
        indices = list(range(n))
        rng.shuffle(indices)
        # Sort by weight, but with some random noise to break ties or slightly reorder
        # Or just use the shuffled order and apply greedy? 
        # Greedy on arbitrary order is worse. 
        # Let's stick to sorting by weight, but randomly break ties or slightly jitter weights.
        jittered_weights = [weights[i] + rng.uniform(-weights[i]*0.01, weights[i]*0.01) for i in range(n)]
        c, s = greedy_construct(range(n), jittered_weights)
        if s != 0 and abs(s) < best_abs_N:
            best_abs_N = abs(s)
            best_N = s
            best_coeffs = c[:]

    # 3. Local Refinement of the best candidate found so far
    # We use the best_coeffs as a starting point for a deterministic local search.
    # Moves: flip sign, zero out, set to +/-1.
    
    current_coeffs = best_coeffs[:]
    current_N = best_N
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            w = weights[i]
            cur_c = current_coeffs[i]
            # Try all 3 values
            for new_c in [-1, 0, 1]:
                if new_c == cur_c:
                    continue
                delta = (new_c - cur_c) * w
                new_N = current_N + delta
                if new_N == 0:
                    continue # Must be nonzero
                if abs(new_N) < abs(current_N):
                    current_coeffs[i] = new_c
                    current_N = new_N
                    improved = True
                    break
            if improved:
                break
        
        if abs(current_N) < best_abs_N:
            best_abs_N = abs(current_N)
            best_N = current_N
            best_coeffs = current_coeffs[:]
            
        # If we hit 1, we can't do better (integer numerator)
        if best_abs_N == 1:
            break

    # Final verification and formatting
    # best_N is the numerator when denominator is L
    # We need to return the coefficients
    # And ensure the sum is nonzero
    
    if best_N == 0:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return best_coeffs

# EVOLVE-BLOCK-END
