import time
from math import isqrt
from bisect import bisect_left

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve of Eratosthenes
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    num_primes = len(primes)
    
    if num_primes < 2:
        return primes
    
    # DP approach: dp[i][j] = longest valid subsequence ending with primes[i], primes[j]
    # where primes[j] is the last prime and primes[i] is the second-to-last prime.
    # The gap from i to j is primes[j] - primes[i].
    # Transition: for k < i, if primes[i] - primes[k] <= primes[j] - primes[i],
    # then dp[i][j] = max(dp[i][j], dp[k][i] + 1)
    # 
    # But this is O(K^3) which is too slow for K=1229.
    # 
    # Alternative: Let's use a different DP.
    # Let dp[j] = the maximum length of a valid subsequence ending at primes[j].
    # But we also need to know the last gap to ensure non-decreasing gaps.
    # So dp[j][g] = max length ending at primes[j] with last gap g? That's not efficient.
    #
    # Better approach: 
    # Let's define dp[i] as a list/dict where dp[i][j] = max length of valid subsequence
    # ending at primes[i] with the previous prime being primes[j] (j < i).
    # Then dp[i][j] = 2 + max{dp[j][k] : k < j and primes[j] - primes[k] <= primes[i] - primes[j]}
    #
    # This is still O(K^3). Let's think of a smarter way.
    #
    # Actually, let's use the beam search but with better pruning and more starts.
    # The key insight from the inspiration is that DP finds longer chains.
    # Let's implement a more efficient DP using the fact that gaps are non-decreasing.
    
    # dp[i][j] = longest subsequence ending with primes[i], primes[j] (i < j)
    # We'll use a 2D array. K = 1229, so K^2 = ~1.5M entries. That's fine.
    # But the transition is O(K) per entry, making it O(K^3) = ~1.8B. Too slow.
    #
    # Let's use a different formulation:
    # For each pair (i, j) with i < j, the gap is gap = primes[j] - primes[i].
    # We want to extend sequences. Let's process in order of the second prime.
    # 
    # Actually, let's use a smarter beam search with the following improvements:
    # 1. Use bisect to find the range of valid next primes quickly
    # 2. Better pruning strategy
    # 3. Try more starting primes
    
    best_sequence = []
    best_length = 0
    
    def run_beam_search(start_indices, beam_width=100):
        nonlocal best_sequence, best_length
        
        # State: (idx, gap, seq_length, seq)
        # We store seq as a tuple for hashing/dedup if needed, but let's just keep list
        beam = []
        for idx in start_indices:
            beam.append((idx, 0, 1, (primes[idx],)))
            
        while beam:
            if time.time() > deadline:
                return
            
            next_beam = []
            for idx, gap, length, seq in beam:
                # Find valid next primes using bisect
                # We need primes[j] >= primes[idx] + gap, so j >= bisect_left(primes, primes[idx] + gap)
                min_val = primes[idx] + gap
                start_j = bisect_left(primes, min_val, idx + 1)
                
                # We want to consider a range of candidates
                # Let's take up to beam_width candidates starting from start_j
                candidates = []
                for j in range(start_j, min(start_j + beam_width * 2, num_primes)):
                    new_gap = primes[j] - primes[idx]
                    # new_gap >= gap is guaranteed by the bisect
                    candidates.append((j, new_gap))
                    
                for j, new_gap in candidates:
                    new_length = length + 1
                    new_seq = seq + (primes[j],)
                    if new_length > best_length:
                        best_length = new_length
                        best_sequence = list(new_seq)
                        
                    next_beam.append((j, new_gap, new_length, new_seq))
                    
            # Prune next_beam
            # Sort by (gap, idx) - smaller gap and smaller prime index is better
            next_beam.sort(key=lambda x: (x[1], x[0]))
            beam = next_beam[:beam_width]
            
    # Try starting from the first 20 primes with a wider beam
    run_beam_search(range(20), beam_width=100)
    
    # If we have time, try even more starts
    if time.time() < deadline - 5.0:
        run_beam_search(range(50), beam_width=50)
    
    return best_sequence
