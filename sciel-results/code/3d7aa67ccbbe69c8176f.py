"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import time
    
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    # 1. Generate all primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # Strategy: Depth-First Search with Heuristic Ordering (Smallest Gap First)
    # This mimics the Greedy approach but allows backtracking to find longer sequences
    # if the greedy path dead-ends early.
    
    best_seq = []
    best_len = 0
    
    # Pre-compute gaps? No, just use indices.
    # To speed up, we can sort potential next moves by gap size.
    
    def dfs(current_seq, last_prime_idx, last_gap):
        nonlocal best_seq, best_len, start_time
        
        if time.time() > DEADLINE:
            return
        
        # Update best if current is longer
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        # Pruning: If we can't beat best_len even with optimistic estimate, stop.
        # Optimistic: Each next gap is at least last_gap. 
        # Max primes remaining is roughly (limit - current_last) / last_gap.
        # This is a loose bound, but helps.
        
        last_prime = primes[last_prime_idx]
        
        # Try all primes j > last_prime_idx
        # Heuristic: Sort candidates by gap (primes[j] - last_prime) ascending.
        # We don't want to sort every time, but we can iterate and pick the smallest valid gap?
        # Or just iterate in index order (which is prime value order, so gap order is NOT sorted).
        # Smallest gap is NOT necessarily the next prime. 
        # However, for small primes, the next prime has the smallest gap.
        
        # To ensure we find long sequences, we prefer small gaps.
        # Let's collect valid candidates and sort by gap.
        candidates = []
        for j in range(last_prime_idx + 1, n):
            gap = primes[j] - last_prime
            if last_gap == -1 or gap >= last_gap:
                candidates.append((gap, j))
        
        # Sort by gap to try smallest gaps first (keeps numbers small, allows more steps)
        candidates.sort()
        
        for gap, j in candidates:
            # Check time again
            if time.time() > DEADLINE:
                return
            
            new_seq = current_seq + [primes[j]]
            dfs(new_seq, j, gap)
            
            # If we found a sequence that is already very long, we can stop?
            # No, we want the MAX length.
            
    # Start DFS from each of the first few primes? 
    # Starting from 2 is best for maximizing length (smallest start).
    # But maybe starting from 3 allows a longer sequence? Unlikely.
    # Let's start from 2.
    
    # Optimization: Limit the depth of search or candidates if it gets too slow.
    # For now, pure DFS with sorting might be slow if branching factor is high.
    # Branching factor is ~N. Depth ~L. 
    # But pruning helps.
    
    # Let's try a simpler approach: Iterative Deepening or just a smarter Greedy.
    # Given the risk of DFS timeout, let's use a **Greedy with Multiple Restarts** 
    # or **A* Search** which is more directed.
    
    # Actually, the simplest fix for the "DP failure" mystery is to just use the 
    # Greedy approach that scored 1.0, but add a small improvement: 
    # Try starting from each of the first 10 primes and pick the longest result.
    
    best_overall = []
    
    for start_idx in range(min(10, n)):
        seq = [primes[start_idx]]
        last_gap = -1
        last_idx = start_idx
        
        # Greedy: pick the smallest valid next prime
        # This is fast and likely good.
        while True:
            next_idx = -1
            min_gap = float('inf')
            
            # Look for the next prime that satisfies the gap condition
            # And has the smallest gap (to keep the sequence "tight")
            for j in range(last_idx + 1, n):
                gap = primes[j] - primes[last_idx]
                if last_gap == -1 or gap >= last_gap:
                    if gap < min_gap:
                        min_gap = gap
                        next_idx = j
                    # Optimization: Since primes are sorted, the first valid prime 
                    # is NOT necessarily the smallest gap? 
                    # Wait, gap = primes[j] - primes[last_idx]. 
                    # As j increases, gap increases. 
                    # So the FIRST valid prime j has the SMALLEST gap.
                    # So we don't need to check all j. We just need the first valid j.
                    break 
                    
            if next_idx == -1:
                break
            
            seq.append(primes[next_idx])
            last_gap = min_gap
            last_idx = next_idx
            
            if time.time() > DEADLINE:
                break
        
        if len(seq) > len(best_overall):
            best_overall = seq
            if len(best_overall) > 100: # Heuristic stop if we get very long
                # We might still find longer, but this is a good candidate
                # Let's continue to be safe, but break if time is tight
                pass

    return best_overall
# EVOLVE-BLOCK-END
