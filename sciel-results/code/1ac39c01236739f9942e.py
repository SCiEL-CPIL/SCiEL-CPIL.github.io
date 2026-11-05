import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe
    
    N = 1000001
    
    # y-smooth number approach:
    # For each y (starting from 2, 3, 5, 7, ...), mark all numbers <= N
    # that have all prime factors <= y.
    # Then check if any consecutive pair [n, n+1] with n >= 10000 are both marked.
    # The first y for which this happens gives the optimal answer.
    
    # Generate primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # Mark y-smooth numbers incrementally
    # smooth[y] will be a boolean array where smooth[y][x] = True iff x is y-smooth
    # Start with y=1: only 1 is 1-smooth
    smooth = np.zeros(N + 1, dtype=bool)
    smooth[1] = True
    
    best_n = None
    best_y = None
    
    # For each prime p, update the smooth mask to include numbers with p as a prime factor
    # A number x is p-smooth (where p is the new prime) if:
    #   - x was already (p_prev)-smooth, OR
    #   - x = s * p^k where s is (p_prev)-smooth and k >= 1
    
    # We process primes in order: 2, 3, 5, 7, 11, ...
    # After processing prime p, smooth marks all numbers with prime factors <= p
    
    for p in primes:
        if p > N:
            break
        
        # Check time
        if time.time() > deadline - 10:
            break
        
        # Add numbers that have p as a prime factor
        # For each power pk = p, p^2, p^3, ... <= N:
        #   For each existing smooth number s (with prime factors < p):
        #     if s * pk <= N, mark s * pk as smooth
        # But we need to be careful: we should only use smooth numbers from BEFORE adding p
        # to avoid using p multiple times incorrectly (though it's fine to use p multiple times
        # since p^k is allowed)
        
        # Actually, the correct approach:
        # A number x is p-smooth (with p being the largest prime <= p) iff
        # x = p^k * s where s is (p_prev)-smooth (prime factors < p) and k >= 0
        # So we can do:
        #   new_smooth = smooth.copy()
        #   For k = 1, 2, 3, ... while p^k <= N:
        #     For each s where smooth[s] is True (s has prime factors < p):
        #       if s * p^k <= N:
        #         new_smooth[s * p^k] = True
        #   smooth = new_smooth
        
        # But this is O(N * number_of_powers) per prime, which could be slow.
        
        # Better: iterate over multiples
        # For each power pk of p, mark all multiples of pk that are already smooth
        # (i.e., have no prime factors > p_prev)
        
        # Let's do it this way:
        # Keep a copy of the current smooth state (before adding p)
        old_smooth = smooth.copy()
        
        # For each power of p: p, p^2, p^3, ...
        pk = p
        while pk <= N:
            # Mark all numbers s * pk where s is old_smooth and s * pk <= N
            # i.e., for s in [1, N//pk], if old_smooth[s], then mark s * pk
            # This is equivalent to: for x in [pk, N], step pk, if old_smooth[x//pk], mark x
            for x in range(pk, N + 1, pk):
                if old_smooth[x // pk]:
                    smooth[x] = True
            pk *= p
        
        # Now check if there's a consecutive pair in [10000, 1000000] that are both smooth
        # We need to check smooth[n] and smooth[n+1] for n in [10000, 999999]
        # Use numpy for speed
        # Check if both smooth[n] and smooth[n+1] are True for any n in range
        
        # Create a boolean array for consecutive pairs
        # pair_ok[n] = smooth[n] AND smooth[n+1] for n in [10000, 999999]
        # We can do: np.logical_and(smooth[10000:1000000], smooth[10001:1000001])
        
        if time.time() > deadline - 5:
            break
        
        pair_ok = np.logical_and(smooth[10000:1000000], smooth[10001:1000001])
        
        if np.any(pair_ok):
            # Find the first (or any) n where this is true
            indices = np.where(pair_ok)[0]
            if len(indices) > 0:
                n = 10000 + indices[0]
                best_n = n
                best_y = p
                # We found a valid pair for this y. Since we're iterating y in increasing order,
                # this is the minimum possible y. But we should continue to see if we can find
                # a smaller n? No, the objective is to minimize the largest prime factor, which
                # is y. So the first y that gives any valid pair is optimal.
                # But wait: we want to minimize max(lpf[n], lpf[n+1]), which equals y if both
                # n and n+1 are y-smooth but not (y_prev)-smooth for some y_prev.
                # Actually, if both are y-smooth, then max(lpf[n], lpf[n+1]) <= y.
                # It could be less than y if both are actually (y_prev)-smooth.
                # But since we're iterating in order, the first y where we find a pair means
                # that for all y' < y, no pair existed. So the pair we find has
                # max(lpf[n], lpf[n+1]) = y (at least one of n, n+1 has lpf = y).
                # So this is optimal.
                break
    
    # If we found a best_n, return it
    if best_n is not None:
        return best_n
    
    # Fallback: if we didn't find anything (shouldn't happen), use the sieve approach
    if time.time() > deadline:
        return 10000
    
    # Linear Sieve to find Smallest Prime Factor (SPF)
    spf = np.zeros(N + 1, dtype=np.int32)
    spf[1] = 1
    primes_list = []
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes_list.append(i)
        for p in primes_list:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
    
    if time.time() > deadline:
        return 10000
    
    # Precompute Largest Prime Factor (LPF)
    lpf = np.zeros(N + 1, dtype=np.int32)
    lpf[1] = 1
    for i in range(2, N + 1):
        p = spf[i]
        if i // p == 1:
            lpf[i] = p
        else:
            lpf[i] = max(p, lpf[i // p])
            
    if time.time() > deadline:
        return 10000
    
    # Search for n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    lpf_local = lpf
    for n in range(10001, 1000001):
        val = lpf_local[n]
        val_next = lpf_local[n + 1]
        if val > val_next:
            current_max = val
        else:
            current_max = val_next
            
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
        if time.time() > deadline:
            break

    return best_n
