"""Optimized program using vectorized NumPy sieve for powerful parts."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    N = 1000001  # We need powerful parts for 10000..1000000 and 100001..1000001
    
    # powerful_part array: for each i, store the product of p^e for all prime powers
    # p^e where e >= 2 in the factorization of i
    pw = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Simple sieve to get primes
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime p, compute the powerful part contribution
    # For each prime power p^k (k>=2), multiply pw[i] by p^k where v_p(i) == k
    # v_p(i) == k means: p^k | i but p^(k+1) !| i
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Check deadline periodically
        if time.time() > deadline - 10:
            break
        
        pk = p2
        k = 2
        while pk <= N:
            # Next power
            pk_next = pk * p
            
            # Get all multiples of pk in [1, N]
            # We want those where v_p(i) == k, i.e., i/pk is NOT divisible by p
            # i.e., (i/pk) % p != 0, i.e., i % (pk*p) != 0
            
            if pk_next <= N:
                # Multiples of pk: pk, 2*pk, 3*pk, ..., floor(N/pk)*pk
                # Among these, exclude multiples of pk_next = pk*p
                # So we want multiples of pk where the multiplier is not divisible by p
                # multipliers: 1, 2, 3, ..., floor(N/pk)
                # Exclude multipliers divisible by p: p, 2p, 3p, ...
                
                num_mults = N // pk
                # Create array of multipliers 1..num_mults
                # Filter out those divisible by p
                # Instead of creating full array, use the pattern:
                # Multipliers not divisible by p: for each block of p, keep p-1 elements
                
                # Efficient: indices = pk * m where m in 1..num_mults and m % p != 0
                # This is equivalent to: all multiples of pk, minus multiples of pk*p
                
                # Use slicing to get multiples of pk
                # Then remove every p-th one (which are multiples of pk*p)
                
                # All multiples of pk: positions pk, 2*pk, 3*pk, ...
                # Count: num_mults
                # Multiples of pk*p: positions pk*p, 2*pk*p, ... = multiples of pk_next
                # Count: N // pk_next
                
                # We want to multiply pw at positions that are multiples of pk but NOT pk_next
                # These are at indices: pk*(m) where m=1..num_mults, m%p!=0
                
                # Vectorized: create the full set of multiples of pk, then mask out multiples of pk_next
                # But this creates large arrays. Alternative:
                
                # For small primes, num_mults is large. Let's use a different approach:
                # Multiply all multiples of pk by pk, then later we'll need to handle the fact that
                # for v_p(i) > k, we should have multiplied by p^(v_p(i)) not p^k.
                # 
                # Actually, the correct approach is:
                # pw[i] should get multiplied by p^k where k = v_p(i) if k >= 2, else by 1.
                # 
                # So for each i, the contribution from prime p is p^max(0, v_p(i) - 1 + 1) = p^v_p(i) if v_p(i)>=2
                # 
                # We can do: for k from 2 upward, for each i with v_p(i) == k, multiply by p^k.
                # 
                # The set {i : v_p(i) == k} = {i : p^k | i} \ {i : p^(k+1) | i}
                
                # Efficient method using numpy:
                # Create boolean mask for multiples of pk
                # Create boolean mask for multiples of pk_next
                # Subtract
                
                # But creating full N+1 boolean masks is expensive. Let's stick with the arange approach
                # but optimize it.
                
                # All multiples of pk:
                idx_all = np.arange(pk, N + 1, pk)
                # Multiples of pk_next within idx_all: these are at positions 0, p, 2p, ... in idx_all
                # i.e., every p-th element of idx_all
                # So we want to keep elements at positions not divisible by p
                
                # idx_all has length num_mults
                # Keep indices where position % p != 0
                # positions: 0, 1, 2, ..., num_mults-1
                # Keep: 0,1,...,p-1, p+1,...,2p-1, 2p+1,...
                
                if num_mults > 0:
                    # Create mask: positions not divisible by p
                    # positions = np.arange(num_mults)
                    # mask = positions % p != 0
                    # This creates an array of size num_mults, which for small p can be large
                    
                    # Alternative: use the fact that we want to exclude every p-th element
                    # We can slice: keep [0:p-1], [p+1:2p-1], etc.
                    # This is complex. Let's just use the modulo approach for now.
                    
                    positions = np.arange(num_mults)
                    mask = positions % p != 0
                    idx_filtered = idx_all[mask]
                    if len(idx_filtered) > 0:
                        pw[idx_filtered] *= pk
            else:
                # pk is the highest power <= N, so all multiples of pk get pk
                # (no higher power exists, so v_p(i) == k for all multiples of pk)
                idx = np.arange(pk, N + 1, pk)
                if len(idx) > 0:
                    pw[idx] *= pk
            
            # Check deadline
            if time.time() > deadline - 5:
                break
            
            pk = pk_next
            k += 1
    
    # Now compute the product of powerful parts of consecutive integers
    # For n in [10000, 1000000], product = pw[n] * pw[n+1]
    products = pw[10000:1000001] * pw[10001:1000002]
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)

# EVOLVE-BLOCK-END
