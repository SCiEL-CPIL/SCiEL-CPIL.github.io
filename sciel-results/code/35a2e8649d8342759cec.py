import math
import time

def construct():
    """
    Finds an integer n in [10000, 1000000] that maximizes powerful_part(n) * powerful_part(n+1).
    powerful_part(x) is the product of p^e for all prime powers p^e dividing x with e >= 2.
    This is equivalent to removing the squarefree part of x.
    """
    t0 = time.time()
    # Safety margin: Evaluator kills at 90s. We aim to finish well before.
    # If we exceed this, we return a safe fallback (though the algorithm below is fast).
    DEADLINE = 85.0

    MAX_N = 1_000_000
    MAX_VAL = MAX_N + 1  # We need to check n and n+1, so up to 1,000,001

    # 1. Sieve of Eratosthenes to find all primes up to MAX_VAL
    is_prime = bytearray(b'\x01') * (MAX_VAL + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i : MAX_VAL+1 : i] = b'\x00' * ((MAX_VAL - i*i)//i + 1)
            
    primes = [i for i in range(2, MAX_VAL + 1) if is_prime[i]]

    # 2. Initialize the array to compute the powerful part
    # powerful_part[i] will initially be i.
    # We will iterate through primes p. For any i where v_p(i) == 1,
    # p is part of the squarefree kernel and should be removed.
    # So, for all multiples of p, we divide by p.
    # For all multiples of p^2, we multiply by p (to restore the factor since v_p >= 2).
    
    pow_part = list(range(MAX_VAL + 1))
    # Using a list of integers is fast enough for 1e6 size in Python if we are careful, 
    # but NumPy is generally faster for vectorized ops. 
    # However, slicing and in-place ops on NumPy int64 is very fast.
    
    import numpy as np
    pow_part_arr = np.arange(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        if time.time() - t0 > DEADLINE:
            # Fallback: return a known valid number, though unlikely to reach here
            return 10000
        
        # Divide by p for all multiples of p
        # Indices: p, 2p, 3p, ...
        mask = np.arange(p, MAX_VAL + 1, p, dtype=np.int64)
        pow_part_arr[mask] //= p
        
        # Multiply by p for all multiples of p^2
        # This cancels the division for numbers where v_p >= 2
        p_sq = p * p
        if p_sq <= MAX_VAL:
            mask_sq = np.arange(p_sq, MAX_VAL + 1, p_sq, dtype=np.int64)
            pow_part_arr[mask_sq] *= p

    # 3. Maximize product pow_part[n] * pow_part[n+1] for n in [10000, 1000000]
    # Slice from 10000 to 1000001 (inclusive)
    # n starts at 10000, ends at 1000000.
    # We need pow_part[10000] * pow_part[10001] ... pow_part[1000000] * pow_part[1000001]
    
    start = 10000
    end = 1_000_000
    
    # Extract relevant range
    # Indices needed: start to end+1
    seg = pow_part_arr[start : end+2]
    
    # Calculate products
    # prod[i] = seg[i] * seg[i+1]
    # Length of seg is (end+2) - start = 990002
    # We want products for i=0 to len(seg)-2
    products = seg[:-1] * seg[1:]
    
    # Find the index of the maximum product
    best_idx = int(np.argmax(products))
    
    # The corresponding n is start + best_idx
    best_n = start + best_idx
    
    # Safety check
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
        
    return best_n
