"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Target: minimize n such that phi(n) % 120 == 0
    # 120 = 2^3 * 3 * 5
    
    # We perform a search over square-free kernels K composed of odd primes.
    # n = 2^a * K
    # phi(n) = phi(2^a) * phi(K) = 2^{a-1} * phi(K) (for a>=1, phi(1)=1 for a=0)
    # We require 120 | phi(n)
    # This splits into:
    # 1. 15 | phi(K)
    # 2. 8 | (2^{a-1} * phi(K))  => a-1 + v_2(phi(K)) >= 3
    
    # Step 1: Generate primes
    limit = 200  # Enough to find small composites
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Filter primes that are useful.
    # A prime p is useful if p-1 contributes factors of 3 or 5, or high powers of 2.
    # However, we can just iterate combinations of small primes.
    
    min_n = float('inf')
    
    # Helper to check condition for a given K (odd, square-free)
    def process_kernel(K, phi_K, v2_phi_K):
        nonlocal min_n
        # Check 15 | phi_K
        if phi_K % 15 != 0:
            return
        
        # Determine minimal a >= 0 such that 8 | 2^a * phi_K
        # If a=0, term is phi_K.
        # If a>=1, term is 2^{a-1} * phi_K.
        
        # Case a=0: n=K
        if v2_phi_K >= 3:
            if K < min_n:
                min_n = K
                return # Can't get smaller for this K with higher a
        
        # Case a>=1
        # We need a-1 + v2_phi_K >= 3
        # a >= 4 - v2_phi_K
        req_a = 4 - v2_phi_K
        if req_a < 1:
            req_a = 1
        # Smallest a is max(1, 4 - v2_phi_K)
        a = max(1, 4 - v2_phi_K)
        # Calculate n
        n_val = K * (2 ** a)
        if n_val < min_n:
            min_n = n_val
            
    # Recursive search for square-free kernels K
    # Start with empty set, add primes in increasing order
    # To keep K small, we stop if K * next_prime > current min_n (rough bound)
    
    def search(idx, current_K, phi_K, v2_phi_K):
        nonlocal min_n
        # Prune if current K already exceeds min_n
        # Note: n can be smaller than K? No, n >= K.
        if current_K > min_n:
            return
            
        # Check current K (if it's > 1)
        if current_K > 1:
            process_kernel(current_K, phi_K, v2_phi_K)
        
        # Try adding more primes
        for i in range(idx, len(primes)):
            p = primes[i]
            new_K = current_K * p
            # Prune if new_K is already too large
            if new_K > min_n:
                break
            # Since we are looking for minimal n, and primes are sorted,
            # if new_K > min_n, subsequent primes will make it larger.
            
            new_phi_K = phi_K * (p - 1)
            # Calculate v2 of new_phi_K
            # v2(x*y) = v2(x) + v2(y)
            p_minus_1 = p - 1
            v2_p = 0
            temp = p_minus_1
            while temp % 2 == 0:
                v2_p += 1
                temp //= 2
            
            new_v2 = v2_phi_K + v2_p
            
            search(i + 1, new_K, new_phi_K, new_v2)
            
    # Initial call
    # K=1, phi=1, v2=0
    # Note: K=1 corresponds to n=2^a.
    # phi(2^a) = 2^{a-1}.
    # 15 does not divide 2^{a-1}. So K=1 yields no solution.
    # But we start the search to build K.
    search(0, 1, 1, 0)
    
    # Also check prime powers p^k separately?
    # We established that for p>5, we need 120 | p-1.
    # Smallest such p is 241.
    # 241 > 143 (which we expect to find).
    # So the combinatorial search covers the minimal case.
    
    if min_n == float('inf'):
        return 1 # Fallback, though 143 is found
    return int(min_n)

# EVOLVE-BLOCK-END
