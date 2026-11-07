import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 210
    MAX_VAL = 10210
    
    # Primes up to MAX_VAL
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(MAX_VAL)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    D = len(primes)
    
    # Target vector for 210
    target = 0
    n = N
    for i, p in enumerate(primes):
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            if exp % 2 == 1:
                target |= (1 << i)
    
    # Function to get vector for x
    def get_vec(x):
        v = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                exp = 0
                while x % p == 0:
                    x //= p
                    exp += 1
                if exp % 2 == 1:
                    v |= (1 << i)
        if x > 1:
            # x is a prime > sqrt(MAX_VAL) or remaining part
            if x in prime_to_idx:
                v |= (1 << prime_to_idx[x])
        return v

    # Basis for span
    # basis[i] is the vector with highest set bit i
    basis = [0] * (D + 1)
    # origin[i] is the set of indices (values) that make up basis[i]
    origin = [set() for _ in range(D + 1)]
    # Null space vectors
    null_space = []
    
    # We iterate M from N+1 to MAX_VAL
    # We need to track which values are added
    # indices in our sets will be the actual values x
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            # Return best found so far? 
            # If we haven't found any, return trivial
            return [4 * N]
            
        v = get_vec(M)
        
        # Try to add v to basis
        temp = v
        temp_origin = {M}
        
        for i in range(D - 1, -1, -1):
            if (temp >> i) & 1:
                if basis[i] == 0:
                    basis[i] = temp
                    origin[i] = temp_origin
                    break
                else:
                    temp ^= basis[i]
                    temp_origin ^= origin[i]
        
        if temp == 0:
            # v is in the span, so temp_origin is a null space vector
            # Actually, temp_origin represents a combination of previous vectors that sums to 0
            # Wait. If temp reduces to 0, it means v is a linear combination of previous basis vectors.
            # The combination is stored in temp_origin.
            # So temp_origin is a set of indices whose vectors sum to 0.
            # This is a null space vector.
            null_space.append(temp_origin)
        
        # Check if target is in span
        # Reduce target by basis
        temp_t = target
        indices = set()
        for i in range(D - 1, -1, -1):
            if (temp_t >> i) & 1:
                if basis[i] != 0:
                    temp_t ^= basis[i]
                    indices ^= origin[i]
                else:
                    # Not in span
                    break
        
        if temp_t == 0:
            # Found a solution
            # indices is a set of values
            if len(indices) > 100:
                # Try to reduce size using null space
                # Greedy reduction
                changed = True
                while changed and len(indices) > 100:
                    changed = False
                    for ns in null_space:
                        if ns.issubset(indices):
                            indices -= ns
                            changed = True
                            break
                        # Also check if ns is mostly in indices?
                        # If |ns \ indices| < |ns \cap indices|, swapping might help?
                        # No, we need to maintain the property that sum is 0.
                        # Adding ns (XOR) is the only way.
                        # If ns is not a subset, XORing it will add some and remove some.
                        # |I ^ ns| = |I| + |ns| - 2|I \cap ns|
                        # We want this to be smaller than |I|.
                        # |I| + |ns| - 2|I \cap ns| < |I|
                        # |ns| < 2|I \cap ns|
                        # So if more than half of ns is in I, XORing reduces size.
                        else:
                            inter = len(ns & indices)
                            if 2 * inter > len(ns):
                                indices ^= ns
                                changed = True
                                break
            
            if len(indices) <= 100 and len(indices) >= 1:
                return list(indices)
            # If still > 100, we might need to continue searching for a better combination
            # or accept that this M doesn't have a valid solution with <=100 elements?
            # The problem implies a solution exists.
            # If we can't find one with <=100, we continue to next M?
            # But M is minimal max. If we skip, we might miss the minimal max.
            # However, if a solution exists, it should be findable.
            # Let's assume the greedy reduction works.
            # If not, we return the trivial one to be safe?
            # No, better to continue to next M and hope for a solution with smaller complexity?
            # Actually, as M increases, null space grows, making it easier to find small solutions.
            # So if this M fails the size check, we can just continue.
            # But wait, if M is the *minimal* max value, there might be *no* other M with smaller max.
            # So if we can't satisfy size <= 100 at minimal M, we are in trouble.
            # But for M close to 210, the space is small, solutions are unique/simple.
            # The risk is low.

    return [4 * N]

# EVOLVE-BLOCK-END
