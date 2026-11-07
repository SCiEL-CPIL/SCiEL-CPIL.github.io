"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# Constants
BASE = 210
MAX_VAL = 10210
MAX_SIZE = 100

def _squarefree_kernel(n):
    """Return the squarefree kernel of n (product of primes with odd exponent)."""
    # Precompute primes up to sqrt(10210) ~ 101
    # We'll factor n by small primes
    n_orig = n
    kernel = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp % 2 == 1:
                kernel *= d
        d += 1 if d == 2 else 2  # 2, 3, 5, 7, ...
    if n > 1:
        kernel *= n
    return kernel

def _is_perfect_square(x):
    if x < 0:
        return False
    r = math.isqrt(x)
    return r * r == x

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s wall-clock
    
    # We need to select 1..100 distinct integers from (210, 10210]
    # such that 210 * product(selected) is a perfect square.
    # Minimize max(selected) - 210.
    
    # Key insight: 210 * prod = square means the squarefree kernel of 210 * prod = 1
    # kernel(210 * prod) = kernel(210) * kernel(prod) (mod squares) = 1 in GF(2)
    # Actually, kernel(a*b) = kernel(a) * kernel(b) / gcd-related... 
    # More precisely: in the group of squarefree numbers under multiplication mod squares,
    # we need kernel(210) * kernel(a1) * ... * kernel(ak) = 1 (mod squares)
    # This is equivalent to: the XOR (symmetric difference) of the prime factor sets = empty
    
    # kernel(210) = kernel(2*3*5*7) = 2*3*5*7 = 210 (since 210 is squarefree)
    
    # So we need the product of kernels of selected numbers to have kernel = 210
    # (since kernel(210 * prod) = kernel(210) * kernel(prod) when we consider the group)
    
    # Actually, let's think in terms of GF(2) vector space over primes.
    # Each number n maps to a vector in GF(2)^primes where entry i = 1 if prime_i appears 
    # with odd exponent in n.
    # We need: vector(210) + vector(a1) + ... + vector(ak) = 0 (mod 2)
    # i.e., vector(a1) + ... + vector(ak) = vector(210)
    
    # To minimize max(selected), we should try small values first.
    # Strategy: iterate through candidate max values M from 211 upward.
    # For each M, check if there exists a subset of {211, ..., M} of size 1..100
    # whose XOR of kernel-vectors equals vector(210).
    
    # For small M, the number of candidates is small, so we can use Gaussian elimination
    # or meet-in-the-middle.
    
    # Let's precompute kernel vectors for numbers 211 to some upper bound.
    # Primes up to 10210: we need primes up to 10210 for kernel computation.
    
    # But for GF(2) representation, we only care about primes that appear.
    # Let's use a dictionary: prime -> index, and represent each number as a Python int
    # bitset where bit i is 1 if prime[i] has odd exponent.
    
    # Precompute primes up to 10210
    def _sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = _sieve(10210)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Compute kernel vector (as Python int bitset) for each number from 211 to MAX_VAL
    def _kernel_vector(n):
        vec = 0
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                exp = 0
                while m % p == 0:
                    m //= p
                    exp += 1
                if exp % 2 == 1:
                    vec |= (1 << prime_to_idx[p])
        if m > 1:
            if m in prime_to_idx:
                vec |= (1 << prime_to_idx[m])
            # else m is prime > 10210, shouldn't happen since n <= 10210
        return vec
    
    # Compute target vector = kernel vector of 210
    target_vec = _kernel_vector(BASE)
    
    # Now search: for M from 211 upward, check if we can form target_vec as XOR of 
    # a subset of {kernel_vector(211), ..., kernel_vector(M)} with size 1..100.
    
    # For efficiency, use Gaussian elimination over GF(2) to find if target is in the span,
    # and also track the minimum-size representation (or just any representation of size <= 100).
    
    # Actually, we just need ANY subset of size 1..100. Since we're adding one element at a time,
    # we can maintain the row-reduced form and a witness for each basis vector.
    
    # Let's use a dict: basis vector -> (witness_vec, witness_list)
    # where witness_vec is the XOR that equals this basis element, and witness_list is the list of numbers used.
    
    # Better approach: maintain a linear basis where each basis element stores:
    #   - the vector (as int)
    #   - the set/list of original numbers that XOR to this vector
    
    # When adding a new number v, we try to reduce it. If it reduces to 0, it's in the span.
    # Otherwise, add it to the basis.
    
    # To check if target is in the span: reduce target using the basis.
    
    # But we also need to extract the actual subset. Let's store for each basis vector
    # the combination of original indices that produces it.
    
    # For small M (up to a few hundred), this is feasible.
    
    # Let's iterate M and maintain an incremental linear basis.
    
    # basis: list of (pivotal_bit, vector, witness_set)
    # We'll use a dict keyed by pivotal bit for fast lookup.
    
    def _search(max_candidates):
        """Search for minimal M such that target_vec is in span of {vec(211)..vec(M)}."""
        # basis: dict pivotal_bit -> (vector, frozenset of indices)
        basis = {}
        
        for M in range(211, max_candidates + 1):
            v = _kernel_vector(M)
            # Try to reduce v using basis
            w = v
            w_set = frozenset([M])  # indices used to form w (0-based: 211+M is the actual number, but we use offset)
            # Actually, let's store the actual number in the witness
            w_set = frozenset([M])
            
            changed = True
            while w != 0:
                # Find highest set bit
                hb = w.bit_length() - 1
                if hb in basis:
                    bv, bset = basis[hb]
                    w ^= bv
                    w_set = w_set ^ bset
                else:
                    break
            
            if w == 0:
                # v is in the span, represented by w_set
                # Check if target is in the span
                tw = target_vec
                tw_set = frozenset()
                while tw != 0:
                    hb = tw.bit_length() - 1
                    if hb in basis:
                        bv, bset = basis[hb]
                        tw ^= bv
                        tw_set = tw_set ^ bset
                    else:
                        break
                if tw == 0 and len(tw_set) <= MAX_SIZE and len(tw_set) >= 1:
                    return M, list(tw_set)
            else:
                # w is new, add to basis
                hb = w.bit_length() - 1
                basis[hb] = (w, w_set)
                # Check if target is now in the span
                tw = target_vec
                tw_set = frozenset()
                while tw != 0:
                    hb = tw.bit_length() - 1
                    if hb in basis:
                        bv, bset = basis[hb]
                        tw ^= bv
                        tw_set = tw_set ^ bset
                    else:
                        break
                if tw == 0 and len(tw_set) <= MAX_SIZE and len(tw_set) >= 1:
                    return M, list(tw_set)
        
        return None, None
    
    # First, try a reasonable upper bound
    # The search space grows with M. Let's cap at something reasonable.
    # If M is too large, we might time out.
    
    result_M, result_set = _search(2000)
    
    if result_M is not None:
        # Verify
        selected = sorted(result_set)
        # Check all are in (210, 10210], distinct, size 1..100
        assert all(210 < x <= MAX_VAL for x in selected)
        assert len(selected) == len(set(selected))
        assert 1 <= len(selected) <= MAX_SIZE
        # Check product * 210 is perfect square
        prod = BASE
        for x in selected:
            prod *= x
        assert _is_perfect_square(prod), f"Product not perfect square for M={result_M}"
        return list(selected)
    
    # If not found in the quick search, do a more thorough search with time limit
    # But this shouldn't happen for small M.
    
    # Fallback: try a single number that makes 210*n a square
    # 210*n = k^2 => n = k^2/210, so k must be divisible by sqrt(210) which isn't integer.
    # 210 = 2*3*5*7, so n = 210 * m^2 for some m.
    # n = 210 * m^2 > 210 => m >= 1
    # m=1: n=210, but must be > 210. m=2: n=840.
    # Check: 210*840 = 176400 = 420^2. Yes!
    
    n_fallback = 210 * 4  # m=2, n=840
    if n_fallback > BASE and n_fallback <= MAX_VAL:
        selected = [n_fallback]
        prod = BASE * n_fallback
        assert _is_perfect_square(prod)
        return list(selected)
    
    # Should not reach here
    return [840]

# EVOLVE-BLOCK-END
