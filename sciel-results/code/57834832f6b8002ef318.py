"""Square-product extension for 210 via exact GF(2) Gaussian elimination.

Strategy (different design family from the seed):
- Represent each candidate n in (210, M] by its square-free parity vector over GF(2)
  (bit i = exponent of prime i mod 2).
- The condition "210 * prod(selected) is a perfect square" is a single linear
  equation system: XOR of selected vectors == parity(210).
- Feasibility for a given M (does a non-empty subset of {211..M} XOR to target?)
  is decided by GF(2) Gaussian elimination in O(M * P^2) bit ops.
- Since feasibility is monotone in M (adding candidates can only help), we binary
  search the minimal M that is feasible -> minimizes max(selected)-210 exactly.
- A particular satisfying subset is then recovered from the elimination basis.
"""

import math
import time

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i] is True]


def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d


def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0  # safety margin under 90s kill

    primes = _sieve_primes(MAX_N)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    P = len(primes)

    # Precompute parity (square-free) bitvector as Python int for 211..MAX_N.
    # Store in a list indexed by (n - 211) for faster access
    parity_list = [0] * (MAX_N - 210)  # index 0 corresponds to n=211
    for i, n in enumerate(range(211, MAX_N + 1)):
        d = _factorize(n, primes)
        v = 0
        for p, e in d.items():
            if e & 1:
                v |= (1 << prime_index[p])
        parity_list[i] = v

    # Target = parity(210) = 2^1*3^1*5^1*7^1
    target = 0
    for p in SMALL_PRIMES:
        target |= (1 << prime_index[p])

    def _feasible(M):
        """Return (bool, list_of_indices) for a non-empty subset of [211..M]
        whose XOR of parity vectors equals target. Uses GF(2) Gaussian elimination
        with single-pass recovery."""
        if M < 211:
            return (False, None)
        num_rows = M - 210  # rows for n = 211..M
        
        # Single elimination pass with solution tracking
        # basis: pivot_bit -> (reduced_vector, solution_mask)
        # solution_mask is a Python int where bit i = 1 means original row i is in the solution
        basis = {}
        
        for i in range(num_rows):
            x = parity_list[i]
            sol = 1 << i
            while x:
                pb = x.bit_length() - 1
                if pb in basis:
                    x ^= basis[pb][0]
                    sol ^= basis[pb][1]
                else:
                    basis[pb] = (x, sol)
                    break
        
        # Reduce target against basis to get solution
        x = target
        sel_mask = 0
        while x:
            pb = x.bit_length() - 1
            if pb in basis:
                x ^= basis[pb][0]
                sel_mask ^= basis[pb][1]
            else:
                return (False, None)
        
        if sel_mask == 0:
            return (False, None)
        
        # Extract selected indices
        selected = []
        i = 0
        while sel_mask:
            if sel_mask & 1:
                selected.append(211 + i)
            sel_mask >>= 1
            i += 1
        
        if not selected:
            return (False, None)
        return (True, selected)

    # --- Binary search for minimal feasible M in [211, MAX_N] ---
    lo, hi = 211, MAX_N
    best_M = None
    best_selected = None

    # First check if any solution exists at all
    ok, sel = _feasible(MAX_N)
    if not ok:
        # No subset works -> fall back to guaranteed 840
        return [840]

    while lo < hi:
        if time.time() > DEADLINE:
            break
        mid = (lo + hi) // 2
        ok, sel = _feasible(mid)
        if ok:
            best_M = mid
            best_selected = sel
            hi = mid
        else:
            lo = mid + 1

    if best_selected is None:
        ok, sel = _feasible(MAX_N)
        if ok:
            best_selected = sel
        else:
            return [840]

    # Validate
    selected = list(best_selected)
    # Enforce distinctness and bounds (should already hold)
    selected = [x for x in selected if isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N]
    # Remove duplicates (shouldn't happen)
    seen = set()
    uniq = []
    for x in selected:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    selected = uniq
    if not selected:
        return [840]

    # Verify exact condition
    prod = N
    for x in selected:
        prod *= x
    if _is_square(prod) and len(selected) <= MAX_K:
        return [int(x) for x in selected]

    # Fallback guaranteed solution
    return [840]


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
