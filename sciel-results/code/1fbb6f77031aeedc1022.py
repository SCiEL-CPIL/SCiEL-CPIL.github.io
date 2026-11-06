"""Search for n in [1000,1000000] minimizing distinct small-prime factors of n+1..n+24."""
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    T = 24  # number of consecutive integers

    def small_factor_mask(x):
        """Return bitmask of small primes dividing x (primes < 24)."""
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if x % p == 0:
                mask |= (1 << i)
        return mask

    def score_for_n(n):
        """Count distinct small primes dividing at least one of n+1..n+24."""
        union = 0
        for i in range(1, T + 1):
            union |= small_factor_mask(n + i)
        return bin(union).count('1')

    # Try to find n where all n+1..n+24 are covered by a small prime set.
    # Use CRT-based construction: assign each position i to a prime p such that n ≡ -i (mod p).
    # Then n+i ≡ 0 (mod p), so p divides n+i.
    # We need a consistent system of congruences.

    def find_n_for_assignment(assign):
        """
        Given assign[i] = prime p for i in 1..T, solve n ≡ -i (mod p) for all i.
        Returns n mod LCM or None if inconsistent.
        """
        from math import gcd
        # Chinese Remainder Theorem with possibly non-coprime moduli
        # n ≡ a_i (mod m_i) where a_i = -i % p_i, m_i = p_i
        n_val = 0
        m_val = 1
        for i in range(1, T + 1):
            p = assign[i]
            a = (-i) % p
            m = p
            # Solve n_val + m_val * t ≡ a (mod m)
            # => m_val * t ≡ a - n_val (mod m)
            diff = (a - n_val) % m
            g = gcd(m_val, m)
            if diff % g != 0:
                return None  # inconsistent
            # Reduce: (m_val/g) * t ≡ (diff/g) (mod m/g)
            m_val_g = m_val // g
            m_g = m // g
            diff_g = diff // g
            # Inverse of m_val_g mod m_g
            try:
                inv = pow(m_val_g, -1, m_g)
            except ValueError:
                return None
            t = (diff_g * inv) % m_g
            n_val = n_val + m_val * t
            m_val = m_val * m_g  # lcm(m_val, m) = m_val * m / g
            n_val %= m_val
        return n_val, m_val  # n mod m_val

    # Strategy: try small prime sets and find assignments via backtracking.
    # For each position i, the prime p must satisfy: p does NOT divide i (otherwise -i ≡ 0 mod p means n ≡ 0 mod p, which is fine actually).
    # Actually any prime p works for any i; the constraint is only CRT consistency.
    
    # Better approach: for a given prime set S, try to assign each i a prime from S such that the CRT system is consistent.
    # Use backtracking with conflict detection.

    def try_prime_set(prime_set):
        """Try to find an assignment of primes from prime_set to positions 1..T that is CRT-consistent."""
        assign = {}
        
        def backtrack(pos):
            if time.time() > deadline:
                return None
            if pos > T:
                # All assigned, verify
                result = find_n_for_assignment(assign)
                if result is None:
                    return None
                n_mod, m_mod = result
                # Find smallest n >= 1000 such that n ≡ n_mod (mod m_mod)
                if n_mod < 1000:
                    k = (1000 - n_mod + m_mod - 1) // m_mod
                    n = n_mod + k * m_mod
                else:
                    n = n_mod
                if n > 1000000:
                    return None
                # Verify and score
                sc = score_for_n(n)
                return (sc, n)
            
            for p in prime_set:
                assign[pos] = p
                # Quick consistency check: check against already-assigned positions with same prime
                consistent = True
                for j, pj in assign.items():
                    if j == pos:
                        continue
                    if pj == p:
                        # n ≡ -j (mod p) and n ≡ -pos (mod p) must be consistent
                        # i.e., -j ≡ -pos (mod p) => j ≡ pos (mod p)
                        if j % p != pos % p:
                            consistent = False
                            break
                if consistent:
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
                del assign[pos]
            return None
        
        return backtrack(1)

    # Try prime sets from smallest to largest
    from itertools import combinations
    
    best_score = 999
    best_n = 1000  # fallback
    
    # First, try the smallest possible sets
    for size in range(1, len(SMALL_PRIMES) + 1):
        if time.time() > deadline:
            break
        for combo in combinations(SMALL_PRIMES, size):
            if time.time() > deadline:
                break
            result = try_prime_set(list(combo))
            if result is not None:
                sc, n = result
                if sc < best_score:
                    best_score = sc
                    best_n = n
                if best_score == size:  # Can't do better than the set size
                    return best_n
    
    # If CRT construction didn't find anything great, do a local search
    if time.time() < deadline - 5:
        # Random sampling
        import random
        rng = random.Random(42)
        for _ in range(200000):
            if time.time() > deadline:
                break
            n = rng.randint(1000, 1000000)
            sc = score_for_n(n)
            if sc < best_score:
                best_score = sc
                best_n = n
    
    # Local improvement: try shifting best_n by small amounts
    if time.time() < deadline - 2:
        base = best_n
        for delta in range(-100, 101):
            if time.time() > deadline:
                break
            n = base + delta
            if 1000 <= n <= 1000000:
                sc = score_for_n(n)
                if sc < best_score:
                    best_score = sc
                    best_n = n
    
    return best_n
