# EVOLVE-BLOCK-START
import time
import bisect
import numpy as np

MAXV = 10000000
WALL_LIMIT = 88.0  # hard stop to ensure we return well before the 90s limit
SAFETY = 0.85

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    # Fallback: a known valid pair with sigma(11)=12, sigma(6)=12.
    # |11/6 - 1.5| = |1.8333 - 1.5| = 0.3333 = 1/3.
    # Score = 1 / (1/3) = 3. This is a very poor score, but valid.
    # The seed score was ~4077490, which corresponds to a raw value of ~1/4077490.
    # Wait, the seed score is 4077490.0000.
    # Score = seed / value? No, "score is the binary64 serialization of that rational".
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value" -> This implies the reported score in the prompt (4077490) is the "search score"?
    # No, the prompt says: "CURRENT PROGRAM (score 4077490.0000)".
    # And "A raw value of exactly 0 cannot be scored... The evaluator measures the raw value (lower is better)."
    # Usually in these tasks, the "score" printed is the metric to be minimized or maximized?
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # If seed value is $V_{seed}$, and seed score is $S_{seed}$, then $S_{seed} = Seed / V_{seed}$?
    # If the current program has score 4077490, and it's an "improvement" over 1.0, then it's likely a large number.
    # But "lower is better" for raw value.
    # If raw value is small, score is large?
    # Let's assume the goal is to MINIMIZE the raw value $|a/b - 3/2|$.
    # The "score" 4077490 might be $1 / \text{raw\_value}$?
    # If raw value is $1/4077490$, then $1/\text{raw} = 4077490$.
    # So we want to MAXIMIZE the score, which means MINIMIZING the raw value.
    # My fallback [11, 6] has raw value $1/3 \approx 0.333$.
    # Score would be $1/0.333 = 3$.
    # This is much WORSE than 4077490.
    # So I MUST find a pair with a very small difference.

    try:
        N = MAXV

        # ---------- Linear Sieve for sigma ----------
        # We need sigma(n) for n up to N.
        # Memory:
        # sigma: list of ints (N+1) -> ~80MB
        # is_prime: bytearray (N+1) -> ~10MB
        # smallest_prime_factor or similar?
        # Linear sieve is O(N).
        
        # Optimization: Use numpy for the sieve if possible?
        # Python loops for linear sieve up to 10^7 are slow (~10-20s).
        # NumPy vectorized sieve for sigma is tricky because of the multiplicative property.
        # However, we can compute sigma using a modified sieve that adds contributions.
        # sigma(n) = sum_{d|n} d.
        # We can iterate d from 1 to N, and add d to sigma[m] for all m multiple of d.
        # This is O(N log N).
        # N=10^7, log N ~ 16. 1.6 * 10^8 operations.
        # In Python, this is too slow.
        # In NumPy, we can do this in chunks or using vectorized slicing?
        # sigma = np.zeros(N+1, dtype=np.int64)
        # for d in range(1, N+1):
        #     sigma[d::d] += d
        # This loop is N iterations. Each iteration is a vectorized add.
        # The inner vectorized add is fast, but the loop overhead is N=10^7.
        # 10^7 Python loop iterations is ~1-2 seconds? No, likely 5-10s.
        # Plus the cost of the vectorized adds.
        # Total time might be 20-40s. Risky.
        
        # Alternative: Linear sieve in Python is ~10-15s for 10^7.
        # Let's stick to Linear Sieve but optimize it heavily.
        
        is_prime = bytearray(b'\x01') * (N + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        
        sigma = [0] * (N + 1)
        sigma[1] = 1
        primes = []
        # max_pow[i] = p^k where p^k is the highest power of p dividing i
        # sigma_pow[i] = 1 + p + ... + p^k
        max_pow = [0] * (N + 1)
        sigma_pow = [0] * (N + 1)
        
        # Local variables for speed
        _is_prime = is_prime
        _sigma = sigma
        _max_pow = max_pow
        _sigma_pow = sigma_pow
        _primes = primes
        _N = N
        
        # Linear Sieve
        for i in range(2, _N + 1):
            if _is_prime[i]:
                _primes.append(i)
                _sigma[i] = 1 + i
                _max_pow[i] = i
                _sigma_pow[i] = 1 + i
            
            for p in _primes:
                ip = i * p
                if ip > _N:
                    break
                _is_prime[ip] = 0
                if i % p == 0:
                    _max_pow[ip] = _max_pow[i] * p
                    _sigma_pow[ip] = _sigma_pow[i] + _max_pow[ip]
                    _sigma[ip] = _sigma[i] // _sigma_pow[i] * _sigma_pow[ip]
                    break
                else:
                    _max_pow[ip] = p
                    _sigma_pow[ip] = 1 + p
                    _sigma[ip] = _sigma[i] * (1 + p)
            
            if (i & 0xFFFFF) == 0 and time_left() < 30.0:
                return [11, 6]

        if time_left() < 20.0:
            return [11, 6]

        # ---------- Group indices by sigma value ----------
        # We want to find a, b such that sigma(a) = sigma(b) and a/b approx 3/2.
        # This implies a approx 1.5 b.
        # So a > b.
        # We can iterate over b, find target a = 1.5 b.
        # We need to quickly find if there is an a in the same sigma group close to 1.5 b.
        
        # Strategy:
        # 1. Build a dict: sigma_val -> list of n's.
        # 2. For each group, sort the list.
        # 3. For each b in the group, look for a in the same group closest to 1.5 b.
        #    Use binary search.
        
        # Memory concern: Storing lists for 10^7 items.
        # Dict of lists is okay.
        
        groups = {}
        for n in range(2, _N + 1):
            s = _sigma[n]
            if s in groups:
                groups[s].append(n)
            else:
                groups[s] = [n]
        
        # We only care about groups with size >= 2.
        # Also, to minimize |a/b - 1.5|, we want large b?
        # No, the error is |a/b - 1.5|.
        # Let a = 1.5 b + eps.
        # |eps/b| is the error.
        # To minimize error, we want small |eps| and large b.
        # So large b is better if |eps| is small.
        
        # Iterate through groups.
        best_val = float('inf')
        best_pair = None
        
        # To speed up, we can process groups in some order?
        # Or just iterate all.
        # Number of groups is roughly N / log N? No, sigma(n) is roughly n.
        # Number of distinct sigma values is smaller.
        
        for s, gn in groups.items():
            L = len(gn)
            if L < 2:
                continue
            
            gn.sort()
            
            # For each b in gn, find a in gn closest to 1.5 b.
            # We can do this in O(L log L) or O(L) with two pointers?
            # Since we want a > b (approx 1.5b), and both are in sorted list.
            # Let's use bisect.
            
            for b in gn:
                target = 1.5 * b
                # Find a in gn closest to target.
                # a must be distinct from b.
                # Since a approx 1.5b, and b >= 2, a >= 3.
                # If b is small, a is small.
                
                # Bisect to find insertion point of target in gn.
                idx = bisect.bisect_left(gn, target)
                
                # Check idx and idx-1.
                # We need to ensure a != b.
                # Since target = 1.5 b, and b is in gn, target > b (for b>0).
                # So a will be > b usually.
                # If a == b, it means 1.5b == b => b=0, impossible.
                # Or if the closest value is b itself?
                # If target is between b and next element, and b is the closest?
                # Distance to b is |1.5b - b| = 0.5b.
                # Distance to next element a_next is |a_next - 1.5b|.
                # If a_next is very close to 1.5b, it's better.
                
                candidates = []
                if idx < L:
                    candidates.append(gn[idx])
                if idx > 0:
                    candidates.append(gn[idx-1])
                
                for a in candidates:
                    if a == b:
                        continue
                    # Calculate error |a/b - 1.5| = |2a - 3b| / (2b)
                    # We want to minimize this.
                    # Equivalent to minimizing |2a - 3b| / b.
                    # Or minimizing |2a - 3b| / (2b).
                    
                    num = abs(2*a - 3*b)
                    if num == 0:
                        return [a, b] # Perfect match?
                        # But "raw value of exactly 0 ... invalid".
                        # So we should skip this?
                        # "A raw value of exactly 0 cannot be scored and counts as invalid."
                        # So we must NOT return a pair where 2a == 3b.
                        # If we find one, we should look for the next best?
                        # Or just return a different one?
                        # If 2a == 3b, then a/b = 1.5 exactly.
                        # We need a/b != 1.5.
                        # So skip this candidate.
                        continue
                    
                    val = num / (2.0 * b)
                    if val < best_val:
                        best_val = val
                        best_pair = (a, b)
            
            if time_left() < 5.0:
                break
        
        if best_pair is None:
            return [11, 6]
        
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= _N and 1 <= b <= _N):
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]
        
        return [a, b]

    except Exception:
        return [11, 6]

# EVOLVE-BLOCK-END
