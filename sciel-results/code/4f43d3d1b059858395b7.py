"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    def compute_score(U):
        if len(U) < 2:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        # Compute |U+U| and |U-U|
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            return 0.0
        score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
        return score
    
    # Start with {0, 1}
    U = [0, 1]
    best_score = compute_score(U)
    best_U = list(U)
    
    # Generate candidate primes
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Greedy: at each step, try adding the next prime and see if it improves score
    current_sum = 1  # last element in U
    next_prime_candidate = 2
    used_primes = set()
    
    while time.time() < deadline:
        # Find next few candidate primes
        candidates = []
        p = next_prime_candidate
        while len(candidates) < 20:
            if is_prime(p) and p not in used_primes:
                candidates.append(p)
            p += 1
        next_prime_candidate = p
        
        best_local_score = best_score
        best_local_prime = None
        
        for prime in candidates:
            new_U = U + [current_sum + prime]
            # Quick check: is this valid (nonnegative, contains 0)?
            # It should be since we're adding positive primes
            s = compute_score(new_U)
            if s > best_local_score:
                best_local_score = s
                best_local_prime = prime
        
        if best_local_prime is None:
            break
        
        # Add the best prime
        U.append(current_sum + best_local_prime)
        current_sum += best_local_prime
        used_primes.add(best_local_prime)
        best_score = best_local_score
        best_U = list(U)
        
        # Check deadline
        if time.time() > deadline - 2:
            break
    
    return best_U
# EVOLVE-BLOCK-END
