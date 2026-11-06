import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    # The optimal n is expected to use the smallest primes.
    # Product of first 15 primes: 2*3*5*...*47 ~ 6e17.
    # Product of first 16 primes: *53 ~ 3.4e19 > 10^18.
    # So k is at most 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_val = 10**18
    
    best_n = 1
    best_score = -1
    
    def tau_perp(n):
        """
        Calculate the score tau_perp(n).
        """
        if n == 1:
            return 0
            
        # Factorize n using the known primes (since n is generated from them)
        divisors = [1]
        temp = n
        
        for p in primes:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                # Extend divisors
                current_divs = list(divisors)
                factor = 1
                for _ in range(count):
                    factor *= p
                    for d in current_divs:
                        divisors.append(d * factor)
            
            # If temp becomes 1, we have factored n completely
            if temp == 1:
                break
        
        # If temp > 1, n has a prime factor larger than our list (should not happen for our candidates)
        # But for safety, handle it if needed. In our construction, n is always from 'primes'.
        
        divisors.sort()
        
        score = 0
        # Optimization: use math.gcd
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
                
        return score

    def get_max_tau_suffix(idx, e_max, current_n):
        """
        Estimate the maximum possible tau for the remaining primes.
        This is an upper bound to allow pruning.
        """
        if idx >= len(primes):
            return 1
        
        # Tighter upper bound: for each remaining prime, the max exponent is min(e_max, floor(log_p(max_val/current_n)))
        # But since exponents must be non-increasing, the next prime can have at most e_max,
        # the one after at most min(e_max, e_max_next), etc.
        # A simple but tighter bound: assume all remaining primes can have exponent e_max
        # but cap by what fits in the remaining budget.
        remaining = len(primes) - idx
        # Very loose but faster: (e_max + 1) ** remaining
        # Better: iterate and compute actual max exponents
        budget = max_val // current_n
        total_tau = 1
        for j in range(idx, len(primes)):
            p = primes[j]
            # Max exponent for this prime given current budget and non-increasing constraint
            # For simplicity, just use e_max as upper bound
            total_tau *= (e_max + 1)
            if total_tau > 10**18:
                break
        return total_tau

    def heuristic_score_estimate(exponents):
        """
        Fast heuristic estimate of tau_perp without generating all divisors.
        Based on the structure: coprime adjacent pairs occur when divisors differ
        in their prime factorization in a way that makes them share no common factors.
        
        For a number with prime factorization p1^e1 * p2^e2 * ... * pk^ek,
        the number of divisors is prod(ei+1). The number of coprime adjacent pairs
        is roughly proportional to the "boundary" between divisors that share primes
        and those that don't.
        
        A simpler heuristic: tau(n) - 1 is an upper bound.
        A lower bound: at least the number of primes used (each prime introduces
        new divisors that are coprime to some existing ones).
        
        For search guidance, we'll use tau(n) as a proxy since maximizing tau
        generally helps maximize tau_perp.
        """
        tau = 1
        for e in exponents:
            tau *= (e + 1)
        return tau - 1

    def dfs(idx, max_exp, current_n, current_tau, exponents):
        nonlocal best_n, best_score
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # If we have included at least one prime, it's a valid candidate
        # We evaluate all candidates found.
        if current_n > 1:
            # Quick heuristic check first
            heur = heuristic_score_estimate(exponents)
            if heur > best_score:
                # Do full evaluation
                score = tau_perp(current_n)
                if score > best_score:
                    best_score = score
                    best_n = current_n
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Determine max exponent for this prime
        # current_n * p^e <= max_val
        # p^e <= max_val // current_n
        limit = max_val // current_n
        
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1
            
        # Try exponents from e_max_possible down to 1
        # Trying higher exponents first might find good candidates sooner
        for e in range(min(e_max_possible, max_exp), 0, -1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            
            # Pruning: Check if we can possibly beat best_score
            # Upper bound on score is tau(n) - 1
            max_future_tau = get_max_tau_suffix(idx + 1, e, next_n)
            upper_bound_score = next_tau * max_future_tau - 1
            
            if upper_bound_score > best_score:
                exponents.append(e)
                dfs(idx + 1, e, next_n, next_tau, exponents)
                exponents.pop()
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    # Max exponent for 2 is log2(10^18) ~ 59
    # We pass max_exp = 60 to allow any reasonable start.
    dfs(0, 60, 1, 1, [])
    
    # Local search / polish: try small perturbations of the best solution found
    if best_n > 1 and time.time() - start_time < TIME_LIMIT - 30:
        # Try to improve by adjusting exponents
        # Factorize best_n to get its exponents
        temp = best_n
        best_exponents = []
        for p in primes:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count > 0:
                best_exponents.append((p, count))
        
        # Try local moves: increase/decrease exponents of individual primes
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 30:
            improved = False
            for i, (p, e) in enumerate(best_exponents):
                # Try decreasing
                if e > 1:
                    new_n = best_n // p
                    if new_n >= 2:
                        score = tau_perp(new_n)
                        if score > best_score:
                            best_score = score
                            best_n = new_n
                            best_exponents[i] = (p, e - 1)
                            improved = True
                # Try increasing
                if best_n * p <= max_val:
                    new_n = best_n * p
                    score = tau_perp(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        best_exponents[i] = (p, e + 1)
                        improved = True
        
        # Try adding a new prime (if we haven't used all 15)
        if len(best_exponents) < len(primes) and time.time() - start_time < TIME_LIMIT - 30:
            for p in primes[len(best_exponents):]:
                if best_n * p <= max_val:
                    new_n = best_n * p
                    score = tau_perp(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        best_exponents.append((p, 1))
    
    return best_n

if __name__ == "__main__":
    # Optional: run a quick self-test
    n = construct()
    # print(f"Best n: {n}")
    # print(f"Score: {tau_perp(n)}")
    pass
