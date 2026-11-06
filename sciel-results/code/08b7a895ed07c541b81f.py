import time
import math

def construct():
    S = 123457
    start_time = time.time()
    deadline = 80.0  # 10s margin from 90s limit

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

    # Generate primes up to S
    # We only need to check primes up to S
    # In worst case, P(S) is S (if S is prime), so we need to go up to S
    primes = []
    for i in range(2, S + 1):
        if is_prime(i):
            primes.append(i)

    # V contains all M-smooth numbers <= S for the current prime threshold M
    # Initially for M=1, V={1}
    V = {1}

    # Check M=1 case (not explicitly looped as 1 is not in primes list)
    # V={1}. a=1, b=123456. 123456 not in V.
    # So no solution for M=1.

    for p in primes:
        # Check time limit
        if time.time() - start_time > deadline:
            # Fallback to a valid solution
            # a=1, b=S is always valid
            return [1, S]

        # Expand V to include numbers with prime factor p
        # New numbers are p^k * v where v in V and p^k * v <= S
        new_nums = set()
        for v in V:
            val = v
            # Multiply by p repeatedly
            while val * p <= S:
                val *= p
                new_nums.add(val)

        # Update V
        V.update(new_nums)

        # Check for pair a, b in V such that a + b = S
        # Iterate through V
        # Optimization: We can break early if we find one
        for a in V:
            b = S - a
            if b > 0 and b in V:
                # Found a valid pair
                # Since we iterate primes in increasing order, this is the minimal max prime factor
                return [a, b]

    # If no solution found in loop (should not happen as a=1, b=S is always a solution for M=P(S))
    return [1, S]

