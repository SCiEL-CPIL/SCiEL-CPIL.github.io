import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    min_val = 211
    max_val = 10210

    # Target: product of selected * 210 must be a perfect square
    # 210 = 2 * 3 * 5 * 7, so the XOR target for parity signatures is {2,3,5,7}
    # For 2 elements a, b: sig(a) ^ sig(b) = {2,3,5,7}
    # We want to minimize max(a,b) - 210

    # Use a fast approach: for each number from 211 upward, compute its square-free signature,
    # and check if its complement has already been seen.

    # First, let's precompute signatures efficiently using trial division by small primes.
    # Since we only need to go up to 10210, we can use a simple approach.

    # Target signature
    target = frozenset({2, 3, 5, 7})

    # We'll store the first (smallest) number seen for each signature
    first_seen = {}

    best_answer = None
    best_max = float('inf')

    # Iterate from smallest to largest, checking if complement exists
    for x in range(min_val, max_val + 1):
        if time.time() > deadline - 3:
            break

        # Compute square-free signature of x
        sig = set()
        temp = x
        # Trial divide by primes up to sqrt(x)
        p = 2
        while p * p <= temp:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                sig.add(p)
            p += 1 if p == 2 else 2  # Skip even numbers after 2
        if temp > 1:
            sig.add(temp)

        s = frozenset(sig)
        complement = s ^ target

        # Check if complement was already seen
        if complement in first_seen:
            a = first_seen[complement]
            b = x
            if a != b:
                current_max = max(a, b)
                if current_max < best_max:
                    best_max = current_max
                    best_answer = [a, b]
                    # Can't do better than max=212 (value=2), so break early if we hit that
                    if best_max == 212:
                        break

        # Store the smallest number for this signature
        if s not in first_seen:
            first_seen[s] = x

    # If no 2-element solution found, use the best 1-element solution
    if best_answer is None:
        best_answer = [840]  # 210 * 2^2 = 840, value = 630
    else:
        best_answer = [int(a) for a in best_answer]

    # Verify the answer
    prod = 210
    for x in best_answer:
        prod *= x
    sqrt_val = math.isqrt(prod)
    if sqrt_val * sqrt_val != prod:
        # Fallback
        best_answer = [840]

    return best_answer
