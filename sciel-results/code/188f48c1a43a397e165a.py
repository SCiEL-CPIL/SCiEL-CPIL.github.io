import math
import time
import heapq

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0

    # First, compute largest prime factor for all numbers up to N using sieve
    lp = [1] * (N + 1)
    lp[0] = 0
    lp[1] = 1
    for i in range(2, N + 1):
        if lp[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i

    # Find the best pair by scanning
    best_max_pf = N + 1
    best_a = 1
    best_b = N - 1
    for a in range(1, N // 2 + 1):
        b = N - a
        current_max = max(lp[a], lp[b])
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b

    # Now try to improve using B-smooth number generation
    # Generate all B-smooth numbers up to N for increasing B
    # We want to find the smallest B such that there exist a+b=N with both a,b being B-smooth

    # Collect all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False

    # Incrementally build B-smooth sets
    # For B=1: only {1}
    # For B=2: {1, 2, 4, 8, ...}
    # For B=3: {1, 2^a * 3^b, ...}
    # etc.

    # We'll generate smooth numbers incrementally
    # Start with B=1, add primes one by one

    # A number n is B-smooth if all prime factors of n are <= B
    # Equivalently, n's largest prime factor <= B

    # For each candidate B (a prime), check if there exist a, b with a+b=N, 
    # both B-smooth (i.e., lp[a] <= B and lp[b] <= B)

    # Instead of scanning all a for each B, we can:
    # 1. Build the set of B-smooth numbers up to N
    # 2. Check if any two sum to N

    # Let's be smart: start from the best we found and see if we can do better
    # Current best_max_pf is the answer from full scan

    # Try to find if we can achieve a lower max by using B-smooth approach
    # For each prime p in increasing order, check if both a and b can be p-smooth

    # Actually, let's just do a smarter search:
    # For each possible threshold T, check if there exist a, b with a+b=N, lp[a]<=T, lp[b]<=T
    # Binary search on T

    # Build list of (lp_value, number) sorted by lp_value
    # Group numbers by their largest prime factor

    # For each prime p, let S_p = {n : lp[n] <= p}
    # We need to check if N can be written as a + b with a, b in S_p

    # Since S_p grows as p increases, we can incrementally add numbers

    # Let's use a different approach: 
    # For each prime p in ascending order, maintain the set of p-smooth numbers
    # and check if any pair sums to N

    # But this could be expensive. Let's think differently.

    # The key insight: we already have lp[] for all numbers.
    # The optimal answer is min over a of max(lp[a], lp[N-a])
    # We already computed this. Let's verify if we can improve it.

    # Actually, the full scan already gives us the optimal answer.
    # The score is based on max(lp[a], lp[b]), and we minimized it.
    # So the current approach should be optimal.

    # Let me double-check: the scan goes from a=1 to a=N//2, checking all pairs.
    # This is exhaustive and correct.

    # The issue might be that we can do better by being smarter about which pairs to check.
    # But since we're checking ALL pairs, we should have the optimal.

    # Wait - let me reconsider. The scan is O(N) which is fine for N=123457.
    # The sieve is O(N log log N). Total time should be well under 1 second.

    # Let's verify the current best
    return [best_a, best_b]
