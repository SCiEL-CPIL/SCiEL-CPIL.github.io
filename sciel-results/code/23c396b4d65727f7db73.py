import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s wall-clock limit

    n = 210
    max_val = 10210
    max_size = 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # We need: product(selected) * 210 is a perfect square
    # Equivalently: XOR of parity-signatures of selected numbers = {2,3,5,7}
    # For a 2-element solution: sig(a) XOR sig(b) = {2,3,5,7}
    # i.e., sig(b) = sig(a) XOR {2,3,5,7}

    # Step 1: Sieve-based parity signature computation
    # For each number, track which primes have odd exponent
    # We only need to track primes that actually appear (up to 10210)
    # Use a dict: for each number, store the set of primes with odd exponent

    # Efficient approach: for each prime p, for each power p^k, 
    # toggle p in the signature of all multiples of p^k but not p^(k+1)
    # Actually simpler: for each prime p, for each multiple m of p,
    # count exponent of p in m, toggle if odd.

    # Even simpler: use smallest prime factor sieve, then factor each number.
    # But the key insight: we can compute signatures incrementally.

    # Let's use a clean approach:
    # 1. Compute smallest prime factor (SPF) for all numbers up to max_val
    # 2. For each number x in (210, max_val], factor using SPF and get parity sig

    # SPF sieve
    spf = list(range(max_val + 1))  # spf[i] = smallest prime factor of i
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_sig(x):
        """Get parity signature using SPF table."""
        sig = set()
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                sig.add(p)
        return frozenset(sig)

    target = frozenset({2, 3, 5, 7})

    # Step 2: Find the optimal 2-element solution
    # Map each signature to the smallest number > 210 with that signature
    # Then for each number a, check if sig(a) XOR target exists in the map

    # Search in increasing order of max element to find minimum
    # Key: we want to minimize max(a, b) - 210
    # So iterate by increasing upper bound

    # First, build the full signature map for all numbers in (210, max_val]
    # But to find the MINIMUM max element, we can do a binary search or
    # simply iterate the upper bound from 211 upward.

    # Actually, the fastest approach:
    # Build map: sig -> list of numbers with that sig (sorted)
    # Then for each pair of complementary signatures, find the pair minimizing max.

    # Let's just build the full map and check all complementary pairs.
    sig_to_numbers = {}
    for x in range(211, max_val + 1):
        s = get_sig(x)
        if s not in sig_to_numbers:
            sig_to_numbers[s] = []
        sig_to_numbers[s].append(x)

    # For 2-element solutions:
    # We need sig(a) ^ sig(b) = target, so sig(b) = sig(a) ^ target
    # For each signature s, if s^target is also in the map,
    # we can pair the smallest from s with the smallest from s^target
    # (ensuring they're distinct numbers)

    best_value = float('inf')
    best_answer = None

    # Check single-element: x such that x*210 is perfect square
    # x = 210 * k^2, x > 210, x <= 10210
    for k in range(1, 100):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            val = x - 210
            if val < best_value:
                best_value = val
                best_answer = [x]
        else:
            break

    # Check 2-element solutions
    # To minimize max(a,b), for each complementary pair of signatures,
    # the best pair is (smallest in sig1, smallest in sig2) if distinct,
    # or (smallest in sig1, second smallest in sig2) if same number.
    # But we need to be careful: the pair that minimizes max is not always
    # (min1, min2). It could be (min1, min2) or we might want a different combo.
    # Actually, to minimize max(a,b), we want both a and b to be as small as possible.
    # The optimal pair for signatures s1, s2 is:
    #   If s1 != s2: (min of s1, min of s2)
    #   If s1 == s2: (min of s1, second min of s1)
    # But wait - we also need to consider that maybe a slightly larger number
    # in one signature paired with a much smaller number in the other gives lower max.
    # No - if we fix the two signatures, the minimum max is achieved by taking
    # the two smallest numbers (one from each, or two from same if s1==s2).

    # Actually, more carefully: we want to minimize max(a, b) where a has sig s1, b has sig s2.
    # The minimum possible max is:
    #   min over a in nums[s1], b in nums[s2] of max(a, b)
    # This is achieved by: take the sorted lists, and the answer is the smaller of:
    #   max(nums[s1][0], nums[s2][0])  [if they're different numbers]
    #   or if same number, max(nums[s1][0], nums[s2][1]) etc.
    # But actually the simplest: the minimum max(a,b) is the minimum value v such that
    # there exists a <= v in s1 and b <= v in s2. That's max(min_s1, min_s2) if distinct.

    # Let me just iterate and check:
    processed_sigs = set()
    for s1, nums1 in sig_to_numbers.items():
        if s1 in processed_sigs:
            continue
        s2 = s1 ^ target
        if s2 not in sig_to_numbers:
            continue
        processed_sigs.add(s1)
        processed_sigs.add(s2)

        nums2 = sig_to_numbers[s2]

        if s1 == s2:
            # Need two distinct numbers from the same signature
            if len(nums1) >= 2:
                a, b = nums1[0], nums1[1]
                max_elem = max(a, b)
                val = max_elem - 210
                if val < best_value:
                    best_value = val
                    best_answer = [a, b]
        else:
            # One from each signature
            a = nums1[0]
            b = nums2[0]
            if a != b:
                max_elem = max(a, b)
                val = max_elem - 210
                if val < best_value:
                    best_value = val
                    best_answer = [a, b]
            else:
                # Same number, need to try next
                # Try (a, nums2[1]) and (nums1[1], b)
                if len(nums2) >= 2:
                    b2 = nums2[1]
                    max_elem = max(a, b2)
                    val = max_elem - 210
                    if val < best_value:
                        best_value = val
                        best_answer = [a, b2]
                if len(nums1) >= 2:
                    a2 = nums1[1]
                    max_elem = max(a2, b)
                    val = max_elem - 210
                    if val < best_value:
                        best_value = val
                        best_answer = [a2, b]

    # Verify the answer
    if best_answer is not None:
        # Final verification
        prod = 210
        for x in best_answer:
            prod *= x
        sqrt_val = math.isqrt(prod)
        if sqrt_val * sqrt_val != prod:
            # Should not happen, but fallback
            best_answer = [840]
            best_value = 630

        return [int(x) for x in best_answer]

    # Fallback
    return [840]
