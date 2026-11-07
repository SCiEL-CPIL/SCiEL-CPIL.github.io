"""Approximate a ratio inside a divisor-sum fiber."""
import time
import math

def construct():
    """
    Find a, b <= 10^7 with sigma(a) = sigma(b) minimizing |a/b - 3/2|.
    Uses SPF sieve + O(N) sigma computation, then two-pointer merge on
    sigma-sorted groups to find the global minimum |2a - 3b| per sigma value.
    """
    deadline = time.time() + 80.0  # 10s safety margin before 90s kill
    N = 10_000_000

    # --- Step 1: SPF sieve (smallest prime factor) ---
    spf = list(range(N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # --- Step 2: Compute sigma(n) for all n in O(N) using recurrence ---
    # sigma(n) = sum of divisors. Use the multiplicative property via SPF.
    # For n = p^k * m where gcd(p,m)=1:
    #   sigma(n) = sigma(p^k) * sigma(m)
    # We compute sigma_powers on the fly.
    sigma = [0] * (N + 1)
    sigma[1] = 1
    # We'll use arrays to track: for each n, the highest power of spf[n] dividing n,
    # and the corresponding sigma(p^k).
    # Actually, a cleaner O(N) approach:
    # sigma[n] can be computed as:
    #   let p = spf[n], m = n // p
    #   if m % p == 0: sigma[n] = sigma[m] + n  (since sigma(p^k * m') = sigma(m')*(1+p+...+p^k))
    #   else: sigma[n] = sigma[m] * (1 + p)
    # Wait, that's not quite right. Let me use a different formulation.
    #
    # Standard linear sieve for sigma:
    # sigma[1] = 1
    # For prime p: sigma[p] = 1 + p
    # For n = p^k * m, p not dividing m:
    #   sigma[n] = sigma[m] * (1 + p + p^2 + ... + p^k)
    #
    # We track:
    #   cnt[n] = exponent of spf[n] in n
    #   spow[n] = spf[n]^cnt[n]
    #   spow_sigma[n] = 1 + spf[n] + ... + spf[n]^cnt[n]
    
    cnt = [0] * (N + 1)
    spow = [0] * (N + 1)  # p^k
    spow_sigma = [0] * (N + 1)  # 1 + p + ... + p^k
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p^(k+1) * m' where m = p^k * m'
            cnt[n] = cnt[m] + 1
            spow[n] = spow[m] * p
            spow_sigma[n] = spow_sigma[m] + spow[n]
            sigma[n] = sigma[m // spow[m]] * spow_sigma[n] if spow[m] > 1 else spow_sigma[n]
            # Wait, this is getting complicated. Let me use a simpler approach.
        else:
            cnt[n] = 1
            spow[n] = p
            spow_sigma[n] = 1 + p
            sigma[n] = sigma[m] * spow_sigma[n]

    # Actually the above has a bug. Let me redo with a cleaner linear sieve for sigma.
    # I'll use a different, well-known approach:
    # 
    # sigma[1] = 1
    # For each n from 2 to N:
    #   p = spf[n]
    #   m = n // p
    #   if m % p == 0:
    #     # n has at least p^2
    #     # sigma[n] = sigma[m] + n  ... NO this is wrong
    #
    # Let me just use the divisor-sum via the identity:
    # sigma(n) = sum_{d|n} d
    # Using the recurrence with prime powers properly:
    #
    # Define:
    #   rest[n] = n with all factors of spf[n] removed
    #   pk[n] = spf[n]^cnt[n]  (the full prime power part)
    #   sigma_pk[n] = 1 + p + p^2 + ... + p^cnt
    #
    # Then sigma[n] = sigma[rest[n]] * sigma_pk[n]
    
    # Recompute properly
    sigma = [0] * (N + 1)
    sigma[1] = 1
    rest = [0] * (N + 1)  # n with spf[n] factors removed
    pk = [0] * (N + 1)    # prime power part
    sigma_pk = [0] * (N + 1)  # sum of geometric series for pk
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so n = p * m where m still has p
            rest[n] = rest[m]
            pk[n] = pk[m] * p
            sigma_pk[n] = sigma_pk[m] + pk[n]
        else:
            # p does not divide m
            rest[n] = m
            pk[n] = p
            sigma_pk[n] = 1 + p
        sigma[n] = sigma[rest[n]] * sigma_pk[n]

    # --- Step 3: Group numbers by sigma value ---
    # We need to find, for each sigma value s, the pair (a, b) from the group
    # that minimizes |2a - 3b| (equivalently |a/b - 3/2| * 2b).
    # Actually we want to minimize |a/b - 3/2| = |2a - 3b| / (2b).
    # For fixed b, minimizing |2a - 3b| is best. But the denominator 2b varies.
    # So we need to minimize |2a - 3b| / (2b) = |2a - 3b| / (2b).
    #
    # Strategy: For each sigma group (sorted list of numbers), use two pointers.
    # For each b in the group, find the a in the group closest to 3b/2.
    # Since the group is sorted, for a given b, the best a is found by binary search
    # for 3b/2 in the sorted group. But that's O(k log k) per group.
    #
    # Better: two-pointer. Sort the group. For each b (in order), advance pointer a
    # toward 3b/2. Since both a and b increase, and 3b/2 increases, we can use
    # a two-pointer sweep.
    #
    # Actually, the cleanest: for each group, sort the elements. Then for each
    # element b, binary search for the element closest to 3*b/2. Track the global
    # minimum of |2a - 3b| / (2b).
    #
    # But with N=10^7, the total number of elements across all groups is N, and
    # binary search per element is O(log k) where k is group size. Total O(N log N).
    #
    # Let's build the groups. We need a dict mapping sigma -> list of n.
    # This uses a lot of memory. Alternative: sort pairs (sigma[n], n) and then
    # process contiguous runs of same sigma.
    
    # Create list of (sigma[n], n) for n=2..N (n=1 has sigma=1, likely no pair)
    # Actually include n=1 too.
    pairs = [(sigma[n], n) for n in range(1, N + 1)]
    pairs.sort()
    
    # Now process runs of same sigma value
    best_error = float('inf')
    best_a = 11
    best_b = 6  # fallback
    
    i = 0
    total = len(pairs)
    while i < total:
        if time.time() > deadline:
            break
        s = pairs[i][0]
        j = i
        while j < total and pairs[j][0] == s:
            j += 1
        # Group is pairs[i:j], the numbers are pairs[k][1] for k in [i,j)
        group_size = j - i
        if group_size >= 2:
            # Extract the numbers (they're already sorted since pairs is sorted by (sigma, n))
            # The numbers in the group are sorted in increasing order.
            nums = [pairs[k][1] for k in range(i, j)]
            # For each b in nums, find a in nums closest to 3*b/2
            # Use two-pointer: since nums is sorted, for increasing b, 3b/2 increases,
            # so the optimal a index is non-decreasing.
            a_ptr = 0
            for b_idx in range(group_size):
                b = nums[b_idx]
                target = 3.0 * b / 2.0
                # Advance a_ptr while nums[a_ptr+1] is closer to target
                while a_ptr < group_size - 1 and abs(nums[a_ptr + 1] - target) <= abs(nums[a_ptr] - target):
                    a_ptr += 1
                # Check a_ptr and possibly a_ptr-1
                for a_candidate in [nums[a_ptr]]:
                    a = a_candidate
                    # Compute |a/b - 3/2| exactly as rational
                    # |a/b - 3/2| = |2a - 3b| / (2b)
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        # Exact match, but task says raw value 0 is invalid
                        continue
                    error = diff / (2.0 * b)
                    if error < best_error:
                        best_error = error
                        best_a = a
                        best_b = b
                # Also check a_ptr - 1 if valid
                if a_ptr > 0:
                    a = nums[a_ptr - 1]
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        continue
                    error = diff / (2.0 * b)
                    if error < best_error:
                        best_error = error
                        best_a = a
                        best_b = b
        i = j

    # Ensure types are Python int
    best_a = int(best_a)
    best_b = int(best_b)
    
    # Validate
    if best_a <= 0 or best_b <= 0 or best_a > 10000000 or best_b > 10000000:
        return [11, 6]
    
    return [best_a, best_b]
