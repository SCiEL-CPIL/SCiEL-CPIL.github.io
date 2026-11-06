import time
from math import lcm
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    # Split into two halves
    mid = n // 2
    half1_idx = list(range(mid))
    half2_idx = list(range(mid, n))

    w1 = [weights[i] for i in half1_idx]
    w2 = [weights[i] for i in half2_idx]

    def gen_sums(weights_list):
        """Generate all subset sums < L with their masks"""
        pairs = [(0, 0)]
        for i, w in enumerate(weights_list):
            if check_time():
                break
            bit = 1 << i
            current_len = len(pairs)
            for j in range(current_len):
                s, m = pairs[j]
                ns = s + w
                if ns < L:
                    pairs.append((ns, m | bit))
        return pairs

    # 1. Generate sums with masks
    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)

    if check_time():
        return [0]  # Fallback

    # 2. Extract sums for sorting
    sums1_vals = [s for s, m in sums1]
    sums2_vals = [s for s, m in sums2]

    sums1_vals.sort()
    sums2_vals.sort()

    # 3. Find Max Sum < L
    j = len(sums2_vals) - 1
    best_sum = -1

    for s1 in sums1_vals:
        while j >= 0 and s1 + sums2_vals[j] >= L:
            j -= 1
        if j < 0:
            break
        current_sum = s1 + sums2_vals[j]
        if current_sum > best_sum:
            best_sum = current_sum

    if best_sum == -1:
        # Fallback: single item with max weight (min fraction)
        max_w_idx = max(range(n), key=lambda i: weights[i])
        return [max_w_idx]

    # 4. Create lookup for sums2 to find masks
    sums2_dict = {}
    for s, m in sums2:
        sums2_dict[s] = m

    target_s1 = None
    mask1 = 0
    mask2 = 0

    for s1, m1 in sums1:
        s2_needed = best_sum - s1
        if s2_needed in sums2_dict:
            target_s1 = s1
            mask1 = m1
            mask2 = sums2_dict[s2_needed]
            break

    if target_s1 is None:
        # Should not happen, but fallback
        return [0]

    # 5. Reconstruct indices
    final_indices = []
    for k in range(mid):
        if mask1 & (1 << k):
            final_indices.append(k)
    for k in range(mid, n):
        if mask2 & (1 << (k - mid)):
            final_indices.append(k)

    if not final_indices:
        return [0]

    return final_indices
