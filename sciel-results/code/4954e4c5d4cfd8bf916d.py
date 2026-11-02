import time
import random
import numpy as np
from itertools import product as iproduct


def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(sorted(set(U)), dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    mx = int(arr.max())
    if mx == 0:
        return 0.0

    N = 2 * mx + 1
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0

    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2

    F = np.fft.rfft(sig, n=fft_size)
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)

    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    diffset_size = np.sum(S_diff > 1e-6)

    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
    return 1.0 + np.log(ratio) / denom


def _build_from_digits(digit_sets, bases):
    """Build U as mixed-radix product of digit sets with given bases.
    U = { sum_i d_i * (prod_{j<i} base_j) : d_i in digit_sets[i] }
    """
    if not digit_sets:
        return [0]
    # Compute place values
    place_vals = [1]
    for i in range(1, len(bases)):
        place_vals.append(place_vals[-1] * bases[i - 1])

    # Generate all combinations
    U = set()
    for combo in iproduct(*digit_sets):
        val = sum(d * pv for d, pv in zip(combo, place_vals))
        U.add(val)
    U = sorted(U)
    if 0 not in U:
        U = [0] + U
    return U


def _score_from_digits(digit_sets, bases):
    """Score a mixed-radix construction."""
    U = _build_from_digits(digit_sets, bases)
    return _score_set(U), U


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    # Strategy: Mixed-radix construction
    # U = { sum_i d_i * P_i : d_i in D_i } where P_i = prod_{j<i} b_j
    # |U+U| <= prod_i |D_i + D_i|, |U-U| <= prod_i |D_i - D_i|
    # We want each digit set to have |D_i - D_i| / |D_i + D_i| > 1
    # This happens when D_i is NOT symmetric around 0.
    # E.g., D_i = {0, 1, k} for some k:
    #   D_i + D_i = {0, 1, k, 2, k+1, 2k} -> size depends on k
    #   D_i - D_i = {-k, -(k-1), -1, 0, 1, k-1, k} -> size 2k+1 (if k>1)
    # For k=2: D={0,1,2}, D+D={0,1,2,3,4} size 5, D-D={-2,-1,0,1,2} size 5. Ratio=1.
    # For k=3: D={0,1,3}, D+D={0,1,2,3,4,6} size 6, D-D={-3,-2,-1,0,1,2,3} size 7. Ratio=7/6.
    # For k=4: D={0,1,4}, D+D={0,1,2,4,5,8} size 6, D-D={-4,-3,-1,0,1,3,4} size 7. Ratio=7/6.
    # For k=5: D={0,1,5}, D+D={0,1,2,5,6,10} size 6, D-D={-5,-4,-1,0,1,4,5} size 7. Ratio=7/6.
    #
    # Better: D_i = {0, a, b} with a < b.
    # D+D = {0, a, b, 2a, a+b, 2b} (6 elements if all distinct and no overlaps)
    # D-D = {-b, a-b, -a, 0, a, b-a, b} (7 elements if all distinct)
    # Ratio = 7/6 when all 6 sums and 7 diffs are distinct.
    #
    # Even better with 4 elements: D = {0, a, b, c}
    # |D+D| can be as low as ~7-9, |D-D| can be up to 2*4-1=7... no, max 2*4-1=7? No.
    # |D-D| for 4 elements: max is 2*4-1 = 7? No, it's at most 2*4-1 = 7? 
    # Actually |A-A| for |A|=n is at most 2n-1. For n=4, max 7.
    # |A+A| for |A|=n is at most n(n+1)/2. For n=4, max 10.
    # So ratio can be at most 7/7=1 for 4 elements? No wait.
    # |A-A| max is 2n-1 = 7. |A+A| min is 2n-1 = 7 (when A is arithmetic progression).
    # So for AP: ratio = 1. For non-AP: |A+A| > 2n-1, so ratio < 1.
    # Hmm, that's a problem. For small digit sets, ratio <= 1.
    #
    # Wait, I need to reconsider. The key insight is that the TOTAL ratio is a product:
    # |U-U|/|U+U| = prod_i |D_i-D_i| / prod_i |D_i+D_i|
    # If each ratio_i < 1, the product is even smaller.
    # But if the bases create "gaps" that prevent overlap in the sumset, the actual
    # |U+U| can be smaller than the product bound.
    #
    # Actually, the product bound IS tight when bases are large enough (no carry).
    # So we need individual digit ratios > 1, which requires...
    #
    # Let me reconsider. For D = {0, 1, k}:
    # |D+D| = |{0,1,k,2,k+1,2k}|. If k >= 3, these are 6 distinct values.
    # |D-D| = |{-k, 1-k, -1, 0, 1, k-1, k}|. If k >= 3, these are 7 distinct values.
    # Ratio = 7/6 > 1. 
    #
    # So with multiple digits each of type {0, 1, k_i}, the total ratio = (7/6)^m.
    # And max U = sum_i k_i * P_i where P_i = prod_{j<i} b_j.
    # To keep max U small, use small bases. But base must be > max digit to avoid carry.
    # If D_i = {0, 1, k}, base must be > k. Let base_i = k_i + 1.
    #
    # Let's try: all digits D_i = {0, 1, 3}, base_i = 4.
    # m digits: |U| = 3^m, max U = 3*(4^m - 1)/3 = 4^m - 1.
    # |U+U| = 6^m (product of 6s), |U-U| = 7^m (product of 7s).
    # Ratio = (7/6)^m.
    # Score = 1 + m*ln(7/6) / ln(2*4^m - 1) ≈ 1 + m*0.154 / (m*ln(4)*2) = 1 + 0.154/(2*1.386) ≈ 1 + 0.0556.
    # That's only ~1.056. Not great.
    #
    # Better: use larger k to get higher ratio per digit.
    # D = {0, 1, k}: ratio = 7/6 always (for k>=3). Same ratio regardless of k.
    #
    # Try D = {0, 2, k} with k > 4:
    # D+D = {0, 2, k, 4, k+2, 2k}. 6 elements if distinct.
    # D-D = {-k, 2-k, -2, 0, 2, k-2, k}. 7 elements if distinct.
    # Same ratio 7/6.
    #
    # Try 4-element digits: D = {0, a, b, c}
    # Max |D-D| = 7, max |D+D| = 10.
    # For D = {0, 1, 4, 9}:
    # D+D = {0,1,2,4,5,8,9,10,13,18} = 10 elements
    # D-D = {-9,-5,-4,-1,0,1,4,5,8,9}... let me compute:
    # 0-0=0, 1-0=1, 4-0=4, 9-0=9, 0-1=-1, 1-1=0, 4-1=3, 9-1=8,
    # 0-4=-4, 1-4=-3, 4-4=0, 9-4=5, 0-9=-9, 1-9=-8, 4-9=-5, 9-9=0
    # D-D = {-9,-8,-5,-4,-3,-1,0,1,3,4,5,8,9} = 13 elements!
    # Wait that's more than 2n-1=7. Let me recount.
    # n=4, |A-A| <= 2n-1 = 7? No! |A-A| <= 2n-1 is WRONG.
    # |A-A| can be up to n^2 in general, but for sets of integers, |A-A| <= 2*max(A)-min(A)+1.
    # Actually |A-A| for a set of n integers is at most 2n-1 ONLY if... no that's not right either.
    # For A = {a_1 < ... < a_n}, A-A has at most n(n+1)/2 distinct values (since a_i - a_j = -(a_j - a_i)).
    # Actually |A-A| <= n(n+1)/2. For n=4, that's 10. But I got 13 above, let me recheck.
    # D = {0, 1, 4, 9}
    # Differences: 0,1,4,9,-1,0,3,8,-4,-3,0,5,-9,-8,-5,0
    # Unique: {-9,-8,-5,-4,-3,-1,0,1,3,4,5,8,9} = 13 values.
    # But n(n+1)/2 = 10. Contradiction? No, n(n+1)/2 counts ordered pairs including both (i,j) and (j,i).
    # The number of DISTINCT values is at most n(n+1)/2 but can be less. Here it's 13 > 10?
    # Wait, I think I'm confusing myself. Let me just count: 13 distinct differences from 4 elements.
    # That's fine. The bound is |A-A| <= 2*max(A) - 2*min(A) + 1 = 2*9 - 0 + 1 = 19.
    # OK so 13 is valid.
    #
    # |D+D| for D={0,1,4,9}: {0,1,2,4,5,8,9,10,13,18} = 10 elements.
    # Ratio = 13/10 = 1.3. Better than 7/6=1.167!
    #
    # Try D = {0, 1, 4, 10}:
    # D+D: 0,1,2,4,5,8,10,11,14,18,20 → 11 elements
    # D-D: 0,1,4,10,-1,0,3,9,-4,-3,0,6,-10,-9,-6,0 → {-10,-9,-6,-4,-3,-1,0,1,3,4,6,9,10} = 13
    # Ratio = 13/11 ≈ 1.18. Worse.
    #
    # Try D = {0, 1, 5, 9}:
    # D+D: 0,1,2,5,6,10,9,10,14,18 → {0,1,2,5,6,9,10,14,18} = 9
    # D-D: 0,1,5,9,-1,0,4,8,-5,-4,0,4... wait
    # 0-0=0,1-0=1,5-0=5,9-0=9,0-1=-1,1-1=0,5-1=4,9-1=8,0-5=-5,1-5=-4,5-5=0,9-5=4,0-9=-9,1-9=-8,5-9=-4,9-9=0
    # D-D = {-9,-8,-5,-4,-1,0,1,4,5,8,9} = 11
    # Ratio = 11/9 ≈ 1.22.
    #
    # Try D = {0, 1, 3, 7}:
    # D+D: 0,1,3,7,2,4,8,6,10,14 → {0,1,2,3,4,6,7,8,10,14} = 10
    # D-D: 0,1,3,7,-1,0,2,6,-3,-2,0,4,-7,-6,-4,0 → {-7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7} = 13
    # Ratio = 13/10 = 1.3.
    #
    # Try D = {0, 2, 5, 9}:
    # D+D: 0,2,5,9,4,7,11,10,14,18 → {0,2,4,5,7,9,10,11,14,18} = 10
    # D-D: 0,2,5,9,-2,0,3,7,-5,-3,0,4,-9,-7,-4,0 → {-9,-7,-5,-4,-3,-2,0,2,3,4,5,7,9} = 13
    # Ratio = 13/10 = 1.3.
    #
    # Great! So 4-element digits with ratio 1.3 are achievable.
    # With m such digits: total ratio = 1.3^m.
    # If base_i = max(D_i) + 1 = 10 (for D={0,1,3,7}), then max U ≈ 7*(10^m - 1)/9.
    # Score ≈ 1 + m*ln(1.3) / ln(2*7*10^m/9) ≈ 1 + m*0.262 / (m*ln(10)*ln(2*7/9))
    # ≈ 1 + 0.262 / (2.303 * 0.448) ≈ 1 + 0.262/1.032 ≈ 1 + 0.254.
    # That's ~1.25! Much better.
    #
    # But wait, we need to verify the product structure actually achieves the product ratio.
    # The key is that with base > max digit, there's no carry, so |U+U| = prod |D_i+D_i| and |U-U| = prod |D_i-D_i|.
    # This is exact when bases are large enough.
    #
    # Let's implement this and then do local search on the digit sets.

    # Start with a promising initial construction
    # Use digits D = {0, 1, 3, 7} with base 8 (need base > 7)
    # Actually base just needs to be > max(D_i) to avoid overlap in the mixed-radix representation.
    # Let's use base = max(D_i) + 1.

    def _try_construction(digit_sets, bases):
        """Try a mixed-radix construction and return (score, U)."""
        # Verify no carry: base_i > max(D_i) for all i
        for i, (D, b) in enumerate(zip(digit_sets, bases)):
            if b <= max(D):
                return 0.0, []
        U = _build_from_digits(digit_sets, bases)
        score = _score_set(U)
        return score, U

    # Initial: one digit set
    best_score = 0.0
    best_U = [0, 1]

    # Try single digit sets
    candidate_digits = [
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 2, 5, 9],
        [0, 1, 3, 8],
        [0, 1, 4, 8],
        [0, 2, 5, 8],
        [0, 1, 2, 7],
        [0, 1, 3, 6],
        [0, 2, 4, 7],
        [0, 1, 5, 8],
    ]

    for D in candidate_digits:
        b = max(D) + 1
        score, U = _try_construction([D], [b])
        if score > best_score:
            best_score = score
            best_U = U

    # Now try multi-digit constructions
    # Start from best single digit and add more
    # We'll do local search: modify digit sets, add/remove digits

    # Current state: list of (digit_set, base)
    # Find best single digit
    best_single_D = None
    best_single_score = 0
    for D in candidate_digits:
        b = max(D) + 1
        score, U = _try_construction([D], [b])
        if score > best_single_score:
            best_single_score = score
            best_single_D = D

    # Start with best single digit
    current_digits = [best_single_D[:]]
    current_bases = [max(best_single_D) + 1]
    current_score = best_single_score

    if current_score > best_score:
        best_score = current_score
        best_U = _build_from_digits(current_digits, current_bases)

    # Local search: try adding digits, modifying digits, changing bases
    time_budget = deadline - start_time

    def _local_search_step(current_digits, current_bases, current_score):
        """Try one local search move. Return (new_digits, new_bases, new_score, improved)."""
        move_type = random.choice(['add_digit', 'modify_digit', 'change_base', 'remove_digit'])

        if move_type == 'add_digit' and len(current_digits) < 6:
            # Add a new digit set
            # Use a random small set containing 0
            size = random.choice([3, 4, 5])
            D = [0] + random.sample(range(1, 15), size - 1)
            D.sort()
            b = max(D) + 1
            new_digits = current_digits[:] + [D]
            new_bases = current_bases[:] + [b]
            # Check no carry with previous
            new_score, _ = _try_construction(new_digits, new_bases)
            return new_digits, new_bases, new_score, new_score > current_score + 1e-9

        elif move_type == 'modify_digit' and len(current_digits) > 0:
            # Modify a random digit in a random digit set
            idx = random.randint(0, len(current_digits) - 1)
            D = current_digits[idx][:]
            if len(D) > 1:
                pos = random.randint(1, len(D) - 1)
                old_val = D[pos]
                new_val = random.randint(1, old_val + 5)
                if new_val not in D:
                    D[pos] = new_val
                    D.sort()
                    new_digits = current_digits[:]
                    new_digits[idx] = D
                    new_bases = current_bases[:]
                    new_bases[idx] = max(D) + 1
                    new_score, _ = _try_construction(new_digits, new_bases)
                    return new_digits, new_bases, new_score, new_score > current_score + 1e-9
            return current_digits, current_bases, current_score, False

        elif move_type == 'change_base' and len(current_digits) > 0:
            # Try a different base (must be > max(D_i))
            idx = random.randint(0, len(current_digits) - 1)
            min_base = max(current_digits[idx]) + 1
            new_base = min_base + random.randint(0, 3)
            new_digits = current_digits[:]
            new_bases = current_bases[:]
            new_bases[idx] = new_base
            new_score, _ = _try_construction(new_digits, new_bases)
            return new_digits, new_bases, new_score, new_score > current_score + 1e-9

        elif move_type == 'remove_digit' and len(current_digits) > 1:
            idx = random.randint(0, len(current_digits) - 1)
            new_digits = current_digits[:idx] + current_digits[idx+1:]
            new_bases = current_bases[:idx] + current_bases[idx+1:]
            new_score, _ = _try_construction(new_digits, new_bases)
            return new_digits, new_bases, new_score, new_score > current_score + 1e-9

        return current_digits, current_bases, current_score, False

    # Run local search
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline and no_improve_count < max_no_improve:
        new_digits, new_bases, new_score, improved = _local_search_step(
            current_digits, current_bases, current_score)

        if improved:
            current_digits = new_digits
            current_bases = new_bases
            current_score = new_score
            no_improve_count = 0

            if current_score > best_score:
                best_score = current_score
                best_U = _build_from_digits(current_digits, current_bases)
        else:
            no_improve_count += 1

        # Occasionally try a random restart with a fresh digit set
        if no_improve_count % 20 == 0 and time.time() < deadline - 10:
            # Try a random multi-digit construction
            n_digits = random.randint(2, 4)
            random_digits = []
            random_bases = []
            for i in range(n_digits):
                size = random.choice([3, 4, 5])
                D = [0] + random.sample(range(1, 12), size - 1)
                D.sort()
                random_digits.append(D)
                random_bases.append(max(D) + 1)
            score, U = _try_construction(random_digits, random_bases)
            if score > best_score:
                best_score = score
                best_U = U
                current_digits = random_digits
                current_bases = random_bases
                current_score = score
                no_improve_count = 0

    # Final: also try some hand-crafted multi-digit constructions
    hand_crafted = [
        ([0, 1, 3, 7], [8, 8]),
        ([0, 1, 3, 7], [8, 9]),
        ([0, 1, 4, 9], [10, 10]),
        ([0, 1, 3, 7], [8, 8, 8]),
        ([0, 1, 3, 7], [8, 9, 10]),
        ([0, 2, 5, 9], [10, 10]),
        ([0, 1, 3, 7], [8, 8, 8, 8]),
    ]

    for digits, bases in hand_crafted:
        if time.time() >= deadline - 5:
            break
        score, U = _try_construction(digits, bases)
        if score > best_score:
            best_score = score
            best_U = U

    # Ensure 0 is in the result
    if 0 not in best_U:
        best_U = [0] + best_U

    return sorted(best_U)


if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:20]}... Size: {len(s)}, Max: {max(s) if s else 0}, Score: {_score_set(s)}")
