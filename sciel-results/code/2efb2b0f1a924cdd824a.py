import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def is_valid(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def count_sq_nonzero(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    best_poly = [1] * 12
    best_score = count_sq_nonzero(best_poly)  # 23

    def try_poly(coeffs):
        nonlocal best_poly, best_score
        if not is_valid(coeffs):
            return
        s = count_sq_nonzero(coeffs)
        if s < best_score:
            best_score = s
            best_poly = list(coeffs)

    # Key insight: For a given support S = {p_1 < p_2 < ... < p_12},
    # the square has terms at positions p_i + p_j for all i,j.
    # There are at most 12*13/2 = 78 distinct positions (with symmetry).
    # The coefficient at position k = sum_{i+j=k} a_i * a_j.
    # We want to choose signs s_i in {-1,1} and magnitudes m_i in {1..64}
    # to minimize the number of nonzero square coefficients.
    #
    # For a FIXED support, the problem reduces to:
    #   Choose s_i in {-1,1} and m_i in {1,...,64} to minimize nonzero count.
    #
    # Strategy: For each support, use a backtracking solver that assigns
    # coefficients one at a time, propagating constraints.
    #
    # But C(41,12) is too large. Instead:
    # 1. Generate structured supports likely to have good cancellation
    # 2. For each, solve the sign/magnitude assignment via backtracking
    #
    # Even better: note that if all m_i = 1, the problem is purely about signs.
    # The square coefficient at k is sum_{i+j=k} s_i * s_j.
    # This is a quadratic form. We can search over sign assignments.
    #
    # For 12 terms, 2^12 = 4096 sign combinations per support.
    # So we can enumerate supports and for each try all 4096 sign patterns.
    #
    # But we also want to try magnitudes. Let's first do signs only,
    # then refine with magnitudes.

    # Phase 1: For many supports, enumerate all 2^12 sign patterns with m=1
    # and find the minimum nonzero count.

    def eval_support_signs(support, signs):
        """Given 12 positions and 12 signs (each +/-1), compute square nonzero count."""
        max_deg = max(support)
        coeffs = np.zeros(max_deg + 1, dtype=np.int64)
        for i in range(12):
            coeffs[support[i]] = signs[i]
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    # Generate candidate supports
    # Strategy: supports with structure that allows cancellation
    # - Consecutive blocks
    # - Arithmetic progressions
    # - Symmetric supports
    # - Random supports

    candidate_supports = []

    # Consecutive
    for start in range(0, 30):
        end = start + 12
        if end <= 41:
            candidate_supports.append(tuple(range(start, end)))

    # Arithmetic progressions
    for a in range(0, 30):
        for d in range(1, 5):
            positions = [a + i * d for i in range(12)]
            if max(positions) <= 40:
                candidate_supports.append(tuple(positions))

    # Symmetric around center
    center = 20
    for offset_set in [
        [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11],
        [-10,-8,-6,-4,-2,0,2,4,6,8,10,12],
        [-11,-8,-5,-2,0,1,2,5,8,11,13,15],
        [-12,-10,-8,-6,-4,-2,2,4,6,8,10,12],
        [-11,-7,-3,0,1,3,7,11,14,17,20,22],
    ]:
        positions = [center + o for o in offset_set]
        if min(positions) >= 0 and max(positions) <= 40 and len(set(positions)) == 12:
            candidate_supports.append(tuple(sorted(positions)))

    # Pairs: group into 6 pairs with same gap (for cancellation)
    for base in range(0, 20):
        for gap in range(1, 5):
            positions = []
            for i in range(6):
                pos = base + i * 3
                if pos + gap <= 40:
                    positions.append(pos)
                    positions.append(pos + gap)
            if len(positions) == 12 and len(set(positions)) == 12:
                candidate_supports.append(tuple(sorted(positions)))

    # Random supports
    random.seed(42)
    for _ in range(200):
        candidate_supports.append(tuple(sorted(random.sample(range(41), 12))))

    # Deduplicate
    candidate_supports = list(set(candidate_supports))

    # Phase 1: For each support, find best sign pattern (all m=1)
    best_sign_score = float('inf')
    best_sign_support = None
    best_sign_signs = None

    # Precompute: for a support, the convolution structure
    # For each pair (i,j) with i<=j, the position is support[i]+support[j]
    # and the contribution is s_i*s_j (or 2*s_i*s_j if i!=j)

    for support in candidate_supports:
        if time.time() > deadline - 5:
            break

        # For this support, enumerate all 2^12 sign patterns
        # But that's 4096 per support, and we have ~500 supports = 2M evaluations
        # Each evaluation is a convolution of length ~41, so ~41*41 = 1681 ops
        # Total: ~3.4B ops - too slow for pure Python

        # Optimize: use the fact that we only need to check which positions are nonzero
        # Precompute the pair positions
        pair_positions = {}
        for i in range(12):
            for j in range(i, 12):
                pos = support[i] + support[j]
                if pos not in pair_positions:
                    pair_positions[pos] = []
                pair_positions[pos].append((i, j))

        # For a given sign vector s, the coefficient at position k is:
        # sum over (i,j) in pair_positions[k] of:
        #   s_i * s_j if i == j
        #   2 * s_i * s_j if i != j
        # We need this to be 0 for as many k as possible.

        # Enumerate sign patterns using bit operations
        best_local = float('inf')
        best_local_signs = None

        # Use Gray code for efficiency
        for mask in range(4096):
            # Convert mask to signs
            # bit i of mask: 0 -> +1, 1 -> -1
            signs = [1 if ((mask >> i) & 1) == 0 else -1 for i in range(12)]

            # Compute square nonzero count
            nonzero_count = 0
            for pos, pairs in pair_positions.items():
                val = 0
                for (i, j) in pairs:
                    if i == j:
                        val += signs[i] * signs[j]
                    else:
                        val += 2 * signs[i] * signs[j]
                if val != 0:
                    nonzero_count += 1
                    # Early termination if already worse
                    if nonzero_count >= best_local:
                        break
            if nonzero_count < best_local:
                best_local = nonzero_count
                best_local_signs = signs
                if nonzero_count <= 13:  # Can't do much better
                    break

        if best_local < best_sign_score:
            best_sign_score = best_local
            best_sign_support = support
            best_sign_signs = best_local_signs
            # Build polynomial and update
            max_deg = max(support)
            coeffs = [0] * (max_deg + 1)
            for i in range(12):
                coeffs[support[i]] = best_local_signs[i]
            try_poly(coeffs)

    # Phase 2: Refine best found with magnitudes
    # For the best support/sign pattern, try varying magnitudes
    if time.time() < deadline - 10:
        # Take the best support found and try magnitude variations
        if best_sign_support is not None:
            support = best_sign_support
            base_signs = best_sign_signs

            # Try small magnitude variations
            for trial in range(2000):
                if time.time() > deadline - 5:
                    break
                mags = [1] * 12
                # Change 1-3 magnitudes
                n_changes = random.randint(1, 3)
                for _ in range(n_changes):
                    idx = random.randint(0, 11)
                    mags[idx] = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64])

                # Also try flipping 0-2 signs
                signs = list(base_signs)
                n_flips = random.randint(0, 2)
                for _ in range(n_flips):
                    idx = random.randint(0, 11)
                    signs[idx] = -signs[idx]

                max_deg = max(support)
                coeffs = [0] * (max_deg + 1)
                for i in range(12):
                    coeffs[support[i]] = signs[i] * mags[i]
                try_poly(coeffs)

    # Phase 3: Broader search with structured magnitude patterns
    if time.time() < deadline - 5:
        # Try supports with "paired" structure where cancellation is easier
        # If support has pairs (p, p+d) with same sign and magnitude,
        # then (a*x^p + a*x^(p+d))^2 = a^2*x^(2p) + 2a^2*x^(2p+d) + a^2*x^(2p+2d)
        # This doesn't cancel. But if signs differ:
        # (a*x^p - a*x^(p+d))^2 = a^2*x^(2p) - 2a^2*x^(2p+d) + a^2*x^(2p+2d)
        # Still 3 terms.

        # Better: if we have 4 terms forming a rectangle in the support:
        # positions p, p+d, p+e, p+d+e
        # (a*x^p + b*x^(p+d) + c*x^(p+e) + d*x^(p+d+e))^2
        # The cross term at p+d+e: 2*(a*d + b*c)
        # If a*d + b*c = 0, that term cancels!
        # So a/b = -c/d. With integers: a=1, b=k, c=-1, d=k gives 1*k + k*(-1) = 0.

        # Let's construct polynomials with this structure
        for base in range(0, 15):
            for d in range(1, 8):
                for e in range(d+1, d+8):
                    if base + d + e > 40:
                        continue
                    # 4-term block: base, base+d, base+e, base+d+e
                    # signs: +1, +k, -1, +k (so cross terms cancel)
                    # We need 12 terms total, so 3 such blocks
                    positions = []
                    for block_offset in [0, d+e+1, 2*(d+e+1)]:
                        if base + block_offset + d + e > 40:
                            break
                        positions.extend([
                            base + block_offset,
                            base + block_offset + d,
                            base + block_offset + e,
                            base + block_offset + d + e
                        ])
                    if len(positions) == 12 and len(set(positions)) == 12:
                        positions = sorted(positions)
                        max_deg = max(positions)
                        coeffs = [0] * (max_deg + 1)
                        for block in range(3):
                            b_off = block * (d + e + 1)
                            coeffs[base + b_off] = 1
                            coeffs[base + b_off + d] = 3
                            coeffs[base + b_off + e] = -1
                            coeffs[base + b_off + d + e] = 3
                        try_poly(coeffs)

    # Phase 4: Direct backtracking on the square coefficients
    # For a given support, the square has positions determined by the support.
    # We want to assign values a_i (in [-64,64], nonzero) such that
    # most of the square coefficients are zero.
    #
    # This is a system of quadratic equations. For a fixed support,
    # let's use a different approach: start with the sign pattern that
    # minimizes nonzeros, then try to introduce magnitudes that create
    # additional cancellations.

    if time.time() < deadline - 3:
        # Try specific algebraic constructions
        # (x^a - x^b)^2 = x^(2a) - 2x^(a+b) + x^(2b) : 3 terms
        # (x^a + x^b - x^c)^2 : up to 6 terms, some may cancel

        # Try: P = sum of 12 terms arranged as 3 groups of 4
        # Each group: (x^i + k*x^(i+d) - x^(i+e) - k*x^(i+d+e))
        # This is (x^i - x^(i+e)) + k*(x^(i+d) - x^(i+d+e))
        # = (x^i - x^(i+e))(1 + k*x^d)
        # Square: (x^i - x^(i+e))^2 * (1 + k*x^d)^2
        # = (x^(2i) - 2x^(2i+e) + x^(2i+2e)) * (1 + 2k*x^d + k^2*x^(2d))
        # This has 9 terms per group, but cross-group terms add more.

        # Simpler: try P = (1 - x^d)^2 * Q where Q has 3 terms
        # (1 - 2x^d + x^(2d)) * (a + b*x^m + c*x^n)
        # = a + b*x^m + c*x^n - 2a*x^d - 2b*x^(m+d) - 2c*x^(n+d)
        #   + a*x^(2d) + b*x^(m+2d) + c*x^(n+2d)
        # That's 9 terms. Need 12, so Q needs 4 terms.
        # (1 - 2x^d + x^(2d)) * (a + b*x^m + c*x^n + e*x^p)
        # = 12 terms (unless some positions collide)
        # Square of this will have many terms though.

        # Let's just do a final random search with the best structure
        random.seed(12345)
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            # Random support with some structure
            positions = sorted(random.sample(range(41), 12))
            signs = [random.choice([-1, 1]) for _ in range(12)]
            mags = [random.choice([1, 1, 1, 2, 3]) for _ in range(12)]

            max_deg = max(positions)
            coeffs = [0] * (max_deg + 1)
            for i in range(12):
                coeffs[positions[i]] = signs[i] * mags[i]
            try_poly(coeffs)

    # Ensure valid result
    if not is_valid(best_poly):
        best_poly = [1] * 12

    return best_poly
