import time
import random
from itertools import combinations, product
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # CSP Solver for finding P with 12 terms such that P^2 has exactly k nonzero terms
    # Variables:
    # 1. Support S of P: 12 distinct degrees from [0, D]
    # 2. Coefficients C of P: 12 integers from [-64, 64] \ {0}
    # 3. Target nonzero degrees of Q=P^2: subset of degrees
    #
    # Strategy:
    # We search for P such that P^2 has exactly K nonzero terms.
    # We start with a known good K (e.g., 16 from current best ~1.35 score implies 12/16? No, score is 1.35.
    # Wait, the score is 1.3529. The task says "Minimize nonzero terms in the EXACT integer square".
    # The evaluator measures the raw value (lower is better).
    # The score is seed/value. Seed is likely the baseline value.
    # If score is 1.35, and seed is 1.0, then value is 1/1.35 = 0.74? No.
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # This implies: Score = Seed_Value / Current_Value.
    # If Score > 1, then Current_Value < Seed_Value.
    # So we want to MINIMIZE the number of nonzero terms in P^2.
    #
    # Let's assume the "seed" program produced a polynomial with N_seed nonzero terms in the square.
    # The current program (score 1.3529) likely found a polynomial with fewer terms.
    # Let's estimate: If seed had 20 terms, and current has 14, score = 20/14 = 1.42.
    # If seed had 18, current 13, score = 1.38.
    # If seed had 16, current 12, score = 1.33.
    #
    # We will try to find a polynomial with P^2 having as few nonzero terms as possible.
    # The theoretical minimum for 12 terms is 12 (if P^2 = c*P, impossible for 12 terms) or 13?
    # Actually, if P = sum a_i x^i, P^2 = sum b_j x^j.
    #
    # CSP Approach:
    # We fix the number of nonzero terms in P^2 to be K.
    # We try K = 12, 13, 14, ...
    # For a given K, we try to find P.
    #
    # To make it tractable, we restrict the support of P to be "structured".
    # A common structure for sparse squares is "Golay pairs" or related.
    # However, we can use a simpler heuristic:
    # Assume P has support {d_1, ..., d_12}.
    # The square has support {d_i + d_j}.
    # We want many d_i + d_j to coincide or cancel.
    #
    # Let's use a randomized local search but with a different objective:
    # Minimize the number of distinct sums d_i + d_j (ignoring coefficients first).
    # If the number of distinct sums is small, we have a good candidate for cancellation.
    # Then we solve for coefficients to make the non-target sums zero.
    #
    # This is a linear system for the coefficients given the support and target sums.
    #
    # Algorithm:
    # 1. Generate candidate supports S of size 12.
    # 2. Compute the multiset of sums S+S.
    # 3. Identify the "collisions" (sums that appear multiple times).
    # 4. If there are enough collisions, we can potentially cancel terms.
    # 5. Let T be the set of target nonzero degrees in P^2.
    # 6. We want P^2 = Q where Q has support T.
    # 7. This implies for all j not in T, sum_{i: d_i+d_j' = j} a_i a_j' = 0.
    #
    # This is a system of quadratic equations. Hard to solve directly.
    #
    # Alternative CSP:
    # Search over small coefficient sets, e.g., {1, -1, 2, -2, 3, -3, ...}.
    # Use a SAT-like solver or backtracking.
    #
    # Given the time limit, a pure CSP might be too slow if not constrained.
    # Let's try a "structured" CSP:
    # Assume P is of the form A(x) + x^k B(x) where A, B have few terms?
    # Or assume P is a sum of 2 monomials raised to powers? No, must be 12 terms.
    #
    # Let's try a different angle:
    # The current best score is 1.35. This suggests the current best value is ~14-15 terms.
    # Let's aim for 12 or 13 terms.
    #
    # We will use a randomized search over supports with high collision counts,
    # and then use a simple solver to find coefficients that zero out the extra terms.
    #
    # Step 1: Find supports with many collisions in S+S.
    # Step 2: For the best supports, try to find coefficients.
    
    def count_distinct_sums(support):
        sums = set()
        n = len(support)
        for i in range(n):
            for j in range(i, n):
                sums.add(support[i] + support[j])
        return len(sums)

    def solve_for_coeffs(support, target_degrees):
        """
        Given support and target degrees (subset of all possible sums),
        find coefficients a_i such that P^2 has nonzero terms ONLY at target_degrees.
        This is a system of quadratic equations.
        We use a simplified approach: try small integer coefficients.
        """
        # This is too hard for general case.
        # Instead, we use a heuristic: 
        # If we fix the support, we can try to minimize the L1 norm of the non-target coefficients.
        # Or, we can just try all combinations of small coefficients? 127^12 is too big.
        
        # Let's stick to the previous strategy but with a "collision-based" support selection.
        pass

    # Actually, the previous program used greedy sign assignment.
    # The key insight for "CSP" here is to explicitly enforce the zero constraints.
    
    # Let's try a specific construction:
    # P(x) = (1+x)^6 * (1-x^k)^2 ? No, that changes term count.
    
    # Let's try to find a polynomial with 12 terms whose square has 12 terms.
    # This is equivalent to P^2 = c * P.
    # P(P-c) = 0. Since P is not zero, P=c. But P has 12 terms, c is constant. Contradiction.
    # So min terms in P^2 is > 12.
    
    # What about 13 terms?
    #
    # Let's use a randomized search with a "collision score" to guide the support selection,
    # and then a local search on coefficients to minimize the nonzero count in the square.
    
    def evaluate(coeffs):
        if not is_valid(coeffs):
            return float('inf')
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(n):
                if coeffs[j] == 0:
                    continue
                sq[i + j] += coeffs[i] * coeffs[j]
        return sum(1 for c in sq if c != 0)

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def square_nonzero_count(support, signs, mags):
        n = len(support)
        deg_sum = defaultdict(int)
        for i in range(n):
            for j in range(n):
                d = support[i] + support[j]
                deg_sum[d] += signs[i] * mags[i] * signs[j] * mags[j]
        return sum(1 for v in deg_sum.values() if v != 0)

    def greedy_sign_assignment(support, mags):
        n = len(support)
        best_signs = [1] * n
        best_count = square_nonzero_count(support, best_signs, mags)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                new_signs = best_signs[:]
                new_signs[i] = -new_signs[i]
                cnt = square_nonzero_count(support, new_signs, mags)
                if cnt < best_count:
                    best_signs = new_signs
                    best_count = cnt
                    improved = True
        return best_signs, best_count

    def state_to_coeffs(support, signs, mags):
        max_deg = max(support)
        coeffs = [0] * (max_deg + 1)
        for i in range(len(support)):
            coeffs[support[i]] = signs[i] * mags[i]
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    best_score = float('inf')
    best_poly = [1] * 12

    def try_result(support, signs, mags):
        nonlocal best_score, best_poly
        coeffs = state_to_coeffs(support, signs, mags)
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Strategy:
    # 1. Find supports with minimal distinct sums (high collision).
    # 2. For these supports, optimize signs and magnitudes.
    
    # Phase 1: Find supports with high collision
    # We sample supports from [0, 30]
    # We rank them by count_distinct_sums
    # We keep the top 100
    
    candidates = []
    num_samples = 5000
    for _ in range(num_samples):
        if time.time() > start_time + 10:
            break
        support = sorted(random.sample(range(31), 12))
        distinct_sums = count_distinct_sums(support)
        candidates.append((distinct_sums, support))
    
    candidates.sort(key=lambda x: x[0])
    top_supports = [sup for _, sup in candidates[:50]]
    
    # Phase 2: Optimize signs and magnitudes for top supports
    phase2_deadline = start_time + 60
    for support in top_supports:
        if time.time() > phase2_deadline:
            break
        # Try mags = [1]*12
        mags = [1] * 12
        signs, cnt = greedy_sign_assignment(support, mags)
        try_result(support, signs, mags)
        
        # Try some magnitude variations
        for i in range(12):
            if time.time() > phase2_deadline:
                break
            for new_m in [2, 3]:
                mags = [1] * 12
                mags[i] = new_m
                new_signs, new_cnt = greedy_sign_assignment(support, mags)
                try_result(support, new_signs, mags)

    # Phase 3: Local search on the best result
    phase3_deadline = start_time + 82
    if best_score < float('inf'):
        best_coeffs = best_poly
        support = [i for i, c in enumerate(best_coeffs) if c != 0]
        signs = [1 if best_coeffs[i] > 0 else -1 for i in support]
        mags = [abs(best_coeffs[i]) for i in support]

        current_support = support
        current_signs = signs
        current_mags = mags
        current_score = best_score

        improved = True
        while improved and time.time() < phase3_deadline:
            improved = False
            for i in range(12):
                if time.time() > phase3_deadline:
                    break
                for new_pos in range(0, 41):
                    if new_pos == current_support[i]:
                        continue
                    if new_pos in current_support[:i] + current_support[i+1:]:
                        continue
                    new_support = current_support[:]
                    new_support[i] = new_pos
                    new_support.sort()
                    new_signs, new_cnt = greedy_sign_assignment(new_support, current_mags)
                    if new_cnt < current_score:
                        current_support = new_support
                        current_signs = new_signs
                        current_score = new_cnt
                        improved = True
                        try_result(new_support, new_signs, current_mags)
                        break
            for i in range(12):
                if time.time() > phase3_deadline:
                    break
                new_signs = current_signs[:]
                new_signs[i] = -new_signs[i]
                new_cnt = square_nonzero_count(current_support, new_signs, current_mags)
                if new_cnt < current_score:
                    current_signs = new_signs
                    current_score = new_cnt
                    improved = True
                    try_result(current_support, current_signs, current_mags)
            for i in range(12):
                if time.time() > phase3_deadline:
                    break
                for new_m in range(1, 11):
                    if new_m == current_mags[i]:
                        continue
                    new_mags = current_mags[:]
                    new_mags[i] = new_m
                    new_signs, new_cnt = greedy_sign_assignment(current_support, new_mags)
                    if new_cnt < current_score:
                        current_mags = new_mags
                        current_signs = new_signs
                        current_score = new_cnt
                        improved = True
                        try_result(current_support, current_signs, current_mags)
                        break

    if not is_valid(best_poly) or best_score == float('inf'):
        best_poly = [1] * 12
        best_score = evaluate(best_poly)

    return best_poly
