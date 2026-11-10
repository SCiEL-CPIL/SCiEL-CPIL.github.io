import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def update_best(c, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = c.copy()
            return best_score <= 12

    # BMM over GF(2) to find collision-free supports
    # For a polynomial with support S (set of degrees), the square has support
    # S+S = {i+j : i,j in S}. We want |S+S| to be small.
    # Over GF(2), the number of pairs (i,j) with i+j=k is the coefficient
    # in the GF(2) convolution. If this is even for all k not in a small set,
    # we can potentially cancel those terms.

    def gf2_convolution_support(s1, s2):
        """Return dict: degree -> count mod 2 of pairs (i in s1, j in s2) with i+j=degree"""
        result = {}
        for i in s1:
            for j in s2:
                k = i + j
                result[k] = result.get(k, 0) ^ 1  # XOR = mod 2
        return result

    def find_sparse_supports(max_degree=11, num_terms=12, max_support_size=20):
        """Find supports S with |S|=num_terms and |S+S| <= max_support_size"""
        all_supports = []
        # Try various base structures
        # 1. Arithmetic progressions
        for start in range(0, 12):
            for step in range(1, 12):
                support = sorted(set((start + i * step) % 12 for i in range(num_terms)))
                if len(support) == num_terms:
                    sum_set = set()
                    for i in support:
                        for j in support:
                            sum_set.add(i + j)
                    if len(sum_set) <= max_support_size:
                        all_supports.append(support)
        
        # 2. Union of two arithmetic progressions
        for start1 in range(0, 6):
            for step1 in range(1, 4):
                for start2 in range(6, 12):
                    for step2 in range(1, 4):
                        s1 = set((start1 + i * step1) % 12 for i in range(6))
                        s2 = set((start2 + i * step2) % 12 for i in range(6))
                        support = sorted(s1 | s2)
                        if len(support) == num_terms:
                            sum_set = set()
                            for i in support:
                                for j in support:
                                    sum_set.add(i + j)
                            if len(sum_set) <= max_support_size:
                                all_supports.append(support)
        
        # 3. Random supports with pruning
        for _ in range(200):
            support = sorted(random.sample(range(12), num_terms))
            sum_set = set()
            for i in support:
                for j in support:
                    sum_set.add(i + j)
            if len(sum_set) <= max_support_size:
                all_supports.append(support)
        
        # Deduplicate
        unique = list(set(tuple(s) for s in all_supports))
        return [list(s) for s in unique]

    # Solve for sign assignments using GF(2) linear algebra
    # For a given support S, the square coefficient at degree k is:
    #   sum_{i+j=k, i<=j} c_i*c_j (with factor 2 if i<j)
    # We want to find c_i in {±1} such that as many of these sums are zero.
    # This is equivalent to: for each k, sum_{i+j=k} c_i*c_j = 0
    # Let x_i = log(c_i) in {0,1} (where 0 means +1, 1 means -1)
    # Then c_i*c_j = (-1)^{x_i+x_j}
    # So sum_{i+j=k} (-1)^{x_i+x_j} = 0

    def solve_signs_gf2(support, target_k_list):
        """Try to find signs for support that zero out coefficients at target_k_list"""
        n = len(support)
        # Build constraint matrix over GF(2)
        # For each target degree k, we need sum_{i+j=k} (-1)^{x_i+x_j} = 0
        # This means an even number of the pairs must have odd sum, even number even sum
        # Actually, let's think differently:
        # sum_{i+j=k} c_i*c_j = 0
        # If there are m pairs, we need sum of m ±1 values = 0
        # This requires m to be even and exactly m/2 to be +1 and m/2 to be -1
        
        # Let's use a direct approach: enumerate sign patterns for small supports
        if n <= 12:
            best_signs = None
            best_count = float('inf')
            
            # Try all 2^12 sign combinations (4096)
            for mask in range(1 << n):
                signs = [(-1) ** ((mask >> i) & 1) for i in range(n)]
                # Build polynomial
                coeff = np.zeros(2 * n + 1, dtype=np.int64)
                for i, s in enumerate(support):
                    coeff[s] = signs[i]
                sq = poly_square(coeff)
                # Count nonzero at target degrees
                nonzero_count = sum(1 for k in target_k_list if sq[k] != 0)
                if nonzero_count < best_count:
                    best_count = nonzero_count
                    best_signs = signs
                    if best_count == 0:
                        break
            
            if best_signs is not None:
                coeff = np.zeros(24, dtype=np.int64)
                for i, s in enumerate(support):
                    coeff[s] = best_signs[i]
                return coeff, best_count
        
        return None, float('inf')

    # Generate diverse initial candidates
    candidates = []

    # Known good patterns
    candidates.append([1]*12)
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    candidates.append([1]*6 + [-1]*6)
    candidates.append([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    candidates.append([1,1,1,1,1,-1,1,1,1,1,1,-1])

    # BMM-based support search
    supports = find_sparse_supports()
    
    for support in supports[:50]:  # Limit to avoid timeout
        if time.time() > deadline - 30:
            break
        # For each support, try to find good signs
        # Target: zero out as many square coefficients as possible
        # The square has degrees 0 to 22 (for 12-term poly)
        target_k = list(range(23))
        
        coeff, nonzero = solve_signs_gf2(support, target_k)
        if coeff is not None:
            # Trim to length 12
            trimmed = coeff[:12]
            if count_nonzero(trimmed) == 12 and trimmed[-1] != 0:
                candidates.append(trimmed.tolist())
                if nonzero < 15:  # Very sparse square
                    candidates.append(trimmed.tolist())  # Duplicate for emphasis

    # Factor-based: P = A * B where A, B are sparse
    for a_pos in range(0, 12, 3):
        for b_base in range(0, 4):
            A = np.zeros(12, dtype=np.int64)
            A[0] = 1
            A[a_pos] = random.choice([-1, 1])
            B = np.zeros(12, dtype=np.int64)
            for j in range(6):
                B[b_base + j] = random.choice([-1, 1])
            P = np.convolve(A, B)[:12]
            if count_nonzero(P) == 12 and P[-1] != 0:
                candidates.append(P.tolist())

    # More factor-based
    for _ in range(50):
        i = random.randint(1, 11)
        a = random.choice([-2, -1, 1, 2])
        B = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        A = np.zeros(12, dtype=np.int64)
        A[0] = 1
        A[i] = a
        B_arr = np.array(B, dtype=np.int64)
        P = np.convolve(A, B_arr)[:12]
        if count_nonzero(P) == 12 and P[-1] != 0 and np.all((P >= -64) & (P <= 64)):
            candidates.append(P.tolist())

    # Structured patterns
    random.seed(42)
    for _ in range(100):
        c = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)

    # Symmetric and antisymmetric
    for _ in range(50):
        base = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        c = base + base[::-1]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)
    for _ in range(50):
        base = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        c = base + [-x for x in base[::-1]]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)

    # Evaluate all candidates
    for init_c in candidates:
        if time.time() > deadline - 20:
            break
        curr_c = np.array(init_c, dtype=np.int64)
        if not validate(curr_c):
            continue
        curr_score = evaluate(curr_c)
        if update_best(curr_c, curr_score):
            break

    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
        best_score = evaluate(best_poly)

    # Constraint-guided local search
    def targeted_local_search(curr_c, curr_score, max_iters=2000):
        nonlocal best_score, best_poly
        no_improvement = 0
        temp = 3.0

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            sq = poly_square(curr_c)
            nz_sq = np.where(sq != 0)[0]

            if len(nz_sq) == 0:
                break

            if random.random() < 0.5:
                idx = random.randint(0, 11)
            else:
                target_deg = random.choice(nz_sq)
                if target_deg <= 11:
                    affecting = list(range(0, target_deg + 1))
                else:
                    affecting = list(range(target_deg - 11, 12))

                if not affecting:
                    idx = random.randint(0, 11)
                else:
                    idx = random.choice(affecting)

            new_val = random.randint(-4, 4)
            if new_val == 0:
                new_val = random.choice([-1, 1])

            if curr_c[idx] == new_val:
                continue

            old_val = curr_c[idx]
            curr_c[idx] = new_val

            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                if update_best(curr_c, curr_score):
                    return True
            else:
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temp, 0.5)):
                    curr_score = new_score
                    no_improvement = 0
                    if curr_score < best_score:
                        update_best(curr_c, curr_score)
                else:
                    curr_c[idx] = old_val
                    no_improvement += 1

            temp *= 0.999
            if no_improvement > 500:
                break

        return best_score <= 12

    # Run targeted local search
    starting_points = [best_poly.copy()]

    for _ in range(10):
        c = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        if c[-1] == 0:
            c[-1] = 1
        c_arr = np.array(c, dtype=np.int64)
        if validate(c_arr):
            starting_points.append(c_arr)

    for sp in starting_points:
        if time.time() > deadline - 10:
            break
        if targeted_local_search(sp.copy(), evaluate(sp), max_iters=1500):
            break

    # Multi-point mutation search
    if time.time() < deadline - 10:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)
        no_improvement = 0

        while time.time() < deadline - 5 and no_improvement < 3000:
            num_mods = random.choice([1, 1, 2, 2, 3])
            indices = random.sample(range(12), num_mods)
            old_vals = [curr_c[i] for i in indices]

            for i in indices:
                val = random.randint(-5, 5)
                if val == 0:
                    val = random.choice([-1, 1])
                curr_c[i] = val

            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                if update_best(curr_c, curr_score):
                    break
            else:
                for i, val in zip(indices, old_vals):
                    curr_c[i] = val
                no_improvement += 1

    # Final refinement
    if time.time() < deadline - 3 and best_poly is not None:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)

        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, 11)
            delta_val = random.choice([-2, -1, 1, 2])
            new_val = curr_c[idx] + delta_val
            if new_val == 0:
                new_val = 1
            if not (-64 <= new_val <= 64):
                continue

            old_val = curr_c[idx]
            curr_c[idx] = new_val
            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                if update_best(curr_c, curr_score):
                    break
            else:
                curr_c[idx] = old_val

    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
