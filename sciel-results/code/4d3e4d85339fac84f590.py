"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _count_nonzero_square_fast(coeffs):
    """Count nonzero terms in the exact square using NumPy convolution."""
    n = len(coeffs)
    if n == 0:
        return 0
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def _validate(coeffs):
    """Check all constraints on the coefficient list."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    return True

def _sparse_to_dense(sparse, max_deg=40):
    """Convert sparse list of (exponent, coeff) to dense coefficient list."""
    coeffs = [0] * (max_deg + 1)
    for exp, c in sparse:
        if 0 <= exp <= max_deg:
            coeffs[exp] = c
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _dense_to_sparse(coeffs):
    """Convert dense coefficient list to sparse representation."""
    return [(i, c) for i, c in enumerate(coeffs) if c != 0]

def _square_nonzero_count(sparse):
    """Count nonzero terms in square from sparse representation directly."""
    dense = _sparse_to_dense(sparse)
    return _count_nonzero_square_fast(dense)

def _pairwise_cancellation_score(sparse):
    """
    Estimate how many cross-terms cancel.
    For each pair (i,j) with i<j, the cross term at degree e_i+e_j is 2*c_i*c_j.
    If two pairs have the same sum degree, their cross terms add up.
    We want many pairs to share sum degrees so cancellations can happen.
    """
    from collections import defaultdict
    pair_sums = defaultdict(int)
    n = len(sparse)
    for i in range(n):
        e_i, c_i = sparse[i]
        for j in range(i + 1, n):
            e_j, c_j = sparse[j]
            pair_sums[e_i + e_j] += 2 * c_i * c_j
    # Count how many pair-sum degrees have zero total
    cancellations = sum(1 for v in pair_sums.values() if v == 0)
    return cancellations

def _hill_climb(sparse_init, deadline, coeff_choices=None):
    """
    Focused hill-climbing with pairwise-cancellation-aware moves.
    """
    if coeff_choices is None:
        coeff_choices = [-2, -1, 1, 2]
    
    current_sparse = list(sparse_init)
    current_sq = _square_nonzero_count(current_sparse)
    
    best_sparse = list(current_sparse)
    best_sq = current_sq
    
    random.seed(42)
    max_exp = 40
    stall_count = 0
    max_stall = 5000
    
    while time.time() < deadline - 3.0 and stall_count < max_stall:
        move_type = random.choices(
            ['flip_sign', 'shift_exp', 'change_coeff', 'shift_pair', 'coeff_pair', 'swap_exps'],
            weights=[25, 25, 15, 15, 10, 10]
        )[0]
        
        trial_sparse = list(current_sparse)
        valid = True
        
        if move_type == 'flip_sign':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)
            
        elif move_type == 'shift_exp':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = e + shift
            if 0 <= new_e <= max_exp:
                existing_exps = set(x[0] for x in trial_sparse if x[0] != e)
                if new_e not in existing_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'change_coeff':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c != c:
                trial_sparse[idx] = (e, new_c)
            else:
                valid = False
                
        elif move_type == 'shift_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                shift_i = random.choice([-2, -1, 1, 2])
                shift_j = random.choice([-2, -1, 1, 2])
                new_ei = ei + shift_i
                new_ej = ej + shift_j
                if 0 <= new_ei <= max_exp and 0 <= new_ej <= max_exp:
                    other_exps = set(x[0] for k, x in enumerate(trial_sparse) if k not in (i, j))
                    if new_ei not in other_exps and new_ej not in other_exps and new_ei != new_ej:
                        trial_sparse[i] = (new_ei, ci)
                        trial_sparse[j] = (new_ej, cj)
                    else:
                        valid = False
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'coeff_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci != ci or new_cj != cj:
                    trial_sparse[i] = (ei, new_ci)
                    trial_sparse[j] = (ej, new_cj)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'swap_exps':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                # Swap exponents
                trial_sparse[i] = (ej, ci)
                trial_sparse[j] = (ei, cj)
        
        if not valid:
            continue
        
        if len(trial_sparse) != 12:
            continue
        exps = [x[0] for x in trial_sparse]
        if len(exps) != len(set(exps)):
            continue
        
        trial_dense = _sparse_to_dense(trial_sparse)
        if not _validate(trial_dense):
            continue
        
        trial_sq = _square_nonzero_count(trial_sparse)
        
        if trial_sq < current_sq:
            current_sparse = trial_sparse
            current_sq = trial_sq
            stall_count = 0
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_sparse = list(trial_sparse)
        else:
            stall_count += 1
            # Accept worse moves with small probability for exploration
            if random.random() < 0.03:
                current_sparse = trial_sparse
                current_sq = trial_sq
    
    return best_sparse, best_sq

def _structured_search(deadline):
    """
    Structured search using factorization: P(x) = A(x)B(x)C(x) where each factor
    has 2-3 terms. Exploit (ABC)^2 = A^2 B^2 C^2 to find structures with maximal
    cross-term cancellation in the final square.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(99)
    
    def _factor_product(factors):
        """Multiply sparse factors together, returning sparse result."""
        result = [(0, 1)]  # start with constant 1
        for factor in factors:
            new_result = []
            for (e1, c1) in result:
                for (e2, c2) in factor:
                    new_result.append((e1 + e2, c1 * c2))
            # Combine like terms
            from collections import defaultdict
            combined = defaultdict(int)
            for e, c in new_result:
                combined[e] += c
            result = [(e, c) for e, c in combined.items() if c != 0]
        return result
    
    def _factor_square_nonzero(factor):
        """Count nonzero terms in the square of a sparse factor."""
        dense = _sparse_to_dense(factor)
        return _count_nonzero_square_fast(dense)
    
    # Strategy 1: Product of 3 binomials
    # P(x) = (x^a + c1*x^b)(x^c + c2*x^d)(x^e + c3*x^f)
    # = 8 terms if all exponents distinct
    # We need exactly 12 nonzero terms, so we need some structure.
    # Try: (x^a + x^b)(x^c + x^d)(x^e + x^f) with careful exponent choices
    # The product has 8 terms. To get 12, we need a different structure.
    
    # Strategy 2: Product of a trinomial and a 4-term polynomial
    # P(x) = (x^a + c1*x^b + c2*x^c) * (x^d + c3*x^e + c4*x^f + c5*x^g)
    # = 12 terms if all products give distinct exponents
    
    # Let's systematically search for factor structures
    # Factor type A: 2 terms (binomial)
    # Factor type B: 3 terms (trinomial)
    # Factor type C: 4 terms
    
    # Search space: choose exponents and coefficients for each factor
    
    # For speed, we'll focus on small coefficient sets and structured exponent patterns
    
    coeff_choices = [-2, -1, 1, 2]
    
    # Pattern 1: Three binomials with geometric-like exponent spacing
    # (x^0 + c1*x^p)(x^q + c2*x^r)(x^s + c3*x^t)
    # The 8 products give exponents: 0, p, q, p+q, s, p+s, q+s, p+q+s
    # For 12 terms, we need a different approach
    
    # Pattern 2: One trinomial times one 4-term polynomial
    # A(x) = x^a + c1*x^b + c2*x^c  (3 terms)
    # B(x) = x^d + c3*x^e + c4*x^f + c5*x^g  (4 terms)
    # Product has up to 12 terms
    
    # Systematic search over exponent patterns
    # A: exponents {0, p, q} with coeffs {1, c1, c2}
    # B: exponents {d, d+r, d+s, d+t} with coeffs {1, c3, c4, c5}
    
    for p in range(1, 8):
        for q in range(p + 1, 12):
            for c1 in coeff_choices:
                for c2 in coeff_choices:
                    for d in range(0, 10):
                        for r in range(1, 6):
                            for s in range(r + 1, 10):
                                for t in range(s + 1, 15):
                                    if d + t > 40:
                                        continue
                                    for c3 in coeff_choices:
                                        for c4 in coeff_choices:
                                            for c5 in coeff_choices:
                                                A = [(0, 1), (p, c1), (q, c2)]
                                                B = [(d, 1), (d + r, c3), (d + s, c4), (d + t, c5)]
                                                product = _factor_product([A, B])
                                                if len(product) != 12:
                                                    continue
                                                exps = [x[0] for x in product]
                                                if len(exps) != len(set(exps)):
                                                    continue
                                                dense = _sparse_to_dense(product)
                                                if not _validate(dense):
                                                    continue
                                                sq = _square_nonzero_count(product)
                                                if sq < best_sq:
                                                    best_sq = sq
                                                    best_sparse = list(product)
                                                if time.time() > deadline - 5:
                                                    break
                                    if time.time() > deadline - 5:
                                        break
                                if time.time() > deadline - 5:
                                    break
                            if time.time() > deadline - 5:
                                break
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    # Pattern 3: Two trinomials
    # A(x) = x^a + c1*x^b + c2*x^c
    # B(x) = x^d + c3*x^e + c4*x^f
    # Product has up to 9 terms, not 12. Need a different structure.
    
    # Pattern 4: One binomial times one 6-term polynomial
    # A(x) = x^a + c1*x^b (2 terms)
    # B(x) = 6 terms
    # Product has up to 12 terms
    
    for a in range(0, 5):
        for b in range(a + 1, a + 10):
            for c1 in coeff_choices:
                A = [(a, 1), (b, c1)]
                # B has 6 terms: exponents {0, p1, p2, p3, p4, p5}
                for p1 in range(1, 4):
                    for p2 in range(p1 + 1, 7):
                        for p3 in range(p2 + 1, 10):
                            for p4 in range(p3 + 1, 13):
                                for p5 in range(p4 + 1, 16):
                                    if b + p5 > 40:
                                        continue
                                    for c2 in coeff_choices:
                                        for c3 in coeff_choices:
                                            for c4 in coeff_choices:
                                                for c5 in coeff_choices:
                                                    for c6 in coeff_choices:
                                                        B = [(0, 1), (p1, c2), (p2, c3), (p3, c4), (p4, c5), (p5, c6)]
                                                        product = _factor_product([A, B])
                                                        if len(product) != 12:
                                                            continue
                                                        exps = [x[0] for x in product]
                                                        if len(exps) != len(set(exps)):
                                                            continue
                                                        dense = _sparse_to_dense(product)
                                                        if not _validate(dense):
                                                            continue
                                                        sq = _square_nonzero_count(product)
                                                        if sq < best_sq:
                                                            best_sq = sq
                                                            best_sparse = list(product)
                                                        if time.time() > deadline - 5:
                                                            break
                                                    if time.time() > deadline - 5:
                                                        break
                                                if time.time() > deadline - 5:
                                                    break
                                            if time.time() > deadline - 5:
                                                break
                                        if time.time() > deadline - 5:
                                            break
                                    if time.time() > deadline - 5:
                                        break
                                if time.time() > deadline - 5:
                                    break
                            if time.time() > deadline - 5:
                                break
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    # Pattern 5: Symmetric pair construction (keep from before)
    for S in range(20, 40):
        sparse = []
        valid = True
        for k in range(6):
            e1 = S - 2 * k - 1
            e2 = S - 2 * k
            if e1 < 0 or e2 > 40:
                valid = False
                break
            sparse.append((e1, 1))
            sparse.append((e2, -1))
        if not valid:
            continue
        if len(sparse) != 12:
            continue
        exps = [x[0] for x in sparse]
        if len(exps) != len(set(exps)):
            continue
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            continue
        sq = _square_nonzero_count(sparse)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sparse)
    
    # Pattern 6: Random search with factor-inspired initialization
    for _ in range(300):
        # Generate random factor structure
        num_factors = random.choice([2, 3])
        if num_factors == 2:
            # Split 12 into two factors: 3+4 or 2+6 or 4+3
            split = random.choice([(3, 4), (2, 6), (4, 3)])
            factors = []
            for size in split:
                exps = sorted(random.sample(range(0, 20), size))
                coeffs = [random.choice(coeff_choices) for _ in range(size)]
                factors.append(list(zip(exps, coeffs)))
        else:
            # 2+2+4 or 2+3+3 or 3+3+3
            split = random.choice([(2, 2, 4), (2, 3, 3), (3, 3, 3)])
            factors = []
            for size in split:
                exps = sorted(random.sample(range(0, 15), size))
                coeffs = [random.choice(coeff_choices) for _ in range(size)]
                factors.append(list(zip(exps, coeffs)))
        
        product = _factor_product(factors)
        if len(product) != 12:
            continue
        exps = [x[0] for x in product]
        if len(exps) != len(set(exps)):
            continue
        dense = _sparse_to_dense(product)
        if not _validate(dense):
            continue
        sq = _square_nonzero_count(product)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(product)
    
    if best_sparse is not None:
        return best_sparse, best_sq
    return None, float('inf')

def _multi_start_hill_climb(deadline):
    """
    Run hill-climbing from multiple diverse starting points.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(777)
    
    # Generate diverse starting points
    start_points = []
    
    # Start 1: All ones
    start_points.append([(i, 1) for i in range(12)])
    
    # Start 2: Alternating signs, spread out
    start_points.append([(i * 3, 1 if i % 2 == 0 else -1) for i in range(12) if i * 3 < 41])
    while len(start_points[-1]) < 12:
        start_points[-1].append((len(start_points[-1]) * 3, 1))
    start_points[-1] = start_points[-1][:12]
    
    # Start 3: Random
    for _ in range(10):
        exps = sorted(random.sample(range(0, 41), 12))
        sparse = [(e, random.choice([-2, -1, 1, 2])) for e in exps]
        start_points.append(sparse)
    
    # Start 4: Symmetric pairs
    for S in [20, 25, 30]:
        sparse = []
        for k in range(6):
            e1 = S - 2 * k - 1
            e2 = S - 2 * k
            if 0 <= e1 <= 40 and 0 <= e2 <= 40 and e1 != e2:
                sparse.append((e1, 1))
                sparse.append((e2, -1))
        if len(sparse) == 12:
            exps = [x[0] for x in sparse]
            if len(exps) == len(set(exps)):
                start_points.append(sparse)
    
    # Run hill-climb from each start point with allocated time
    num_starts = len(start_points)
    time_per_start = max(2.0, (deadline - time.time() - 5.0) / num_starts)
    
    for i, sp in enumerate(start_points):
        if time.time() >= deadline - 5.0:
            break
        if time_per_start < 1.0:
            time_per_start = 1.0
        
        # Validate start point
        dense = _sparse_to_dense(sp)
        if not _validate(dense):
            continue
        
        sq = _square_nonzero_count(sp)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sp)
        
        # Run hill-climb
        try:
            found_sparse, found_sq = _hill_climb(sp, time.time() + time_per_start)
            if found_sq < best_sq:
                best_sq = found_sq
                best_sparse = list(found_sparse)
        except Exception:
            pass
    
    if best_sparse is not None:
        return best_sparse, best_sq
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial candidate
    initial_sparse = [(i * 3, 1 if i % 2 == 0 else -1) for i in range(12)]
    initial_sparse = [(e, c) for e, c in initial_sparse if e < 41]
    while len(initial_sparse) < 12:
        initial_sparse.append((len(initial_sparse) * 3, 1))
    initial_sparse = initial_sparse[:12]
    
    initial_dense = _sparse_to_dense(initial_sparse)
    if not _validate(initial_dense):
        initial_dense = [1] * 12
        initial_sparse = _dense_to_sparse(initial_dense)
    
    best_sparse = list(initial_sparse)
    best_sq = _square_nonzero_count(best_sparse)
    
    # Phase 1: Structured search
    if time.time() < deadline - 40.0:
        try:
            struct_sparse, struct_sq = _structured_search(deadline - 35.0)
            if struct_sparse is not None and struct_sq < best_sq:
                best_sq = struct_sq
                best_sparse = list(struct_sparse)
        except Exception:
            pass
    
    # Phase 2: Multi-start hill climbing
    if time.time() < deadline - 10.0:
        try:
            found_sparse, found_sq = _multi_start_hill_climb(deadline)
            if found_sparse is not None and found_sq < best_sq:
                best_sq = found_sq
                best_sparse = list(found_sparse)
        except Exception:
            pass
    
    # Final: One more hill-climb from best
    if time.time() < deadline - 5.0:
        try:
            final_sparse, final_sq = _hill_climb(best_sparse, deadline)
            if final_sq < best_sq:
                best_sq = final_sq
                best_sparse = list(final_sparse)
        except Exception:
            pass
    
    # Convert to dense and validate
    best_coeffs = _sparse_to_dense(best_sparse)
    if not _validate(best_coeffs):
        best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
