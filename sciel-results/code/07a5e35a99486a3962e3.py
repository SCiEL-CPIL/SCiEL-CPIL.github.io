"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product
from collections import defaultdict

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

def _factor_product_square_count(factors):
    """
    Given a list of sparse factors (each a list of (exp, coeff)),
    compute the number of nonzero terms in the square of their product.
    P = prod(factors), P^2 = prod(factors^2)
    We compute each factor's square, then convolve them all.
    """
    # Compute square of each factor
    factor_squares = []
    for f in factors:
        dense = _sparse_to_dense(f)
        sq = np.convolve(dense, dense)
        # Convert back to sparse
        sq_sparse = [(i, int(c)) for i, c in enumerate(sq) if c != 0]
        factor_squares.append(sq_sparse)
    
    # Convolve all factor squares together
    # Start with first factor square
    result_dense = np.array([_sparse_to_dense([factor_squares[0]]) if factor_squares[0] else [0]], dtype=np.int64)
    if len(factor_squares) == 0:
        return 0
    if len(factor_squares) == 1:
        return len(factor_squares[0])
    
    result_dense = np.array(_sparse_to_dense(factor_squares[0]), dtype=np.int64)
    for fs in factor_squares[1:]:
        fs_dense = np.array(_sparse_to_dense(fs), dtype=np.int64)
        result_dense = np.convolve(result_dense, fs_dense)
    
    return int(np.count_nonzero(result_dense))

def _factor_product_sparse(factors):
    """Compute the sparse representation of the product of sparse factors."""
    result = [(0, 1)]  # start with 1
    for f in factors:
        new_result = defaultdict(int)
        for (e1, c1) in result:
            for (e2, c2) in f:
                new_result[e1 + e2] += c1 * c2
        result = [(e, c) for e, c in new_result.items() if c != 0]
    return result

def _structured_search(deadline):
    """
    Structured search using factorization: P = A * B * C where each factor
    has 2-3 terms. Exploit P^2 = A^2 * B^2 * C^2 for sparse squares.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(99)
    
    # Construction 1: Product of three 2-term factors (2*2*2=8 terms, need 12)
    # Skip - gives only 8 terms
    
    # Construction 2: 2-term * 2-term * 3-term = 12 terms
    # A = x^a + c1*x^b, B = x^c + c2*x^d, C = x^e + c3*x^f + c4*x^g
    # P^2 = A^2 * B^2 * C^2
    # A^2 has 3 terms, B^2 has 3 terms, C^2 has up to 5 terms
    # Total products: 3*3*5 = 45 max, but many can collide
    
    # Try to find exponent choices where many products collide
    # Key: if A^2 exponents are {2a, a+b, 2b}, B^2 are {2c, c+d, 2d}, C^2 are {2e, e+f, e+g, f+g, 2f, 2g}
    # We want many sums to be equal.
    
    # Strategy: Make A^2, B^2, C^2 have overlapping structure
    # If all factors are of form x^p + x^q (binary), then squares are x^{2p} + 2x^{p+q} + x^{2q}
    # For C with 3 terms: x^e + x^f + x^g, square has 5 terms
    
    # Let's try specific constructions where exponents are arithmetic progressions
    # to maximize collisions in the product of squares
    
    # Construction 2a: A={0,1}, B={2,3}, C={4,5,6} with all coeff=1
    # A^2 = {0:1, 1:2, 2:1}, B^2 = {4:1, 5:2, 6:1}, C^2 = {8:1, 9:2, 10:3, 11:2, 12:1}
    # P^2 = A^2 * B^2 * C^2: convolution of these
    # This is a specific case, let's compute it
    
    # More systematic: try various offset patterns
    for a_base in range(0, 5):
        for a_span in [1, 2, 3]:
            for b_base in range(a_base + a_span, 15):
                for b_span in [1, 2, 3]:
                    for c_base in range(b_base + b_span, 25):
                        for c_span in [1, 2, 3]:
                            # A: 2 terms at a_base, a_base+a_span
                            # B: 2 terms at b_base, b_base+b_span  
                            # C: 3 terms at c_base, c_base+c_span, c_base+2*c_span
                            A = [(a_base, 1), (a_base + a_span, 1)]
                            B = [(b_base, 1), (b_base + b_span, 1)]
                            C = [(c_base, 1), (c_base + c_span, 1), (c_base + 2*c_span, 1)]
                            
                            P_sparse = _factor_product_sparse([A, B, C])
                            if len(P_sparse) != 12:
                                continue
                            exps = [x[0] for x in P_sparse]
                            if len(exps) != len(set(exps)):
                                continue
                            if max(exps) > 40:
                                continue
                            
                            dense = _sparse_to_dense(P_sparse)
                            if not _validate(dense):
                                continue
                            
                            sq_count = _factor_product_square_count([A, B, C])
                            if sq_count < best_sq:
                                best_sq = sq_count
                                best_sparse = list(P_sparse)
    
    # Construction 3: With signs to create cancellations
    # A = x^a - x^b, B = x^c - x^d, C = x^e + x^f + x^g
    # A^2 = x^{2a} - 2x^{a+b} + x^{2b}
    # B^2 = x^{2c} - 2x^{c+d} + x^{2d}
    # C^2 = x^{2e} + 2x^{e+f} + 2x^{e+g} + x^{2f} + 2x^{f+g} + x^{2g}
    
    for a_base in range(0, 5):
        for a_span in [1, 2, 3]:
            for b_base in range(a_base + a_span, 15):
                for b_span in [1, 2, 3]:
                    for c_base in range(b_base + b_span, 25):
                        for c_span in [1, 2, 3]:
                            for sign_a in [1, -1]:
                                for sign_b in [1, -1]:
                                    A = [(a_base, 1), (a_base + a_span, sign_a)]
                                    B = [(b_base, 1), (b_base + b_span, sign_b)]
                                    C = [(c_base, 1), (c_base + c_span, 1), (c_base + 2*c_span, 1)]
                                    
                                    P_sparse = _factor_product_sparse([A, B, C])
                                    if len(P_sparse) != 12:
                                        continue
                                    exps = [x[0] for x in P_sparse]
                                    if len(exps) != len(set(exps)):
                                        continue
                                    if max(exps) > 40:
                                        continue
                                    
                                    dense = _sparse_to_dense(P_sparse)
                                    if not _validate(dense):
                                        continue
                                    
                                    sq_count = _factor_product_square_count([A, B, C])
                                    if sq_count < best_sq:
                                        best_sq = sq_count
                                        best_sparse = list(P_sparse)
    
    # Construction 4: 3-term * 3-term * 2-term = 18 terms (too many, skip)
    # Construction 5: 4-term * 3-term * 1-term = 12 terms
    # A has 4 terms, B has 3 terms, C has 1 term (just a constant shift)
    # This reduces to 4*3=12 terms
    
    # Let's also try: 2-term * 3-term * 2-term where the 2-term factors
    # have coefficients that create more cancellation in their squares
    
    # Construction 6: Random factor-based search
    for _ in range(2000):
        # Random 2-term factor A
        ea1 = random.randint(0, 8)
        ea2 = random.randint(ea1+1, ea1+5)
        ca1 = random.choice([-2, -1, 1, 2])
        ca2 = random.choice([-2, -1, 1, 2])
        A = [(ea1, ca1), (ea2, ca2)]
        
        # Random 2-term factor B
        eb1 = random.randint(ea2+1, ea2+8)
        eb2 = random.randint(eb1+1, eb1+5)
        cb1 = random.choice([-2, -1, 1, 2])
        cb2 = random.choice([-2, -1, 1, 2])
        B = [(eb1, cb1), (eb2, cb2)]
        
        # Random 3-term factor C
        ec1 = random.randint(eb2+1, eb2+8)
        ec2 = random.randint(ec1+1, ec1+4)
        ec3 = random.randint(ec2+1, ec2+4)
        cc1 = random.choice([-2, -1, 1, 2])
        cc2 = random.choice([-2, -1, 1, 2])
        cc3 = random.choice([-2, -1, 1, 2])
        C = [(ec1, cc1), (ec2, cc2), (ec3, cc3)]
        
        P_sparse = _factor_product_sparse([A, B, C])
        if len(P_sparse) != 12:
            continue
        exps = [x[0] for x in P_sparse]
        if len(exps) != len(set(exps)):
            continue
        if max(exps) > 40:
            continue
        
        dense = _sparse_to_dense(P_sparse)
        if not _validate(dense):
            continue
        
        sq_count = _factor_product_square_count([A, B, C])
        if sq_count < best_sq:
            best_sq = sq_count
            best_sparse = list(P_sparse)
    
    # Construction 7: Original symmetric pairs (keep for comparison)
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
    
    if best_sparse is not None:
        return best_sparse, best_sq
    return None, float('inf')

def _multi_start_hill_climb(deadline):
    """
    Run hill-climbing from multiple diverse starting points,
    including factor-based constructions.
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
    
    # Start 5: Factor-based constructions (2*2*3 = 12 terms)
    for _ in range(8):
        ea1 = random.randint(0, 5)
        ea2 = ea1 + random.randint(1, 3)
        eb1 = ea2 + random.randint(1, 5)
        eb2 = eb1 + random.randint(1, 3)
        ec1 = eb2 + random.randint(1, 5)
        ec2 = ec1 + random.randint(1, 3)
        ec3 = ec2 + random.randint(1, 3)
        
        if ec3 > 40:
            continue
        
        ca = random.choice([-1, 1])
        cb = random.choice([-1, 1])
        cc = random.choice([-1, 1])
        cd = random.choice([-1, 1])
        ce = random.choice([-1, 1])
        cf = random.choice([-1, 1])
        cg = random.choice([-1, 1])
        
        A = [(ea1, 1), (ea2, ca)]
        B = [(eb1, 1), (eb2, cb)]
        C = [(ec1, 1), (ec2, cc), (ec3, cd)]
        
        P_sparse = _factor_product_sparse([A, B, C])
        if len(P_sparse) == 12:
            exps = [x[0] for x in P_sparse]
            if len(exps) == len(set(exps)) and max(exps) <= 40:
                start_points.append(P_sparse)
    
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

def _search_structured_Q(deadline):
    """
    Search for Q(y) with 12 terms such that Q(y)^2 has few nonzero terms.
    Try exponents 0..11 scaled by d=1,2,3.
    Coeffs from {-2, -1, 1, 2}.
    """
    best_sq = float('inf')
    best_coeffs = None
    random.seed(12345)
    
    for d in [1, 2, 3]:
        base_exps = list(range(12))
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            if all(c == 0 for c in coeffs):
                continue
            
            P_sparse = [(i * d, c) for i, c in enumerate(coeffs) if c != 0]
            if len(P_sparse) != 12:
                continue
            
            max_exp = max(e for e, c in P_sparse)
            if max_exp > 40:
                continue
            
            dense = _sparse_to_dense(P_sparse)
            if not _validate(dense):
                continue
            
            sq_count = _square_nonzero_count(P_sparse)
            if sq_count < best_sq:
                best_sq = sq_count
                best_coeffs = dense
    
    return best_coeffs, best_sq

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
    
    # Phase 0: Structured Q search
    if time.time() < deadline - 60.0:
        try:
            q_dense, q_sq = _search_structured_Q(deadline - 55.0)
            if q_dense is not None and q_sq < best_sq:
                best_sq = q_sq
                best_sparse = _dense_to_sparse(q_dense)
        except Exception:
            pass
    
    # Phase 1: Factor-based structured search
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
