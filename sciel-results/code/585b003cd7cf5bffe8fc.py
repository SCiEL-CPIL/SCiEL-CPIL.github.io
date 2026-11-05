"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product, combinations

def _square_nonzero_count_from_pairs(pairs):
    """Compute number of nonzero terms in square of polynomial given as list of (exponent, coeff)."""
    n = len(pairs)
    sq = {}
    for i in range(n):
        ei, ci = pairs[i]
        for j in range(i, n):
            ej, cj = pairs[j]
            pos = ei + ej
            val = ci * cj
            if i != j:
                val *= 2
            sq[pos] = sq.get(pos, 0) + val
    return sum(1 for v in sq.values() if v != 0)

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

def _factor_product_search(deadline):
    """
    Model P = A*B*C where A, B, C are sparse factors with 2-4 terms each.
    Search for factor structures where the squares of individual factors
    have minimal support and their exponents align to maximize collisions.
    """
    best_coeffs = None
    best_sq = float('inf')
    
    random.seed(42)
    
    # Factor term count splits: (2,2,4), (2,3,3), (3,3,3), (2,4,2), (3,2,3), (4,2,2)
    splits = [(2,2,4), (2,3,3), (3,3,3), (2,4,2), (3,2,3), (4,2,2)]
    
    def eval_sparse(sparse):
        if len(sparse) != 12:
            return float('inf')
        exps = [x[0] for x in sparse]
        if len(set(exps)) != 12:
            return float('inf')
        if any(e < 0 or e > 40 for e in exps):
            return float('inf')
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            return float('inf')
        return _square_nonzero_count_from_pairs(sparse)
    
    for trial in range(80000):
        if time.time() > deadline - 3.0:
            break
        
        split = random.choice(splits)
        
        # Generate random exponents for each factor
        factor_exps = []
        factor_coeffs = []
        valid = True
        
        for terms in split:
            # Choose base offset for this factor
            base = random.randint(0, 30)
            exps = []
            for k in range(terms):
                e = base + random.randint(0, 15)
                if e > 40:
                    valid = False
                    break
                exps.append(e)
            if not valid:
                break
            factor_exps.append(sorted(exps))
            
            # Choose coefficients for this factor
            mags = [random.choice([1, 1, 1, 2, 2, 3]) for _ in range(terms)]
            signs = [random.choice([1, -1]) for _ in range(terms)]
            factor_coeffs.append([s * m for s, m in zip(signs, mags)])
        
        if not valid:
            continue
        
        # Multiply the factors together
        # Start with first factor
        current_exp_coeff = list(zip(factor_exps[0], factor_coeffs[0]))
        
        for fi in range(1, 3):
            new_exp_coeff = []
            for e1, c1 in current_exp_coeff:
                for e2, c2 in zip(factor_exps[fi], factor_coeffs[fi]):
                    new_exp_coeff.append((e1 + e2, c1 * c2))
            current_exp_coeff = new_exp_coeff
        
        # Check we have exactly 12 terms
        if len(current_exp_coeff) != 12:
            continue
        
        # Check all exponents are distinct
        exps = [x[0] for x in current_exp_coeff]
        if len(set(exps)) != 12:
            continue
        
        # Check all within bounds
        if any(e < 0 or e > 40 for e in exps):
            continue
        
        # Check coefficients within bounds
        if any(abs(c) > 64 for e, c in current_exp_coeff):
            continue
        
        # Evaluate
        sq = eval_sparse(current_exp_coeff)
        if sq < best_sq:
            best_sq = sq
            best_coeffs = _sparse_to_dense(current_exp_coeff)
            if best_sq <= 14:
                break
    
    return best_coeffs, best_sq

def _design_cancellation_polynomial(deadline):
    """
    Structured design: group 12 terms into 6 pairs.
    Each pair has terms at exponents (e, e+d) with coefficients (a, -a).
    Search over pair structures to maximize cross-pair cancellation.
    """
    best_coeffs = None
    best_sq = float('inf')
    
    random.seed(777)
    
    def eval_sparse(sparse):
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            return float('inf')
        return _square_nonzero_count_from_pairs(sparse)
    
    gap_choices = [1, 2, 3, 4, 5, 6, 8, 10]
    mag_choices = [1, 2, 3, 4]
    
    for trial in range(40000):
        if time.time() > deadline - 3.0:
            break
        
        strategy = trial % 4
        
        if strategy == 0:
            base_spacing = random.choice([4, 5, 6, 7, 8])
            offsets = [0] * 6
            for i in range(1, 6):
                offsets[i] = offsets[i-1] + base_spacing + random.randint(-1, 1)
            if offsets[5] > 35:
                continue
            bases = [random.randint(0, 3) + off for off in offsets]
            gaps = [random.choice(gap_choices) for _ in range(6)]
            
        elif strategy == 1:
            cluster_size = random.choice([2, 3])
            num_clusters = 6 // cluster_size
            bases = []
            gaps = []
            for c in range(num_clusters):
                cluster_base = c * (cluster_size * 3 + random.randint(2, 5))
                for k in range(cluster_size):
                    bases.append(cluster_base + k * random.randint(1, 3))
                    gaps.append(random.choice(gap_choices[:3]))
            if len(bases) != 6 or max(bases) + max(gaps) > 40:
                continue
                
        elif strategy == 2:
            bases = sorted(random.sample(range(0, 30), 6))
            gaps = [random.choice(gap_choices) for _ in range(6)]
            if max(bases) + max(gaps) > 40:
                continue
                
        else:
            gap = random.choice(gap_choices)
            base_spacing = random.choice([gap+1, gap+2, gap+3])
            start = random.randint(0, 3)
            bases = [start + i * base_spacing for i in range(6)]
            if bases[5] + gap > 40:
                continue
            gaps = [gap] * 6
        
        sparse = []
        valid = True
        for i in range(6):
            e1 = bases[i]
            e2 = bases[i] + gaps[i]
            if e1 < 0 or e2 > 40:
                valid = False
                break
            mag = random.choice(mag_choices)
            sign = random.choice([1, -1])
            sparse.append((e1, sign * mag))
            sparse.append((e2, -sign * mag))
        
        if not valid:
            continue
        
        exps = [x[0] for x in sparse]
        if len(set(exps)) != 12:
            continue
        
        sq = eval_sparse(sparse)
        if sq < best_sq:
            best_sq = sq
            best_coeffs = _sparse_to_dense(sparse)
            if best_sq <= 14:
                break
    
    return best_coeffs, best_sq

def _design_symmetric_polynomial(deadline):
    """Try anti-palindromic structures for additional cancellation."""
    best_coeffs = None
    best_sq = float('inf')
    
    random.seed(888)
    
    def eval_sparse(sparse):
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            return float('inf')
        return _square_nonzero_count_from_pairs(sparse)
    
    for trial in range(20000):
        if time.time() > deadline - 3.0:
            break
        
        center = random.randint(12, 20)
        sparse = []
        valid = True
        
        for i in range(1, 7):
            e1 = center - i
            e2 = center + i
            if e1 < 0 or e2 > 40:
                valid = False
                break
            mag = random.choice([1, 2, 3, 4])
            sign = random.choice([1, -1])
            sparse.append((e1, sign * mag))
            sparse.append((e2, -sign * mag))
        
        if not valid or len(sparse) != 12:
            continue
        
        exps = [x[0] for x in sparse]
        if len(set(exps)) != 12:
            continue
        
        sq = eval_sparse(sparse)
        if sq < best_sq:
            best_sq = sq
            best_coeffs = _sparse_to_dense(sparse)
            if best_sq <= 14:
                break
    
    return best_coeffs, best_sq

def _brute_force_signs_consecutive(deadline):
    """Brute-force all 2^12 sign patterns for consecutive exponents 0..11."""
    best_coeffs = None
    best_sq = float('inf')
    
    exps = list(range(12))
    for pattern in product([-1, 1], repeat=12):
        if time.time() > deadline - 2.0:
            break
        pairs = [(exps[i], pattern[i]) for i in range(12)]
        cnt = _square_nonzero_count_from_pairs(pairs)
        if cnt < best_sq:
            best_sq = cnt
            best_coeffs = list(pattern)
            if best_sq <= 14:
                break
    
    return best_coeffs, best_sq

def _local_refine(sparse_init, deadline):
    """Fast local search with incremental tracking and stagnation resets."""
    current_sparse = list(sparse_init)
    current_sq = _square_nonzero_count_from_pairs(current_sparse)
    
    best_sparse = list(current_sparse)
    best_sq = current_sq
    
    random.seed(999)
    max_exp = 40
    coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
    exp_set = set(x[0] for x in current_sparse)
    
    no_improve = 0
    
    while time.time() < deadline - 1.0:
        no_improve += 1
        
        if no_improve > 3000:
            current_sparse = list(best_sparse)
            current_sq = _square_nonzero_count_from_pairs(current_sparse)
            exp_set = set(x[0] for x in current_sparse)
            no_improve = 0
            for _ in range(10):
                idx = random.randrange(12)
                e, c = current_sparse[idx]
                new_e = max(0, min(40, e + random.randint(-5, 5)))
                if new_e not in exp_set or new_e == e:
                    current_sparse[idx] = (new_e, c)
                    exp_set.remove(e)
                    exp_set.add(new_e)
                    current_sq = _square_nonzero_count_from_pairs(current_sparse)
                    break
        
        move_type = random.random()
        
        if move_type < 0.4:
            idx = random.randrange(12)
            e, c = current_sparse[idx]
            shift = random.randint(-4, 4)
            if shift == 0:
                continue
            new_e = e + shift
            if 0 <= new_e <= max_exp and new_e not in exp_set:
                old_e = e
                current_sparse[idx] = (new_e, c)
                exp_set.remove(old_e)
                exp_set.add(new_e)
                trial_sq = _square_nonzero_count_from_pairs(current_sparse)
                if trial_sq < current_sq:
                    current_sq = trial_sq
                    if trial_sq < best_sq:
                        best_sq = trial_sq
                        best_sparse = list(current_sparse)
                        no_improve = 0
                else:
                    current_sparse[idx] = (old_e, c)
                    exp_set.remove(new_e)
                    exp_set.add(old_e)
                    
        elif move_type < 0.7:
            idx = random.randrange(12)
            e, c = current_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c == c:
                continue
            old_c = c
            current_sparse[idx] = (e, new_c)
            trial_sq = _square_nonzero_count_from_pairs(current_sparse)
            if trial_sq < current_sq:
                current_sq = trial_sq
                if trial_sq < best_sq:
                    best_sq = trial_sq
                    best_sparse = list(current_sparse)
                    no_improve = 0
            else:
                current_sparse[idx] = (e, old_c)
                
        elif move_type < 0.9:
            i, j = random.sample(range(12), 2)
            ei, ci = current_sparse[i]
            ej, cj = current_sparse[j]
            other_exps = set(x[0] for k, x in enumerate(current_sparse) if k not in (i, j))
            if ej not in other_exps and ei not in other_exps:
                current_sparse[i] = (ej, ci)
                current_sparse[j] = (ei, cj)
                exp_set = set(x[0] for x in current_sparse)
                trial_sq = _square_nonzero_count_from_pairs(current_sparse)
                if trial_sq < current_sq:
                    current_sq = trial_sq
                    if trial_sq < best_sq:
                        best_sq = trial_sq
                        best_sparse = list(current_sparse)
                        no_improve = 0
                else:
                    current_sparse[i] = (ei, ci)
                    current_sparse[j] = (ej, cj)
                    exp_set = set(x[0] for x in current_sparse)
        else:
            i, j = random.sample(range(12), 2)
            ei, ci = current_sparse[i]
            ej, cj = current_sparse[j]
            si = random.randint(-3, 3)
            sj = random.randint(-3, 3)
            if si == 0 and sj == 0:
                continue
            new_ei = ei + si
            new_ej = ej + sj
            if 0 <= new_ei <= max_exp and 0 <= new_ej <= max_exp:
                other_exps = set(x[0] for k, x in enumerate(current_sparse) if k not in (i, j))
                if new_ei not in other_exps and new_ej not in other_exps and new_ei != new_ej:
                    current_sparse[i] = (new_ei, ci)
                    current_sparse[j] = (new_ej, cj)
                    exp_set = set(x[0] for x in current_sparse)
                    trial_sq = _square_nonzero_count_from_pairs(current_sparse)
                    if trial_sq < current_sq:
                        current_sq = trial_sq
                        if trial_sq < best_sq:
                            best_sq = trial_sq
                            best_sparse = list(current_sparse)
                            no_improve = 0
                    else:
                        current_sparse[i] = (ei, ci)
                        current_sparse[j] = (ej, cj)
                        exp_set = set(x[0] for x in current_sparse)
    
    return best_sparse, best_sq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_coeffs = None
    best_sq = float('inf')
    
    # Phase 1: Brute force signs on consecutive exponents (fast, 4096 patterns)
    if time.time() < deadline - 50.0:
        try:
            coeffs, sq = _brute_force_signs_consecutive(deadline - 50.0)
            if coeffs is not None and sq < best_sq:
                best_sq = sq
                best_coeffs = coeffs
        except Exception:
            pass
    
    # Phase 2: Factor product search (structural: P = A*B*C)
    if time.time() < deadline - 30.0:
        try:
            coeffs, sq = _factor_product_search(deadline - 30.0)
            if coeffs is not None and sq < best_sq:
                best_sq = sq
                best_coeffs = coeffs
        except Exception:
            pass
    
    # Phase 3: Pair cancellation design
    if time.time() < deadline - 15.0:
        try:
            coeffs, sq = _design_cancellation_polynomial(deadline - 15.0)
            if coeffs is not None and sq < best_sq:
                best_sq = sq
                best_coeffs = coeffs
        except Exception:
            pass
    
    # Phase 4: Symmetric design
    if time.time() < deadline - 8.0:
        try:
            coeffs, sq = _design_symmetric_polynomial(deadline - 8.0)
            if coeffs is not None and sq < best_sq:
                best_sq = sq
                best_coeffs = coeffs
        except Exception:
            pass
    
    # Phase 5: Local refinement from best found
    if best_coeffs is not None and time.time() < deadline - 2.0:
        try:
            sparse = _dense_to_sparse(best_coeffs)
            refined_sparse, sq = _local_refine(sparse, deadline)
            if sq < best_sq:
                best_sq = sq
                best_coeffs = _sparse_to_dense(refined_sparse)
        except Exception:
            pass
    
    if best_coeffs is None:
        best_coeffs = [1] * 12
    
    if not _validate(best_coeffs):
        best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
