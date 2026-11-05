"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
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

def _square_nonzero_count_from_sparse(sparse):
    """Count nonzero terms in square from sparse representation."""
    dense = _sparse_to_dense(sparse)
    return _count_nonzero_square_fast(dense)

def _find_collision_exponents(deadline):
    """
    Find exponent sets where many pairs share the same sum degree.
    This creates opportunities for coefficient-based cancellation.
    
    Strategy: Use symmetric structures. If exponents come in pairs (a_i, b_i)
    where a_i + b_j = constant for various (i,j), then cross-terms at that
    degree can cancel if coefficients are chosen appropriately.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(12345)
    
    # Method 1: Arithmetic progression with step k
    # For exponents 0, k, 2k, ..., 11k, the pair sums range from 0 to 22k
    # Many pairs share the same sum: sum s has (s//k + 1) pairs if s <= 11k
    # We want coefficients c_i such that for each sum s, sum_{i+j=s} c_i*c_j = 0
    # This is a system of equations. Try specific patterns.
    
    for k in range(1, 4):
        max_start = max(0, 40 - 11 * k)
        for start in range(0, max_start + 1):
            # Try various coefficient patterns
            patterns = []
            # Alternating
            patterns.append([1 if i % 2 == 0 else -1 for i in range(12)])
            # Period 3
            for phase in range(3):
                patterns.append([1 if (i + phase) % 3 == 0 else -1 for i in range(12)])
            # Period 4
            for phase in range(4):
                patterns.append([1 if (i + phase) % 4 < 2 else -1 for i in range(12)])
            # Specific known-good patterns
            patterns.append([1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1])
            patterns.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
            patterns.append([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
            
            for pat in patterns:
                if time.time() > deadline:
                    return best_sparse, best_sq
                sparse = [(start + i * k, pat[i]) for i in range(12)]
                exps = [x[0] for x in sparse]
                if len(exps) != len(set(exps)) or any(e < 0 or e > 40 for e in exps):
                    continue
                sq = _square_nonzero_count_from_sparse(sparse)
                if sq < best_sq:
                    best_sq = sq
                    best_sparse = list(sparse)
    
    # Method 2: Paired symmetric exponents around a center
    # (C-d, C-d+1), (C-d-2, C-d-1), ... creates many shared sums
    for C in range(15, 35):
        for d in range(5, 15):
            sparse = []
            valid = True
            for i in range(6):
                e1 = C - d - 2 * i
                e2 = C - d + 1 - 2 * i
                if e1 < 0 or e2 > 40:
                    valid = False
                    break
                sparse.append((e1, 1))
                sparse.append((e2, -1))
            if not valid or len(sparse) != 12:
                continue
            exps = [x[0] for x in sparse]
            if len(exps) != len(set(exps)):
                continue
            sq = _square_nonzero_count_from_sparse(sparse)
            if sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
    
    # Method 3: Three clusters of 4 with internal structure
    for base in range(0, 15):
        for gap in range(8, 15):
            sparse = []
            valid = True
            for g in range(3):
                b = base + g * gap
                for k, c in enumerate([1, -1, -1, 1]):
                    e = b + k
                    if e > 40:
                        valid = False
                        break
                    sparse.append((e, c))
                if not valid:
                    break
            if not valid or len(sparse) != 12:
                continue
            exps = [x[0] for x in sparse]
            if len(exps) != len(set(exps)):
                continue
            sq = _square_nonzero_count_from_sparse(sparse)
            if sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
    
    # Method 4: Random with targeted collision structure
    # Pick exponents so that many pairs have the same sum
    for _ in range(500):
        if time.time() > deadline:
            break
        # Pick a "collision degree" S and build exponents around it
        S = random.randint(10, 35)
        exps = set()
        # Add pairs that sum to S
        for _ in range(random.randint(3, 6)):
            a = random.randint(0, S)
            b = S - a
            if a != b and a not in exps and b not in exps:
                exps.add(a)
                exps.add(b)
        # Fill remaining with random exponents
        while len(exps) < 12:
            e = random.randint(0, 40)
            if e not in exps:
                exps.add(e)
        exps = sorted(exps)[:12]
        if len(exps) != 12:
            continue
        # Try a few coefficient patterns
        for _ in range(3):
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            sparse = list(zip(exps, coeffs))
            sq = _square_nonzero_count_from_sparse(sparse)
            if sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
    
    return best_sparse, best_sq

def _directed_cancellation_search(deadline):
    """
    Directly search for configurations where cross-term cancellations occur.
    
    Key idea: For a given set of exponents, the square has terms at degrees
    e_i + e_j for all pairs (i,j). If multiple pairs share the same sum degree,
    their contributions add. We want to choose coefficients so that at each
    shared-sum degree, the total contribution is zero.
    
    This is a constrained optimization: fix exponents, optimize coefficients.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(54321)
    
    # Generate candidate exponent sets with high collision density
    candidate_exps = []
    
    # AP-based
    for k in [1, 2, 3]:
        for start in range(0, max(1, 41 - 11 * k)):
            candidate_exps.append([start + i * k for i in range(12)])
    
    # Symmetric pairs
    for C in range(10, 35):
        for spread in range(3, 12):
            exps = []
            for i in range(6):
                e1 = C - spread + i
                e2 = C + spread - i
                if 0 <= e1 <= 40 and 0 <= e2 <= 40 and e1 != e2:
                    exps.extend([e1, e2])
            if len(exps) == 12 and len(set(exps)) == 12:
                candidate_exps.append(sorted(exps))
    
    # Random collision-structured
    for _ in range(200):
        S = random.randint(10, 35)
        exps = set()
        for _ in range(6):
            a = random.randint(0, S)
            b = S - a
            if a != b:
                exps.add(a)
                exps.add(b)
        while len(exps) < 12:
            exps.add(random.randint(0, 40))
        candidate_exps.append(sorted(exps)[:12])
    
    # For each candidate exponent set, do local coefficient optimization
    for exps in candidate_exps[:300]:  # Limit to avoid timeout
        if time.time() > deadline:
            break
        if len(exps) != 12 or len(set(exps)) != 12:
            continue
        if any(e < 0 or e > 40 for e in exps):
            continue
        
        # Start with a reasonable coefficient assignment
        best_local = None
        best_local_sq = float('inf')
        
        # Try several initial coefficient patterns
        initial_patterns = [
            [1] * 12,
            [1 if i % 2 == 0 else -1 for i in range(12)],
            [1 if i % 3 == 0 else -1 for i in range(12)],
            [random.choice([-2, -1, 1, 2]) for _ in range(12)],
        ]
        
        for init_coeffs in initial_patterns:
            if time.time() > deadline:
                break
            current_coeffs = list(init_coeffs)
            current_sq = _square_nonzero_count_from_sparse(list(zip(exps, current_coeffs)))
            
            if current_sq < best_local_sq:
                best_local_sq = current_sq
                best_local = list(current_coeffs)
            
            # Local search: flip signs or change coefficients
            improved = True
            max_iter = 50
            while improved and max_iter > 0 and time.time() < deadline:
                improved = False
                max_iter -= 1
                for i in range(12):
                    for new_c in [-2, -1, 1, 2]:
                        if new_c == current_coeffs[i]:
                            continue
                        trial = list(current_coeffs)
                        trial[i] = new_c
                        trial_sq = _square_nonzero_count_from_sparse(list(zip(exps, trial)))
                        if trial_sq < current_sq:
                            current_coeffs = trial
                            current_sq = trial_sq
                            improved = True
                            if current_sq < best_local_sq:
                                best_local_sq = current_sq
                                best_local = list(current_coeffs)
        
        if best_local is not None and best_local_sq < best_sq:
            best_sq = best_local_sq
            best_sparse = list(zip(exps, best_local))
    
    return best_sparse, best_sq

def _fast_hill_climb(sparse_init, deadline):
    """
    Fast hill-climbing with focus on moves that create or destroy cancellations.
    Uses incremental evaluation where possible.
    """
    current_sparse = list(sparse_init)
    current_sq = _square_nonzero_count_from_sparse(current_sparse)
    best_sparse = list(current_sparse)
    best_sq = current_sq
    
    random.seed(999)
    max_exp = 40
    stall_count = 0
    max_stall = 5000
    
    while time.time() < deadline - 2.0 and stall_count < max_stall:
        # Choose move type
        r = random.random()
        trial_sparse = list(current_sparse)
        valid = True
        
        if r < 0.3:
            # Flip sign of one term
            idx = random.randrange(12)
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)
        elif r < 0.5:
            # Change coefficient
            idx = random.randrange(12)
            e, c = trial_sparse[idx]
            new_c = random.choice([-2, -1, 1, 2])
            if new_c != c:
                trial_sparse[idx] = (e, new_c)
            else:
                continue
        elif r < 0.7:
            # Shift one exponent
            idx = random.randrange(12)
            e, c = trial_sparse[idx]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = e + shift
            if 0 <= new_e <= max_exp:
                other_exps = set(x[0] for i, x in enumerate(trial_sparse) if i != idx)
                if new_e not in other_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    continue
            else:
                continue
        elif r < 0.85:
            # Shift two exponents (potential collision creation)
            i, j = random.sample(range(12), 2)
            ei, ci = trial_sparse[i]
            ej, cj = trial_sparse[j]
            # Try to make ei + ej equal to some other pair's sum
            other_sums = set()
            for a in range(12):
                for b in range(a + 1, 12):
                    if a not in (i, j) and b not in (i, j):
                        other_sums.add(trial_sparse[a][0] + trial_sparse[b][0])
            target_sum = random.choice(list(other_sums)) if other_sums else random.randint(0, 80)
            new_ei = random.randint(0, min(max_exp, target_sum))
            new_ej = target_sum - new_ei
            if 0 <= new_ej <= max_exp and new_ei != new_ej:
                other_exps = set(x[0] for k, x in enumerate(trial_sparse) if k not in (i, j))
                if new_ei not in other_exps and new_ej not in other_exps:
                    trial_sparse[i] = (new_ei, ci)
                    trial_sparse[j] = (new_ej, cj)
                else:
                    continue
            else:
                continue
        else:
            # Swap exponents of two terms
            i, j = random.sample(range(12), 2)
            ei, ci = trial_sparse[i]
            ej, cj = trial_sparse[j]
            trial_sparse[i] = (ej, ci)
            trial_sparse[j] = (ei, cj)
        
        # Validate
        exps = [x[0] for x in trial_sparse]
        if len(exps) != len(set(exps)):
            continue
        if any(e < 0 or e > max_exp for e in exps):
            continue
        
        trial_sq = _square_nonzero_count_from_sparse(trial_sparse)
        
        if trial_sq < current_sq:
            current_sparse = trial_sparse
            current_sq = trial_sq
            stall_count = 0
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_sparse = list(trial_sparse)
        else:
            stall_count += 1
            # Occasionally accept worse solutions to escape local minima
            if random.random() < 0.05 and trial_sq <= current_sq + 3:
                current_sparse = trial_sparse
                current_sq = trial_sq
    
    return best_sparse, best_sq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Phase 1: Collision-based structured search
    best_sparse = None
    best_sq = float('inf')
    
    if time.time() < deadline - 50.0:
        try:
            cs, csq = _find_collision_exponents(deadline - 45.0)
            if cs is not None and csq < best_sq:
                best_sq = csq
                best_sparse = list(cs)
        except Exception:
            pass
    
    # Phase 2: Directed cancellation search
    if time.time() < deadline - 20.0:
        try:
            ds, dsq = _directed_cancellation_search(deadline - 15.0)
            if ds is not None and dsq < best_sq:
                best_sq = dsq
                best_sparse = list(ds)
        except Exception:
            pass
    
    # Phase 3: Fast hill-climbing from best
    if best_sparse is None:
        # Fallback initial
        best_sparse = [(i * 3, 1 if i % 2 == 0 else -1) for i in range(12) if i * 3 < 41]
        while len(best_sparse) < 12:
            best_sparse.append((len(best_sparse) * 3, 1))
        best_sparse = best_sparse[:12]
        best_sq = _square_nonzero_count_from_sparse(best_sparse)
    
    if time.time() < deadline - 5.0:
        try:
            fs, fsq = _fast_hill_climb(best_sparse, deadline)
            if fsq < best_sq:
                best_sq = fsq
                best_sparse = list(fs)
        except Exception:
            pass
    
    # Convert to dense and validate
    best_coeffs = _sparse_to_dense(best_sparse)
    if not _validate(best_coeffs):
        # Fallback
        best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
