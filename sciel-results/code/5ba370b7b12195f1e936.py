import time
import random
import numpy as np

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square using numpy convolution."""
    n = len(coefs)
    if n == 0:
        return 0
    while n > 1 and coefs[n-1] == 0:
        n -= 1
    arr = np.array(coefs[:n], dtype=np.int64)
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _count_nz(coefs):
    return sum(1 for c in coefs if c != 0)

def _sq_full(coefs):
    """Return the full square coefficient array."""
    n = len(coefs)
    while n > 1 and coefs[n-1] == 0:
        n -= 1
    arr = np.array(coefs[:n], dtype=np.int64)
    return np.convolve(arr, arr, mode='full')

def _gen_product_factors(rng, n_factors, max_exp=40):
    """Generate a polynomial as product of (1 + s_i * x^{e_i}) factors."""
    P = [0] * 41
    P[0] = 1
    current_max = 0
    for i in range(n_factors):
        s = rng.choice([-1, 1])
        e = rng.randint(1, max_exp)
        if current_max + e >= 41:
            e = max(1, 41 - current_max - 1)
            if e <= 0:
                break
        new_P = [0] * 41
        for d in range(41):
            if P[d] != 0:
                new_P[d] += P[d]
                if d + e < 41:
                    new_P[d + e] += s * P[d]
        P = new_P
        current_max += e
    return P

def _gen_structured_candidates(rng, count=200):
    """Generate structured candidates with algebraic properties."""
    candidates = []
    for _ in range(count):
        r = rng.random()
        if r < 0.35:
            # Product of 2-5 binomial factors
            n_f = rng.randint(2, 5)
            P = _gen_product_factors(rng, n_f)
        elif r < 0.55:
            # Product with shared exponents (more cancellation potential)
            n_f = rng.randint(3, 6)
            exponents = rng.sample(range(1, 25), min(n_f, 12))
            P = [0] * 41
            P[0] = 1
            for e in exponents:
                s = rng.choice([-1, 1])
                new_P = [0] * 41
                for d in range(41):
                    if P[d] != 0:
                        new_P[d] += P[d]
                        if d + e < 41:
                            new_P[d + e] += s * P[d]
                P = new_P
        elif r < 0.75:
            # Arithmetic progression with alternating signs
            start = rng.randint(0, 5)
            step = rng.randint(2, 8)
            P = [0] * 41
            for i in range(12):
                idx = start + i * step
                if idx < 41:
                    P[idx] = 1 if i % 2 == 0 else -1
            nz = _count_nz(P)
            if nz == 12 and P[40] != 0:
                candidates.append(P[:])
            elif nz == 12 and P[40] == 0:
                nz_idx = [i for i in range(41) if P[i] != 0]
                if nz_idx:
                    last = nz_idx[-1]
                    P[40] = P[last]
                    P[last] = 0
                    if _count_nz(P) == 12:
                        candidates.append(P[:])
        else:
            # Random sparse with ±1 coefficients
            P = [0] * 41
            indices = rng.sample(range(41), 12)
            for idx in indices:
                P[idx] = rng.choice([-1, 1])
            if P[40] == 0:
                nz_idx = [i for i in range(41) if P[i] != 0]
                if nz_idx:
                    k = rng.choice(nz_idx)
                    P[40] = P[k]
                    P[k] = 0
            if _count_nz(P) == 12:
                candidates.append(P[:])
    return candidates

def _fast_local_search(P_init, rng, time_limit=20.0):
    """
    Fast local search with incremental square update.
    When we change P[i] from old to new, the square changes by:
    delta_sq[d] = (new^2 - old^2) * P[d-i] + 2*(new - old) * P[d-i] for cross terms
    Actually: P^2 = sum_j P[j]^2 x^{2j} + 2*sum_{j<k} P[j]P[k] x^{j+k}
    If we change P[i] from a to b:
    - P[i]^2 term: a^2 -> b^2, delta at degree 2i is b^2 - a^2
    - Cross terms involving i: 2*a*P[k] -> 2*b*P[k] for all k != i
      delta at degree i+k is 2*(b-a)*P[k]
    """
    current = P_init[:]
    current_score = _sq_nonzero_count(current)
    best = current[:]
    best_score = current_score
    deadline = time.time() + time_limit
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    # Precompute the square
    cur_sq = _sq_full(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        nz_idx = [i for i in range(41) if current[i] != 0]
        
        # Try changing coefficient values
        for idx in nz_idx:
            if time.time() > deadline:
                break
            old_val = current[idx]
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                if abs(new_val) > 64:
                    continue
                
                # Compute new square incrementally
                # Old contribution of idx to square:
                # - At degree 2*idx: old_val^2
                # - At degree idx + k for each k != idx: 2 * old_val * P[k]
                # New contribution:
                # - At degree 2*idx: new_val^2
                # - At degree idx + k for each k != idx: 2 * new_val * P[k]
                
                new_sq = cur_sq.copy()
                # Remove old contribution
                new_sq[2*idx] -= old_val * old_val
                for k in range(41):
                    if k != idx and current[k] != 0:
                        new_sq[idx + k] -= 2 * old_val * current[k]
                # Add new contribution
                new_sq[2*idx] += new_val * new_val
                for k in range(41):
                    if k != idx and current[k] != 0:
                        new_sq[idx + k] += 2 * new_val * current[k]
                
                # Check validity: need exactly 12 nonzero terms, last nonzero
                current[idx] = new_val
                if not _valid(current):
                    current[idx] = old_val
                    continue
                
                sc = int(np.count_nonzero(new_sq))
                if sc < best_score:
                    best_score = sc
                    best = current[:]
                    cur_sq = new_sq
                    improved = True
                else:
                    current[idx] = old_val
        
        # Try support moves: move a term to a different position
        for idx in nz_idx:
            if time.time() > deadline:
                break
            old_val = current[idx]
            zero_idx = [i for i in range(41) if current[i] == 0]
            for new_idx in zero_idx:
                if time.time() > deadline:
                    break
                if new_idx == idx:
                    continue
                
                # Remove old term at idx, add new term at new_idx
                new_sq = cur_sq.copy()
                # Remove old contribution of idx
                new_sq[2*idx] -= old_val * old_val
                for k in range(41):
                    if k != idx and current[k] != 0:
                        new_sq[idx + k] -= 2 * old_val * current[k]
                # Add new contribution at new_idx
                new_sq[2*new_idx] += old_val * old_val
                for k in range(41):
                    if k != idx and k != new_idx and current[k] != 0:
                        new_sq[new_idx + k] += 2 * old_val * current[k]
                
                current[idx] = 0
                current[new_idx] = old_val
                if not _valid(current):
                    current[new_idx] = 0
                    current[idx] = old_val
                    continue
                
                sc = int(np.count_nonzero(new_sq))
                if sc < best_score:
                    best_score = sc
                    best = current[:]
                    cur_sq = new_sq
                    improved = True
                else:
                    current[new_idx] = 0
                    current[idx] = old_val
    
    return best, best_score

def _simulated_annealing(P_init, rng, time_limit=30.0):
    """
    Simulated annealing to minimize nonzero terms in P^2.
    """
    current = P_init[:]
    current_score = _sq_nonzero_count(current)
    best = current[:]
    best_score = current_score
    
    T = 2.0
    T_min = 0.05
    cooling = 0.9995
    deadline = time.time() + time_limit
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    no_improve = 0
    
    while time.time() < deadline:
        no_improve += 1
        if no_improve > 5000:
            current = best[:]
            current_score = best_score
            T = 1.0
            no_improve = 0
            continue
        
        neighbor = current[:]
        move_type = rng.random()
        
        if move_type < 0.4:
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if not nz_idx:
                continue
            idx = rng.choice(nz_idx)
            old_val = neighbor[idx]
            new_val = rng.choice(val_pool)
            if new_val == old_val:
                new_val = old_val + rng.choice([-1, 1])
            if abs(new_val) > 64:
                continue
            neighbor[idx] = new_val
        elif move_type < 0.6:
            zero_idx = [i for i in range(41) if neighbor[i] == 0]
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if not zero_idx or not nz_idx:
                continue
            idx_remove = rng.choice(nz_idx)
            idx_add = rng.choice(zero_idx)
            if idx_add == idx_remove:
                continue
            neighbor[idx_remove] = 0
            neighbor[idx_add] = rng.choice(val_pool)
        elif move_type < 0.8:
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if not nz_idx:
                continue
            idx = rng.choice(nz_idx)
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            new_idx = idx + shift
            if 0 <= new_idx < 41 and neighbor[new_idx] == 0:
                neighbor[new_idx] = neighbor[idx]
                neighbor[idx] = 0
            else:
                neighbor[idx] = rng.choice(val_pool)
        else:
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if len(nz_idx) >= 2:
                i1, i2 = rng.sample(nz_idx, 2)
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
        
        if not _valid(neighbor):
            continue
        
        neighbor_score = _sq_nonzero_count(neighbor)
        delta = neighbor_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best = current[:]
                no_improve = 0
        
        T = max(T_min, T * cooling)
    
    return best, best_score

def _try_split_structure(rng, time_budget=5.0):
    """Try P(x) = A(x) + x^k * B(x) with disjoint supports."""
    deadline = time.time() + time_budget
    best_score = 10**9
    best_poly = None
    
    splits = [(1,11),(2,10),(3,9),(4,8),(5,7),(6,6)]
    
    for a_n, b_n in splits:
        if time.time() > deadline:
            break
        for k in range(max(a_n, 1), 41 - b_n + 1):
            if time.time() > deadline:
                break
            max_attempts = 30
            for _ in range(max_attempts):
                if time.time() > deadline:
                    break
                a_idx = sorted(rng.sample(range(k), min(a_n, k)))
                if len(a_idx) < a_n:
                    continue
                A = [0] * 41
                for idx in a_idx:
                    A[idx] = rng.choice([-1, 1, 2, -2, 3, -3])
                
                b_idx = sorted(rng.sample(range(41-k), min(b_n, 41-k)))
                if len(b_idx) < b_n:
                    continue
                B = [0] * 41
                for idx in b_idx:
                    B[idx] = rng.choice([-1, 1, 2, -2, 3, -3])
                
                P = [0] * 41
                for i in range(41):
                    P[i] += A[i]
                    if i >= k:
                        P[i] += B[i-k]
                
                if not _valid(P):
                    continue
                sc = _sq_nonzero_count(P)
                if sc < best_score:
                    best_score = sc
                    best_poly = P[:]
    
    return best_poly, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    rng = random.Random(42)
    
    best_score = 10**9
    best_poly = None
    
    # Phase 1: Structured candidates
    if time.time() < deadline - 30:
        candidates = _gen_structured_candidates(rng, count=400)
        for p in candidates:
            if time.time() > deadline - 30:
                break
            if not _valid(p):
                continue
            sc = _sq_nonzero_count(p)
            if sc < best_score:
                best_score = sc
                best_poly = p[:]
    
    # Phase 2: Split structure
    if time.time() < deadline - 25:
        poly, score = _try_split_structure(rng, time_budget=5.0)
        if poly is not None and score < best_score:
            best_score = score
            best_poly = poly
    
    # Fallback
    if best_poly is None:
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        best_poly[40] = 1
        # Ensure exactly 12 nonzero
        best_score = _sq_nonzero_count(best_poly)
    
    # Phase 3: Fast local search
    if time.time() < deadline - 20:
        best_poly, best_score = _fast_local_search(best_poly, rng, time_limit=min(15.0, deadline - time.time() - 15))
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5:
        sa_poly, sa_score = _simulated_annealing(best_poly, rng, time_limit=min(35.0, deadline - time.time() - 3))
        if sa_score < best_score:
            best_score = sa_score
            best_poly = sa_poly
    
    # Phase 5: Final fast local search
    if time.time() < deadline - 1:
        best_poly, best_score = _fast_local_search(best_poly, rng, time_limit=min(8.0, deadline - time.time() - 0.5))
    
    return best_poly
