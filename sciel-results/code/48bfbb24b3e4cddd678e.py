import time
import random
import numpy as np

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square using numpy convolution."""
    n = len(coefs)
    if n == 0:
        return 0
    # Trim trailing zeros for speed
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

def _try_split_structure(rng, time_budget=5.0):
    """
    Try P(x) = A(x) + x^k * B(x) where A and B have disjoint support.
    The square is A^2 + 2*x^k*A*B + x^(2k)*B^2.
    If supports are chosen so cross terms cancel or are sparse, we get low total.
    
    We search over small A and B (with few terms each) and combine.
    """
    deadline = time.time() + time_budget
    best_score = 10**9
    best_poly = None
    
    # Try various splits: A has a terms, B has b terms, a+b=12
    splits = [(1,11),(2,10),(3,9),(4,8),(5,7),(6,6),(7,5),(8,4),(9,3),(10,2),(11,1)]
    
    for a_n, b_n in splits:
        if time.time() > deadline:
            break
        
        # Generate A with a_n terms
        # A is in low degrees [0, k)
        # B is in high degrees [k, 41)
        
        # Try different shift values
        for k in range(1, 35):
            if time.time() > deadline:
                break
            
            # A must fit in [0, k), so A has degree < k
            # B must fit in [k, 41), so B shifted by k has degree < 41-k
            # B itself has degree < 41-k
            
            if k < a_n:  # Need at least a_n positions for A
                continue
            if 41 - k < b_n:  # Need at least b_n positions for B
                continue
            
            # Generate random A with a_n nonzero terms in [0, k)
            # Generate random B with b_n nonzero terms in [0, 41-k)
            
            max_attempts = 50
            for _ in range(max_attempts):
                if time.time() > deadline:
                    break
                
                # A: a_n terms in [0, k)
                a_idx = sorted(rng.sample(range(k), min(a_n, k)))
                if len(a_idx) < a_n:
                    continue
                A = [0] * 41
                for idx in a_idx:
                    A[idx] = rng.choice([-1, 1, 2, -2, 3, -3])
                
                # B: b_n terms in [0, 41-k), then shifted by k
                b_idx = sorted(rng.sample(range(41-k), min(b_n, 41-k)))
                if len(b_idx) < b_n:
                    continue
                B = [0] * 41
                for idx in b_idx:
                    B[idx] = rng.choice([-1, 1, 2, -2, 3, -3])
                
                # P = A + x^k * B
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

def _try_product_structure(rng, time_budget=5.0):
    """
    Try P(x) = A(x) * B(x) where A and B are sparse.
    P^2 = A^2 * B^2. If A^2 and B^2 have few terms, P^2 may too.
    """
    deadline = time.time() + time_budget
    best_score = 10**9
    best_poly = None
    
    # A has a terms, B has b terms, a*b = 12 (approximately)
    factor_pairs = [(1,12),(2,6),(3,4),(4,3),(6,2),(12,1)]
    
    for a_n, b_n in factor_pairs:
        if time.time() > deadline:
            break
        
        max_attempts = 100
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate A with a_n terms
            A_idx = sorted(rng.sample(range(41), min(a_n, 41)))
            if len(A_idx) < a_n:
                continue
            A = [0] * 41
            for idx in A_idx:
                A[idx] = rng.choice([-1, 1, 2, -2])
            
            # Generate B with b_n terms
            B_idx = sorted(rng.sample(range(41), min(b_n, 41)))
            if len(B_idx) < b_n:
                continue
            B = [0] * 41
            for idx in B_idx:
                B[idx] = rng.choice([-1, 1, 2, -2])
            
            # P = A * B
            P = [0] * 41
            for i in range(41):
                if A[i] != 0:
                    for j in range(41-i):
                        if B[j] != 0:
                            P[i+j] += A[i] * B[j]
            
            if not _valid(P):
                continue
            
            sc = _sq_nonzero_count(P)
            if sc < best_score:
                best_score = sc
                best_poly = P[:]
    
    return best_poly, best_score

def _try_geometric_structure(rng, time_budget=3.0):
    """
    Try polynomials with geometric-like structure: terms at positions forming
    arithmetic or geometric progressions with carefully chosen signs.
    """
    deadline = time.time() + time_budget
    best_score = 10**9
    best_poly = None
    
    # Arithmetic progression with varying step and sign patterns
    for step in range(1, 10):
        if time.time() > deadline:
            break
        for start in range(0, 5):
            if time.time() > deadline:
                break
            # Need 12 terms: start, start+step, ..., start+11*step < 41
            if start + 11*step >= 41:
                continue
            
            # Try different sign patterns
            for sign_pattern in range(min(16, 2**min(12, 4))):
                if time.time() > deadline:
                    break
                
                P = [0] * 41
                for i in range(12):
                    idx = start + i * step
                    if idx >= 41:
                        break
                    # Sign based on pattern
                    if (sign_pattern >> i) & 1:
                        P[idx] = -1
                    else:
                        P[idx] = 1
                
                # Ensure p[40] != 0
                if P[40] == 0:
                    # Move last term to position 40
                    nz_idx = [i for i in range(41) if P[i] != 0]
                    if nz_idx:
                        last = nz_idx[-1]
                        P[40] = P[last]
                        P[last] = 0
                
                if not _valid(P):
                    continue
                
                sc = _sq_nonzero_count(P)
                if sc < best_score:
                    best_score = sc
                    best_poly = P[:]
    
    return best_poly, best_score

def _local_search(P_init, rng, time_limit=15.0):
    """Greedy local search for quick improvement."""
    current = P_init[:]
    current_score = _sq_nonzero_count(current)
    best = current[:]
    best_score = current_score
    deadline = time.time() + time_limit
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try changing coefficient values
        for idx in range(41):
            if time.time() > deadline:
                break
            if current[idx] == 0:
                continue
            old_val = current[idx]
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                current[idx] = new_val
                if not _valid(current):
                    current[idx] = old_val
                    continue
                sc = _sq_nonzero_count(current)
                if sc < best_score:
                    best_score = sc
                    best = current[:]
                    improved = True
                else:
                    current[idx] = old_val
        
        # Try support moves: move a term to a different position
        for idx in range(41):
            if time.time() > deadline:
                break
            if current[idx] == 0:
                continue
            old_val = current[idx]
            for new_idx in range(41):
                if time.time() > deadline:
                    break
                if current[new_idx] != 0:
                    continue
                if new_idx == idx:
                    continue
                current[idx] = 0
                current[new_idx] = old_val
                if _valid(current):
                    sc = _sq_nonzero_count(current)
                    if sc < best_score:
                        best_score = sc
                        best = current[:]
                        improved = True
                current[new_idx] = 0
                current[idx] = old_val
        
        # Try swapping two nonzero terms' values
        nz_idx = [i for i in range(41) if current[i] != 0]
        for i in range(len(nz_idx)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(nz_idx)):
                if time.time() > deadline:
                    break
                idx1, idx2 = nz_idx[i], nz_idx[j]
                v1, v2 = current[idx1], current[idx2]
                current[idx1], current[idx2] = v2, v1
                if _valid(current):
                    sc = _sq_nonzero_count(current)
                    if sc < best_score:
                        best_score = sc
                        best = current[:]
                        improved = True
                current[idx1], current[idx2] = v1, v2
    
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
            # Restart
            current = best[:]
            current_score = best_score
            T = 1.0
            no_improve = 0
            continue
        
        # Generate neighbor
        neighbor = current[:]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Change coefficient value of a random nonzero term
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
            # Add a new term (remove one to keep 12)
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
            # Shift a term
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
            # Swap two terms' positions
            nz_idx = [i for i in range(41) if neighbor[i] != 0]
            if len(nz_idx) >= 2:
                i1, i2 = rng.sample(nz_idx, 2)
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
        
        # Validate
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

def _gen_structured_candidates(rng, count=200):
    """Generate structured candidates with algebraic properties."""
    candidates = []
    
    for _ in range(count):
        r = rng.random()
        
        if r < 0.3:
            # Product of 2-4 binomial factors
            n_f = rng.randint(2, 4)
            P = [0] * 41
            P[0] = 1
            current_max = 0
            for i in range(n_f):
                s = rng.choice([-1, 1])
                e = rng.randint(1, 40)
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
        elif r < 0.5:
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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    rng = random.Random(42)
    
    best_score = 10**9
    best_poly = None
    
    # Phase 1: Try split structure P = A + x^k * B
    poly, score = _try_split_structure(rng, time_budget=10.0)
    if poly is not None and score < best_score:
        best_score = score
        best_poly = poly
    
    # Phase 2: Try product structure P = A * B
    if time.time() < deadline - 20:
        poly, score = _try_product_structure(rng, time_budget=8.0)
        if poly is not None and score < best_score:
            best_score = score
            best_poly = poly
    
    # Phase 3: Try geometric structure
    if time.time() < deadline - 15:
        poly, score = _try_geometric_structure(rng, time_budget=5.0)
        if poly is not None and score < best_score:
            best_score = score
            best_poly = poly
    
    # Phase 4: Generate structured candidates
    if time.time() < deadline - 10:
        candidates = _gen_structured_candidates(rng, count=300)
        for p in candidates:
            if time.time() > deadline - 10:
                break
            if not _valid(p):
                continue
            sc = _sq_nonzero_count(p)
            if sc < best_score:
                best_score = sc
                best_poly = p[:]
    
    # Fallback
    if best_poly is None:
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        best_score = _sq_nonzero_count(best_poly)
    
    # Phase 5: Local search
    if time.time() < deadline - 10:
        best_poly, best_score = _local_search(best_poly, rng, time_limit=min(15.0, deadline - time.time() - 5))
    
    # Phase 6: Simulated annealing
    if time.time() < deadline - 5:
        sa_poly, sa_score = _simulated_annealing(best_poly, rng, time_limit=min(40.0, deadline - time.time() - 2))
        if sa_score < best_score:
            best_score = sa_score
            best_poly = sa_poly
    
    # Phase 7: Final local search
    if time.time() < deadline - 2:
        best_poly, best_score = _local_search(best_poly, rng, time_limit=min(10.0, deadline - time.time() - 1))
    
    return best_poly
