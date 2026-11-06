"""Optimized 12-stage product formula for Heisenberg chain time evolution."""
import time
import numpy as np
from scipy.linalg import expm

_start_time = time.time()
_TIMEOUT = 140.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ Heisenberg chain."""
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    delta = 0.8  # anisotropy
    h = 0.3      # field
    
    n = 6
    I2 = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        """Apply operator op to site in n-site chain."""
        result = np.array([[1.0+0j]], dtype=complex)
        for i in range(n):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, I2)
        return result
    
    # A: odd bonds (0-1, 2-3, 4-5) + half field
    # B: even bonds (1-2, 3-4) + half field
    A = np.zeros((2**n, 2**n), dtype=complex)
    B = np.zeros((2**n, 2**n), dtype=complex)
    
    # XXZ interaction: delta*(Sx_i Sx_j + Sy_i Sy_j) + Sz_i Sz_j
    # For bond (i, j):
    def bond_op(i, j):
        Sxi = site_op(Sx, i)
        Sxj = site_op(Sx, j)
        Syi = site_op(Sy, i)
        Syj = site_op(Sy, j)
        Szi = site_op(Sz, i)
        Szj = site_op(Sz, j)
        return delta * (Sxi @ Sxj + Syi @ Syj) + Szi @ Szj
    
    # Odd bonds: (0,1), (2,3), (4,5)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_op(i, j)
    
    # Even bonds: (1,2), (3,4)
    for i, j in [(1, 2), (3, 4)]:
        B += bond_op(i, j)
    
    # Half field for each: -h/2 * sum_i Sz_i
    half_field = 0.0
    for i in range(n):
        half_field += site_op(Sz, i)
    A -= (h / 2) * half_field
    B -= (h / 2) * half_field
    
    return A, B

def _score_coefficients(a, b, A, B, times=(0.25, 0.5, 1.0)):
    """Compute mean spectral norm error."""
    total_dim = 2 ** 6
    errors = []
    H = A + B
    for t in times:
        # Build product formula
        U = np.eye(total_dim, dtype=complex)
        for k in range(12):
            U = U @ expm(-1j * b[k] * t * B)
            U = U @ expm(-1j * a[k] * t * A)
        # Exact propagator
        U_exact = expm(-1j * t * H)
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    """Return optimal 12-stage product formula coefficients."""
    global _start_time
    A, B = _build_hamiltonians()
    
    # Start with symmetric 12-stage coefficients
    # Use a 6th-order symmetric composition
    # Symmetric: a_k = a_{13-k}, b_k = b_{13-k}
    
    # Initial guess: equal weights (2nd order Strang-like)
    a = [1.0 / 12] * 12
    b = [1.0 / 12] * 12
    
    # Try known high-order symmetric patterns
    # 6th order symmetric 12-stage: use composition weights
    # For 3-step composition with 4 stages each = 12 stages
    # Outer weights: [w, 1-2w, w] where w ~ 0.1 for 6th order
    
    # Try several candidate coefficient sets
    candidates = []
    
    # Candidate 1: Equal weights
    candidates.append((a[:], b[:]))
    
    # Candidate 2: Symmetric with emphasized middle
    # w_outer, w_mid for 3 groups of 4
    for w_out in [0.05, 0.1, 0.15, 0.2]:
        w_mid = (1 - 2 * w_out) / 3
        if w_mid <= 0 or w_out <= 0:
            continue
        # 3 groups, each with 4 stages
        # Group structure: each group has 4 A-B pairs
        # Within each group, use Strang-like: [0.25, 0.5, 0.5, 0.25] for A, [0.5, 0.25, 0.25, 0.5] for B (adjusted)
        # Actually, for each group of 4, use symmetric 4-stage
        a_cand = []
        b_cand = []
        for gi in range(3):
            gw = [w_out, w_mid, w_out][gi] if gi != 1 else w_mid
            # Actually: group 0 weight w_out, group 1 weight w_mid, group 2 weight w_out
            gw = [w_out, w_mid, w_out][gi]
            # Within group, 4 stages with local symmetry
            local_a = [0.25, 0.5, 0.5, 0.25]
            local_b = [0.5, 0.25, 0.25, 0.5]
            for la, lb in zip(local_a, local_b):
                a_cand.append(gw * la)
                b_cand.append(gw * lb)
        # Normalize to ensure sums are 1
        sa = sum(a_cand)
        sb = sum(b_cand)
        if abs(sa) > 1e-12:
            a_cand = [x / sa for x in a_cand]
        if abs(sb) > 1e-12:
            b_cand = [x / sb for x in b_cand]
        candidates.append((a_cand, b_cand))
    
    # Candidate 3: Use known 6th-order Suzuki coefficients pattern
    # For 12 stages, use symmetric pattern
    # a = [c1, c2, c3, c4, c4, c3, c3, c4, c4, c3, c2, c1]
    # b = [d1, d2, d3, d3, d4, d5, d5, d4, d3, d3, d2, d1]
    # with constraints sum(a)=1, sum(b)=1
    
    # Try a simple symmetric high-order pattern
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25]:
        # a: [alpha, alpha, 0.5-2*alpha, 0.5-2*alpha, 0.5-2*alpha, 0.5-2*alpha, 
        #            0.5-2*alpha, 0.5-2*alpha, 0.5-2*alpha, 0.5-2*alpha, alpha, alpha]
        # That doesn't sum right. Let me be more careful.
        
        # Symmetric a: a1=a12, a2=a11, a3=a10, a4=a9, a5=a8, a6=a7
        # sum = 2*(a1+a2+a3+a4+a5+a6) = 1, so a1+a2+a3+a4+a5+a6 = 0.5
        a_sym = [alpha, alpha, 0.5 - 2*alpha, 0.5 - 2*alpha, 0.5 - 2*alpha, 0.5 - 2*alpha]
        # Check sum
        s = 2 * sum(a_sym)
        if abs(s - 1) > 1e-10 or any(x < 0 for x in a_sym):
            continue
        a_cand = a_sym + a_sym[::-1]
        
        # b: symmetric, sum to 1
        b_sym = [0.5 - 2*alpha, 0.5 - 2*alpha, alpha, alpha, 0.5 - 2*alpha, 0.5 - 2*alpha]
        s = 2 * sum(b_sym)
        if abs(s - 1) > 1e-10 or any(x < 0 for x in b_sym):
            continue
        b_cand = b_sym + b_sym[::-1]
        
        candidates.append((a_cand, b_cand))
    
    # Evaluate all candidates and pick the best
    best_score = float('inf')
    best_a, best_b = a, b
    
    for a_cand, b_cand in candidates:
        if time.time() - _start_time > _TIMEOUT - 10:
            break
        try:
            score = _score_coefficients(a_cand, b_cand, A, B)
            if score < best_score:
                best_score = score
                best_a, best_b = a_cand, b_cand
        except:
            continue
    
    # If time remains, try a local optimization
    if time.time() - _start_time < _TIMEOUT - 20:
        # Simple gradient-free local optimization
        best_a, best_b = best_a[:], best_b[:]
        for it in range(50):
            if time.time() - _start_time > _TIMEOUT - 10:
                break
            # Perturb one coefficient at a time
            improved = False
            for i in range(12):
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    a_test = best_a[:]
                    b_test = best_b[:]
                    a_test[i] += delta
                    # Adjust to maintain sum = 1: add -delta to a symmetric partner
                    partner = 11 - i
                    if partner != i:
                        a_test[partner] -= delta
                    else:
                        a_test[(i+1) % 12] -= delta
                    # Clip to valid range
                    if any(x < -50 or x > 50 for x in a_test):
                        continue
                    try:
                        score = _score_coefficients(a_test, b_test, A, B)
                        if score < best_score - 1e-12:
                            best_score = score
                            best_a, best_b = a_test, b_test
                            improved = True
                            break
                    except:
                        continue
            if not improved:
                break
    
    # Final validation: ensure constraints are met
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    # Renormalize to ensure exact sum = 1
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-12:
        best_a = [x / sa for x in best_a]
    if abs(sb) > 1e-12:
        best_b = [x / sb for x in best_b]
    
    return best_a, best_b
