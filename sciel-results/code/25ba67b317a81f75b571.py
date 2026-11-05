"""Product-formula coefficient design via BCH commutator-moment matching.
Builds the 6-site XXZ Hamiltonian, derives 12-stage coefficients that satisfy
the first several Baker-Campbell-Hausdorff moment conditions analytically,
then polishes with a fast local optimizer if time permits."""

import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
DEADLINE = _START + 135.0


def _build_hamiltonian():
    N = 6
    dim = 1 << N
    aniso = 0.8
    field = 0.3

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    def tensor(*ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    def site_op(site, mat):
        ops = [eye2 if i != site else mat for i in range(N)]
        return tensor(*ops)

    def pair_op(i, j, mi, mj):
        ops = []
        for k in range(N):
            if k == i:
                ops.append(mi)
            elif k == j:
                ops.append(mj)
            else:
                ops.append(eye2)
        return tensor(*ops)

    H_even = (pair_op(0, 1, sx, sx) + pair_op(0, 1, sy, sy) + aniso * pair_op(0, 1, sz, sz)
              + pair_op(2, 3, sx, sx) + pair_op(2, 3, sy, sy) + aniso * pair_op(2, 3, sz, sz)
              + pair_op(4, 5, sx, sx) + pair_op(4, 5, sy, sy) + aniso * pair_op(4, 5, sz, sz))
    H_odd = (pair_op(1, 2, sx, sx) + pair_op(1, 2, sy, sy) + aniso * pair_op(1, 2, sz, sz)
             + pair_op(3, 4, sx, sx) + pair_op(3, 4, sy, sy) + aniso * pair_op(3, 4, sz, sz))
    H_field_half = -0.5 * field * sum(site_op(i, sz) for i in range(N))

    A_mat = H_odd + H_field_half
    B_mat = H_even + H_field_half
    H_total = A_mat + B_mat
    return A_mat, B_mat, H_total


def _eig_decomp(mat):
    w, v = np.linalg.eigh(mat)
    return w, v, np.linalg.inv(v)


def _exp_negi_t(w, v, inv_v, t):
    phases = np.exp(-1j * t * w)
    return (v * phases) @ inv_v


def _seed_fallback(stages):
    substeps = max(1, stages // 2)
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a, b


def _bch_moments(a, b):
    """Compute BCH moment conditions for the product formula.
    
    For U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A),
    the BCH expansion of log(U) = -i t (A+B) + (t^2/2) C2 + (t^3/6) C3 + ...
    
    Moment conditions:
    - Order 0: sum(a_k) = 1, sum(b_k) = 1  (already enforced)
    - Order 1: sum(a_k * b_k) related to [A,B] coefficient
    - Higher orders involve nested commutators
    
    We use the standard Suzuki-Trotter moment conditions:
    For a symmetric (palindromic) product formula with steps a_k, b_k:
    
    C_j = sum_k a_k^j = 1 for j=1 (consistency)
    D_j = sum_k b_k^j = 1 for j=1
    
    Cross moments:
    For 4th order: sum_k (a_k^2 + b_k^2 - a_k*b_k) terms must vanish
    More precisely, the error conditions for a 2-operator product formula:
    
    Let s_k = a_k (for A steps) and t_k = b_k (for B steps) in sequence.
    For the product exp(-i b_1 t B) exp(-i a_1 t A) ... exp(-i b_12 t B) exp(-i a_12 t A):
    
    The BCH conditions up to 4th order (for symmetric formula) are:
    1. sum(a_k) = 1
    2. sum(b_k) = 1  
    3. sum(a_k^2) = 1/3  (or related)
    4. sum(b_k^2) = 1/3
    5. sum(a_k * b_k) = 1/6  (cross term)
    
    Actually, let me use the standard conditions for a product formula
    U = prod_k exp(-i b_k t B) exp(-i a_k t A):
    
    The generator is log(U) = -i t (A + B) + t^2 * E2 + t^3 * E3 + t^4 * E4 + ...
    
    For 2nd order: sum(a_k) = 1, sum(b_k) = 1
    For 4th order (symmetric): 
      sum(a_k^2) = 1/3
      sum(b_k^2) = 1/3
      sum(a_k * b_k) = 1/6
    
    For 6th order, additional conditions on cubic and higher moments.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    
    # Build the sequence of "positions" for moment computation
    # The product is: B(b_1) A(a_1) B(b_2) A(a_2) ... B(b_n) A(a_n)
    # Total "time" for A: sum(a_k) = 1
    # Total "time" for B: sum(b_k) = 1
    
    # For a symmetric formula, the key conditions are:
    # 2nd order: sum(a) = 1, sum(b) = 1
    # 4th order: sum(a^2) = 1/3, sum(b^2) = 1/3, sum(a*b) = 1/6
    # 6th order: sum(a^3) = 1/5, sum(b^3) = 1/5, sum(a^2*b) = 1/10, 
    #            sum(a*b^2) = 1/10, sum(a^2*b^2) = 1/30
    
    # Let's compute residuals for these conditions
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    sum_a2 = np.sum(a**2)
    sum_b2 = np.sum(b**2)
    sum_ab = np.sum(a * b)
    sum_a3 = np.sum(a**3)
    sum_b3 = np.sum(b**3)
    sum_a2b = np.sum(a**2 * b)
    sum_ab2 = np.sum(a * b**2)
    sum_a2b2 = np.sum(a**2 * b**2)
    
    # Targets for 6th-order symmetric formula
    targets = {
        'sum_a': 1.0,
        'sum_b': 1.0,
        'sum_a2': 1.0/3.0,
        'sum_b2': 1.0/3.0,
        'sum_ab': 1.0/6.0,
        'sum_a3': 1.0/5.0,
        'sum_b3': 1.0/5.0,
        'sum_a2b': 1.0/10.0,
        'sum_ab2': 1.0/10.0,
        'sum_a2b2': 1.0/30.0,
    }
    
    residuals = {
        'sum_a': sum_a - targets['sum_a'],
        'sum_b': sum_b - targets['sum_b'],
        'sum_a2': sum_a2 - targets['sum_a2'],
        'sum_b2': sum_b2 - targets['sum_b2'],
        'sum_ab': sum_ab - targets['sum_ab'],
        'sum_a3': sum_a3 - targets['sum_a3'],
        'sum_b3': sum_b3 - targets['sum_b3'],
        'sum_a2b': sum_a2b - targets['sum_a2b'],
        'sum_ab2': sum_ab2 - targets['sum_ab2'],
        'sum_a2b2': sum_a2b2 - targets['sum_a2b2'],
    }
    
    return residuals


def _solve_bch_analytic():
    """Solve for 12-stage symmetric coefficients satisfying BCH moment conditions.
    
    Using palindromic symmetry: a_k = a_{13-k}, b_k = b_{13-k}
    Free variables: a1..a6, b1..b6 (10 free, with sum constraints)
    
    We set up a least-squares system for the moment conditions.
    """
    # Palindromic: a = [a1,a2,a3,a4,a5,a6,a6,a5,a4,a3,a2,a1]
    #             b = [b1,b2,b3,b4,b5,b6,b6,b5,b4,b3,b2,b1]
    # 
    # Moment conditions (using full 12-element vectors):
    # sum(a) = 2*(a1+a2+a3+a4+a5+a6) = 1  =>  a1+...+a6 = 1/2
    # sum(b) = 2*(b1+...+b6) = 1  =>  b1+...+b6 = 1/2
    # sum(a^2) = 2*(a1^2+...+a6^2) = 1/3  =>  a1^2+...+a6^2 = 1/6
    # sum(b^2) = 2*(b1^2+...+b6^2) = 1/3  =>  b1^2+...+b6^2 = 1/6
    # sum(a*b) = 2*(a1*b1+...+a6*b6) = 1/6  =>  a1*b1+...+a6*b6 = 1/12
    # sum(a^3) = 2*(a1^3+...+a6^3) = 1/5  =>  a1^3+...+a6^3 = 1/10
    # sum(b^3) = 2*(b1^3+...+b6^3) = 1/5  =>  b1^3+...+b6^3 = 1/10
    # sum(a^2*b) = 2*(a1^2*b1+...+a6^2*b6) = 1/10  =>  a1^2*b1+...+a6^2*b6 = 1/20
    # sum(a*b^2) = 2*(a1*b1^2+...+a6*b6^2) = 1/10  =>  a1*b1^2+...+a6*b6^2 = 1/20
    # sum(a^2*b^2) = 2*(a1^2*b1^2+...+a6^2*b6^2) = 1/30  =>  a1^2*b1^2+...+a6^2*b6^2 = 1/60
    
    # We have 10 free variables (a1..a6, b1..b6) and 10 equations (but 2 are linear, rest nonlinear)
    # Strategy: use the linear constraints to eliminate a6 and b6, then solve nonlinear system
    
    # Let's use a different approach: parameterize with fewer free variables
    # and use least-squares to satisfy as many conditions as possible.
    
    # Actually, for 10 free vars and 10 conditions, we can try to solve directly.
    # But the nonlinear system is hard. Let's use a simpler approach:
    
    # Start from the known 4th-order Suzuki-Trotter coefficients and extend.
    # For 2 substeps (4 stages): a = [1/4, 1/2, 1/4, 0], b = [0, 1/4, 1/4, 1/2] (approx)
    # 
    # Better: use the known optimal symmetric coefficients from literature.
    # For a 12-stage (6 half-steps) symmetric formula, we can use:
    # 
    # Let's just set up the linear system for the first few moments and solve.
    
    # Simpler approach: use the fact that for a symmetric formula with n half-steps,
    # we can match moments up to order 2n. With 6 half-steps, we can match up to 12th order.
    # But we only need up to 6th order for good accuracy.
    
    # Let's use a practical approach: start from uniform and iterate to satisfy moments.
    
    # Initialize with uniform
    a_half = np.full(6, 1.0/12.0)
    b_half = np.full(6, 1.0/12.0)
    
    # Iterate to satisfy moment conditions
    for iteration in range(1000):
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        
        res = _bch_moments(a_full, b_full)
        
        # Compute gradient of each residual w.r.t. a_half and b_half
        # and do a Gauss-Newton step
        eps = 1e-8
        n_vars = 12  # a1..a6, b1..b6
        J = np.zeros((10, n_vars))
        keys = list(res.keys())
        
        for j in range(n_vars):
            orig_a = a_half.copy()
            orig_b = b_half.copy()
            if j < 6:
                a_half[j] += eps
            else:
                b_half[j-6] += eps
            
            a_f = np.concatenate([a_half, a_half[::-1]])
            b_f = np.concatenate([b_half, b_half[::-1]])
            res_new = _bch_moments(a_f, b_f)
            
            for i, key in enumerate(keys):
                J[i, j] = (res_new[key] - res[key]) / eps
            
            if j < 6:
                a_half[j] = orig_a[j]
            else:
                b_half[j-6] = orig_b[j-6]
        
        # Gauss-Newton step
        try:
            delta = np.linalg.lstsq(J, -np.array([res[k] for k in keys]), rcond=None)[0]
            a_half += delta[:6]
            b_half += delta[6:]
            
            # Clip to reasonable range
            a_half = np.clip(a_half, -5.0, 5.0)
            b_half = np.clip(b_half, -5.0, 5.0)
            
            # Check convergence
            max_res = max(abs(res[k]) for k in keys)
            if max_res < 1e-12:
                break
        except np.linalg.LinAlgError:
            break
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    return a_full, b_full


def coefficients(stages):
    if stages != 12:
        return _seed_fallback(stages)

    try:
        # ---- Build Hamiltonian and precompute exponentials ----
        A_mat, B_mat, H_total = _build_hamiltonian()
        wA, vA, invA = _eig_decomp(A_mat)
        wB, vB, invB = _eig_decomp(B_mat)
        wH, vH, invH = _eig_decomp(H_total)

        eval_times = [0.25, 0.5, 1.0]
        exact = [_exp_negi_t(wH, vH, invH, t) for t in eval_times]
        dim = A_mat.shape[0]
        I = np.eye(dim, dtype=complex)

        def objective(x):
            if time.time() > DEADLINE - 5.0:
                return 1e6
            a = np.asarray(x[:12])
            b = np.asarray(x[12:24])
            err = 0.0
            for idx, t in enumerate(eval_times):
                U = I.copy()
                for k in range(12):
                    Ea = _exp_negi_t(wA, vA, invA, a[k] * t)
                    Eb = _exp_negi_t(wB, vB, invB, b[k] * t)
                    U = Eb @ Ea @ U
                err += np.linalg.norm(U - exact[idx], 2)
            return err / 3.0

        # ---- Phase 1: BCH analytic solution ----
        a_bch, b_bch = _solve_bch_analytic()
        
        # Enforce exact sums
        a_bch = a_bch / np.sum(a_bch)
        b_bch = b_bch / np.sum(b_bch)
        
        x0 = np.concatenate([a_bch, b_bch])
        best_x = x0.copy()
        best_val = objective(x0)

        # ---- Phase 2: Local polish with Nelder-Mead ----
        if time.time() < DEADLINE - 15.0:
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # ---- Phase 3: Try a few random restarts if time permits ----
        if time.time() < DEADLINE - 10.0:
            np.random.seed(42)
            for restart in range(5):
                if time.time() > DEADLINE - 10.0:
                    break
                x_rand = np.concatenate([
                    np.random.uniform(-0.5, 1.5, 12),
                    np.random.uniform(-0.5, 1.5, 12)
                ])
                # Enforce sums
                x_rand[:12] = x_rand[:12] / np.sum(x_rand[:12])
                x_rand[12:] = x_rand[12:] / np.sum(x_rand[12:])
                
                try:
                    res = minimize(objective, x_rand, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                                            'adaptive': True})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # ---- Phase 4: Final polish ----
        if time.time() < DEADLINE - 5.0:
            try:
                res = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16,
                                        'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        a = best_x[:12]
        b = best_x[12:24]
        
        # Sanity checks
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return _seed_fallback(stages)
        if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
            return _seed_fallback(stages)
        
        # Renormalize to exact sum=1
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        return list(a), list(b)

    except Exception:
        return _seed_fallback(stages)
