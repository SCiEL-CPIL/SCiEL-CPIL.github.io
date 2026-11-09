"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Strategy: Direct numerical optimization of 12-stage product formula coefficients.
- Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain.
- Cache eigen-decompositions for fast matrix exponential evaluation.
- Use Nelder-Mead to minimize mean spectral-norm error at t = 0.25, 0.5, 1.0.
- Multiple random restarts to escape local minima.
- Deadline guard: return best found before 140s wall-clock.
"""
import numpy as np
import time
import random

_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
def _build_hamiltonians():
    N = 6
    delta = 0.8
    field = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)

    def bond_op(i, j):
        ops_sz = [I] * N; ops_sz[i] = sz; ops_sz[j] = sz
        ops_sx = [I] * N; ops_sx[i] = sx; ops_sx[j] = sx
        ops_sy = [I] * N; ops_sy[i] = sy; ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))

    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)   # even bonds
    A = bond_op(1, 2) + bond_op(3, 4)                   # odd bonds

    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)

    A = A + field_op
    B = B + field_op
    return A, B

# ---------------------------------------------------------------------------
# Cached eigen-decompositions and fast error
# ---------------------------------------------------------------------------
class _EigCache:
    def __init__(self, A, B):
        self.dim = A.shape[0]
        self.I = np.eye(self.dim, dtype=complex)
        self.H = A + B
        self.eig_H, self.vec_H = np.linalg.eigh(self.H)
        self.eig_A, self.vec_A = np.linalg.eigh(A)
        self.eig_B, self.vec_B = np.linalg.eigh(B)
        self.times = [0.25, 0.5, 1.0]
        self.expH = {}
        self.expA = {}
        self.expB = {}
        for t in self.times:
            self.expH[t] = (self.vec_H * np.exp(-1j * t * self.eig_H)) @ self.vec_H.conj().T
            self.expA[t] = (self.vec_A * np.exp(-1j * t * self.eig_A)) @ self.vec_A.conj().T
            self.expB[t] = (self.vec_B * np.exp(-1j * t * self.eig_B)) @ self.vec_B.conj().T

    def error(self, a, b):
        err_sum = 0.0
        for t in self.times:
            U = self.I.copy()
            for k in range(11, -1, -1):
                eb = (self.vec_B * np.exp(-1j * b[k] * t * self.eig_B)) @ self.vec_B.conj().T
                ea = (self.vec_A * np.exp(-1j * a[k] * t * self.eig_A)) @ self.vec_A.conj().T
                U = eb @ ea @ U
            err_sum += np.linalg.norm(U - self.expH[t], ord=2)
        return err_sum / len(self.times)

# ---------------------------------------------------------------------------
# Nelder-Mead optimizer
# ---------------------------------------------------------------------------
def _nelder_mead(objective, x0, max_iter=200, tol=1e-10):
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        step = 0.1 if abs(xi[i]) < 1e-8 else 0.1 * np.sign(xi[i])
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(x) for x in simplex]

    for it in range(max_iter):
        if _remaining() < 20:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[-1] - fvals[0] < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        x_ref = centroid + 1.0 * (centroid - simplex[n])
        f_ref = objective(x_ref)

        if f_ref < fvals[0]:
            x_exp = centroid + 2.0 * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp; fvals[n] = f_exp
            else:
                simplex[n] = x_ref; fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            simplex[n] = x_ref; fvals[n] = f_ref
        else:
            x_con = centroid + 0.5 * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con; fvals[n] = f_con
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    best_idx = int(np.argmin(fvals))
    return simplex[best_idx], fvals[best_idx]

# ---------------------------------------------------------------------------
# Initial coefficient generators
# ---------------------------------------------------------------------------
def _st4_coeffs():
    """4th-order Suzuki-Trotter mapped to 12 (B,A) pairs."""
    r = 2.0 ** (1.0/3.0)
    alpha1 = 1.0 / (4.0 - r)
    alpha2 = -r / (4.0 - r)
    alpha3 = 1.0 - 2.0 * alpha1
    beta1 = 1.0 / (2.0 - r)
    beta2 = 1.0 - 2.0 * beta1
    
    # Map ST4 to 12 stages: 4 substeps x 3 (B,A) pairs each
    # Per substep: B=[beta1/4, beta2/4, beta1/4], A=[alpha1/4, alpha2/4, alpha3/4]
    a = []
    b = []
    for _ in range(4):
        b.extend([beta1/4, beta2/4, beta1/4])
        a.extend([alpha1/4, alpha2/4, alpha3/4])
    
    # Normalize to ensure sums = 1
    sa = sum(a); sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x/sa for x in a]
    if abs(sb) > 1e-15:
        b = [x/sb for x in b]
    return a, b

def _strang_coeffs():
    """2nd-order Strang: 6 substeps of (B/2, A, B/2) mapped to (B,A) pairs."""
    a = []
    b = []
    for _ in range(6):
        b.append(1.0/12)
        a.append(1.0/6)
        b.append(1.0/12)
        a.append(0.0)
    # Adjust: we need exactly 12 a's and 12 b's
    a = [1.0/6]*6 + [0.0]*6
    b = [1.0/12]*12
    return a, b

def _random_coeffs():
    """Random coefficients with sums normalized to 1."""
    a = [random.uniform(-0.5, 1.5) for _ in range(12)]
    b = [random.uniform(-0.5, 1.5) for _ in range(12)]
    sa = sum(a); sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x/sa for x in a]
    if abs(sb) > 1e-15:
        b = [x/sb for x in b]
    return a, b

# ---------------------------------------------------------------------------
# Main coefficient function
# ---------------------------------------------------------------------------
def coefficients(stages):
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        a = a[:stages]; b = b[:stages]
        return a, b

    A, B = _build_hamiltonians()
    cache = _EigCache(A, B)

    def objective(a, b):
        return cache.error(a, b)

    # Generate initial candidates
    candidates = []
    
    # ST4-based
    a_st4, b_st4 = _st4_coeffs()
    candidates.append((a_st4, b_st4))
    
    # Strang-based
    a_str, b_str = _strang_coeffs()
    candidates.append((a_str, b_str))
    
    # Random restarts
    random.seed(42)
    for _ in range(5):
        a_r, b_r = _random_coeffs()
        candidates.append((a_r, b_r))

    # Evaluate all candidates
    best_a, best_b = None, None
    best_err = float('inf')
    for a, b in candidates:
        err = objective(a, b)
        if err < best_err:
            best_err = err
            best_a, best_b = a[:], b[:]

    # Nelder-Mead refinement on the best candidate
    if _remaining() > 30:
        x0 = best_a + best_b  # 24 parameters
        def obj_vec(x):
            a = x[:12]
            b = x[12:]
            return objective(a, b)
        
        best_params, best_f = _nelder_mead(obj_vec, x0, max_iter=300)
        a_opt = best_params[:12]
        b_opt = best_params[12:]
        err_opt = objective(a_opt, b_opt)
        if err_opt < best_err:
            best_err = err_opt
            best_a, best_b = a_opt, b_opt

    # Final normalization
    sa = sum(best_a); sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    return best_a, best_b
