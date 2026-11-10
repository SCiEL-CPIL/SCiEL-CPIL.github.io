"""Frequency-domain inverse design for 4-5 TM band gap in 32x32 square photonic crystal."""
import time
import numpy as np

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

# --- 4-fold symmetry basis ---
# Square lattice, cell centre at (16,16) in 0..31 coords.
# 4-fold group: identity, 90-deg rotation, 180-deg, 270-deg.
# A 4-fold invariant cell is determined by its values on an orbit of size 4 (or 1/2 on axes).
# We parameterize the cell as a sum of 4-fold-symmetric basis functions:
#   B_k(x,y) = cos(2*pi*kx*x/L)*cos(2*pi*ky*y/L)  with (kx,ky) in a 4-fold-symmetric set
# This automatically gives 4-fold symmetry about the centre.

L = 32  # cell size

# Precompute pixel coordinates (centre at (L/2, L/2))
_x = np.arange(L, dtype=float) - L / 2.0 + 0.5
_y = np.arange(L, dtype=float) - L / 2.0 + 0.5
_X, _Y = np.meshgrid(_x, _y)  # _X[i,j] = x of pixel (i,j)

# 4-fold-symmetric basis: use cos-cos modes that are invariant under 90-deg rotation.
# Under 90-deg rotation about centre: (x,y) -> (-y, x).
# cos(2*pi*kx*x/L)*cos(2*pi*ky*y/L) -> cos(2*pi*kx*(-y)/L)*cos(2*pi*ky*x/L)
#   = cos(2*pi*ky*x/L)*cos(2*pi*kx*y/L)
# So the mode (kx,ky) maps to (ky,kx). For invariance we need (kx,ky) ~ (ky,kx),
# i.e. we sum over the orbit {(kx,ky),(ky,kx),(-kx,ky),...} but cos is even so:
#   B(kx,ky) = cos(2*pi*kx*x/L)*cos(2*pi*ky*y/L) + cos(2*pi*ky*x/L)*cos(2*pi*kx*y/L)
# is 4-fold symmetric (check: under 90-deg, x->-y, y->x:
#   first term -> cos(2*pi*kx*(-y)/L)*cos(2*pi*ky*x/L) = cos(2*pi*ky*x/L)*cos(2*pi*kx*y/L) = second term
#   second term -> cos(2*pi*ky*(-y)/L)*cos(2*pi*kx*x/L) = cos(2*pi*kx*x/L)*cos(2*pi*ky*y/L) = first term
#   sum is invariant. Good.)
# Also need invariance under reflection x->-x (part of square group): cos is even in x, so yes.

# Choose a set of (kx,ky) pairs. Use small integers for smoothness.
_kx_list = [0, 1, 2, 3]
_ky_list = [0, 1, 2, 3]
# Build basis: for each (kx,ky) with kx<=ky to avoid duplication, but we'll just include all and let optimizer handle it.
# Actually simpler: include all (kx,ky) in a small grid, the basis function is the symmetrized version.
# To keep parameter count small, use a truncated set.

# Basis functions: B_m(x,y) for m indexing (kx,ky) pairs
_basis_kx = []
_basis_ky = []
for kx in range(0, 4):
    for ky in range(0, 4):
        # Only include if this is the "first" in its orbit to reduce redundancy? 
        # No - just include all; the optimizer will find the right combination.
        # But to save parameters, note B(kx,ky) = B(ky,kx) by construction? No, they're different basis functions.
        # Let's just use a small set: (0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,2),(1,2),(2,1),(3,0),(0,3),(3,3),(2,3),(3,2)
        _basis_kx.append(kx)
        _basis_ky.append(ky)

N_basis = len(_basis_kx)

# Precompute basis values at each pixel: _B[m, i, j]
_B = np.zeros((N_basis, L, L), dtype=float)
for m, (kx, ky) in enumerate(zip(_basis_kx, _basis_ky)):
    if kx == ky:
        _B[m] = np.cos(2 * np.pi * kx * _X / L) * np.cos(2 * np.pi * ky * _Y / L)
    else:
        _B[m] = (np.cos(2 * np.pi * kx * _X / L) * np.cos(2 * np.pi * ky * _Y / L) +
                 np.cos(2 * np.pi * ky * _X / L) * np.cos(2 * np.pi * kx * _Y / L))

# --- Fast surrogate band-structure computation ---
# TM mode: eigenvalue problem  (k+G)^2 / eps(k+G)  ->  use 1/eps Fourier coefficients.
# We use a small set of G-vectors for speed.

# G-vectors: (gx, gy) in units of 2*pi/a. Use a 3x3 grid: gx,gy in {-1,0,1}
_G_list = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
N_G = len(_G_list)

# k-points in irreducible BZ (30 points as per evaluator, but we use a subset for speed)
# Irreducible BZ: 0<=kx<=ky, kx+ky<=0.5 (in units of 2*pi/a)
_k_points = []
# Gamma-X: (t,0) t in [0,0.5]
for t in np.linspace(0, 0.5, 6):
    _k_points.append((t, 0.0))
# X-M: (0.5, t) t in [0,0.5]
for t in np.linspace(0, 0.5, 6):
    _k_points.append((0.5, t))
# Gamma-M: (t,t) t in [0,0.25]
for t in np.linspace(0, 0.25, 6):
    _k_points.append((t, t))
# Add a few interior points
_k_points.append((0.2, 0.3))
_k_points.append((0.15, 0.35))
_k_points.append((0.1, 0.4))
_k_points.append((0.25, 0.25))
_k_points.append((0.3, 0.2))

N_k = len(_k_points)

def _compute_fourier_inv_eps(cell_float):
    """Compute Fourier coefficients of 1/eps for the given 32x32 float cell (values in [0,1] mapped to eps).
    Returns a 2D array indexed by (i,j) for _G_list[i]-_G_list[j] differences, but we store as a dict or 2D.
    Actually we need eps_inv[G] for G in a larger set. Let's compute all needed G differences.
    """
    # cell_float is L x L, values in [0,1] where 0=air(eps=1), 1=dielectric(eps=12)
    eps = 1.0 + 11.0 * cell_float  # linear interpolation
    inv_eps = 1.0 / eps
    
    # Compute Fourier coefficients for all G in _G_list
    # F[G] = sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(Gx*i + Gy*j)/L) / L^2
    F = np.zeros(N_G, dtype=complex)
    for gi, (gx, gy) in enumerate(_G_list):
        phase_x = np.exp(-2j * np.pi * gx * np.arange(L) / L)
        phase_y = np.exp(-2j * np.pi * gy * np.arange(L) / L)
        phase = np.outer(phase_x, phase_y)
        F[gi] = np.sum(inv_eps * phase) / (L * L)
    return F

def _surrogate_gap(params, cell_float=None):
    """Compute the surrogate 4-5 TM band gap-midgap ratio.
    params: 1D array of length N_basis, coefficients for the basis expansion.
    Returns the gap ratio (can be negative).
    """
    if cell_float is None:
        # cell = sigmoid(sum_m params[m] * B[m]) mapped to [0,1]
        # Use a scaling to keep values reasonable
        field = np.tensordot(params, _B, axes=(0,0))  # L x L
        cell_float = 1.0 / (1.0 + np.exp(-10.0 * (field - 0.5)))
    
    F = _compute_fourier_inv_eps(cell_float)
    
    # Build the 9x9 matrix at each k-point and find bands 4,5 (0-indexed 3,4)
    band4_max = 0.0
    band5_min = float('inf')
    
    for kx, ky in _k_points:
        A = np.zeros((N_G, N_G), dtype=complex)
        for i in range(N_G):
            gx_i, gy_i = _G_list[i]
            kx_i = kx + gx_i
            ky_i = ky + gy_i
            k2_i = kx_i * kx_i + ky_i * ky_i
            for j in range(N_G):
                gx_j, gy_j = _G_list[j]
                dgx = gx_i - gx_j
                dgy = gy_i - gy_j
                # Find index for (dgx, dgy) in _G_list
                found = False
                for idx in range(N_G):
                    if _G_list[idx][0] == dgx and _G_list[idx][1] == dgy:
                        A[i, j] = k2_i * F[idx]
                        found = True
                        break
                if not found:
                    A[i, j] = 0.0
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w4 > band4_max:
                    band4_max = w4
                if w5 < band5_min:
                    band5_min = w5
        except:
            continue
    
    if band5_min == float('inf') or band4_max <= 0:
        return 0.0
    
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap <= 0:
        return 0.0
    return gap / midgap

def _params_to_cell(params):
    """Convert parameter vector to a 32x32 float cell in [0,1]."""
    field = np.tensordot(params, _B, axes=(0, 0))
    return 1.0 / (1.0 + np.exp(-10.0 * (field - 0.5)))

def _cell_to_binary(cell_float):
    """Threshold float cell to 0/1."""
    return (cell_float > 0.5).astype(int).tolist()

def _gradient_ascent(params0, lr, n_steps, deadline):
    """Simple gradient ascent using finite differences (Jacobian-free)."""
    params = params0.copy()
    best_params = params.copy()
    best_score = _surrogate_gap(params)
    
    # Compute gradient via central differences
    for step in range(n_steps):
        if _elapsed() > deadline:
            break
        
        score = _surrogate_gap(params)
        if score > best_score:
            best_score = score
            best_params = params.copy()
        
        # Gradient estimation
        grad = np.zeros(N_basis)
        h = 0.01
        for m in range(N_basis):
            if _elapsed() > deadline:
                break
            p_plus = params.copy()
            p_plus[m] += h
            p_minus = params.copy()
            p_minus[m] -= h
            s_plus = _surrogate_gap(p_plus)
            s_minus = _surrogate_gap(p_minus)
            grad[m] = (s_plus - s_minus) / (2 * h)
        
        # Update
        params = params + lr * grad
        
        # Adaptive learning rate: reduce if score decreased
        new_score = _surrogate_gap(params)
        if new_score < best_score - 0.01:
            lr *= 0.8
            params = best_params.copy()
    
    return best_params, best_score

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Deadline: 200s wall-clock, use 190s for safety
    deadline = 190.0
    
    best_cell = None
    best_score = -1.0
    
    # Multi-start gradient ascent
    n_starts = 5
    for start in range(n_starts):
        if _elapsed() > deadline - 10:
            break
        
        if start == 0:
            # Start from a disk-like configuration: large (0,0) coefficient
            params0 = np.zeros(N_basis)
            params0[0] = 1.5  # (0,0) mode
        elif start == 1:
            # Start from a ring-like configuration
            params0 = np.zeros(N_basis)
            params0[0] = 0.5
            params0[1] = -0.5  # (1,0) mode
            params0[3] = -0.5  # (1,1) mode
        elif start == 2:
            # Start from a cross pattern
            params0 = np.zeros(N_basis)
            params0[0] = 0.3
            params0[2] = 0.8  # (0,2) mode
            params0[5] = 0.8  # (2,0) mode
        elif start == 3:
            # Start from a square-like pattern
            params0 = np.zeros(N_basis)
            params0[0] = 0.8
            params0[4] = -0.3  # (2,0)
            params0[6] = -0.3  # (0,2)
        else:
            # Random start
            np.random.seed(start * 100)
            params0 = np.random.randn(N_basis) * 0.5
        
        # Scale parameters to reasonable range
        params0 = np.clip(params0, -2, 2)
        
        # Gradient ascent with decreasing steps
        lr = 0.05
        n_steps = 50
        params, score = _gradient_ascent(params0, lr, n_steps, deadline - 5)
        
        if score > best_score:
            best_score = score
            best_cell = _cell_to_binary(_params_to_cell(params))
    
    # If no valid cell found, fall back to a simple disk
    if best_cell is None:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 * 0.2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        best_cell = cell
    
    # Verify symmetry (should be automatic, but ensure)
    # The basis is 4-fold symmetric, so the cell should be. But sigmoid thresholding might break it slightly.
    # Enforce exact symmetry by averaging over the 4-fold group.
    arr = np.array(best_cell, dtype=float)
    # 90-deg rotation about centre: new[i,j] = old[L-1-j, i]
    rot90 = np.rot90(arr, k=1)  # This rotates 90 deg CCW
    rot180 = np.rot90(arr, k=2)
    rot270 = np.rot90(arr, k=3)
    # Average
    sym = (arr + rot90 + rot180 + rot270) / 4.0
    # Threshold
    best_cell = (sym > 0.5).astype(int).tolist()
    
    return best_cell
