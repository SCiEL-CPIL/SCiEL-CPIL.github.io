"""Photonic-crystal unit cell design: C4v harmonic-mask + stochastic hill-climbing."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from scipy.linalg import eigh

_DEADLINE = time.time() + 205  # wall-clock safety margin vs 230 s kill

# Precompute pixel coordinates and symmetry orbit indices once.
_N = 32
_PX = (np.arange(_N) + 0.5) / _N - 0.5
_GX, _GY = np.meshgrid(_PX, _PX, indexing='ij')
# Orbit: (i,j) -> (N-1-i, j), (i, N-1-j), (N-1-i, N-1-j)
_ORBIT = np.stack([
    np.arange(_N * _N).reshape(_N, _N),
    (_N - 1 - _GX).astype(int) * _N + _GY.astype(int),
    _GX.astype(int) * _N + (_N - 1 - _GY).astype(int),
    (_N - 1 - _GX).astype(int) * _N + (_N - 1 - _GY).astype(int),
], axis=0)
# Canonical (top-left-most) index per orbit
_CANON = _ORBIT.min(axis=0)
# For each canonical index, the set of orbit members (flattened list)
_orbit_members = {}
for _idx in range(_N * _N):
    _c = _CANON[_idx]
    _orbit_members.setdefault(_c, []).append(_idx)
# Unique canonical indices (upper-left triangle of orbits)
_CANON_UNIQUE = sorted(_orbit_members.keys())
# Map canonical -> list of member indices (for fast symmetric assignment)
_CANON_TO_MEMBERS = {c: np.array(m, dtype=int) for c, m in _orbit_members.items()}

# Precompute harmonic basis values at each pixel for C4v-symmetric functions.
# A C4v-symmetric scalar field can be written as sum of terms:
#   r^p * cos(k*theta) for k in {0,4,8,...} and r^q * sin(k*theta) for k in {4,8,...}
# But to keep C4v (invariant under 90-deg rotation AND reflections), we need
# terms that are invariant under the full C4v group. The simplest invariant
# polynomials in (x,y) under C4v are:
#   x^2 + y^2, x^4 + y^4, x^2*y^2, x^6 + y^6, x^4*y^2 + x^2*y^4, ...
# More generally, any polynomial in u = x^2 + y^2 and v = x^2 - y^2 ... no,
# v is not C4v-invariant (it flips sign under 90-deg rotation).
# C4v-invariant monomials: x^a * y^b where a+b is even AND a ≡ b (mod 4)? 
# Actually under (x,y)->(y,-x) [90 deg rot]: x^a y^b -> y^a (-x)^b = (-1)^b x^b y^a.
# For invariance we need a=b and b even, OR we use combinations.
# Simplest: use basis { (x^2+y^2)^m, (x^4 - 6x^2y^2 + y^4)^n * (x^2+y^2)^k, ... }
# Let's just use: f(x,y) = sum_j a_j * phi_j(x,y) where phi_j are C4v invariants.
# Practical C4v invariants:
#   phi_0 = 1
#   phi_1 = x^2 + y^2
#   phi_2 = x^4 + 6*x^2*y^2 + y^4  (= (x^2+y^2)^2)
#   phi_3 = x^6 + 15*x^4*y^2 + 15*x^2*y^4 + y^6 = (x^2+y^2)^3
#   phi_4 = x^8 + 28*x^6*y^2 + 70*x^4*y^4 + 28*x^2*y^6 + y^8 = (x^2+y^2)^4
#   phi_5 = x^4 - 6*x^2*y^2 + y^4  (this IS C4v invariant: under 90-deg rot, x^4+y^4 unchanged, x^2y^2 unchanged)
#   phi_6 = (x^4 - 6x^2y^2 + y^4)(x^2+y^2)
#   phi_7 = (x^4 - 6x^2y^2 + y^4)^2
#   phi_8 = (x^2+y^2)^5
#   phi_9 = (x^4 - 6x^2y^2 + y^4)(x^2+y^2)^2
# Let me verify phi_5: under (x,y)->(-y,x): (-y)^4 - 6(-y)^2 x^2 + x^4 = y^4 - 6y^2x^2 + x^4 = same. Yes!
# Under (x,y)->(y,x): y^4 - 6y^2x^2 + x^4 = same. Yes!
# So phi_5 is C4v invariant.

def _compute_basis():
    """Compute C4v-invariant basis functions at all pixels."""
    x, y = _GX, _GY
    r2 = x*x + y*y
    # C4v invariants
    basis = [
        np.ones_like(r2),
        r2,
        r2*r2,
        r2*r2*r2,
        r2**4,
        x**4 - 6*x**2*y**2 + y**4,
        (x**4 - 6*x**2*y**2 + y**4) * r2,
        (x**4 - 6*x**2*y**2 + y**4)**2,
        r2**5,
        (x**4 - 6*x**2*y**2 + y**4) * r2 * r2,
        (x**4 - 6*x**2*y**2 + y**4)**2 * r2,
        r2**6,
    ]
    return np.stack(basis, axis=-1)  # shape (32,32,12)

_BASIS = _compute_basis()
_NBASIS = _BASIS.shape[-1]

def _params_to_mask(params, threshold=0.0):
    """Convert parameter vector to a C4v-symmetric binary mask."""
    # params: array of shape (NBASIS,)
    f = _BASIS * params[None, None, :]  # (32,32,NBASIS)
    f = f.sum(axis=-1)  # (32,32)
    # Apply threshold to get binary
    mask = (f > threshold).astype(float)
    # Ensure C4v symmetry by taking the orbit-max (or just enforce via canonical)
    # Actually the basis is already C4v invariant, so mask should be symmetric.
    # But floating point might cause tiny asymmetries; enforce explicitly.
    mask_flat = mask.ravel()
    for c in _CANON_UNIQUE:
        members = _CANON_TO_MEMBERS[c]
        val = 1.0 if mask_flat[c] > 0.5 else 0.0
        for m in members:
            mask_flat[m] = val
    return mask_flat.reshape(_N, _N)

# Fast local band-structure evaluator using a small plane-wave set.
# For TM polarization, the eigenvalue problem is:
#   sum_k' eps_inv(k-k') * |k'|^2 * c_k' = omega^2 * c_k
# where eps_inv is the inverse dielectric function.
# We use a 7x7 = 49 plane wave basis for speed.

_PW_N = 7  # 7x7 = 49 plane waves
_pw_indices = []
for _dx in range(-(_PW_N//2), _PW_N//2 + 1):
    for _dy in range(-(_PW_N//2), _PW_N//2 + 1):
        _pw_indices.append((_dx, _dy))
_PW_NW = len(_pw_indices)
_pw_kx = np.array([p[0] for p in _pw_indices], dtype=float)
_pw_ky = np.array([p[1] for p in _pw_indices], dtype=float)
_pw_k2 = _pw_kx**2 + _pw_ky**2

# Precompute G-difference indices
_G_DIFF = []
for i in range(_PW_NW):
    for j in range(_PW_NW):
        _G_DIFF.append(_pw_indices[i][0] - _pw_indices[j][0], _pw_indices[i][1] - _pw_indices[j][1])
_G_DIFF = np.array(_G_DIFF)

# High-symmetry points in the irreducible BZ (fractional coordinates in units of 2π/a)
# Square lattice: BZ is [-0.5,0.5]^2 in fractional coords. Irreducible: 0 <= ky <= kx, kx in [0,0.5]
# Key points: Gamma(0,0), M(0.5,0), X(0.5,0.5), and some interior points.
_HSP = [
    (0.0, 0.0),       # Gamma
    (0.5, 0.0),       # M (or X depending on convention)
    (0.5, 0.5),       # X
    (0.25, 0.0),
    (0.25, 0.25),
    (0.5, 0.25),
    (0.125, 0.0),
    (0.125, 0.125),
    (0.375, 0.125),
    (0.375, 0.375),
]

def _eps_inv_from_mask(mask_flat):
    """Compute Fourier coefficients of 1/epsilon from the pixel mask.
    For a pixelated dielectric, we approximate eps(x,y) as piecewise constant
    per pixel, then compute its Fourier series.
    """
    # eps(x,y) = eps_low + (eps_high - eps_low) * mask(x,y)
    # For TM: we need 1/eps
    eps_map = 1.0 + 11.0 * mask_flat.reshape(_N, _N)  # eps_low=1, eps_high=12
    inv_eps = 1.0 / eps_map
    
    # Compute Fourier coefficients for the G-vectors we need
    # c_G = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i G . (x_i, y_j))
    # where x_i = (i+0.5)/N - 0.5, etc.
    # For the plane-wave method, we need eps_inv(G_i - G_j) for all pairs.
    
    # Precompute the exponential factors for each pixel and each needed G
    # Needed G's are all differences: _G_DIFF
    # But we only need unique G values
    unique_G = np.unique(_G_DIFF, axis=0)
    nG = len(unique_G)
    
    # exp(-i * G . r) for each pixel and each G
    # r_i = (x_i, y_j)
    rx = _PX[None, :]  # (1, N)
    ry = _PX[:, None]  # (N, 1)
    # Actually let's do it more efficiently
    # For each G=(gx,gy): exp(-i*gx*x_i) * exp(-i*gy*y_j)
    # We can separate: E[i,j,G] = exp(-i*gx*x_i) * exp(-i*gy*y_j)
    
    # Precompute for each unique G
    # This might be memory-heavy; let's compute on the fly per evaluation
    
    # For speed, precompute the pixel positions
    px = _PX  # (N,)
    py = _PX  # (N,)
    
    # inv_eps_flat = inv_eps.ravel()  # (N*N,)
    # For each G: c_G = mean(inv_eps * exp(-i G . r))
    
    # Let's precompute exp(-i * gx * px) and exp(-i * gy * py) for all needed gx, gy
    all_gx = np.unique(_G_DIFF[:, 0])
    all_gy = np.unique(_G_DIFF[:, 1])
    
    # exp(-i * gx * px) for each gx
    exp_gx = {}
    for gx in all_gx:
        exp_gx[int(gx)] = np.exp(-1j * gx * 2 * math.pi * px)  # G in units of 1/a, r in units of a
    exp_gy = {}
    for gy in all_gy:
        exp_gy[int(gy)] = np.exp(-1j * gy * 2 * math.pi * py)
    
    # Now compute c_G for each G in unique_G
    c_G = np.zeros(nG, dtype=complex)
    for ig, (gx, gy) in enumerate(unique_G):
        # c_G = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-i gx 2pi x_i) exp(-i gy 2pi y_j)
        # = (1/N^2) sum_i exp(-i gx 2pi x_i) sum_j inv_eps[i,j] exp(-i gy 2pi y_j)
        egx = exp_gx[int(gx)]
        egy = exp_gy[int(gy)]
        # inv_eps is (N,N): inv_eps[i,j]
        # sum_j inv_eps[i,j] * egy[j]
        tmp = inv_eps @ egy  # (N,)
        c_G[ig] = np.vdot(egx, tmp) / (_N * _N)
    
    # Map from (gx,gy) to index in unique_G
    G_to_idx = {}
    for ig, (gx, gy) in enumerate(unique_G):
        G_to_idx[(int(gx), int(gy))] = ig
    
    return c_G, G_to_idx, _pw_k2

def _compute_bands_fast(mask_flat, n_bands=6):
    """Compute the first n_bands TM bands at high-symmetry points using fast PW method."""
    c_G, G_to_idx, pw_k2 = _eps_inv_from_mask(mask_flat)
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for (kx_frac, ky_frac) in _HSP:
        # k in units of 2π/a
        kx = 2 * math.pi * kx_frac
        ky = 2 * math.pi * ky_frac
        
        # Build the eigenvalue matrix A where A_{ij} = (k_i + G_i)^2 * eps_inv(G_i - G_j) / (2π)^2 ... 
        # Actually the standard formulation:
        # sum_j (k + G_i)^2 eps_inv(G_i - G_j) c_j = omega^2 c_i
        # Let's use: eigenvalues are (omega*a/(2π))^2 or similar. Let's just compute ratios.
        
        # (k + G_i) in units of 2π/a: kx_frac + G_i_x, ky_frac + G_i_y
        kx_total = kx_frac + _pw_kx
        ky_total = ky_frac + _pw_ky
        k2_total = kx_total**2 + ky_total**2  # in units of (2π/a)^2
        
        # Build matrix M where M[i,j] = k2_total[i] * eps_inv(G_i - G_j)
        # But this is not symmetric in general. The standard symmetric form uses:
        # sqrt(k2_i) * eps_inv(G_i-G_j) * sqrt(k2_j)
        # Let's use the symmetric formulation.
        
        sqrt_k2 = np.sqrt(np.maximum(k2_total, 1e-12))
        
        M = np.zeros((_PW_NW, _PW_NW), dtype=complex)
        for i in range(_PW_NW):
            for j in range(_PW_NW):
                gdx = _pw_indices[i][0] - _pw_indices[j][0]
                gdy = _pw_indices[i][1] - _pw_indices[j][1]
                idx = G_to_idx[(gdx, gdy)]
                M[i, j] = sqrt_k2[i] * c_G[idx] * sqrt_k2[j]
        
        # M should be Hermitian (eps_inv is real and symmetric, so c_G(-G) = conj(c_G(G)))
        # Take the real part or use eigh on the Hermitian part
        M_herm = (M + M.conj().T) / 2
        M_real = M_herm.real  # Should be real since eps is real
        
        try:
            eigvals = eigh(M_real, eigvals_only=True)
        except:
            continue
        
        eigvals = np.sort(eigvals)
        # eigvals are in units of (2π/a)^2, so omega = sqrt(eigval) * c/a ... 
        # For the ratio, we can just use sqrt(eigvals) as proportional to omega
        omegas = np.sqrt(np.maximum(eigvals, 0))
        
        if len(omegas) >= n_bands:
            # Bands are 1-indexed: band 4 = index 3, band 5 = index 4
            max_omega4 = max(max_omega4, omegas[3])
            min_omega5 = min(min_omega5, omegas[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap <= 0:
        return -1.0
    return 2 * gap / midgap

def _random_params():
    """Generate random initial parameters."""
    params = np.zeros(_NBASIS)
    # Center term
    params[0] = random.uniform(-1.0, 1.0)
    # Radial terms
    for i in range(1, 5):
        params[i] = random.uniform(-0.5, 0.5)
    # Angular terms
    for i in range(5, _NBASIS):
        params[i] = random.uniform(-0.3, 0.3)
    return params

def design(problem):
    """Design a C4v-symmetric photonic crystal unit cell with a large TM band gap."""
    start_time = time.time()
    
    # Initial random parameters
    rng = random.Random(42)
    params = _random_params()
    mask_flat = _params_to_mask(params)
    best_score = _compute_bands_fast(mask_flat)
    best_params = params.copy()
    best_mask = mask_flat.copy()
    
    # Stochastic hill-climbing
    step_size = 0.15
    iteration = 0
    no_improve = 0
    
    while time.time() < _DEADLINE - 10:  # Leave 10s safety margin
        iteration += 1
        
        # Propose a perturbation
        new_params = best_params.copy()
        # Perturb 1-3 random parameters
        n_perturb = rng.randint(1, 3)
        for _ in range(n_perturb):
            idx = rng.randint(0, _NBASIS - 1)
            new_params[idx] += rng.gauss(0, step_size)
        
        new_mask = _params_to_mask(new_params)
        new_score = _compute_bands_fast(new_mask)
        
        if new_score > best_score:
            best_score = new_score
            best_params = new_params
            best_mask = new_mask
            no_improve = 0
            # Slightly increase step size on success
            step_size = min(step_size * 1.05, 0.3)
        else:
            no_improve += 1
            # Decrease step size on failure
            step_size = max(step_size * 0.95, 0.01)
        
        # Occasionally try a larger jump (restart-like)
        if no_improve > 200:
            no_improve = 0
            step_size = 0.15
            # Partial restart: keep some parameters, randomize others
            for _ in range(_NBASIS // 2):
                idx = rng.randint(0, _NBASIS - 1)
                best_params[idx] = rng.gauss(best_params[idx], 0.5)
            best_mask = _params_to_mask(best_params)
            best_score = _compute_bands_fast(best_mask)
    
    return best_mask.tolist()
# EVOLVE-BLOCK-END
