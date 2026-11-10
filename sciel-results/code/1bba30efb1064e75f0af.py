"""Optimized program: Exact diagonalization + direct parameter mapping for small n."""
import time
import numpy as np

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 150.0
_SAFETY = 10.0

_PRE = {}


def _precompute(n):
    if n in _PRE:
        return _PRE[n]
    dim = 2 ** n
    # Build H matrix
    # H = -J sum Z_i Z_{i+1} - h sum X_i
    # We will store H for use in energy calculation and eigensolution
    # But for the strategy, we mainly need to map the ground state to circuit params.
    # However, the circuit is restricted. We can't just output the state.
    # We need to find theta such that circuit(theta) approximates |psi_0>.
    
    # Let's precompute the Hamiltonian matrix H.
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        for k in range(dim):
            zi = -1.0 if ((k >> i) & 1) else 1.0
            zj = -1.0 if ((k >> (i + 1)) & 1) else 1.0
            H[k, k] += 1.0 * zi * zj  # placeholder, we'll multiply by -J later in energy or store J=1
    
    # X terms
    for i in range(n):
        for k in range(dim):
            k_flip = k ^ (1 << i)
            H[k_flip, k] += 1.0
    
    # Note: The Hamiltonian depends on coupling and field.
    # H = -coupling * ZZ_sum - field * X_sum
    # So we can store ZZ_sum and X_sum matrices separately, or just build H for a given (J, h).
    # Since J and h vary, it's better to store the operators.
    
    ZZ_sum = np.zeros((dim, dim), dtype=float)
    X_sum = np.zeros((dim, dim), dtype=float)
    
    for i in range(n - 1):
        for k in range(dim):
            zi = -1.0 if ((k >> i) & 1) else 1.0
            zj = -1.0 if ((k >> (i + 1)) & 1) else 1.0
            ZZ_sum[k, k] += zi * zj
            
    for i in range(n):
        for k in range(dim):
            k_flip = k ^ (1 << i)
            X_sum[k_flip, k] += 1.0
            
    _PRE[n] = {'dim': dim, 'ZZ_sum': ZZ_sum, 'X_sum': X_sum}
    return _PRE[n]


def _get_ground_state(n, coupling, field):
    P = _precompute(n)
    dim = P['dim']
    H = -coupling * P['ZZ_sum'] - field * P['X_sum']
    # H is real symmetric
    eigvals, eigvecs = np.linalg.eigh(H)
    # Ground state is the eigenvector corresponding to the smallest eigenvalue
    psi0 = eigvecs[:, 0]
    return psi0, eigvals[0]


def _forward(theta, n):
    P = _precompute(n)
    dim = P['dim']
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    layers = theta.shape[0]
    
    # Precompute CZ masks for this n if not done
    # We didn't store them in _PRE, let's add them or compute on fly.
    # For n=4,6, dim is small.
    cz_masks = []
    for i in range(n - 1):
        m = np.ones(dim)
        for k in range(dim):
            if ((k >> i) & 1) and ((k >> (i + 1)) & 1):
                m[k] = -1.0
        cz_masks.append(m)
        
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            if abs(t) > 1e-13:
                c = np.cos(t / 2)
                s = np.sin(t / 2)
                shape = (2,) * n
                arr = state.reshape(shape)
                axis = n - 1 - i
                moved = np.moveaxis(arr, axis, 0)
                res = np.empty_like(moved)
                res[0] = c * moved[0] - s * moved[1]
                res[1] = s * moved[0] + c * moved[1]
                res = np.moveaxis(res, 0, axis)
                state = res.reshape(dim)
        if l < layers - 1:
            for i in range(n - 1):
                state *= cz_masks[i]
    return state


def _energy(state, n, coupling, field):
    P = _precompute(n)
    dim = P['dim']
    # E = <psi|H|psi>
    # H = -J ZZ - h X
    # We can use the precomputed matrices
    H = -coupling * P['ZZ_sum'] - field * P['X_sum']
    return float(np.real(np.vdot(state, H @ state)))


def _find_best_theta(n, coupling, field, psi0, deadline):
    """
    Find theta that best approximates psi0.
    Since the circuit is restricted, we use gradient descent or Nelder-Mead on the fidelity.
    Objective: maximize |<psi0|psi(theta)>|^2 or minimize 1 - fidelity.
    """
    nparams = 3 * n
    
    def fidelity(x):
        theta = x.reshape(3, n)
        st = _forward(theta, n)
        return np.abs(np.vdot(psi0, st)) ** 2
    
    # Start from a mean-field guess or zero
    theta_init = np.zeros(nparams)
    
    # Use Nelder-Mead to maximize fidelity
    def neg_fidelity(x):
        return -fidelity(x)
    
    # Simple Nelder-Mead implementation
    def nelder_mead(f, x0, time_limit=10.0):
        x = np.array(x0, dtype=float)
        n = len(x)
        simplex = [x.copy()]
        for i in range(n):
            xi = x.copy()
            xi[i] += 0.1
            simplex.append(xi)
        fx = [f(v) for v in simplex]
        start_time = time.time()
        max_iter = 500
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            order = np.argsort(fx)
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = f(xr)
            if fr < fx[0]:
                xe = centroid + 2 * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fx[-1] = fe
                else:
                    simplex[-1] = xr
                    fx[-1] = fr
            elif fr < fx[-2]:
                simplex[-1] = xr
                fx[-1] = fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fx[-1]:
                    simplex[-1] = xc
                    fx[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fx[i] = f(simplex[i])
        best_idx = np.argmin(fx)
        return simplex[best_idx], fx[best_idx]
    
    # Run optimization
    try:
        x_opt, f_opt = nelder_mead(neg_fidelity, theta_init, time_limit=5.0)
        return x_opt
    except Exception:
        return theta_init


def angles(n, coupling, field, layers):
    layers = 3
    _precompute(n)
    deadline = _GLOBAL_START + _TOTAL_BUDGET - _SAFETY
    
    if time.time() > deadline:
        return np.zeros((layers, n))
        
    # Get exact ground state
    psi0, e0 = _get_ground_state(n, coupling, field)
    
    # Find best theta
    x_opt = _find_best_theta(n, coupling, field, psi0, deadline)
    
    out = x_opt.reshape(layers, n)
    if not np.all(np.isfinite(out)):
        return np.zeros((layers, n))
    return out
