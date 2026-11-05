import numpy as np
import time

def _build_hamiltonians():
    """
    Build the Hamiltonian parts A and B for the 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    N = 6
    dim = 2 ** N

    def apply_op(op, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def apply_two_op(op1, op2, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op1
        ops[k+1] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    J = 1.0
    delta = 0.8
    h = 0.3

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    field_sum = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_sum += apply_op(Sz, k)

    A += 0.5 * h * field_sum
    B += 0.5 * h * field_sum

    for k in range(0, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            A += term

    for k in range(1, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            B += term

    return A, B

def _expm_cached(M, cache):
    """Compute matrix exponential with simple cache."""
    key = id(M)
    if key not in cache:
        cache[key] = np.linalg.matrix_exp(M)
    return cache[key]

def _nelder_mead(f, x0, maxiter=200, xatol=1e-8, fatol=1e-10, deadline=None):
    """Pure NumPy Nelder-Mead with deadline support."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.05 * max(abs(xi[i]), 1.0)
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    
    for it in range(maxiter):
        if deadline is not None and time.time() > deadline:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < fatol:
            break
        if np.max(np.abs(np.array(simplex) - np.array(simplex[0]))) < xatol:
            break
        
        xw = simplex[-1]
        xc = np.mean(simplex[:-1], axis=0)
        
        xr = xc + (xc - xw)
        fr = f(xr)
        
        if fr < fvals[0]:
            xe = xc + 2 * (xc - xw)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc2 = xc + 0.5 * (xc - xw)
            else:
                xc2 = xc - 0.5 * (xc - xw)
            fc = f(xc2)
            if fc < fvals[-1]:
                simplex[-1] = xc2
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def _particle_swarm(f, bounds_low, bounds_high, n_particles=20, max_iter=50, deadline=None):
    """Simple particle swarm optimization."""
    n_dims = len(bounds_low)
    n_particles = min(n_particles, 50)
    
    # Initialize particles
    positions = []
    velocities = []
    personal_best_pos = []
    personal_best_val = []
    
    for _ in range(n_particles):
        pos = np.random.uniform(bounds_low, bounds_high)
        vel = np.random.uniform(-0.01, 0.01, n_dims)
        positions.append(pos)
        velocities.append(vel)
        val = f(pos)
        personal_best_pos.append(pos.copy())
        personal_best_val.append(val)
    
    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_val = personal_best_val[global_best_idx]
    
    w = 0.8  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n_particles):
            r1 = np.random.random()
            r2 = np.random.random()
            
            vel = w * velocities[i] + c1 * r1 * (personal_best_pos[i] - positions[i]) + \
                  c2 * r2 * (global_best_pos - positions[i])
            velocities[i] = np.clip(vel, -0.1, 0.1)
            
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], bounds_low, bounds_high)
            
            val = f(positions[i])
            if val < personal_best_val[i]:
                personal_best_val[i] = val
                personal_best_pos[i] = positions[i].copy()
                if val < global_best_val:
                    global_best_val = val
                    global_best_idx = i
                    global_best_pos = positions[i].copy()
    
    return global_best_pos, global_best_val

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using multi-start
    Nelder-Mead with particle swarm initialization.
    """
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    try:
        A, B = _build_hamiltonians()
        H = A + B
        dim = A.shape[0]

        times = [0.25, 0.5, 1.0]
        U_exacts = []
        for t in times:
            U_exacts.append(np.linalg.matrix_exp(-1j * t * H))

        # Pre-compute exponentials for each stage to speed up evaluation
        # For a given coefficient a_k and time t, we need exp(-1j * a_k * t * A)
        # We'll cache these based on (a_k * t) pairs

        def compute_error_fast(a, b, expm_cache_A, expm_cache_B):
            """Compute mean spectral-norm error with cached exponentials."""
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            total_err = 0.0
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                for k in range(12):
                    # Get cached exponential for a_k * t
                    key_A = (round(a[k] * t, 10), 'A')
                    if key_A not in expm_cache_A:
                        expm_cache_A[key_A] = np.linalg.matrix_exp(-1j * a[k] * t * A)
                    
                    key_B = (round(b[k] * t, 10), 'B')
                    if key_B not in expm_cache_B:
                        expm_cache_B[key_B] = np.linalg.matrix_exp(-1j * b[k] * t * B)
                    
                    Ea = expm_cache_A[key_A]
                    Eb = expm_cache_B[key_B]
                    U = Eb @ Ea @ U
                err = np.linalg.norm(U - U_exacts[i], 2)
                total_err += err
            return total_err / len(times)

        def objective(x):
            """Objective: x is 24-dim [a1..a12, b1..b12]."""
            if time.time() > deadline:
                return 1e10
            a = x[:12]
            b = x[12:]
            # Normalize
            sa = np.sum(a)
            sb = np.sum(b)
            if abs(sa) > 1e-15:
                a = a / sa
            else:
                a = np.ones(12) / 12
            if abs(sb) > 1e-15:
                b = b / sb
            else:
                b = np.ones(12) / 12
            # Clip
            a = np.clip(a, -50.0, 50.0)
            b = np.clip(b, -50.0, 50.0)
            # Re-normalize after clipping
            sa = np.sum(a)
            sb = np.sum(b)
            if abs(sa) > 1e-15:
                a = a / sa
            if abs(sb) > 1e-15:
                b = b / sb
            
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return 1e10
            
            return compute_error_fast(a, b, expm_cache_A, expm_cache_B)

        # Initialize caches
        expm_cache_A = {}
        expm_cache_B = {}

        # --- Multi-start optimization ---
        best_x = None
        best_err = float('inf')
        rng = np.random.RandomState(42)

        # Start 1: Uniform
        a_u = np.ones(12) / 12.0
        b_u = np.ones(12) / 12.0
        x_u = np.concatenate([a_u, b_u])
        err_u = objective(x_u)
        if err_u < best_err:
            best_err = err_u
            best_x = x_u.copy()

        # Start 2-10: Random perturbations around uniform
        for trial in range(9):
            if time.time() > deadline - 60.0:
                break
            a_r = 1.0/12 + rng.normal(0, 0.02, 12)
            b_r = 1.0/12 + rng.normal(0, 0.02, 12)
            x_r = np.concatenate([a_r, b_r])
            err_r = objective(x_r)
            if err_r < best_err:
                best_err = err_r
                best_x = x_r.copy()

        # Start 11-15: Structured patterns
        # Cosine profiles
        for freq in [1, 2, 3]:
            if time.time() > deadline - 50.0:
                break
            ks = np.arange(12)
            a_c = 1.0 + 0.3 * np.cos(2 * np.pi * freq * ks / 12.0)
            b_c = 1.0 + 0.3 * np.cos(2 * np.pi * freq * ks / 12.0)
            x_c = np.concatenate([a_c, b_c])
            err_c = objective(x_c)
            if err_c < best_err:
                best_err = err_c
                best_x = x_c.copy()

        # Asymmetric patterns
        for sign in [1, -1]:
            if time.time() > deadline - 40.0:
                break
            ks = np.arange(12)
            a_asym = 1.0 + 0.3 * np.cos(2 * np.pi * ks / 12.0)
            b_asym = 1.0 + sign * 0.3 * np.cos(2 * np.pi * ks / 12.0)
            x_asym = np.concatenate([a_asym, b_asym])
            err_asym = objective(x_asym)
            if err_asym < best_err:
                best_err = err_asym
                best_x = x_asym.copy()

        # --- Nelder-Mead refinement from best start ---
        try:
            x_opt, f_opt = _nelder_mead(objective, best_x, maxiter=150, deadline=deadline - 20.0)
            if f_opt < best_err:
                best_err = f_opt
                best_x = x_opt
        except Exception:
            pass

        # --- Second Nelder-Mead from another random start ---
        if time.time() < deadline - 30.0:
            try:
                a_r2 = 1.0/12 + rng.normal(0, 0.03, 12)
                b_r2 = 1.0/12 + rng.normal(0, 0.03, 12)
                x_r2 = np.concatenate([a_r2, b_r2])
                x_opt2, f_opt2 = _nelder_mead(objective, x_r2, maxiter=100, deadline=deadline - 15.0)
                if f_opt2 < best_err:
                    best_err = f_opt2
                    best_x = x_opt2
            except Exception:
                pass

        # --- Final refinement ---
        if time.time() < deadline - 10.0:
            try:
                x_opt3, f_opt3 = _nelder_mead(objective, best_x, maxiter=100, xatol=1e-9, deadline=deadline - 5.0)
                if f_opt3 < best_err:
                    best_err = f_opt3
                    best_x = x_opt3
            except Exception:
                pass

        # Extract and normalize final coefficients
        a_final = best_x[:12]
        b_final = best_x[12:]
        sa = np.sum(a_final)
        sb = np.sum(b_final)
        if abs(sa) > 1e-15:
            a_final = a_final / sa
        if abs(sb) > 1e-15:
            b_final = b_final / sb
        a_final = np.clip(a_final, -50.0, 50.0)
        b_final = np.clip(b_final, -50.0, 50.0)
        sa = np.sum(a_final)
        sb = np.sum(b_final)
        if abs(sa) > 1e-15:
            a_final = a_final / sa
        if abs(sb) > 1e-15:
            b_final = b_final / sb

        if not (np.all(np.isfinite(a_final)) and np.all(np.isfinite(b_final))):
            return [1.0/12.0]*12, [1.0/12.0]*12

        return list(a_final), list(b_final)

    except Exception:
        return [1.0/12.0]*12, [1.0/12.0]*12
