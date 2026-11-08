"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END

def _validate_and_fix(xyz):
    """Ensure circles are inside [0,1]^2 with positive radii. Returns (xyz, valid)."""
    xyz = np.array(xyz, dtype=float).copy()
    if xyz.shape != (N, 3):
        return None, False
    # Clip positions to [0,1]
    xyz[:, 0] = np.clip(xyz[:, 0], 0.0, 1.0)
    xyz[:, 1] = np.clip(xyz[:, 1], 0.0, 1.0)
    # Ensure radii positive
    xyz[:, 2] = np.clip(xyz[:, 2], 1e-6, 0.5)
    return xyz, True

def _check_validity(xyz, tol=1e-7):
    """Check if all circles are disjoint and inside the unit square."""
    if xyz is None:
        return False
    for i in range(N):
        x, y, r = xyz[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol or r <= 0:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = xyz[i, 0] - xyz[j, 0]
            dy = xyz[i, 1] - xyz[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < xyz[i, 2] + xyz[j, 2] - tol:
                return False
    return True

def _score(xyz):
    """Compute the objective: sum of radii."""
    if xyz is None:
        return 0.0
    return float(np.sum(xyz[:, 2]))

def _make_grid_initial():
    """Grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
    return np.array(cells[:N])

def _make_hex_initial():
    """Hexagonal packing initial configuration."""
    # Hexagonal lattice for better packing
    rows = int(np.ceil(N / 4))
    cells = []
    for row in range(rows):
        for col in range(4):
            if len(cells) >= N:
                break
            x = (col + 0.5 + (row % 2) * 0.5) / 4.0
            y = (row + 0.5) / float(rows)
            r = 0.22
            cells.append([x, y, r])
    return np.array(cells[:N])

def _make_random_initial(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    xyz = np.zeros((N, 3))
    xyz[:, 0] = rng.uniform(0.1, 0.9, N)
    xyz[:, 1] = rng.uniform(0.1, 0.9, N)
    xyz[:, 2] = rng.uniform(0.05, 0.15, N)
    return xyz

def _compute_loss_and_grad(params, beta=100.0):
    """
    Compute loss and gradient for SGD-based optimization.
    Loss = -sum(radii) + penalty for constraint violations.
    """
    xyz = params.reshape(N, 3)
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    loss = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    
    # Boundary penalties using softplus
    for i in range(N):
        for boundary_x in [x[i], 1.0 - x[i]]:
            v = r[i] - boundary_x
            if v > -1.0 / beta:  # avoid underflow
                sp = np.log1p(np.exp(beta * v)) / beta
                sig = 1.0 / (1.0 + np.exp(-beta * v))
                loss += sp
                if boundary_x == x[i]:
                    grad[3*i] -= sig
                else:
                    grad[3*i] += sig
                grad[3*i+2] += sig
        
        for boundary_y in [y[i], 1.0 - y[i]]:
            v = r[i] - boundary_y
            if v > -1.0 / beta:
                sp = np.log1p(np.exp(beta * v)) / beta
                sig = 1.0 / (1.0 + np.exp(-beta * v))
                loss += sp
                if boundary_y == y[i]:
                    grad[3*i+1] -= sig
                else:
                    grad[3*i+1] += sig
                grad[3*i+2] += sig
    
    # Circle-circle penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) + 1e-10
            rsum = r[i] + r[j]
            v = rsum - dist
            if v > -1.0 / beta:
                sp = np.log1p(np.exp(beta * v)) / beta
                sig = 1.0 / (1.0 + np.exp(-beta * v))
                loss += sp
                gx = -sig * dx / dist
                gy = -sig * dy / dist
                grad[3*i] += gx
                grad[3*i+1] += gy
                grad[3*j] -= gx
                grad[3*j+1] -= gy
                grad[3*i+2] += sig
                grad[3*j+2] += sig
    
    return loss, grad

def _sgd_optimize(xyz_init, deadline, max_iter=5000):
    """
    SGD with Adam-like adaptive learning rates and simulated annealing for beta.
    """
    params = xyz_init.flatten().copy()
    n_params = len(params)
    
    # Adam hyperparameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Momentum buffers
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    
    # Annealing schedule for penalty weight
    beta_start = 20.0
    beta_end = 500.0
    
    best_xyz = xyz_init.copy()
    best_score = _score(xyz_init)
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        # Anneal beta
        progress = it / max_iter
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Compute loss and gradient
        loss, grad = _compute_loss_and_grad(params, beta=beta)
        
        # Add small noise to escape local minima (annealing)
        noise_scale = 0.005 * (1.0 - progress)
        grad = grad + noise_scale * rng.randn(n_params)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Clamp to reasonable bounds
        xyz_trial = params.reshape(N, 3)
        xyz_trial[:, 0] = np.clip(xyz_trial[:, 0], 0.0, 1.0)
        xyz_trial[:, 1] = np.clip(xyz_trial[:, 1], 0.0, 1.0)
        xyz_trial[:, 2] = np.clip(xyz_trial[:, 2], 1e-6, 0.5)
        params = xyz_trial.flatten()
        
        # Check validity and update best
        if it % 50 == 0 or it == max_iter - 1:
            if _check_validity(xyz_trial):
                s = _score(xyz_trial)
                if s > best_score:
                    best_score = s
                    best_xyz = xyz_trial.copy()
    
    return best_xyz, best_score

def _high_precision_polish(xyz, deadline):
    """Final polish with high beta using SGD."""
    if time.time() > deadline - 10.0:
        return xyz, _score(xyz)
    
    params = xyz.flatten().copy()
    n_params = len(params)
    
    # Adam hyperparameters for fine-tuning
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    
    best_xyz = xyz.copy()
    best_score = _score(xyz)
    
    rng = np.random.RandomState(7)
    
    max_iter = 3000
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        # Use high beta for tight constraints
        beta = 1000.0 + 1000.0 * (it / max_iter)
        
        loss, grad = _compute_loss_and_grad(params, beta=beta)
        
        # Small noise for exploration
        noise_scale = 0.001 * (1.0 - it / max_iter)
        grad = grad + noise_scale * rng.randn(n_params)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        xyz_trial = params.reshape(N, 3)
        xyz_trial[:, 0] = np.clip(xyz_trial[:, 0], 0.0, 1.0)
        xyz_trial[:, 1] = np.clip(xyz_trial[:, 1], 0.0, 1.0)
        xyz_trial[:, 2] = np.clip(xyz_trial[:, 2], 1e-6, 0.5)
        params = xyz_trial.flatten()
        
        if it % 100 == 0 or it == max_iter - 1:
            if _check_validity(xyz_trial):
                s = _score(xyz_trial)
                if s > best_score:
                    best_score = s
                    best_xyz = xyz_trial.copy()
    
    return best_xyz, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s kill
    
    # Generate multiple initial configurations
    inits = [
        _make_grid_initial(),
        _make_hex_initial(),
        _make_random_initial(42),
        _make_random_initial(123),
        _make_random_initial(999),
    ]
    
    best_xyz = None
    best_score = 0.0
    
    # Allocate time per initial config
    time_per_init = min(20.0, (deadline - start_time - 15.0) / len(inits))
    
    # Try each initial configuration with SGD
    for i, xyz_init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        xyz_init, valid = _validate_and_fix(xyz_init)
        if not valid:
            continue
        
        init_deadline = min(start_time + time_per_init, deadline - 15.0)
        xyz_res, score = _sgd_optimize(xyz_init, init_deadline, max_iter=3000)
        if score > best_score:
            best_score = score
            best_xyz = xyz_res
    
    # Final high-precision polish
    if best_xyz is not None and time.time() < deadline - 10.0:
        best_xyz, best_score = _high_precision_polish(best_xyz, deadline)
    
    if best_xyz is None:
        # Fallback to grid
        best_xyz = _make_grid_initial()
    
    # Final validation
    best_xyz, valid = _validate_and_fix(best_xyz)
    if not valid or not _check_validity(best_xyz):
        best_xyz = _make_grid_initial()
    
    return best_xyz

# EVOLVE-BLOCK-END
