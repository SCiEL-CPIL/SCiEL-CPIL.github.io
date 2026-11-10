"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START = time.time()

def _check_deadline():
    return (time.time() - _START) > _DEADLINE

def _maxcut_exact(n, edges):
    if n == 0:
        return 0
    best = 0
    # Use bit manipulation for speed
    # Precompute edge bit masks? For n<=10, simple loop is fine.
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    for mask in range(1, 1 << n):
        cut = 0
        for u, v in edge_set:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _qaoa_expectation_fast(n, edges, p, gammas, betas):
    """
    Computes <H_C> using tensor network / state vector contraction.
    Optimization: Instead of applying gates sequentially which is O(L * E * 2^n),
    we can leverage the fact that the cost function is a sum of 2-body terms.
    However, for p=2 and n<=10, the state vector approach is still fast if vectorized.
    
    Let's stick to the state vector but optimize the gate application.
    The previous code used a loop over qubits for the X-rotation.
    We can vectorize the X-rotation using slicing or reshape tricks.
    
    Key Insight for Speed:
    The cost function H_C = sum (1 - Z_u Z_v)/2.
    The expectation value is E[C] = sum_{(u,v) in E} 0.5 * (1 - <Z_u Z_v>).
    <Z_u Z_v> can be computed from the state vector probabilities and bit values.
    
    Let's optimize the state evolution.
    1. Initialize |psi> = |+>^n.
    2. For each layer:
       a. Apply U_C = exp(-i gamma H_C). This is diagonal in Z basis.
          H_C eigenvalues are integers (number of cut edges).
          We can compute the phase for each basis state.
       b. Apply U_B = exp(-i beta sum X_i). This is a product of single-qubit rotations.
          R_x(beta) = cos(beta) I - i sin(beta) X.
          Applying this to each qubit independently.
    
    Vectorization of U_B:
    Instead of looping q in range(n) and reshaping, we can use the fact that
    R_x is a 2x2 matrix.
    For qubit q, the state tensor has shape (2, 2, ..., 2).
    We can move axis q to the front, apply the matrix, and move it back.
    Or better, use numpy einsum or simple slicing if we flatten cleverly.
    
    Actually, for n<=10, the overhead of reshape is small. The bottleneck is likely the Python loop in the optimizer.
    Let's focus on making the expectation calculation as fast as possible and using a better optimizer.
    """
    dim = 1 << n
    # State vector
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge list
    edge_list = list(edges)
    
    # Precompute bit masks for edges to speed up phase calculation?
    # For each edge (u,v), we need to know if bits differ.
    # We can precompute a 'cut_mask' for each edge? No, it depends on the state.
    # But we can precompute the indices where bits differ?
    # For edge (u,v), the states where bit u != bit v form a specific pattern.
    # Let's just compute the phase vector for each edge and sum them.
    
    # Precompute bit arrays for all qubits?
    # bits[q][i] = (i >> q) & 1
    # This is 10 * 1024 = 10k integers. Very fast.
    indices = np.arange(dim)
    bits = np.array([[(i >> q) & 1 for i in range(dim)] for q in range(n)], dtype=np.int8)
    
    # Precompute the "diff" vectors for each edge
    # diff_edge[e][i] = 1 if bits[u][i] != bits[v][i] else 0
    # We can store these as boolean arrays or float arrays.
    edge_diffs = []
    for u, v in edge_list:
        diff = (bits[u] != bits[v]).astype(np.float64)
        edge_diffs.append(diff)
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply U_C
        if abs(gamma) > 1e-10:
            # Phase = -gamma * (number of cut edges)
            # H_C = sum (1 - ZuZv)/2. Eigenvalue is 1 if cut, 0 if not.
            # So phase is -gamma * sum(diff)
            total_cut = np.sum(edge_diffs, axis=0)
            phases = -gamma * total_cut
            state *= np.exp(1j * phases) # exp(-i * (-gamma * cut)) = exp(i * gamma * cut)? 
            # Wait. U_C = exp(-i gamma H_C).
            # H_C |s> = c_s |s>.
            # U_C |s> = exp(-i gamma c_s) |s>.
            # c_s is the number of cut edges.
            # So state *= exp(-1j * gamma * total_cut).
            state *= np.exp(-1j * gamma * total_cut)
            
        # Apply U_B
        if abs(beta) > 1e-10:
            c = np.cos(beta)
            s = np.sin(beta)
            # R_x = [[c, -is], [-is, c]]
            # Apply to each qubit.
            # We can do this in one go using reshape and broadcasting if we are careful,
            # but a loop over n is only 10 iterations.
            for q in range(n):
                # Reshape state to isolate qubit q
                # q is bit q.
                # shape: (2**q, 2, 2**(n-q-1))
                shape = (2**q, 2, 2**(n-q-1))
                state_reshaped = state.reshape(shape)
                
                # Apply rotation
                # new[...0...] = c * old[...0...] - i s * old[...1...]
                # new[...1...] = -i s * old[...0...] + c * old[...1...]
                # We can do this without creating a new array if we use temporary variables?
                # Or just create new.
                # To save memory/time, we can update in place if we store old values.
                old_0 = state_reshaped[..., 0, :].copy()
                old_1 = state_reshaped[..., 1, :].copy()
                
                state_reshaped[..., 0, :] = c * old_0 - 1j * s * old_1
                state_reshaped[..., 1, :] = -1j * s * old_0 + c * old_1
                
                state = state_reshaped.reshape(dim)
                
        if _check_deadline():
            break
            
    # Calculate Expectation
    # E[C] = sum_{(u,v)} 0.5 * (1 - <Z_u Z_v>)
    # <Z_u Z_v> = sum_s |psi_s|^2 * z_u(s) z_v(s)
    # z_u = 1 - 2*bit_u.
    # z_u z_v = (1 - 2b_u)(1 - 2b_v) = 1 - 2b_u - 2b_v + 4b_u b_v
    # Note: if b_u != b_v, b_u b_v = 0. If b_u == b_v, b_u b_v = b_u.
    # Alternatively, just compute z_u and z_v.
    
    probs = np.abs(state)**2
    
    exp_val = 0.0
    for u, v in edge_list:
        z_u = 1 - 2 * bits[u]
        z_v = 1 - 2 * bits[v]
        zz = z_u * z_v
        exp_zz = np.sum(probs * zz)
        exp_val += 0.5 * (1 - exp_zz)
        
    return exp_val

def angles(n, edges, p):
    if _check_deadline():
        # Fallback
        return np.tile([0.35, 0.25], (p, 1))
    
    # Optimization
    # The landscape of QAOA expectation is smooth but has local minima.
    # Nelder-Mead is robust but slow.
    # L-BFGS-B with multiple random restarts is often faster for smooth functions.
    # We will use a multi-start strategy.
    
    best_val = -np.inf
    best_params = np.zeros(p * 2)
    
    # Initial guesses
    # Known good ranges: gamma ~ 0.1-0.5, beta ~ 0.1-0.5 for small p.
    # Let's try a few deterministic starting points and random ones.
    
    starts = []
    # Heuristic starts
    starts.append(np.tile([0.2, 0.3], p))
    starts.append(np.tile([0.4, 0.2], p))
    starts.append(np.tile([0.1, 0.5], p))
    
    # Random starts
    np.random.seed(42) # Reproducibility for dev, though eval might differ
    for _ in range(10):
        g = np.random.uniform(0, np.pi/2, p)
        b = np.random.uniform(0, np.pi/2, p)
        starts.append(np.concatenate([g, b]))
        
    for x0 in starts:
        if _check_deadline():
            break
            
        # Clip to valid range? Angles are periodic, but optimizer works better in bounded box.
        # [0, pi] is a standard fundamental domain.
        x0 = np.clip(x0, 0, np.pi)
        
        def objective(x):
            if _check_deadline():
                return 0.0
            gammas = x[:p]
            betas = x[p:]
            try:
                val = _qaoa_expectation_fast(n, edges, p, gammas, betas)
                return -val
            except Exception:
                return 0.0
                
        try:
            # Use L-BFGS-B for faster convergence on smooth functions
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(0, np.pi)] * (2 * p),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
                
            # Check if we found a very high value (near optimal?)
            # If best_val is close to the number of edges, we might be done?
            # But we don't know the optimal cut size easily without brute force.
            # However, for small n, we can estimate.
            
        except Exception:
            continue
            
    # If L-BFGS-B failed or didn't improve, try one Nelder-Mead from best found
    if _check_deadline():
        pass
    else:
        x0 = best_params
        if x0 is None or np.all(x0 == 0):
            x0 = np.tile([0.3, 0.3], p)
            
        def objective_nm(x):
            if _check_deadline():
                return 0.0
            gammas = x[:p]
            betas = x[p:]
            try:
                val = _qaoa_expectation_fast(n, edges, p, gammas, betas)
                return -val
            except Exception:
                return 0.0
                
        try:
            res = minimize(
                objective_nm, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            pass
            
    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.column_stack((gammas, betas))
# EVOLVE-BLOCK-END
