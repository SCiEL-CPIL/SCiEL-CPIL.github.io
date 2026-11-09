import numpy as np
import scipy.optimize as opt
import time

def _apply_gate(state, qubits, gate):
    n = int(np.log2(len(state)))
    k = len(qubits)
    
    if k == 0:
        return state
        
    tensor = state.reshape([2] * n)
    
    others = [i for i in range(n) if i not in qubits]
    perm = list(qubits) + others
    
    tensor_perm = tensor.transpose(perm)
    tensor_2d = tensor_perm.reshape(2**k, 2**(n-k))
    
    new_2d = gate @ tensor_2d
    
    new_tensor = new_2d.reshape([2] * n)
    
    inv_perm = [0] * n
    for idx, axis in enumerate(perm):
        inv_perm[axis] = idx
        
    final_tensor = new_tensor.transpose(inv_perm)
    
    return final_tensor.reshape(-1)

def _expected_cut(n, edges, gammas, betas):
    # State initialization
    dim = 2**n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute gates
    # U_B(beta) = exp(-i beta X)
    # U_C(gamma) = prod exp(-i gamma/2 (1 - Z_u Z_v))
    
    # For each layer
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply U_B(beta)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # X = [[0, 1], [1, 0]]
        # Gate matrix:
        # [[cos(beta), -i sin(beta)],
        #  [-i sin(beta), cos(beta)]]
        
        c_b = np.cos(beta)
        s_b = np.sin(beta)
        gate_b = np.array([
            [c_b, -1j * s_b],
            [-1j * s_b, c_b]
        ], dtype=np.complex128)
        
        for q in range(n):
            state = _apply_gate(state, (q,), gate_b)
            
        # Apply U_C(gamma)
        # For each edge (u, v)
        # exp(-i gamma/2 (1 - Z_u Z_v)) = exp(-i gamma/2) exp(i gamma/2 Z_u Z_v)
        # The global phase exp(-i gamma/2) is the same for all edges.
        # Total global phase: exp(-i gamma * |E| / 2)
        # We can ignore global phase for expectation value?
        # <psi| C |psi>. If psi -> e^{i phi} psi, then <psi| -> e^{-i phi} <psi|.
        # e^{-i phi} e^{i phi} = 1. So yes, global phase cancels.
        # So we only need to apply exp(i gamma/2 Z_u Z_v) for each edge.
        
        # exp(i theta Z_u Z_v) where theta = gamma/2
        # Z_u Z_v is diagonal: 1, -1, -1, 1 in basis |00>, |01>, |10>, |11>
        # Eigenvalues: 1, -1, -1, 1
        # exp(i theta Z Z) = diag(exp(i theta), exp(-i theta), exp(-i theta), exp(i theta))
        
        theta = gamma / 2.0
        e_pos = np.exp(1j * theta)
        e_neg = np.exp(-1j * theta)
        
        gate_c = np.array([
            [e_pos, 0, 0, 0],
            [0, e_neg, 0, 0],
            [0, 0, e_neg, 0],
            [0, 0, 0, e_pos]
        ], dtype=np.complex128)
        
        for u, v in edges:
            state = _apply_gate(state, (u, v), gate_c)
            
    # Calculate Expected Cut
    # C = sum_{(u,v)} 0.5 (1 - Z_u Z_v)
    # <C> = sum_{(u,v)} 0.5 (1 - <Z_u Z_v>)
    # <Z_u Z_v> = <psi| Z_u Z_v |psi>
    
    # We can calculate <Z_u Z_v> by applying Z_u Z_v to the state and taking the inner product.
    # Or, since Z is diagonal, we can compute the expectation value directly from the state vector components.
    # <Z_u Z_v> = sum_x |psi_x|^2 * z_u(x) * z_v(x)
    # where z_q(x) is 1 if bit q of x is 0, -1 if bit q is x is 1.
    
    # This is O(2^n * n).
    # For n=10, 1024 * 10 = 10k ops. Very fast.
    
    exp_cut = 0.0
    for u, v in edges:
        # Calculate <Z_u Z_v>
        # Create an array of values 1 or -1 for Z_u Z_v for each basis state
        # Basis state x is an integer 0..2^n-1
        # bit u of x: (x >> u) & 1
        # If 0, Z=1. If 1, Z=-1.
        
        # Vectorized calculation:
        # x = np.arange(2**n)
        # z_u = np.where((x >> u) & 1, -1, 1)
        # z_v = np.where((x >> v) & 1, -1, 1)
        # zz = z_u * z_v
        # exp_zz = np.sum(np.abs(state)**2 * zz)
        
        # This allocates arrays of size 2^n.
        # Doing this for each edge is O(E * 2^n).
        # E ~ 25, 2^n = 1024. 25k ops. Fine.
        
        x = np.arange(2**n)
        z_u = np.where((x >> u) & 1, -1.0, 1.0)
        z_v = np.where((x >> v) & 1, -1.0, 1.0)
        zz = z_u * z_v
        probs = np.abs(state)**2
        exp_zz = np.sum(probs * zz)
        
        exp_cut += 0.5 * (1.0 - exp_zz)
        
    return exp_cut

def angles(n, edges, p):
    start_time = time.time()
    time_limit = 8.0 # seconds
    
    # Objective function
    def objective(params):
        gammas = params[0::2]
        betas = params[1::2]
        # Ensure p matches
        if len(gammas) != p:
            return 0
        return -_expected_cut(n, edges, gammas, betas)
        
    # Multi-start optimization
    best_val = np.inf
    best_params = np.random.uniform(0, 2*np.pi, 2*p)
    
    # Initial guess from seed
    seed_params = np.array([0.35, 0.25, 0.35, 0.25])
    if len(seed_params) == 2*p:
        best_params = seed_params
        best_val = objective(seed_params)
        
    # Random restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        x0 = np.random.uniform(0, 2*np.pi, 2*p)
        
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    # Convert back to (p, 2) array
    gammas = best_params[0::2]
    betas = best_params[1::2]
    
    # Ensure correct shape
    result = np.column_stack((gammas, betas))
    
    return result
