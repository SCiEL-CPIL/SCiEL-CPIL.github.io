"""Optimize shallow circuit parameters for Ising ground states using vectorized simulation and global search."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the transverse-field Ising model."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for k in range(dim):
            # z_i is 1 if bit i is 0, -1 if bit i is 1
            z_i = 1 if (k >> i) & 1 == 0 else -1
            z_j = 1 if (k >> (i + 1)) & 1 == 0 else -1
            H[k, k] += -coupling * z_i * z_j
            
    # X_i terms
    for i in range(n):
        for k in range(dim):
            # X_i flips bit i
            k_flip = k ^ (1 << i)
            H[k, k_flip] += -field / 2.0
            H[k_flip, k] += -field / 2.0
            
    return H

def _get_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])

def _simulate_energy(params, n, layers, H, init_state):
    """
    Vectorized simulation of the circuit to compute energy <H>.
    params: (layers, n) array of Ry angles.
    """
    dim = 1 << n
    # Reshape state to tensor (2, 2, ..., 2)
    # Qubit i corresponds to axis i.
    # Index k = sum b_i 2^i.
    # state_tensor[i_0, i_1, ..., i_{n-1}] corresponds to bitstring i_0 i_1 ... i_{n-1}
    # where i_0 is qubit 0 (LSB).
    
    state = init_state.copy()
    
    # Precompute CZ masks for vectorization? 
    # Actually, for n=4,6, iterating over qubits is fine if we use slicing.
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = params[l, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # We need to apply Ry on qubit i.
            # In the flat state vector, qubit i is bit i.
            # We can use advanced indexing or reshape.
            # Reshape is cleaner.
            shape = [2] * n
            state_tensor = state.reshape(shape)
            
            # Move axis i to front
            state_tensor = np.moveaxis(state_tensor, i, 0)
            
            # Apply Ry matrix to the first axis
            # Ry = [[c, s], [-s, c]]
            # new_state[0] = c*old[0] + s*old[1]
            # new_state[1] = -s*old[0] + c*old[1]
            
            # We operate on the first axis (size 2)
            # state_tensor[0] and state_tensor[1] are slices of the remaining n-1 axes
            s0 = state_tensor[0].copy()
            s1 = state_tensor[1].copy()
            
            new_s0 = c * s0 + s * s1
            new_s1 = -s * s0 + c * s1
            
            # Assign back
            state_tensor[0] = new_s0
            state_tensor[1] = new_s1
            
            # Move axis back
            state_tensor = np.moveaxis(state_tensor, 0, i)
            state = state_tensor.reshape(-1)
            
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                # CZ between i and i+1
                # CZ is diagonal: -1 if both bits are 1.
                # We can do this with a mask on the flat vector.
                mask_i = 1 << i
                mask_j = 1 << (i + 1)
                mask = mask_i | mask_j
                
                # Create an array of indices where both bits are set
                # Or simpler: iterate? No, vectorize.
                # indices = np.where((np.arange(dim) & mask) == mask)[0]
                # state[indices] *= -1
                
                # Even faster: use the property that CZ only affects subspace where both are 1.
                # For n=6, dim=64. Iteration is negligible.
                # But let's stick to vectorized where easy.
                # Actually, for n=6, a simple loop over 64 elements is faster than np.where overhead.
                # However, we are in a tight loop of optimization.
                # Let's use the mask approach.
                # We can precompute the mask indices for each pair once outside the loop?
                # Yes, but params change, structure doesn't.
                
                # Precomputation of CZ indices would be better.
                # But for now, simple logic:
                # state = state * (1 - 2 * indicator)
                # indicator[k] = 1 if (k & mask) == mask else 0
                
                # Let's just do the simple loop for CZ, it's O(dim) per CZ, and we have n-1 CZs.
                # n=6, dim=64. 5 * 64 = 320 ops. Trivial.
                for k in range(dim):
                    if (k & mask) == mask:
                        state[k] *= -1.0

    # Compute energy
    # H is (dim, dim). state is (dim,).
    # H @ state
    Hs = H.dot(state)
    energy = np.real(np.vdot(state, Hs))
    return energy

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    dim = 1 << n
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(n, coupling, field)
    
    # Initial state |0>^n
    init_state = np.zeros(dim, dtype=complex)
    init_state[0] = 1.0
    
    def objective(params):
        # params is flat (layers * n,)
        thetas = params.reshape((layers, n))
        try:
            e = _simulate_energy(thetas, n, layers, H, init_state)
            # Objective is mean energy gap per spin
            # The task says "minimize the mean energy gap per spin relative to an independent exact eigensolution"
            # Objective: mean (circuit energy - exact ground energy)/n
            # Since ground energy is constant, minimizing energy is sufficient.
            # But returning the gap helps if we want to check convergence.
            # For optimizer, just return energy.
            return e
        except Exception:
            return 1e10

    # Differential Evolution
    # Bounds: [-pi, pi] for each angle
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Seed
    rng = np.random.default_rng(42)
    
    # DE parameters
    # popsize: 15 * (layers*n) is default, might be too large for small n.
    # For n=4, layers=3, params=12. popsize=15*12=180.
    # For n=6, layers=3, params=18. popsize=15*18=270.
    # This is manageable.
    
    # maxiter: 100?
    # tol: 1e-8
    
    # We can run DE, and if time permits, polish with Nelder-Mead or L-BFGS-B.
    
    best_params = np.zeros(layers * n)
    best_energy = np.inf
    
    # Try a few different initial populations or just run DE with a decent popsize.
    # DE is stochastic, so fixed seed is good for reproducibility.
    
    try:
        res = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=50, 
            popsize=20, # 20 * 12 = 240 evals per gen. 50 gens = 12000 evals.
            tol=1e-8,
            init='sobol', # Better initialization
            polish=True # Uses L-BFGS-B at the end
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
            
    except Exception as e:
        # Fallback to simple random search if DE fails
        print(f"DE failed: {e}")
        for _ in range(100):
            if time.time() > deadline:
                break
            p = rng.uniform(-np.pi, np.pi, size=layers * n)
            e = objective(p)
            if e < best_energy:
                best_energy = e
                best_params = p
                
    # Check time, if we have plenty left, do local polish on best_params
    if time.time() < deadline - 10.0:
        try:
            # Use L-BFGS-B for final polish
            res_polish = scipy.optimize.minimize(
                objective, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_polish.fun < best_energy:
                best_energy = res_polish.fun
                best_params = res_polish.x
        except Exception:
            pass
            
    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
