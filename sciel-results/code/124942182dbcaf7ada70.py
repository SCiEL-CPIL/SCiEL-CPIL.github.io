import numpy as np
import time
from scipy.optimize import differential_evolution

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Strategy:
    1. Precompute the cut values for all 2^n states.
    2. Recognize that the QAOA state vector for p layers can be expressed 
       using a sequence of unitary operations. Instead of simulating 
       state vectors repeatedly during optimization, we will use a 
       highly optimized simulation routine that leverages the sparsity 
       of the problem Hamiltonian (diagonal) and the tensor-product 
       structure of the mixer (X rotations).
       
    3. Crucially, we switch from local gradient-based optimization 
       (L-BFGS-B) to a global population-based stochastic optimizer 
       (Differential Evolution). This avoids local minima and requires 
       no gradient computation, trading off precision per step for 
       robustness and global exploration.
       
    4. To make the evaluation fast enough for DE (which needs many 
       function evaluations), we optimize the inner simulation loop 
       using NumPy vectorization and avoiding unnecessary Python loops 
       where possible. For n<=10, 2^n=1024, which is small, so 
       vectorized operations on arrays of size 1024 are very fast.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precompute cut values
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    
    # Efficiently compute cut values using bit manipulation
    # For each edge (u, v), check if bits differ
    # We can vectorize this if we represent states as binary arrays
    # But for n<=10, a simple loop is fine, or we can use lookup tables.
    # Let's use a lookup table for edge contributions.
    
    # Actually, let's just compute it once.
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Precompute the diagonal phases for C?
    # exp(-i gamma C) is diagonal with entries exp(-i gamma cut_vals[s])
    
    def simulate_fast(gammas, betas):
        """
        Fast simulation of QAOA circuit.
        Returns expected cut value.
        """
        # Initial state: uniform superposition
        # |psi> = 1/sqrt(2^n) * sum |s>
        # We can work with amplitudes directly.
        # Amplitude of |s> is initially 1/sqrt(2^n) for all s.
        
        # Instead of tracking the full state vector (complex), 
        # note that for p=2, the state is a linear combination of basis states.
        # However, the mixer couples all states.
        # So we must track the full state vector.
        
        # Optimization: Use the fact that exp(-i beta B) is a tensor product of Rx(beta).
        # Rx(beta) = cos(beta) I - i sin(beta) X
        # This is a 2x2 matrix applied to each qubit.
        
        # We will simulate the state vector.
        # state is a complex array of size 2^n.
        # Initial state: all 1/sqrt(2^n)
        
        # To speed up, we can precompute the effect of the mixer on the basis states?
        # No, the mixer is global (sum of X_i), but it's a tensor product of single-qubit ops.
        
        # Let's use the standard state vector simulation but optimized with NumPy.
        
        state = np.ones(max_states, dtype=np.complex128)
        state /= np.sqrt(max_states)
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase: state[s] *= exp(-i g cut_vals[s])
            # Vectorized
            state *= np.exp(-1j * g * cut_vals)
            
            # Apply B mixer: exp(-i b sum X_i) = tensor_product(Rx(b))
            # This is equivalent to applying Rx(b) on each qubit.
            # We can do this by reshaping the state into a tensor of shape (2, 2, ..., 2)
            # and applying the 2x2 rotation matrix to each axis.
            
            t = state.reshape([2] * n)
            
            c_b = np.cos(b)
            s_b = np.sin(b)
            
            # Rx(b) = [[c_b, -i s_b], [-i s_b, c_b]]
            # t_new[i] = c_b * t[i] - i s_b * t[1-i]
            
            for i in range(n):
                # Move axis i to front
                t = np.moveaxis(t, i, 0)
                
                t0 = t[0]
                t1 = t[1]
                
                new_t0 = c_b * t0 - 1j * s_b * t1
                new_t1 = -1j * s_b * t0 + c_b * t1
                
                # Reconstruct
                # We need to be careful with memory views.
                # Creating a new array is safer for correctness in this context.
                t_new = np.empty_like(t)
                t_new[0] = new_t0
                t_new[1] = new_t1
                t = t_new
                
                # Move axis back
                t = np.moveaxis(t, 0, i)
            
            state = t.reshape(max_states)
            
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut

    def objective(params):
        """
        Objective function for minimization (we minimize -ratio).
        """
        if time.time() > deadline:
            return 1e10
            
        gammas = params[:p]
        betas = params[p:]
        
        exp_cut = simulate_fast(gammas, betas)
        return -exp_cut / opt_cut

    # Optimization using Differential Evolution
    # Bounds for gamma and beta are [0, 2*pi] (or [0, pi] for beta due to symmetry, but [0, 2*pi] is safer)
    bounds = [(0, 2 * np.pi)] * (2 * p)
    
    # DE parameters:
    # popsize: 15 * len(params)
    # maxiter: 100
    # tol: 1e-6
    # seed: for reproducibility
    
    best_params = None
    best_score = -1e10
    
    try:
        # Use a moderate popsize to balance exploration and speed
        # 2*p = 4 parameters for p=2. popsize=15*4=60.
        # Each evaluation is fast (~100-500 us for n=10).
        # 100 iterations * 60 individuals = 6000 evaluations.
        # 6000 * 0.0005 s = 3 s. This is well within budget.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True  # Use L-BFGS-B on the best result to polish
        )
        
        if res.success or res.fun < best_score:
            best_score = -res.fun
            best_params = res.x.copy()
            
    except Exception:
        pass
    
    # Fallback if DE failed or timed out
    if best_params is None or best_score < 0.5: # If score is very bad
        # Try a few random restarts with L-BFGS-B or just random search
        # Since we have time, let's do a quick random search
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            x0 = np.random.uniform(0, 2 * np.pi, 2 * p)
            val = objective(x0)
            if -val > best_score:
                best_score = -val
                best_params = x0.copy()
        
        if best_params is None:
            best_params = np.zeros(2 * p)

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T
