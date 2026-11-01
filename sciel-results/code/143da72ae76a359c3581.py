"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The metric is defined as: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    where conv is the discrete autoconvolution.
    """
    # signed_step is an array of N values, typically in {-1, 1} or continuous
    conv = np.convolve(signed_step, signed_step, mode='full')
    sum_abs_h = np.sum(np.abs(signed_step))
    
    if sum_abs_h < 1e-12:
        return float('inf')
        
    return np.max(np.abs(conv)) / (sum_abs_h ** 2)

def _parallel_tempering_search(start_time):
    """
    Structural change: Formulate as a 1D Ising spin glass ground state search.
    Uses Parallel Tempering (Simulated Annealing with multiple replicas)
    to escape local minima in the rugged energy landscape of the C3 objective.
    
    The 'couplings' are implicitly derived from the convolution structure,
    but we search directly in the spin space {+1, -1}^N.
    """
    # Initialize replicas with diverse structures
    # Replica 0: All +1 (Seed)
    # Replica 1: Alternating signs (High freq)
    # Replica 2: Random
    # Replica 3: Blocky random
    replicas = []
    
    # Start time check
    if time.time() - start_time > DEADLINE - 10.0:
        return np.ones(N, dtype=np.float64)
        
    # Define initial states
    init_states = [
        np.ones(N, dtype=np.int8),
        np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8),
        np.random.choice([1, -1], size=N).astype(np.int8),
        np.repeat(np.random.choice([1, -1], size=N//10), 10).astype(np.int8)
    ]
    
    for s in init_states:
        replicas.append(s.copy())
        
    # Initial energies
    energies = [_compute_c3(r.astype(np.float64)) for r in replicas]
    
    # Temperatures for parallel tempering
    # T_high allows more moves, T_low refines
    temps = [1.0, 0.5, 0.1, 0.01]
    
    best_h = replicas[0].astype(np.float64)
    best_val = energies[0]
    
    max_iter = 20000
    swap_interval = 10
    
    for it in range(max_iter):
        # Deadline check
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        # Update each replica
        for i in range(len(replicas)):
            T = temps[i]
            s = replicas[i]
            E = energies[i]
            
            # Choose a random index to flip
            idx = np.random.randint(N)
            
            # Calculate delta E efficiently? 
            # For N=400, full recompute is ~400*800 ops = 320k ops. 
            # Python overhead for np.convolve might be significant.
            # Let's try to be smarter or just rely on it being fast enough for 20k iters * 4 replicas?
            # 20k * 4 * 320k = 25.6 billion ops. Too slow in pure Python/NumPy loop if not optimized.
            # We need to limit iterations or optimize the delta calculation.
            
            # Optimization: 
            # conv_new = conv_old + h[idx]*(h * delta_flip) + ...
            # Actually, flipping h[idx] from s to -s changes the convolution.
            # Let's just do the full recompute but limit iterations strictly.
            # Or, use a cheaper local search for most steps.
            
            # Let's switch to a simpler local search with simulated annealing logic 
            # to ensure we stay within time limits, keeping the "Ising" structural idea 
            # but with a more efficient update rule or fewer iterations.
            
            # For this implementation, we will do a few flips per replica per iteration.
            # If it's too slow, we reduce max_iter.
            
            # Try 1 flip
            old_val = s[idx]
            s[idx] = -old_val
            
            new_E = _compute_c3(s.astype(np.float64))
            
            # Metropolis criterion
            if new_E < E or np.random.rand() < np.exp(-(new_E - E) / T):
                E = new_E
                # Keep flip
            else:
                # Revert
                s[idx] = old_val
                
            replicas[i] = s
            energies[i] = E
            
            # Track global best
            if E < best_val:
                best_val = E
                best_h = s.astype(np.float64)
                
        # Swap replicas if needed (Parallel Tempering swap)
        if it % swap_interval == 0 and len(replicas) > 1:
            # Try swapping adjacent replicas
            for i in range(len(replicas) - 1):
                i1, i2 = i, i + 1
                E1, E2 = energies[i1], energies[i2]
                T1, T2 = temps[i1], temps[i2]
                
                # Delta E for swap: E1/T2 + E2/T1 - (E1/T1 + E2/T2)
                # Accept if delta < 0 or with prob exp(-delta)
                delta = (E1 / T2 + E2 / T1) - (E1 / T1 + E2 / T2)
                
                if delta < 0 or np.random.rand() < np.exp(-delta):
                    # Swap
                    replicas[i1], replicas[i2] = replicas[i2], replicas[i1]
                    energies[i1], energies[i2] = energies[i2], energies[i1]

    return best_h

def _gradient_refinement(h_start, start_time):
    """
    Refinement machinery from the current program: 
    Use L-BFGS-B with tanh parameterization to polish the discrete solution
    or find a continuous local minimum nearby.
    """
    def _objective(x):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        eps = 1e-12
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Stable soft-max
        max_abs = np.max(abs_conv)
        if max_abs < 1e-9:
            return 1e6
            
        # Temperature for soft-max
        T = 2.0
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h < 1e-6:
            return 1e6
            
        value = log_max / (sum_abs_h**2)
        return value

    def _exact_value(h):
        conv = np.convolve(h, h, mode='full')
        sum_abs_h = np.sum(np.abs(h))
        if sum_abs_h < 1e-6:
            return float('inf')
        return np.max(np.abs(conv)) / (sum_abs_h**2)

    # Initialize x from h_start
    # h_start is in [-1, 1]
    # x = arctanh(h)
    # Clip to avoid infinities
    h_clipped = np.clip(h_start, -0.9999, 0.9999)
    x0 = np.arctanh(h_clipped)
    
    try:
        res = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        h_opt = np.tanh(res.x)
        return h_opt
        
    except Exception:
        return h_start

def construct():
    start_time = time.time()
    
    # 1. Structural Search: Parallel Tempering / Ising Spin Glass
    # This explores the discrete {+1, -1} space effectively.
    best_discrete = _parallel_tempering_search(start_time)
    
    # Check time
    if time.time() - start_time > DEADLINE - 10.0:
        return best_discrete
        
    # 2. Refinement: Gradient-based polishing
    # Take the best discrete solution and refine it in continuous space
    # This might find a slightly better local minimum in the tanh-parameterized space.
    
    # Also try a few other initializations for the gradient step
    candidates = [best_discrete]
    
    # Add a few other diverse starts for gradient refinement
    candidates.append(np.ones(N))
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float64))
    
    best_h = best_discrete
    best_val = _compute_c3(best_h)
    
    for h_start in candidates:
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        h_refined = _gradient_refinement(h_start, start_time)
        val = _compute_c3(h_refined)
        
        if val < best_val:
            best_val = val
            best_h = h_refined
            
    # Final check: ensure we return a valid array
    # If best_h is still the discrete one, that's fine.
    # The task asks for a "sequence of signed step heights".
    # Returning the refined continuous values is valid as per "signed step function" 
    # usually implying the heights can be real, but the seed uses integers.
    # The metric is continuous.
    
    return best_h

# EVOLVE-BLOCK-END
