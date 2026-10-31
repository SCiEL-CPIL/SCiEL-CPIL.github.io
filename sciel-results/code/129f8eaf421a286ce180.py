# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping, minimize

N = 38

def _lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    if min_r2 < 0.25:
        return 1e10
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _step(x, stepsize):
    """Random step for basin hopping."""
    return x + stepsize * np.random.randn(*x.shape)

def construct():
    best_energy = float('inf')
    best_pos = None
    
    # Run multiple basin hopping trials to increase chance of finding global min
    for trial in range(3):
        # Random initialization in a box [-5, 5]^3
        x0 = np.random.uniform(-5, 5, (N, 3))
        x0_flat = x0.flatten()
        
        def energy_func(x):
            pos = x.reshape(-1, 3)
            return _lj_energy(pos)
        
        def step_func(x, stepsize):
            return _step(x, stepsize)
        
        try:
            res = basinhopping(
                f=energy_func,
                x0=x0_flat,
                stepfunc=step_func,
                T=50.0,
                stepsize=0.5,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 30}},
                niter=50,
                minsteps=5,
                maxfun=2000
            )
            
            e = res.fun
            if e < best_energy:
                best_energy = e
                best_pos = res.x.reshape(-1, 3)
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback: return random positions
        return np.random.uniform(-5, 5, (N, 3))
        
    # Final local relaxation to polish
    def energy_func_final(x):
        pos = x.reshape(-1, 3)
        return _lj_energy(pos)
        
    result = minimize(energy_func_final, best_pos.flatten(), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
    final_pos = result.x.reshape(-1, 3)
    
    # Ensure minimum distance constraint
    diff = final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e10)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        final_pos = final_pos * (0.5 / min_r)
        
    return final_pos
# EVOLVE-BLOCK-END
