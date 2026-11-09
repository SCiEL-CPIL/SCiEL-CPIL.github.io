"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _precompute_cuts(n, edges):
    """
    Precompute the cut value for every basis state |x> for the given graph.
    Returns a numpy array of shape (2^n,) containing the number of cut edges.
    """
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    edge_masks = [(1 << i, 1 << j) for i, j in edges]
    
    for x in range(num_states):
        c = 0
        for mask_i, mask_j in edge_masks:
            bi = (x & mask_i) != 0
            bj = (x & mask_j) != 0
            if bi != bj:
                c += 1
        cuts[x] = c
    return cuts

def _precompute_mixer_indices(n):
    """
    Precompute index pairs for applying X_i gates.
    Returns a list of (idx_0, idx_1) for each qubit i.
    """
    num_states = 1 << n
    range_arr = np.arange(num_states)
    pairs = []
    for i in range(n):
        bit = 1 << i
        mask = (range_arr & bit) == 0
        idx_0 = range_arr[mask]
        idx_1 = idx_0 + bit
        pairs.append((idx_0, idx_1))
    return pairs

def _expected_cut_fast(n, edges, cuts, mixer_indices, gamma1, beta1, gamma2, beta2):
    """
    Compute the expected cut for the QAOA circuit with p=2.
    Uses precomputed indices for speed.
    """
    num_states = 1 << n
    amps = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    def apply_mixer(amps, beta):
        cb = np.cos(beta)
        sb = -1j * np.sin(beta)
        for idx_0, idx_1 in mixer_indices:
            old_0 = amps[idx_0].copy()
            old_1 = amps[idx_1].copy()
            amps[idx_0] = cb * old_0 + sb * old_1
            amps[idx_1] = sb * old_0 + cb * old_1
        return amps

    # Apply Layer 1: Cost
    amps *= np.exp(-1j * gamma1 * cuts)
    
    # Apply Layer 1: Mixer
    amps = apply_mixer(amps, beta1)
    
    # Apply Layer 2: Cost
    amps *= np.exp(-1j * gamma2 * cuts)
    
    # Apply Layer 2: Mixer
    amps = apply_mixer(amps, beta2)
    
    # Measure
    probs = np.abs(amps) ** 2
    expected_cut = np.sum(probs * cuts)
    return expected_cut

def _exact_optimum(n, edges, cuts):
    """
    Compute the exact maximum cut size for the given graph.
    """
    return np.max(cuts)

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if n > 12:
        return np.array([[np.pi/4, np.pi/8], [np.pi/4, np.pi/8]])
        
    # Precompute cuts
    cuts = _precompute_cuts(n, edges)
    opt_val = _exact_optimum(n, edges, cuts)
    
    # Precompute mixer indices
    mixer_indices = _precompute_mixer_indices(n)
    
    # Objective function to minimize (negative of expected cut)
    def objective(params):
        g1, b1, g2, b2 = params
        val = _expected_cut_fast(n, edges, cuts, mixer_indices, g1, b1, g2, b2)
        return -val
    
    # Bounds for parameters
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-4, polish=True)
        best_params = res.x
    except Exception:
        best_params = np.array([np.pi/4, np.pi/8, np.pi/4, np.pi/8])
        
    return np.array([[best_params[0], best_params[1]], 
                     [best_params[2], best_params[3]]])

# EVOLVE-BLOCK-END
