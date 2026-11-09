"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _precompute_cuts(n, edges):
    """
    Precompute the cut value for every basis state |x> for the given graph.
    Returns a numpy array of shape (2^n,) containing the number of cut edges.
    """
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    # Vectorized computation of cuts is tricky with variable edge lists, 
    # but n is small (<=10), so a loop over states is acceptable if optimized, 
    # or we can use bit operations.
    # For n=10, 1024 states. Edges ~ 25. 1024*25 = 25k operations. Very fast.
    
    # Optimization: Precompute edge bit masks
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

def _expected_cut_fast(n, edges, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute the expected cut for the QAOA circuit with p=2.
    Circuit: |+>^n -> exp(-i gamma1 C) -> exp(-i beta1 H_B) -> exp(-i gamma2 C) -> exp(-i beta2 H_B)
    
    Args:
        n: number of qubits
        edges: list of edges (not used directly, cuts is precomputed)
        cuts: precomputed cut values for all 2^n states
        gamma1, beta1, gamma2, beta2: QAOA angles
    
    Returns:
        Expected cut value (float)
    """
    num_states = 1 << n
    # Initial state: |+>^n
    # Amplitudes are 1/sqrt(2^n) for all states
    # We can work with normalized amplitudes or unnormalized. 
    # Let's use unnormalized amps = ones(2^n) and divide at the end?
    # No, unitary evolution preserves norm. 
    # amps = np.ones(num_states) / np.sqrt(num_states)
    
    # To save memory and time, we can represent the state as a complex array.
    amps = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Helper to apply mixer exp(-i beta sum X_i)
    # This is a product of single-qubit rotations.
    # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
    # In Z-basis, this mixes amplitudes of states differing by bit i.
    
    def apply_mixer(amps, beta):
        cb = np.cos(beta)
        sb = -1j * np.sin(beta)
        # We need to apply for each qubit i
        # For qubit i, pair (x, x ^ (1<<i))
        # new_x = cb * old_x + sb * old_y
        # new_y = sb * old_x + cb * old_y
        
        # Vectorized approach:
        # We can reshape the array to separate the bit dimension.
        # But iterating over n qubits (max 10) is fast enough.
        
        # To avoid in-place modification issues, we might need a copy or careful ordering.
        # Since the operations for different qubits commute, we can process them sequentially.
        # However, processing qubit i modifies amps such that processing qubit j (j!=i) is independent.
        # We can do it in-place if we are careful, but copying is safer and fast for 2^10=1024.
        
        for i in range(n):
            bit = 1 << i
            # Create masks for indices where bit i is 0
            # Indices: 0..2^n-1
            # We can create a boolean mask or just loop.
            # Looping 1024 * 10 = 10k iterations is trivial.
            
            # Vectorized pair processing:
            # Indices where bit i is 0:
            # We can construct these indices efficiently.
            # For n=10, let's just use a simple loop for clarity and speed is sufficient.
            
            # Actually, we can use numpy slicing if we reshape.
            # Reshape to shape (2^(n-1), 2) where the last dimension is the bit i?
            # The bit order in the integer representation matters.
            # If we use standard binary representation, bit i is not the last axis in a simple reshape unless we permute axes.
            
            # Let's stick to the loop over states for the mixer, it's robust.
            # Optimization: Process pairs.
            # For each x where bit i is 0, y = x + bit.
            
            # We can vectorize this by creating arrays of indices.
            # indices_0 = [x for x in range(num_states) if (x & bit) == 0]
            # indices_1 = [x + bit for x in indices_0]
            
            # Generating indices_0 efficiently:
            # Pattern: 0, 1, ..., 2^(i-1)-1, then 2^i, ..., etc.
            # It's easier to just use a boolean mask on the range.
            # range_arr = np.arange(num_states)
            # mask = (range_arr & bit) == 0
            # idx_0 = range_arr[mask]
            # idx_1 = idx_0 + bit
            
            # This creates arrays of size 2^(n-1).
            # Let's do this.
            range_arr = np.arange(num_states)
            mask = (range_arr & bit) == 0
            idx_0 = range_arr[mask]
            idx_1 = idx_0 + bit
            
            old_0 = amps[idx_0].copy()
            old_1 = amps[idx_1].copy()
            
            amps[idx_0] = cb * old_0 + sb * old_1
            amps[idx_1] = sb * old_0 + cb * old_1
            
    # Apply Layer 1: Cost
    # exp(-i gamma1 C) is diagonal in Z basis.
    phase1 = np.exp(-1j * gamma1 * cuts)
    amps = amps * phase1
    
    # Apply Layer 1: Mixer
    apply_mixer(amps, beta1)
    
    # Apply Layer 2: Cost
    phase2 = np.exp(-1j * gamma2 * cuts)
    amps = amps * phase2
    
    # Apply Layer 2: Mixer
    apply_mixer(amps, beta2)
    
    # Measure
    probs = np.abs(amps) ** 2
    expected_cut = np.sum(probs * cuts)
    return expected_cut

def _exact_optimum(n, edges):
    """
    Compute the exact maximum cut size for the given graph.
    """
    num_states = 1 << n
    best = 0
    edge_masks = [(1 << i, 1 << j) for i, j in edges]
    for x in range(num_states):
        c = 0
        for mask_i, mask_j in edge_masks:
            bi = (x & mask_i) != 0
            bj = (x & mask_j) != 0
            if bi != bj:
                c += 1
        if c > best:
            best = c
    return best

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of edges (i, j)
        p: number of layers (always 2)
    
    Returns:
        numpy array of shape (p, 2), rows (gamma, beta) in radians.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if n > 12:
        # Fallback for larger n if needed, though task says n<=10
        return np.array([[np.pi/4, np.pi/8], [np.pi/4, np.pi/8]])
        
    # Precompute cuts
    cuts = _precompute_cuts(n, edges)
    
    # Objective function to minimize (negative of expected cut)
    def objective(params):
        g1, b1, g2, b2 = params
        # Constraints: angles are in [0, pi] typically, but can be extended.
        # QAOA angles are periodic. We can bound them to [0, pi] for gamma and [0, pi/2] for beta?
        # Actually, beta in [0, pi] is sufficient due to symmetry? 
        # Let's bound to [0, pi] for all to be safe and let optimizer find local maxima.
        # To avoid boundary issues, we might use unbounded and rely on local minima, 
        # but bounded is safer for Nelder-Mead.
        
        # Ensure parameters are within reasonable bounds for numerical stability
        # If optimizer goes out of bounds, penalty? 
        # scipy minimize with bounds handles this.
        
        val = _expected_cut_fast(n, edges, cuts, g1, b1, g2, b2)
        return -val
    
    # Initial guesses
    # Common good starts:
    # 1. Uniform small angles
    # 2. Heuristic based on density
    # 3. Random
    
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # Heuristic guess:
    # gamma ~ 1/sqrt(m) ? 
    # Let's try a few diverse starts.
    starts = [
        [np.pi/4, np.pi/8, np.pi/4, np.pi/8],
        [0.5, 0.2, 0.5, 0.2],
        [1.0, 0.5, 1.0, 0.5],
        [0.1, 0.1, 0.1, 0.1],
        [np.pi/2, np.pi/4, np.pi/2, np.pi/4],
    ]
    
    # Add a random start
    np.random.seed(42) # Fixed seed for reproducibility in evaluation? 
    # The task says "graph-dependent", so using random might be okay if we pick the best.
    # But deterministic is better for evaluation consistency.
    # Let's add a deterministic pseudo-random start based on graph hash?
    # Or just stick to the list above.
    
    best_params = starts[0]
    best_score = -np.inf
    
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
    
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # Ensure best_params is valid
    if not np.isfinite(best_score):
        best_params = starts[0]
        
    # Format output: (p, 2) array
    # Row 0: gamma1, beta1
    # Row 1: gamma2, beta2
    return np.array([[best_params[0], best_params[1]], 
                     [best_params[2], best_params[3]]])

# EVOLVE-BLOCK-END
