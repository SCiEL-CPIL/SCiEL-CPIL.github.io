"""
Broadband AR coating design using Multi-Resolution Coordinate Descent.
Strategy: 
1. Initialize with an analytical quarter-wave stack heuristic.
2. Use a deterministic multi-resolution coordinate descent to optimize thicknesses.
3. Periodically attempt to improve the material sequence via local search (flips/swaps).
4. Dynamically adjust the number of layers to find the optimal trade-off between 
   number of layers and reflectance performance, respecting constraints.
"""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _reflectance_calc(mats, thicks, n0, ns, materials, lambdas):
    """
    Fast vectorized reflectance calculation.
    mats: list of 'L'/'H'
    thicks: numpy array of thicknesses
    """
    N = len(thicks)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for i in range(N):
        n_i = materials[mats[i]]
        d_i = thicks[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s budget (evaluator limit 120s, but keep tight)
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Helper to evaluate a specific state
    def evaluate(mats, thicks):
        if len(mats) == 0:
            return np.mean(_reflectance_calc(mats, np.array([]), n0, ns, materials, lambdas)) * 100.0
        return np.mean(_reflectance_calc(mats, thicks, n0, ns, materials, lambdas)) * 100.0

    # Global best
    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # We will test a range of layer counts.
    # Typically, optimal broadband AR coatings with 2 materials use 5-15 layers.
    # We will iterate through potential layer counts and optimize each.
    
    # Heuristic initialization function
    def init_stack(num_layers):
        """
        Initialize a stack with alternating L/H layers, starting with L (air side).
        Thicknesses initialized to quarter-wave at center wavelength.
        """
        center_lambda = 0.5 * (band[0] + band[1])
        qw_L = center_lambda / (4.0 * materials['L'])
        qw_H = center_lambda / (4.0 * materials['H'])
        
        mats = []
        thicks = np.zeros(num_layers)
        
        # Ensure total thickness is within limits.
        # If num_layers is large, we might need to scale down.
        # For now, use standard QW.
        for i in range(num_layers):
            if i % 2 == 0:
                mats.append('L')
                thicks[i] = qw_L
            else:
                mats.append('H')
                thicks[i] = qw_H
        
        # Check total thickness constraint
        total_t = np.sum(thicks)
        if total_t > max_total_nm:
            scale = max_total_nm / total_t
            thicks *= scale
            
        # Clip individual layers
        thicks = np.clip(thicks, 0, max_layer_nm)
        
        return mats, thicks

    # Coordinate Descent Optimization for a fixed material sequence
    def optimize_thicknesses(mats, thicks, time_limit):
        """
        Perform multi-resolution coordinate descent on thicknesses.
        """
        current_thicks = thicks.copy()
        current_score = evaluate(mats, current_thicks)
        
        # Multi-resolution steps: coarse to fine
        # Start with large steps to find the basin, then refine.
        steps = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
        
        for step in steps:
            # Iterate until no improvement or time limit
            improved = True
            while improved and (time.time() < time_limit):
                improved = False
                for i in range(len(mats)):
                    # Try +/- step
                    for delta in [-step, step]:
                        trial_thicks = current_thicks.copy()
                        trial_thicks[i] += delta
                        trial_thicks[i] = np.clip(trial_thicks[i], 0, max_layer_nm)
                        
                        # Check total thickness constraint
                        if np.sum(trial_thicks) > max_total_nm:
                            continue
                        
                        trial_score = evaluate(mats, trial_thicks)
                        if trial_score < current_score:
                            current_thicks = trial_thicks
                            current_score = trial_score
                            improved = True
                            
        return current_thicks, current_score

    # Local Search for Material Sequence
    def optimize_materials(mats, thicks, time_limit):
        """
        Try to improve the material sequence by flipping or swapping.
        """
        current_mats = mats.copy()
        current_thicks = thicks.copy()
        current_score = evaluate(current_mats, current_thicks)
        
        # We'll do a few passes of local search
        for _ in range(10):
            if time.time() > time_limit:
                break
            improved = False
            for i in range(len(current_mats)):
                # Try flipping material
                new_mat = 'L' if current_mats[i] == 'H' else 'H'
                trial_mats = current_mats.copy()
                trial_mats[i] = new_mat
                trial_score = evaluate(trial_mats, current_thicks)
                
                if trial_score < current_score:
                    # If flip improves, accept and re-optimize thicknesses locally?
                    # To save time, just accept and maybe do a quick thickness polish
                    current_mats = trial_mats
                    current_score = trial_score
                    improved = True
                    
                    # Quick thickness polish after material change
                    # Use a small step size
                    for j in range(len(current_mats)):
                        for delta in [-2.0, 2.0]:
                            trial_thicks = current_thicks.copy()
                            trial_thicks[j] += delta
                            trial_thicks[j] = np.clip(trial_thicks[j], 0, max_layer_nm)
                            if np.sum(trial_thicks) > max_total_nm:
                                continue
                            t_score = evaluate(current_mats, trial_thicks)
                            if t_score < current_score:
                                current_thicks = trial_thicks
                                current_score = t_score
                                improved = True
                            
            if not improved:
                break
                
        return current_mats, current_thicks, current_score

    # Main Loop: Iterate over layer counts
    # We'll prioritize layer counts that are likely optimal.
    # Common optimal counts for broadband AR with 2 materials: 5, 7, 9, 11, 13.
    # We'll test a range but spend more time on promising ones.
    
    layer_counts_to_test = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    # Sort by "promisingness" - usually mid-range is good.
    # Let's just iterate.
    
    for num_layers in layer_counts_to_test:
        if time.time() > deadline - 5.0:
            break
            
        # Time budget for this layer count
        # We want to spend more time on counts that might be better.
        # Let's allocate time proportional to 1/num_layers or just equal?
        # Equal is safer.
        time_per_count = (deadline - time.time()) / len(layer_counts_to_test)
        # But we have already spent time on previous ones.
        # Let's just use a fixed fraction of remaining time.
        # Actually, let's just run until deadline - 5s, but check inside loops.
        
        # Initialize
        mats, thicks = init_stack(num_layers)
        
        # Optimize thicknesses first
        mats, thicks, score = optimize_materials(mats, thicks, deadline - 2.0)
        
        # If this is better than global best, update
        if score < best_score:
            best_score = score
            best_mats = mats.copy()
            best_thicks = thicks.copy()
            
        # If we have time, do a more thorough optimization for this specific count
        # by re-running thickness optimization with finer steps if it was close to best
        if score < best_score * 1.05: # Within 5% of best
             # Re-optimize thicknesses with finer steps
             thicks, score = optimize_thicknesses(mats, thicks, deadline - 1.0)
             if score < best_score:
                 best_score = score
                 best_mats = mats.copy()
                 best_thicks = thicks.copy()

    # Final Polish: If we have time, do a final coordinate descent on the best solution
    if time.time() < deadline - 1.0 and len(best_mats) > 0:
        best_thicks, best_score = optimize_thicknesses(best_mats, best_thicks, deadline - 0.5)
        
    # Ensure constraints are strictly met in the final output
    # Clip thicknesses
    best_thicks = np.clip(best_thicks, 0, max_layer_nm)
    
    # Check total thickness
    if np.sum(best_thicks) > max_total_nm:
        scale = max_total_nm / np.sum(best_thicks)
        best_thicks *= scale
        best_thicks = np.clip(best_thicks, 0, max_layer_nm)
        
    # If no layers found (shouldn't happen), return empty
    if len(best_mats) == 0:
         return [], []
        
    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
