"""
Broadband AR coating design using Multi-Resolution Coordinate Descent.
Strategy: Deterministic local optimization.
1. Initialize with a heuristic alternating stack.
2. Perform coordinate descent with a shrinking step size.
3. Include material switching and layer addition/removal in the search.
"""
import numpy as np
import time

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
    deadline = start_time + 55.0  # Safety margin
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    
    def evaluate(mats, thicks):
        if len(mats) == 0:
            return np.mean(_reflectance_calc(mats, np.array([]), n0, ns, materials, lambdas)) * 100.0
        return np.mean(_reflectance_calc(mats, thicks, n0, ns, materials, lambdas)) * 100.0

    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # Heuristic Initialization
    # Start with a standard quarter-wave stack
    center_lambda = 0.5 * (band[0] + band[1])
    qw_L = center_lambda / (4.0 * materials['L'])
    qw_H = center_lambda / (4.0 * materials['H'])
    
    # Try a few different initial configurations
    initial_configs = [
        # (mats, thicks)
        (['L', 'H', 'L', 'H', 'L'], [qw_L, qw_H, qw_L, qw_H, qw_L]),
        (['L', 'H', 'L', 'H', 'L', 'H', 'L'], [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]),
        (['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'], [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]),
        (['L', 'H', 'L'], [qw_L, qw_H, qw_L]),
    ]
    
    # We will run coordinate descent on the best of these, or sequentially if time permits.
    # Given the smooth landscape, a single good run of CD is often sufficient, but we'll 
    # implement a robust CD that also handles material changes and layer count changes.
    
    # Select the best initial config
    for mats, thicks in initial_configs:
        thicks = np.array(thicks)
        if np.sum(thicks) > max_total_nm:
            thicks *= (max_total_nm / np.sum(thicks))
        score = evaluate(mats, thicks)
        if score < best_score:
            best_score = score
            best_mats = mats.copy()
            best_thicks = thicks.copy()
            
    # Multi-Resolution Coordinate Descent
    # We define a "state" as (mats, thicks).
    # Step sizes will shrink.
    
    current_mats = best_mats.copy()
    current_thicks = best_thicks.copy()
    current_score = best_score
    
    # Step sizes for thicknesses (nm)
    # Start coarse, end fine
    step_sizes = [50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
    
    # We will iterate through step sizes. For each step size, we do a full pass of coordinate descent.
    # We also allow material flips and layer insert/delete if it improves score.
    
    for step in step_sizes:
        if time.time() > deadline - 2.0:
            break
            
        # Flag to indicate if any improvement was found in this resolution level
        improved = False
        
        # 1. Thickness Optimization
        # Iterate until no improvement in a full pass (with current step size)
        # To save time, limit passes per step size
        max_passes = 10
        for _ in range(max_passes):
            pass_improved = False
            N = len(current_thicks)
            for i in range(N):
                # Try +/- step
                for delta in [-step, step]:
                    trial_thicks = current_thicks.copy()
                    trial_thicks[i] = np.clip(current_thicks[i] + delta, 0, max_layer_nm)
                    
                    # Check total thickness constraint
                    if np.sum(trial_thicks) > max_total_nm:
                        # If constraint violated, try to scale down? 
                        # Better: just skip if it violates, as CD assumes valid moves.
                        # However, scaling might be a valid move in itself.
                        # For simplicity in CD, we only accept valid moves.
                        # If current is valid, and trial is invalid, skip.
                        continue
                    
                    trial_score = evaluate(current_mats, trial_thicks)
                    if trial_score < current_score - 1e-9:
                        current_thicks = trial_thicks
                        current_score = trial_score
                        pass_improved = True
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_mats = current_mats.copy()
                            best_thicks = current_thicks.copy()
                        # Move to next delta for this i? Or break?
                        # Usually we just take the best move found for this coordinate.
                        # Let's check the other delta as well? No, coordinate descent usually takes one step.
                        # But if we take -step and it improves, we update. Then we check +step from new position?
                        # Standard CD: evaluate all neighbors, pick best.
                        # Let's stick to: if it improves, accept and break out of delta loop for this i?
                        # Or evaluate both and pick best?
                        # Evaluating both is safer for local minima.
                        pass # We'll continue to check the other delta? No, let's just accept the first improvement for speed, 
                             # as we will come back in the next pass.
                             # Actually, let's just accept the improvement and break the delta loop for this i.
                        break
            
            if not pass_improved:
                break
        
        # 2. Material Optimization (Flips)
        # Try flipping each layer's material
        for i in range(len(current_mats)):
            trial_mats = current_mats.copy()
            trial_mats[i] = 'H' if trial_mats[i] == 'L' else 'L'
            trial_score = evaluate(trial_mats, current_thicks)
            if trial_score < current_score - 1e-9:
                current_mats = trial_mats
                current_score = trial_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_mats = current_mats.copy()
                    best_thicks = current_thicks.copy()
        
        # 3. Structure Optimization (Add/Remove layers)
        # Only try if we have time and are at a finer resolution
        if step <= 10.0:
            # Try removing a layer (if > 1)
            if len(current_mats) > 1:
                for i in range(len(current_mats)):
                    trial_mats = current_mats[:i] + current_mats[i+1:]
                    trial_thicks = np.delete(current_thicks, i)
                    if np.sum(trial_thicks) <= max_total_nm:
                        trial_score = evaluate(trial_mats, trial_thicks)
                        if trial_score < current_score - 1e-9:
                            current_mats = trial_mats
                            current_thicks = trial_thicks
                            current_score = trial_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_mats = current_mats.copy()
                                best_thicks = current_thicks.copy()
                            break # Restart structure search after change? 
                                 # For simplicity, just break and let next iteration handle it.
            
            # Try adding a layer (if < max_layers)
            if len(current_mats) < max_layers:
                # Try inserting at each position
                for pos in range(len(current_mats) + 1):
                    # Try adding L and H
                    for new_mat in ['L', 'H']:
                        trial_mats = current_mats[:pos] + [new_mat] + current_mats[pos:]
                        # Initial thickness: quarter wave? Or small?
                        # Use a small thickness to not violate constraints immediately
                        # Or use a heuristic thickness
                        trial_thicks = np.insert(current_thicks, pos, 50.0) 
                        
                        if np.sum(trial_thicks) <= max_total_nm:
                            trial_score = evaluate(trial_mats, trial_thicks)
                            if trial_score < current_score - 1e-9:
                                current_mats = trial_mats
                                current_thicks = trial_thicks
                                current_score = trial_score
                                improved = True
                                if current_score < best_score:
                                    best_score = current_score
                                    best_mats = current_mats.copy()
                                    best_thicks = current_thicks.copy()
                                # Break out of loops to restart structure optimization
                                break
                    else:
                        continue
                    break
        
        # If no improvement at this resolution, we can stop early?
        # No, finer steps might still help even if coarser ones didn't improve much?
        # Actually, if no improvement at step=10, step=5 is unlikely to help unless we're in a very narrow valley.
        # But material flips and structure changes don't depend on step size.
        # Let's continue to the next step size regardless, but if we haven't improved in 3 steps, we might break.
        # For now, just proceed.

    # Final Polish: Very fine step
    # The loop above ends at 0.1. Let's do one more pass with 0.05 if time permits.
    if time.time() < deadline - 1.0:
        current_mats = best_mats.copy()
        current_thicks = best_thicks.copy()
        current_score = best_score
        
        for i in range(len(current_thicks)):
            for delta in [-0.05, 0.05]:
                trial_thicks = current_thicks.copy()
                trial_thicks[i] = np.clip(current_thicks[i] + delta, 0, max_layer_nm)
                if np.sum(trial_thicks) > max_total_nm:
                    continue
                trial_score = evaluate(current_mats, trial_thicks)
                if trial_score < current_score - 1e-9:
                    current_thicks = trial_thicks
                    current_score = trial_score
                    if current_score < best_score:
                        best_score = current_score
                        best_mats = current_mats.copy()
                        best_thicks = current_thicks.copy()

    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
