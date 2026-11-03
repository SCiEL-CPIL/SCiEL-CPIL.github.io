"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _max_radius(pos, radii, i, n):
    """Calculate max possible radius for circle i given current state of others."""
    x, y = pos[i]
    # Boundary constraints
    r_bound = min(x, y, 1.0 - x, 1.0 - y)
    if r_bound <= 0:
        return 0.0
    
    # Overlap constraints with other circles
    r_max = r_bound
    for j in range(n):
        if i == j:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        # Circle i must not overlap circle j: r_i + r_j <= dist => r_i <= dist - r_j
        r_candidate = dist - radii[j]
        if r_candidate < r_max:
            r_max = r_candidate
    return max(0.0, r_max)

def _optimize_packing(pos, radii, n, deadline):
    """
    Iteratively maximize radii. 
    1. Greedy radius update.
    2. Local search on positions to potentially allow larger radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_sum = np.sum(radii)
    no_improve_count = 0
    
    # Greedy radius expansion phase
    # Iterate a few times to let radii settle
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            old_r = radii[i]
            new_r = _max_radius(pos, radii, i, n)
            if new_r > old_r + 1e-6:
                radii[i] = new_r
                improved = True
        if not improved:
            break
            
    current_sum = np.sum(radii)
    if current_sum > best_sum:
        best_sum = current_sum
        
    # Local search on positions
    # Try to move circles to increase the sum of max radii
    rng = np.random.RandomState(42)
    step_size = 0.05
    for it in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random circle to move
        i = rng.randint(n)
        orig_x, orig_y = pos[i]
        
        # Try a few random moves
        best_local_sum = current_sum
        best_local_pos = pos.copy()
        best_local_radii = radii.copy()
        
        for _ in range(10):
            # Random move
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            new_x = np.clip(orig_x + dx, 0.0, 1.0)
            new_y = np.clip(orig_y + dy, 0.0, 1.0)
            
            # If no movement, skip
            if new_x == orig_x and new_y == orig_y:
                continue
                
            # Update pos
            pos[i] = [new_x, new_y]
            
            # Recalculate radii for all circles based on new pos
            # This is expensive O(N^2) but N=26 is small
            # We just need to check if sum(radii) increases
            # To be efficient, we can just recalculate the max radius for the moved circle 
            # and check if it allows others to grow? 
            # Simpler: Recalculate all radii greedily once.
            
            trial_radii = np.zeros(n)
            # Order of update matters. Let's just do one pass.
            # Actually, to get a fair comparison, we should run the greedy expansion again from this new pos.
            # But that's slow. 
            # Approximation: Just calculate max radius for i and check if it's better, 
            # assuming others stay same? No, moving i might allow j to grow.
            
            # Let's do a full greedy update for this trial.
            trial_radii = np.zeros(n)
            for k in range(n):
                trial_radii[k] = _max_radius(pos, trial_radii, k, n)
            
            trial_sum = np.sum(trial_radii)
            
            if trial_sum > best_local_sum:
                best_local_sum = trial_sum
                best_local_pos = pos.copy()
                best_local_radii = trial_radii.copy()
                
        # Accept best local move
        if best_local_sum > current_sum + 1e-6:
            pos = best_local_pos
            radii = best_local_radii
            current_sum = best_local_sum
            if current_sum > best_sum:
                best_sum = current_sum
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            # Reduce step size if stuck
            if no_improve_count > 50:
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 0.001:
                    break
                    
    return pos, radii

def _validate(pos, radii, n, tol=1e-9):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_and_validate(pos, radii, n):
    """
    Ensure validity by clamping radii. 
    This is a safety net.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
        
    # Fix overlaps by reducing radii
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - 1e-9:
                    excess = md - d
                    # Reduce both radii
                    reduction = excess / 2.0 + 1e-6
                    radii[i] = max(0.0, radii[i] - reduction)
                    radii[j] = max(0.0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
            
    # Final boundary clamp
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
        
    if _validate(pos, radii, n):
        return pos, radii, np.sum(radii)
    else:
        # If still invalid (should be rare), return zeros or best effort
        # For safety, return a known valid small packing
        return None, None, -1.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = -np.inf
    
    # Initial configurations
    configs = []
    
    # 1. Grid 5x5 (25 circles) + 1 extra? No, N=26.
    # Grid 5x6 = 30. Take 26.
    # Grid 4x7 = 28. Take 26.
    
    # Hexagonal
    # Try to create a hex layout
    
    # Random
    rng = np.random.RandomState(0)
    
    # Generate 10 random starts
    for _ in range(10):
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        # Ensure min separation to avoid starting with huge overlaps
        # Simple rejection sampling for initial pos
        valid_pos = []
        attempts = 0
        while len(valid_pos) < N and attempts < 500:
            p = rng.uniform(0.1, 0.9, size=2)
            ok = True
            for vp in valid_pos:
                if np.linalg.norm(p - vp) < 0.1:
                    ok = False
                    break
            if ok:
                valid_pos.append(p)
            attempts += 1
        while len(valid_pos) < N:
            valid_pos.append(rng.uniform(0.1, 0.9, size=2))
        pos = np.array(valid_pos)
        radii = np.full(N, 0.01)
        configs.append((pos, radii))
        
    # Grid start
    pos_grid = []
    for i in range(5):
        for j in range(6):
            if len(pos_grid) >= N:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 6.0
            pos_grid.append([x, y])
    pos_grid = np.array(pos_grid[:N])
    radii_grid = np.full(N, 0.01)
    configs.append((pos_grid, radii_grid))
    
    # Hexagonal start
    pos_hex = []
    # 5 rows of 5?
    # Approximate hex
    for r in range(6):
        for c in range(5):
            if len(pos_hex) >= N:
                break
            x = (c + 0.5) * 0.2
            y = (r + 0.5) * 0.2
            if r % 2 == 1:
                x += 0.1
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pos_hex.append([x, y])
    pos_hex = np.array(pos_hex[:N])
    radii_hex = np.full(N, 0.01)
    configs.append((pos_hex, radii_hex))
    
    # Process configs
    for pos, radii in configs:
        if time.time() > deadline - 10:
            break
            
        try:
            opt_pos, opt_radii = _optimize_packing(pos, radii, N, deadline - 5)
            valid_pos, valid_radii, score = _fix_and_validate(opt_pos, opt_radii, N)
            
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            continue
            
    if best_result is None:
        # Fallback: Small circles in a grid
        result = np.zeros((N, 3))
        # 5x5 grid + 1
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
        
    return best_result

# EVOLVE-BLOCK-END
