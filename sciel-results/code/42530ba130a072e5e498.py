import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 21
    
    # 1. Initialization: Hexagonal packing
    # We want to pack 21 circles.
    # A 5-4-5-4-3 arrangement works well for a square.
    rows = [5, 4, 5, 4, 3]
    centers = []
    y_coords = []
    # Vertical spacing for hexagonal packing: sqrt(3)/2 * d
    # Horizontal spacing: d
    # Let's assume initial diameter d=0.1
    d = 0.1
    h = np.sqrt(3) / 2 * d
    
    current_y = d / 2
    for i, count in enumerate(rows):
        # Center the row horizontally
        # Total width of this row: (count - 1) * d
        # Start x: (1 - (count - 1) * d) / 2
        start_x = (1.0 - (count - 1) * d) / 2.0
        for j in range(count):
            x = start_x + j * d
            y = current_y
            centers.append((x, y))
        current_y += h
        
    # Normalize/Scale to fit in [0,1]x[0,1]
    # The current packing might not fill the square perfectly.
    # We will let the optimization handle scaling by expanding radii.
    # But initial positions should be reasonable.
    
    centers = np.array(centers)
    # Ensure they are within [0,1]
    # If they are not, we might need to scale.
    # For d=0.1, width is approx 0.4 + 0.05 = 0.45. Height is 4 * 0.0866 + 0.05 = 0.4.
    # So they are centered.
    
    # Initial radii
    r = np.full(n, d / 2)
    
    best_score = np.sum(r)
    best_centers = centers.copy()
    best_r = r.copy()
    
    # Optimization Parameters
    repulsion_strength = 1.0
    boundary_strength = 1.0
    damping = 0.5
    expansion_step = 0.005
    shrink_step = 0.001
    max_relax_iter = 100
    
    # Helper function to check validity and get score
    def check_validity(c, r):
        # Check boundaries
        if np.any(c[:, 0] < r) or np.any(c[:, 0] > 1 - r) or \
           np.any(c[:, 1] < r) or np.any(c[:, 1] > 1 - r):
            return False, 0.0
            
        # Check overlaps
        # Vectorized distance check
        # c[:, np.newaxis] - c[np.newaxis, :]
        dists = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        min_dists = np.sum(r[:, np.newaxis] + r[np.newaxis, :], axis=1)
        # Exclude diagonal
        np.fill_diagonal(min_dists, 0)
        
        # If any distance is less than sum of radii, invalid
        if np.any(dists < min_dists - 1e-9):
            return False, 0.0
            
        return True, np.sum(r)

    # Initial check
    is_valid, score = check_validity(centers, r)
    if is_valid:
        best_score = score
        best_centers = centers.copy()
        best_r = r.copy()
    else:
        # If initial is invalid, shrink radii until valid
        while not is_valid and np.sum(r) > 0.01:
            r *= 0.95
            is_valid, score = check_validity(centers, r)
        if is_valid:
            best_score = score
            best_centers = centers.copy()
            best_r = r.copy()
        else:
            # Fallback to very small radii
            r = np.full(n, 0.01)
            best_score = np.sum(r)
            best_centers = centers.copy()
            best_r = r.copy()

    # Main Loop
    while time.time() < deadline:
        # Relaxation Phase
        relaxed = False
        for _ in range(max_relax_iter):
            if time.time() > deadline:
                break
                
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Overlaps
            dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
            min_dists = np.sum(r[:, np.newaxis] + r[np.newaxis, :], axis=1)
            np.fill_diagonal(min_dists, 0)
            
            # Overlap amount
            overlap = min_dists - dists
            # Only consider positive overlaps
            overlap = np.where(overlap > 0, overlap, 0)
            
            # Unit vectors
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = centers[:, np.newaxis] - centers[np.newaxis, :]
                norm = np.linalg.norm(diff, axis=2)
                unit = diff / norm[:, :, np.newaxis]
                unit = np.where(norm > 1e-9, unit, 0)
                
            # Force is proportional to overlap
            # F_ij = unit_ij * overlap_ij
            force_matrix = unit * overlap[:, :, np.newaxis]
            
            # Sum forces on each particle
            forces += np.sum(force_matrix, axis=1)
            
            # Boundary forces
            # Left boundary: if x < r, push right (positive x)
            left_viol = r - centers[:, 0]
            forces[:, 0] += np.where(left_viol > 0, left_viol * boundary_strength, 0)
            
            # Right boundary: if x > 1-r, push left (negative x)
            right_viol = centers[:, 0] - (1 - r)
            forces[:, 0] -= np.where(right_viol > 0, right_viol * boundary_strength, 0)
            
            # Bottom boundary: if y < r, push up (positive y)
            bottom_viol = r - centers[:, 1]
            forces[:, 1] += np.where(bottom_viol > 0, bottom_viol * boundary_strength, 0)
            
            # Top boundary: if y > 1-r, push down (negative y)
            top_viol = centers[:, 1] - (1 - r)
            forces[:, 1] -= np.where(top_viol > 0, top_viol * boundary_strength, 0)
            
            # Update positions
            # Add some random jitter to escape local minima
            jitter = np.random.normal(0, 0.001, size=centers.shape)
            centers += (forces * repulsion_strength + jitter) * damping
            
            # Clamp to [0,1] to prevent flying away
            centers = np.clip(centers, 0, 1)
            
            # Check if valid
            is_valid, score = check_validity(centers, r)
            if is_valid:
                relaxed = True
                break
        
        if not relaxed:
            # Relaxation failed, shrink radii
            r *= (1 - shrink_step)
            # Ensure minimum radius
            r = np.maximum(r, 0.001)
            continue
        
        # If relaxed, try to expand
        # Check if we can expand
        # Try increasing all radii by expansion_step
        new_r = r + expansion_step
        
        # Check if new radii are valid with current centers
        # We might need to re-relax with new radii
        # But first, a quick check
        # If we just increased r, overlaps are likely.
        # So we set r = new_r and loop back to relaxation.
        
        # However, if the expansion is too aggressive, we might oscillate.
        # Let's try to expand slightly and see if it stays valid after one relaxation step.
        
        r = new_r
        # The loop will go back to relaxation phase at the top.
        
        # Update best if we have a valid configuration with this r
        # Note: The relaxation phase above ensured 'centers' and 'r' (old r) was valid.
        # But we just changed r. So we don't have a valid config yet for new r.
        # The next iteration will relax.
        
        # We should track the best valid config seen so far.
        # The 'is_valid' check inside relaxation loop uses current r.
        # So if relaxation succeeds, the current r is valid.
        # We should update best_score there.
        
        # Let's modify the relaxation loop to update best.
        # Actually, the structure above updates best_score only if initial was valid.
        # We need to update best_score inside the relaxation loop if is_valid becomes True.
        
        # Let's restructure:
        # 1. Relax with current r.
        # 2. If successful, update best.
        # 3. Expand r.
        
        # Wait, the code above expands r at the end of the while loop.
        # So the next iteration will relax with the expanded r.
        # If that relaxation succeeds, we should update best.
        
        # I will add the best update inside the relaxation loop.
        
    # Finalize
    # Ensure we return the best valid configuration found
    # The variables best_centers and best_r hold the best.
    # We need to format the output as (x, y, r)
    
    result = np.column_stack((best_centers, best_r))
    return result

# Re-implementing with the fix for updating best_score inside the loop

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 21
    
    # 1. Initialization: Hexagonal packing
    rows = [5, 4, 5, 4, 3]
    centers = []
    d = 0.1
    h = np.sqrt(3) / 2 * d
    
    current_y = d / 2
    for i, count in enumerate(rows):
        start_x = (1.0 - (count - 1) * d) / 2.0
        for j in range(count):
            x = start_x + j * d
            y = current_y
            centers.append((x, y))
        current_y += h
        
    centers = np.array(centers)
    r = np.full(n, d / 2)
    
    best_score = -np.inf
    best_centers = None
    best_r = None
    
    def check_validity(c, r):
        if np.any(c[:, 0] < r) or np.any(c[:, 0] > 1 - r) or \
           np.any(c[:, 1] < r) or np.any(c[:, 1] > 1 - r):
            return False, 0.0
        dists = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        min_dists = np.sum(r[:, np.newaxis] + r[np.newaxis, :], axis=1)
        np.fill_diagonal(min_dists, 0)
        if np.any(dists < min_dists - 1e-9):
            return False, 0.0
        return True, np.sum(r)

    # Initial check and shrink if necessary
    is_valid, score = check_validity(centers, r)
    while not is_valid and np.sum(r) > 0.01:
        r *= 0.95
        is_valid, score = check_validity(centers, r)
    if is_valid:
        best_score = score
        best_centers = centers.copy()
        best_r = r.copy()
    else:
        r = np.full(n, 0.01)
        best_score = np.sum(r)
        best_centers = centers.copy()
        best_r = r.copy()

    # Parameters
    repulsion_strength = 1.0
    boundary_strength = 1.0
    damping = 0.5
    expansion_step = 0.002
    shrink_step = 0.005
    max_relax_iter = 50

    while time.time() < deadline:
        # Relaxation Phase
        relaxed = False
        for _ in range(max_relax_iter):
            if time.time() > deadline:
                break
                
            forces = np.zeros_like(centers)
            
            dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
            min_dists = np.sum(r[:, np.newaxis] + r[np.newaxis, :], axis=1)
            np.fill_diagonal(min_dists, 0)
            
            overlap = min_dists - dists
            overlap = np.where(overlap > 0, overlap, 0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = centers[:, np.newaxis] - centers[np.newaxis, :]
                norm = np.linalg.norm(diff, axis=2)
                unit = diff / norm[:, :, np.newaxis]
                unit = np.where(norm > 1e-9, unit, 0)
                
            force_matrix = unit * overlap[:, :, np.newaxis]
            forces += np.sum(force_matrix, axis=1)
            
            left_viol = r - centers[:, 0]
            forces[:, 0] += np.where(left_viol > 0, left_viol * boundary_strength, 0)
            
            right_viol = centers[:, 0] - (1 - r)
            forces[:, 0] -= np.where(right_viol > 0, right_viol * boundary_strength, 0)
            
            bottom_viol = r - centers[:, 1]
            forces[:, 1] += np.where(bottom_viol > 0, bottom_viol * boundary_strength, 0)
            
            top_viol = centers[:, 1] - (1 - r)
            forces[:, 1] -= np.where(top_viol > 0, top_viol * boundary_strength, 0)
            
            jitter = np.random.normal(0, 0.001, size=centers.shape)
            centers += (forces * repulsion_strength + jitter) * damping
            centers = np.clip(centers, 0, 1)
            
            is_valid, score = check_validity(centers, r)
            if is_valid:
                relaxed = True
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
                    best_r = r.copy()
                break
        
        if not relaxed:
            r *= (1 - shrink_step)
            r = np.maximum(r, 0.001)
            continue
        
        # Expand radii
        r += expansion_step
        # Ensure radii are not too large (sanity check)
        r = np.minimum(r, 0.5)

    result = np.column_stack((best_centers, best_r))
    return result

