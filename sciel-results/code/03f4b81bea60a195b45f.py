"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Uses Simulated Annealing to pack 21 circles in a rectangle of perimeter 4.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_score = -1.0
    best_result = None
    
    # Perimeter 4 => w + h = 2
    # We search over aspect ratio ar = w/h
    # h = 2 / (1 + ar), w = 2 * ar / (1 + ar)
    
    # Define aspect ratios to try
    # Square is often good for packing, but elongated might allow larger radii if arranged in rows
    aspect_ratios = np.linspace(0.5, 2.0, 15)
    
    # SA Parameters
    # Number of iterations per aspect ratio depends on time budget
    # Let's allocate time dynamically
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initial State: Random
        # Radii: Start with a reasonable guess, e.g., fitting N circles in a grid
        # Grid estimate:
        # Assume roughly square packing for initial guess
        # side ~ sqrt(N)
        # r ~ min(w, h) / sqrt(N) * 0.8
        init_r = min(w, h) / np.sqrt(N) * 0.7
        
        # Random positions inside box
        xs = np.random.uniform(init_r, w - init_r, N)
        ys = np.random.uniform(init_r, h - init_r, N)
        rs = np.full(N, init_r)
        
        # State vector
        state = np.concatenate([xs, ys, rs])
        
        # Objective Function (Vectorized)
        def get_score(state):
            xs = state[0:N]
            ys = state[N:2*N]
            rs = state[2*N:3*N]
            
            # Boundary Penalties
            # Left: x < r -> penalty (r - x)^2
            # Right: x > w - r -> penalty (x - (w - r))^2
            # Bottom: y < r -> penalty (r - y)^2
            # Top: y > h - r -> penalty (y - (h - r))^2
            
            pen_left = np.maximum(0, rs - xs) ** 2
            pen_right = np.maximum(0, xs - (w - rs)) ** 2
            pen_bottom = np.maximum(0, rs - ys) ** 2
            pen_top = np.maximum(0, ys - (h - rs)) ** 2
            
            bound_pen = np.sum(pen_left + pen_right + pen_bottom + pen_top)
            
            # Overlap Penalties
            # dist^2 = (xi-xj)^2 + (yi-yj)^2
            # min_dist^2 = (ri+rj)^2
            # penalty = max(0, min_dist - dist)^2
            
            # Vectorized pair calculation
            # Create distance matrix
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist_sq = dx**2 + dy**2
            
            r_sum = rs[:, None] + rs[None, :]
            min_dist = r_sum
            
            # dist = sqrt(dist_sq)
            # To avoid sqrt for all, we can compare squares?
            # No, penalty is (min_dist - dist)^2. 
            # We need dist.
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Only upper triangle
            iu = np.triu_indices(N, 1)
            dists = dist[iu]
            min_dists = min_dist[iu]
            
            overlaps = np.maximum(0, min_dists - dists)
            overlap_pen = np.sum(overlaps**2)
            
            total_pen = bound_pen + overlap_pen
            
            # Maximize sum of radii, minimize penalty
            score = np.sum(rs) - 50.0 * total_pen
            
            return score, total_pen
        
        # Initial Score
        cur_score, _ = get_score(state)
        best_state = state.copy()
        best_local_score = cur_score
        
        # SA Loop
        T = 0.05 # Initial temperature
        T_min = 0.001
        alpha = 0.999 # Cooling rate
        
        # Estimate iterations
        # We have ~110s / 15 ratios = ~7s per ratio
        # Vectorized eval is fast, maybe 1000-5000 iters/s
        # Let's run for a fixed number of iterations or until time runs out
        max_iters = 5000
        
        for i_iter in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Generate new state
            new_state = state.copy()
            
            # Choose a random circle to perturb
            idx = np.random.randint(N)
            
            # Move type:
            # 1. Move position
            # 2. Change radius
            if np.random.rand() < 0.7:
                # Move position
                dx = np.random.normal(0, 0.1)
                dy = np.random.normal(0, 0.1)
                new_state[idx] += dx
                new_state[N + idx] += dy
            else:
                # Change radius
                dr = np.random.normal(0, 0.05)
                new_state[2*N + idx] += dr
                # Keep radius positive
                if new_state[2*N + idx] < 1e-4:
                    new_state[2*N + idx] = 1e-4
            
            # Evaluate
            new_score, new_pen = get_score(new_state)
            
            # Acceptance Criterion
            delta = new_score - cur_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                state = new_state
                cur_score = new_score
                if cur_score > best_local_score:
                    best_local_score = cur_score
                    best_state = state.copy()
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
        
        # Post-processing: Polish the best local state
        # Use a simple gradient-free local search or just check validity
        # The penalty method might leave tiny overlaps.
        # We will run a few iterations of "relaxation" to push circles apart if overlapping.
        
        final_state = best_state.copy()
        # Simple relaxation: if overlapping, push apart.
        # This is heuristic but helps clean up.
        for _ in range(50):
            xs = final_state[0:N]
            ys = final_state[N:2*N]
            rs = final_state[2*N:3*N]
            
            # Check overlaps
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12)
            r_sum = rs[:, None] + rs[None, :]
            
            overlaps = np.maximum(0, r_sum - dist)
            
            # If no overlaps, break
            if np.sum(overlaps) < 1e-6:
                break
            
            # Move circles apart
            # For each pair, push in opposite directions
            # This is O(N^2) update
            moves = np.zeros_like(xs)
            moves_y = np.zeros_like(ys)
            
            for i in range(N):
                for j in range(i+1, N):
                    if overlaps[i, j] > 1e-6:
                        # Unit vector from i to j
                        d = dist[i, j]
                        if d < 1e-6:
                            # Random push
                            angle = np.random.rand() * 2 * np.pi
                            ux, uy = np.cos(angle), np.sin(angle)
                        else:
                            ux, uy = (xs[j] - xs[i]) / d, (ys[j] - ys[i]) / d
                        
                        # Push amount: half the overlap
                        push = overlaps[i, j] * 0.5 * 0.9 # Damping
                        moves[i] -= ux * push
                        moves_y[i] -= uy * push
                        moves[j] += ux * push
                        moves_y[j] += uy * push
            
            final_state[0:N] += moves
            final_state[N:2*N] += moves_y
            
            # Clamp to box
            # This might introduce new overlaps or boundary violations, but relaxation will fix overlaps.
            # Boundary violations are handled by penalty in SA, but here we just clamp.
            # Actually, clamping might cause overlaps. It's a trade-off.
            # Given the high penalty weight in SA, the solution should be mostly valid.
        
        # Final Validation
        xs = final_state[0:N]
        ys = final_state[N:2*N]
        rs = final_state[2*N:3*N]
        
        valid = True
        # Check radii
        if np.any(rs <= 0):
            valid = False
        
        # Check boundaries (strict)
        if valid:
            if np.any(xs < rs - 1e-5) or np.any(xs > w - rs + 1e-5):
                valid = False
            if np.any(ys < rs - 1e-5) or np.any(ys > h - rs + 1e-5):
                valid = False
        
        # Check overlaps (strict)
        if valid:
            dist = np.sqrt((xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2)
            min_dist = rs[:, None] + rs[None, :]
            # Check upper triangle
            iu = np.triu_indices(N, 1)
            if np.any(dist[iu] < min_dist[iu] - 1e-5):
                valid = False
        
        if valid:
            score = np.sum(rs)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([xs, ys, rs])
    
    # Fallback
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        # w+h=2. Let's use a square 1x1 for simplicity in fallback
        w_fb, h_fb = 1.0, 1.0
        r_fb = 0.5 / side - 1e-3
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(j + 0.5) / side, (i + 0.5) / side, r_fb])
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
