import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50

def score_centres(c):
    # c is (N, 3)
    # Compute pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    # Compute boundary clearances
    # c is in [-0.5, 0.5]
    # distance to wall is 0.5 - |c_i|
    clearances = 0.5 - np.abs(c)
    min_clearance = np.min(clearances)
    
    return min(min_pair_dist, min_clearance) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_r = 0.0
    best_c = None
    
    # Initialize with a few random restarts
    # We can also try a grid start
    starts = []
    
    # 1. Grid start
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        starts.append(grid[:N].copy())
    
    # 2. Random starts
    for i in range(10):
        starts.append(np.random.uniform(-0.5, 0.5, (N, 3)))
        
    for c_init in starts:
        if time.time() > deadline:
            break
            
        c = c_init.copy()
        # Ensure initial validity (push inside if needed, though random is inside)
        
        # Local Search Loop
        # We will use a step size that decays
        step_size = 0.01
        min_step = 1e-8
        
        for it in range(2000): # Max iterations per restart
            if time.time() > deadline:
                break
                
            r = score_centres(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
            # Compute gradient direction for the active constraints
            # Active constraints are those with value close to min
            d = cdist(c, c)
            np.fill_diagonal(d, np.inf)
            
            clearances = 0.5 - np.abs(c)
            
            min_d = np.min(d)
            min_c = np.min(clearances)
            
            r_curr = min(min_d, min_c) / 2.0
            
            # If we are stuck (step too small), break
            if step_size < min_step:
                break
                
            # Identify active pairs
            # Pairs with distance <= min_d + tol
            tol = 1e-6
            active_pairs = np.argwhere(d <= min_d + tol)
            
            # Identify active boundaries
            # For each point i and dimension k, if clearance is small
            active_bounds = []
            for i in range(N):
                for k in range(3):
                    if clearances[i, k] <= min_c + tol:
                        active_bounds.append((i, k))
                        
            # Compute gradient
            # We want to maximize r.
            # d(r)/d(c) is sum of contributions from active constraints.
            
            grad = np.zeros_like(c)
            
            # Pair contributions
            # For pair (i, j) with distance d_ij, gradient of d_ij w.r.t c_i is (c_i - c_j)/d_ij
            # We want to increase d_ij, so we move c_i away from c_j.
            for i, j in active_pairs:
                if i == j: continue
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation if coincident
                    diff = np.random.normal(size=3)
                    dist = np.linalg.norm(diff)
                grad[i] += diff / dist
                grad[j] -= diff / dist
                
            # Boundary contributions
            # For point i, dimension k, if c_ik > 0, wall is at 0.5.
            # clearance = 0.5 - c_ik. To increase clearance, decrease c_ik.
            # If c_ik < 0, wall is at -0.5.
            # clearance = -0.5 - c_ik. To increase clearance, increase c_ik.
            # Gradient of clearance w.r.t c_ik is -sign(c_ik).
            # We want to increase clearance, so we move in direction of gradient of clearance.
            # So grad[i, k] += -sign(c_ik) ?
            # Let's check:
            # If c_ik = 0.4, clearance = 0.1. To increase, move to 0.3. Delta c = -0.1.
            # sign(c_ik) = 1. -sign(c_ik) = -1.
            # So we add -1 to grad. This moves c_ik down. Correct.
            # If c_ik = -0.4, clearance = 0.1. To increase, move to -0.3. Delta c = +0.1.
            # sign(c_ik) = -1. -sign(c_ik) = 1.
            # So we add 1 to grad. This moves c_ik up. Correct.
            
            for i, k in active_bounds:
                if c[i, k] > 0:
                    grad[i, k] -= 1.0
                elif c[i, k] < 0:
                    grad[i, k] += 1.0
                else:
                    # On the center line, either direction increases clearance initially?
                    # No, if c=0, clearance=0.5. Moving to 0.1 -> 0.4. Moving to -0.1 -> 0.4.
                    # Gradient is undefined/discontinuous. Pick random or 0.
                    grad[i, k] += np.random.choice([-1, 1])
                    
            # Normalize gradient
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-10:
                # No active constraints or zero gradient?
                # This shouldn't happen if r is not infinite.
                # Maybe all distances are large?
                # If min_d is large, we are not constrained by pairs?
                # But we are always constrained by the cube.
                # If min_c is large, we are not constrained by bounds?
                # If both are large, we can move anywhere?
                # But we want to maximize the MIN.
                # If min_d > min_c, only bounds are active.
                # If min_c > min_d, only pairs are active.
                # If both are active, we have both.
                # If norm is 0, it means no active constraints? Impossible.
                # Maybe numerical issue.
                break
                
            grad /= norm_grad
            
            # Take step
            c_new = c + step_size * grad
            
            # Project back to cube?
            # The gradient step might push points outside.
            # We should clip. But clipping changes the direction.
            # Better: if a point goes outside, we should have had a stronger boundary gradient.
            # Let's just clip for safety, but this might cause issues.
            # Actually, if we are at the boundary, the gradient points IN.
            # So step should move IN.
            # If we are NOT at boundary, gradient is 0 for that dim.
            # So we should not go outside if we are careful.
            # But if we are close, we might overshoot.
            # Clip is safe.
            c_new = np.clip(c_new, -0.5, 0.5)
            
            # Check if we improved
            r_new = score_centres(c_new)
            
            if r_new > r_curr:
                c = c_new
                # Increase step size slightly if improving?
                # step_size *= 1.05
            else:
                # Decrease step size
                step_size *= 0.5
                # If step is too small, we are at a local max of the non-smooth function
                # or stuck.
                # Try a random kick to escape?
                if step_size < 1e-6:
                    # Small random perturbation
                    c = c + np.random.normal(0, 1e-5, size=c.shape)
                    c = np.clip(c, -0.5, 0.5)
                    step_size = 0.01 # Reset step size
                    # Limit resets to avoid infinite loop
                    # We can track resets, but for now, let it run.
                    
        # End local search loop
        
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 50

def score_centres(c):
    # c is (N, 3)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    clearances = 0.5 - np.abs(c)
    min_clearance = np.min(clearances)
    
    return min(min_pair_dist, min_clearance) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_r = 0.0
    best_c = None
    
    starts = []
    
    # 1. Grid start
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        starts.append(grid[:N].copy())
        
    # 2. Random starts
    np.random.seed(42)
    for i in range(15):
        starts.append(np.random.uniform(-0.5, 0.5, (N, 3)))
        
    for c_init in starts:
        if time.time() > deadline:
            break
            
        c = c_init.copy()
        
        step_size = 0.01
        min_step = 1e-9
        resets = 0
        
        for it in range(3000):
            if time.time() > deadline:
                break
                
            r = score_centres(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
            d = cdist(c, c)
            np.fill_diagonal(d, np.inf)
            clearances = 0.5 - np.abs(c)
            
            min_d = np.min(d)
            min_c = np.min(clearances)
            
            r_curr = min(min_d, min_c) / 2.0
            
            if step_size < min_step:
                break
                
            tol = 1e-6
            active_pairs = np.argwhere(d <= min_d + tol)
            
            active_bounds = []
            for i in range(N):
                for k in range(3):
                    if clearances[i, k] <= min_c + tol:
                        active_bounds.append((i, k))
                        
            grad = np.zeros_like(c)
            
            for i, j in active_pairs:
                if i == j: continue
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.normal(size=3)
                    dist = np.linalg.norm(diff)
                grad[i] += diff / dist
                grad[j] -= diff / dist
                
            for i, k in active_bounds:
                if c[i, k] > 0:
                    grad[i, k] -= 1.0
                elif c[i, k] < 0:
                    grad[i, k] += 1.0
                else:
                    grad[i, k] += np.random.choice([-1, 1])
                    
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-10:
                break
                
            grad /= norm_grad
            
            c_new = c + step_size * grad
            c_new = np.clip(c_new, -0.5, 0.5)
            
            r_new = score_centres(c_new)
            
            if r_new > r_curr:
                c = c_new
                # step_size *= 1.02
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    c = c + np.random.normal(0, 1e-4, size=c.shape)
                    c = np.clip(c, -0.5, 0.5)
                    step_size = 0.01
                    resets += 1
                    if resets > 10:
                        break
                        
    if best_c is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c
# EVOLVE-BLOCK-END
