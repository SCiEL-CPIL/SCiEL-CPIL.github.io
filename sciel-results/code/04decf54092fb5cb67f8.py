import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def _elapsed():
        return time.time() - start_time

    def _is_valid(xy, r, tol=1e-9):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def _sum_r(r):
        return np.sum(r)

    def _objective(params, lam=10.0):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Objective: Maximize sum(r) - Penalty(overlaps)
        # We want to maximize sum(r), so minimize -sum(r) + penalty
        penalty = 0.0
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        penalty = lam * np.sum(overlap**2)
        
        # Boundary penalty (soft, to keep optimizer stable)
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        b_pen = lam * (np.sum(np.maximum(0, -d_left)**2) + 
                       np.sum(np.maximum(0, -d_right)**2) + 
                       np.sum(np.maximum(0, -d_bottom)**2) + 
                       np.sum(np.maximum(0, -d_top)**2))
        
        return -np.sum(r) + penalty + b_pen

    def _physics_optimize(xy_init, r_init, max_time=10.0):
        """
        Physics-based relaxation.
        Forces:
        1. Repulsive force between overlapping circles.
        2. Boundary repulsion.
        3. 'Growth' force: We don't have a direct force for growth, 
           but we can periodically increase the target radius or 
           use a potential that rewards larger radii?
           Actually, standard MD minimizes energy. 
           If we define Energy = -Sum(r) + Penalty(overlap), 
           then the force on r_i is -dE/dr_i = 1 - dPenalty/dr_i.
           dPenalty/dr_i for overlap with j is 2*lam*(r_i+r_j-d_ij) (if overlapping).
           So if not overlapping, force is +1 (grow).
           If overlapping, force is 1 - 2*lam*overlap. If lam is large, this is negative (shrink).
           This naturally balances growth and overlap.
        """
        n = N
        xy = xy_init.copy()
        r = r_init.copy()
        
        # Parameters
        dt = 0.01
        damping = 0.9
        lam = 50.0 # Strong penalty to enforce separation
        v_xy = np.zeros_like(xy)
        v_r = np.zeros_like(r)
        
        end_time = time.time() + max_time
        
        while time.time() < end_time:
            f_xy = np.zeros_like(xy)
            f_r = np.zeros_like(r)
            
            # Pairwise interactions
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Force magnitude for penalty 0.5 * lam * overlap^2
                        # d/dx (0.5 * lam * (r_i+r_j-d)^2) = -lam * (r_i+r_j-d) * d/dx (r_i+r_j-d)
                        # d/dx_i (d) = -dx/d => d/dx_i (overlap) = dx/d
                        # Force on i: -dE/dx_i = -(-lam * overlap * (dx/d)) = lam * overlap * dx/d
                        # Wait, E = -sum(r) + lam * sum(overlap^2)
                        # dE/dx_i = lam * 2 * overlap * d(overlap)/dx_i
                        # d(overlap)/dx_i = d(min_dist - dist)/dx_i = -d(dist)/dx_i = dx/d
                        # So dE/dx_i = 2 * lam * overlap * (dx/d)
                        # Force = -dE/dx_i = -2 * lam * overlap * (dx/d)
                        # This pushes i in direction -dx, i.e., away from j. Correct.
                        
                        if dist > 1e-10:
                            fx = 2.0 * lam * overlap * (dx / dist)
                            fy = 2.0 * lam * overlap * (dy / dist)
                            f_xy[i, 0] -= fx
                            f_xy[i, 1] -= fy
                            f_xy[j, 0] += fx
                            f_xy[j, 1] += fy
                        else:
                            # Random push
                            angle = np.random.uniform(0, 2*np.pi)
                            f_xy[i, 0] -= 2.0 * lam * overlap * np.cos(angle)
                            f_xy[i, 1] -= 2.0 * lam * overlap * np.sin(angle)
                            f_xy[j, 0] += 2.0 * lam * overlap * np.cos(angle)
                            f_xy[j, 1] += 2.0 * lam * overlap * np.sin(angle)
                        
                        # Force on radii
                        # dE/dr_i = -1 + 2 * lam * overlap * d(overlap)/dr_i
                        # d(overlap)/dr_i = 1
                        # dE/dr_i = -1 + 2 * lam * overlap
                        # Force = -dE/dr_i = 1 - 2 * lam * overlap
                        f_r[i] += 1.0 - 2.0 * lam * overlap
                        f_r[j] += 1.0 - 2.0 * lam * overlap
                    else:
                        # No overlap, force on radii is just 1 (grow)
                        f_r[i] += 1.0
                        f_r[j] += 1.0

            # Boundary forces
            # Penalty: lam * max(0, r - x)^2
            # dE/dx = -2 * lam * (r - x) if x < r
            # Force = -dE/dx = 2 * lam * (r - x)
            for i in range(n):
                if xy[i, 0] < r[i]:
                    f_xy[i, 0] += 2.0 * lam * (r[i] - xy[i, 0])
                elif xy[i, 0] > 1 - r[i]:
                    f_xy[i, 0] -= 2.0 * lam * (xy[i, 0] - (1 - r[i]))
                
                if xy[i, 1] < r[i]:
                    f_xy[i, 1] += 2.0 * lam * (r[i] - xy[i, 1])
                elif xy[i, 1] > 1 - r[i]:
                    f_xy[i, 1] -= 2.0 * lam * (xy[i, 1] - (1 - r[i]))
            
            # Update velocities and positions
            v_xy += f_xy * dt
            v_r += f_r * dt
            v_xy *= damping
            v_r *= damping
            
            xy += v_xy * dt
            r += v_r * dt
            
            # Clip radii to reasonable bounds
            r = np.clip(r, 0.0, 0.5)
            
            # Hard clip positions to box to prevent escape
            xy[:, 0] = np.clip(xy[:, 0], 0, 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, 1)

        return xy, r

    def _make_valid(xy, r, max_iters=200):
        """
        Projection step to ensure validity.
        If physics leaves slight overlaps, fix them.
        """
        n = N
        xy = xy.copy()
        r = r.copy()
        
        for _ in range(max_iters):
            moved = False
            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist > 1e-10:
                            # Push apart
                            push_x = (dx / dist) * overlap
                            push_y = (dy / dist) * overlap
                            # Weight by inverse radius? Or equal? Equal is safer for stability
                            xy[i, 0] -= push_x * 0.5
                            xy[i, 1] -= push_y * 0.5
                            xy[j, 0] += push_x * 0.5
                            xy[j, 1] += push_y * 0.5
                        else:
                            xy[i, 0] -= 0.01
                            xy[j, 0] += 0.01
                        moved = True
            
            # Check boundaries
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                elif xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                elif xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True
            
            if not moved:
                break
        
        return xy, r

    def _tight_polish(xy, r, max_iter=200):
        """
        L-BFGS polish on the feasible set (using penalty for safety, but starting close).
        """
        n = N
        params = np.concatenate([xy.flatten(), r])
        
        def obj(p):
            xy_p = p[:2*n].reshape(n, 2)
            r_p = p[2*n:]
            # Strong penalty for any violation to keep it feasible
            penalty = 0.0
            dx = xy_p[:, 0][:, None] - xy_p[:, 0][None, :]
            dy = xy_p[:, 1][:, None] - xy_p[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r_p[:, None] + r_p[None, :]
            iu = np.triu_indices(n, k=1)
            overlap = np.maximum(0, min_dist[iu] - dist[iu])
            penalty += 1000.0 * np.sum(overlap**2)
            
            d_left = xy_p[:, 0] - r_p
            d_right = 1 - xy_p[:, 0] - r_p
            d_bottom = xy_p[:, 1] - r_p
            d_top = 1 - xy_p[:, 1] - r_p
            b_pen = 1000.0 * (np.sum(np.maximum(0, -d_left)**2) + 
                              np.sum(np.maximum(0, -d_right)**2) + 
                              np.sum(np.maximum(0, -d_bottom)**2) + 
                              np.sum(np.maximum(0, -d_top)**2))
            
            return -np.sum(r_p) + penalty + b_pen
        
        try:
            res = minimize(obj, params, method='L-BFGS-B', 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                xy_new = res.x[:2*n].reshape(n, 2)
                r_new = res.x[2*n:]
                # Verify
                if _is_valid(xy_new, r_new):
                    return xy_new, r_new
        except Exception:
            pass
        return xy, r

    # Main Loop
    best_xy = None
    best_r = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Initializations
    # 1. Hexagonal
    def get_hex_init():
        n = N
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        # 6x6 grid
        side = 6
        for i in range(n):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            if row % 2 == 1:
                x += 0.5 / side
            xy[i] = [x, y]
            r[i] = 0.5 / side * 0.9
        return xy, r

    # 2. Random
    def get_rand_init():
        n = N
        xy = rng.uniform(0.1, 0.9, size=(n, 2))
        r = rng.uniform(0.05, 0.1, size=n)
        return xy, r

    inits = [get_hex_init(), get_rand_init(), get_hex_init()] # Multiple hex starts
    
    for idx, (xy_init, r_init) in enumerate(inits):
        if _elapsed() > deadline - 20:
            break
            
        # Phase 1: Physics Growth
        # Allocate time based on remaining budget
        phys_time = min(10.0, deadline - _elapsed() - 5.0)
        if phys_time < 1.0:
            continue
            
        xy_phys, r_phys = _physics_optimize(xy_init, r_init, max_time=phys_time)
        
        # Phase 2: Make Valid
        xy_valid, r_valid = _make_valid(xy_phys, r_phys)
        
        # Phase 3: Tight Polish
        if _elapsed() < deadline - 5.0:
            xy_polished, r_polished = _tight_polish(xy_valid, r_valid)
            if _is_valid(xy_polished, r_polished):
                score = _sum_r(r_polished)
                if score > best_score:
                    best_score = score
                    best_xy = xy_polished
                    best_r = r_polished
        else:
            if _is_valid(xy_valid, r_valid):
                score = _sum_r(r_valid)
                if score > best_score:
                    best_score = score
                    best_xy = xy_valid
                    best_r = r_valid

    if best_xy is not None:
        return np.column_stack([best_xy, best_r])
    
    # Fallback
    side = 6
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
        r[i] = 0.5/side - 0.01
    return np.column_stack([xy, r])

if __name__ == "__main__":
    res = construct()
    print(res)
