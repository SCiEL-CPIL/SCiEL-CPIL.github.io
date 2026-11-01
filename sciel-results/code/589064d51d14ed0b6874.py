import numpy as np
from scipy.optimize import linprog
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Initial Configuration: Hexagonal packing approximation
    # For N=32, a 6x6 grid is too sparse. 4x8 or 5x7?
    # Let's try to generate a dense initial packing using a simple grid and then optimize.
    # A good initial guess for 32 circles in unit square:
    # If we assume equal radius R, max R is approx 1/sqrt(N) * packing_density_factor.
    # sqrt(32) ~ 5.6. 1/5.6 ~ 0.18. But packing density < 1.
    # Typical R for N=32 is around 0.10 - 0.12.
    
    # We will use an iterative linearization approach.
    # Variables: x_i, y_i, r_i for i in 0..N-1. Total 3N variables.
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    # Constraints:
    # 1. x_i - r_i >= 0
    # 2. x_i + r_i <= 1
    # 3. y_i - r_i >= 0
    # 4. y_i + r_i <= 1
    # 5. r_i >= 0
    # 6. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    #    This is non-linear. We linearize it using the current positions (x_k, y_k) and radii (r_k).
    #    Tangent plane at (x_k, y_k, r_k):
    #    2(x_k - x_j_k)(x_i - x_j) + 2(y_k - y_j_k)(y_i - y_j) >= (r_k + r_j_k)^2 - ((x_k-x_j_k)^2 + (y_k-y_j_k)^2) + 2(r_k+r_j_k)(r_i+r_j)
    #    Wait, the standard linearization for ||u|| >= v is:
    #    u_k^T u >= v_k^2 + 2 v_k (v - v_k)
    #    Here u = [x_i-x_j, y_i-y_j], v = r_i + r_j.
    #    Let dx_k = x_i_k - x_j_k, dy_k = y_i_k - y_j_k, rsum_k = r_i_k + r_j_k.
    #    Constraint: dx_k*(x_i-x_j) + dy_k*(y_i-y_j) + rsum_k*(r_i+r_j) >= (dx_k^2 + dy_k^2)/2 + rsum_k^2/2
    #    Actually, the tangent plane to the cone x^2+y^2 >= z^2 at (x0, y0, z0) is:
    #    x0*x + y0*y >= z0*z + (x0^2+y0^2-z0^2)/2
    #    If we are at a feasible point, x0^2+y0^2 >= z0^2.
    #    If we start with a feasible configuration, we can iteratively solve the LP.
    
    # Step 1: Generate an initial feasible configuration.
    # Use a simple grid with small radius.
    grid_side = 6
    # 32 circles in 6x6 grid (36 cells). Pick 32.
    # Or just use 4x8 grid.
    rows = 4
    cols = 8
    xy_init = np.zeros((N, 2))
    r_init = np.zeros(N)
    
    # Hexagonal offset for better initial packing
    for i in range(N):
        row = i // cols
        col = i % cols
        offset = 0.5/cols if row % 2 == 1 else 0
        xy_init[i, 0] = (col + 0.5)/cols + offset
        xy_init[i, 1] = (row + 0.5)/rows
        # Clamp to [0,1]
        xy_init[i, 0] = np.clip(xy_init[i, 0], 0.01, 0.99)
        xy_init[i, 1] = np.clip(xy_init[i, 1], 0.01, 0.99)
        
    # Calculate max possible radius for this grid to ensure feasibility
    # Min distance between centers:
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(xy_init[i] - xy_init[j])
            min_dist = min(min_dist, d)
    # Also check boundaries
    for i in range(N):
        min_dist = min(min_dist, xy_init[i, 0], 1-xy_init[i, 0], xy_init[i, 1], 1-xy_init[i, 1])
        
    r_init[:] = min_dist / 2 * 0.95 # Slightly smaller to be safe
    # Ensure r_init is not too small
    r_init[:] = np.maximum(r_init, 0.01)
    
    # Verify initial feasibility
    def is_valid(xy, r, tol=1e-7):
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

    if not is_valid(xy_init, r_init):
        # Shrink radii until valid
        scale = 1.0
        while not is_valid(xy_init, r_init * scale) and scale > 0.1:
            scale *= 0.9
        r_init *= scale
        if not is_valid(xy_init, r_init):
            # Fallback: very small circles
            r_init[:] = 0.01
            xy_init[:, 0] = np.linspace(0.05, 0.95, N)
            xy_init[:, 1] = np.linspace(0.05, 0.95, N)
            # This might still be invalid if N is large, but for 32 it's likely ok with 0.01
            # Actually, let's just use a grid with very small radius
            for i in range(N):
                row = i // 8
                col = i % 8
                xy_init[i, 0] = (col + 0.5)/8
                xy_init[i, 1] = (row + 0.5)/4
            r_init[:] = 0.02

    current_xy = xy_init.copy()
    current_r = r_init.copy()
    best_score = np.sum(current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    rng = np.random.RandomState(42)
    
    # Iterative Linearization
    # We will iterate:
    # 1. Build LP with current linearization
    # 2. Solve LP
    # 3. Check if new solution is feasible for the original problem
    # 4. If yes, update current solution. If no, we might need to repair or backtrack.
    # Since LP solution is a vertex, it might violate the non-linear constraints if the linearization is too loose.
    # However, if we start with a feasible point, the tangent plane constraints are satisfied by the current point.
    # The LP maximizes sum(r). The new point might be infeasible for the true constraints.
    # Strategy: If new point is infeasible, we can try to project it back or reduce the step size.
    # A simpler approach: Use the LP solution as the new linearization point ONLY IF it is feasible.
    # If it's not feasible, we can try to move halfway towards it? Or just keep the old point and try a different restart?
    # Actually, a common technique is to solve the LP, and if the solution is infeasible for the true constraints,
    # we can treat it as a "best effort" and try to repair it, or simply reject it and perturb the current point to find a new linearization.
    
    # Given the complexity, let's try a simpler "Iterative Linear Programming" (ILP) approach:
    # Solve LP. If solution is feasible, accept it.
    # If not, we might try to blend it with the previous feasible solution.
    
    max_iters = 100
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Build LP
        # Variables: x_0..x_31, y_0..y_31, r_0..r_31
        # c = -[0...0, 0...0, 1...1] (minimize -sum r)
        c = np.zeros(3*N)
        c[2*N:] = -1.0
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints:
        # x_i - r_i >= 0  => -x_i + r_i <= 0
        # -x_i + r_i <= 0
        # x_i + r_i <= 1
        # y_i - r_i >= 0  => -y_i + r_i <= 0
        # y_i + r_i <= 1
        # r_i >= 0 => -r_i <= 0
        
        # We can handle boundaries with A_ub and b_ub, or bounds.
        # Bounds are better for x, y, r.
        # x_i >= r_i is not a simple bound. It's a constraint.
        # So we must use A_ub for x_i - r_i >= 0.
        
        # Constraint 1: x_i - r_i >= 0  => -x_i + r_i <= 0
        # Constraint 2: 1 - x_i - r_i >= 0 => x_i + r_i <= 1
        # Constraint 3: y_i - r_i >= 0 => -y_i + r_i <= 0
        # Constraint 4: 1 - y_i - r_i >= 0 => y_i + r_i <= 1
        
        # Let's build these.
        # For each i:
        #   -x_i + r_i <= 0
        #    x_i + r_i <= 1
        #   -y_i + r_i <= 0
        #    y_i + r_i <= 1
        
        # And separation constraints:
        # For each pair (i, j):
        # dx_k*(x_i - x_j) + dy_k*(y_i - y_j) + rsum_k*(r_i + r_j) >= rhs_k
        # -dx_k*x_i + dx_k*x_j - dy_k*y_i + dy_k*y_j - rsum_k*r_i - rsum_k*r_j <= -rhs_k
        
        n_pairs = 0
        iu = np.triu_indices(N, k=1)
        
        # Pre-allocate if possible, but list append is fine for N=32
        # 4*N boundary constraints + N pairs separation constraints
        # 4*32 + 496 = 128 + 496 = 624 constraints.
        
        for i in range(N):
            # -x_i + r_i <= 0
            row1 = np.zeros(3*N)
            row1[i] = -1.0
            row1[2*N+i] = 1.0
            A_ub.append(row1)
            b_ub.append(0.0)
            
            # x_i + r_i <= 1
            row2 = np.zeros(3*N)
            row2[i] = 1.0
            row2[2*N+i] = 1.0
            A_ub.append(row2)
            b_ub.append(1.0)
            
            # -y_i + r_i <= 0
            row3 = np.zeros(3*N)
            row3[N+i] = -1.0
            row3[2*N+i] = 1.0
            A_ub.append(row3)
            b_ub.append(0.0)
            
            # y_i + r_i <= 1
            row4 = np.zeros(3*N)
            row4[N+i] = 1.0
            row4[2*N+i] = 1.0
            A_ub.append(row4)
            b_ub.append(1.0)
            
        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            
            dx_k = current_xy[i, 0] - current_xy[j, 0]
            dy_k = current_xy[i, 1] - current_xy[j, 1]
            rsum_k = current_r[i] + current_r[j]
            
            # Avoid division by zero or instability if points are identical
            if dx_k == 0 and dy_k == 0:
                dx_k = 1e-6
                dy_k = 1e-6
                
            # RHS of the inequality:
            # dx_k^2 + dy_k^2 >= rsum_k^2 is the true constraint.
            # Linearization:
            # dx_k*(x_i-x_j) + dy_k*(y_i-y_j) + rsum_k*(r_i+r_j) >= (dx_k^2 + dy_k^2)/2 + rsum_k^2/2
            # Wait, the tangent plane to the second-order cone x^2+y^2 >= z^2 at (x0,y0,z0) is:
            # x0*x + y0*y >= z0*z + (x0^2+y0^2-z0^2)/2
            # Here x = x_i-x_j, y = y_i-y_j, z = r_i+r_j.
            # So:
            # dx_k*(x_i-x_j) + dy_k*(y_i-y_j) >= rsum_k*(r_i+r_j) + (dx_k^2 + dy_k^2 - rsum_k^2)/2
            
            # Rearranging to A_ub * vars <= b_ub:
            # -dx_k*(x_i-x_j) - dy_k*(y_i-y_j) - rsum_k*(r_i+r_j) <= -(dx_k^2 + dy_k^2 - rsum_k^2)/2
            
            row = np.zeros(3*N)
            row[i] = -dx_k
            row[j] = dx_k
            row[N+i] = -dy_k
            row[N+j] = dy_k
            row[2*N+i] = -rsum_k
            row[2*N+j] = -rsum_k
            
            rhs = -(dx_k**2 + dy_k**2 - rsum_k**2) / 2.0
            
            A_ub.append(row)
            b_ub.append(rhs)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for variables
        # x_i in [0, 1]
        # y_i in [0, 1]
        # r_i in [0, 1]
        bounds = [(0, 1)] * (3*N)
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                new_params = res.x
                new_xy = new_params[:2*N].reshape(N, 2)
                new_r = new_params[2*N:]
                
                # Check if new solution is feasible for the true non-linear constraints
                if is_valid(new_xy, new_r, tol=1e-6):
                    new_score = np.sum(new_r)
                    if new_score > best_score:
                        best_score = new_score
                        best_xy = new_xy.copy()
                        best_r = new_r.copy()
                        
                    # Update current solution for next linearization
                    current_xy = new_xy
                    current_r = new_r
                    no_improve_count = 0
                else:
                    # Solution from LP is not feasible for true problem.
                    # This can happen if the linearization is not tight enough or we moved too far.
                    # Strategy: Try to move halfway towards the new solution?
                    # Or just perturb the current solution to break symmetry and try again.
                    # Let's try a small perturbation of the current feasible solution.
                    current_xy += rng.normal(0, 0.001, size=(N, 2))
                    current_r += rng.normal(0, 0.001, size=N)
                    # Ensure feasibility after perturbation
                    if not is_valid(current_xy, current_r):
                        # Revert to last known good if perturbation broke it
                        current_xy = best_xy
                        current_r = best_r
                    no_improve_count += 1
            else:
                # LP failed
                no_improve_count += 1
        except Exception:
            no_improve_count += 1
            
        # If we haven't improved in a while, try a restart with a different initial config?
        # Or just break and return best.
        if no_improve_count > 20:
            # Try a random restart with a different grid orientation or random init
            # For now, just break to save time, as we have a valid best.
            # A more robust approach would be to try a few random restarts here.
            pass
            
    # Final verification and return
    if is_valid(best_xy, best_r):
        return np.column_stack([best_xy, best_r])
    else:
        # Fallback
        return np.column_stack([xy_init, r_init])

# Test
if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(np.sum(res[:, 2]))
