import numpy as np
from scipy.optimize import fsolve
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_result = None
    best_score = -np.inf

    def is_valid(xy, r, tol=1e-7):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1.0 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1.0 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def update_best(xy, r):
        nonlocal best_result, best_score
        if is_valid(xy, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_result = (xy.copy(), r.copy())

    def greedy_contact_init(rng):
        """Place circles one-by-one; each new circle grows until it touches
        an existing circle or a wall, then is placed at the optimal spot."""
        n = N
        # Start with a coarse grid of tiny seeds
        side = int(np.ceil(np.sqrt(n)))
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.005

        # Now iteratively grow each circle to its maximum feasible radius
        # given the current positions of all circles.
        # We do multiple passes, growing one circle at a time.
        for _pass in range(3):
            order = rng.permutation(n)
            for i in order:
                # Find max radius for circle i given others fixed
                # Constraints:
                #   r_i <= min_j (dist(i,j) - r_j) for all j != i
                #   r_i <= x_i, 1-x_i, y_i, 1-y_i
                max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                for j in range(n):
                    if j == i:
                        continue
                    d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                    max_r = min(max_r, d - r[j])
                max_r = max(max_r, 0.001)
                r[i] = max_r

        return xy, r

    def solve_contact_system(xy, r, contact_pairs, wall_contacts):
        """
        Solve the system where for each contact pair (i,j): dist(i,j) = r_i + r_j
        and for each wall contact, the circle touches the wall.
        
        Unknowns: xy (2n) and r (n) -> 3n unknowns.
        Equations: len(contact_pairs) + len(wall_contacts) should be ~ 3n.
        
        We use a subset of the most critical contacts to form a square system.
        """
        n = len(r)
        # We'll set up a system with exactly 3n equations.
        # Prioritize: wall contacts first, then circle-circle contacts.
        # If we have more than 3n, take the first 3n.
        # If fewer, we need to add more (e.g., force some circles to touch).
        
        # Build residual function
        def residuals(params):
            xy_p = params[:2*n].reshape(n, 2)
            r_p = params[2*n:]
            res = np.zeros(3*n)
            idx = 0
            
            # Wall contacts: for each (i, wall) where wall in {0:left,1:right,2:bottom,3:top}
            for (i, wall) in wall_contacts:
                if wall == 0:  # left
                    res[idx] = xy_p[i, 0] - r_p[i]
                elif wall == 1:  # right
                    res[idx] = 1.0 - xy_p[i, 0] - r_p[i]
                elif wall == 2:  # bottom
                    res[idx] = xy_p[i, 1] - r_p[i]
                elif wall == 3:  # top
                    res[idx] = 1.0 - xy_p[i, 1] - r_p[i]
                idx += 1
            
            # Circle-circle contacts
            for (i, j) in contact_pairs:
                dx = xy_p[j, 0] - xy_p[i, 0]
                dy = xy_p[j, 1] - xy_p[i, 1]
                d = np.sqrt(dx*dx + dy*dy)
                res[idx] = d - r_p[i] - r_p[j]
                idx += 1
            
            # Pad with zeros if not enough equations
            if idx < 3*n:
                # Add boundary constraints for circles not yet constrained
                for k in range(idx, 3*n):
                    # Force some circles to stay within bounds softly
                    ci = k % n
                    res[k] = 0.0  # placeholder, should not happen often
            
            return res[:3*n]
        
        # Initial guess
        x0 = np.concatenate([xy.flatten(), r])
        
        try:
            sol, info, ier, msg = fsolve(residuals, x0, full_output=True, maxfev=2000)
            if ier == 1:
                xy_sol = sol[:2*n].reshape(n, 2)
                r_sol = sol[2*n:]
                if np.all(r_sol > 0):
                    return xy_sol, r_sol
        except Exception:
            pass
        
        return None, None

    def find_contacts(xy, r, threshold=1e-3):
        """Identify which circles are in contact (touching) and which walls they touch."""
        n = len(r)
        contact_pairs = []
        wall_contacts = []
        
        # Circle-circle
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
                if d < r[i] + r[j] + threshold:
                    contact_pairs.append((i, j))
        
        # Wall contacts
        for i in range(n):
            if xy[i, 0] - r[i] < threshold:
                wall_contacts.append((i, 0))
            if 1.0 - xy[i, 0] - r[i] < threshold:
                wall_contacts.append((i, 1))
            if xy[i, 1] - r[i] < threshold:
                wall_contacts.append((i, 2))
            if 1.0 - xy[i, 1] - r[i] < threshold:
                wall_contacts.append((i, 3))
        
        return contact_pairs, wall_contacts

    def refine_with_contacts(xy, r, max_rounds=5):
        """Iteratively identify contacts, solve the contact system, and refine."""
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            
            contact_pairs, wall_contacts = find_contacts(xy, r, threshold=1e-2)
            
            # Ensure we have enough equations: 3n total
            # If too few, add more contacts with slightly larger threshold
            total_eqs = len(contact_pairs) + len(wall_contacts)
            if total_eqs < 3*N:
                # Expand threshold
                contact_pairs, wall_contacts = find_contacts(xy, r, threshold=0.05)
                total_eqs = len(contact_pairs) + len(wall_contacts)
            
            if total_eqs >= 3*N:
                # Trim to exactly 3n if needed
                if len(wall_contacts) > 3*N:
                    wall_contacts = wall_contacts[:3*N]
                elif len(wall_contacts) + len(contact_pairs) > 3*N:
                    remaining = 3*N - len(wall_contacts)
                    contact_pairs = contact_pairs[:remaining]
                
                xy_new, r_new = solve_contact_system(xy, r, contact_pairs, wall_contacts)
                if xy_new is not None:
                    if is_valid(xy_new, r_new, tol=1e-5):
                        xy, r = xy_new, r_new
                    else:
                        # If solution is slightly invalid, try to repair
                        # by shrinking radii slightly
                        r_try = r_new * 0.99
                        if is_valid(xy_new, r_try, tol=1e-5):
                            xy, r = xy_new, r_try
                        else:
                            break
                else:
                    break
            else:
                break
        
        return xy, r

    # Main loop: multiple random restarts with greedy init + contact refinement
    rng = np.random.RandomState(123)
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Standard grid
            side = int(np.ceil(np.sqrt(N)))
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
                r[i] = 0.5 / side - 0.001
        elif restart == 1:
            # Hexagonal-like grid
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            cols = 6
            for i in range(N):
                row = i // cols
                col = i % cols
                x = (col + 0.5) / cols + (0.5 / cols if row % 2 == 1 else 0)
                y = (row + 0.5) / (N // cols + 1)
                xy[i] = [min(max(x, 0.05), 0.95), min(max(y, 0.05), 0.95)]
                r[i] = 0.08
        else:
            # Random perturbation of grid
            side = int(np.ceil(np.sqrt(N)))
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
                r[i] = 0.5 / side - 0.001
            xy += rng.uniform(-0.03, 0.03, size=(N, 2))
            xy = np.clip(xy, 0.05, 0.95)
        
        # Greedy grow-to-contact
        xy, r = greedy_contact_init(rng)
        update_best(xy, r)
        
        # Contact-based refinement
        xy, r = refine_with_contacts(xy, r, max_rounds=3)
        update_best(xy, r)
    
    if best_result is not None:
        xy, r = best_result
        # Final safety clamp
        r = np.clip(r, 0.001, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        if is_valid(xy, r):
            return np.column_stack([xy, r])
        # If clamping broke validity, shrink radii
        r *= 0.9
        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        if is_valid(xy, r):
            return np.column_stack([xy, r])
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        rad = 0.5 / side - 0.002
        cells.append([x, y, rad])
    return np.array(cells)
