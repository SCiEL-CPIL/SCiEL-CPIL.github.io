"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
from scipy.optimize import minimize, fsolve
import time

N = 26

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    best_result = None
    best_score = -np.inf
    
    def evaluate(x, y, r):
        """Check validity and compute score."""
        if np.any(r < 0) or np.any(r > 0.5):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)
    
    def solve_contact_system(x0, y0, r0, contact_pairs, boundary_contacts):
        """
        Solve for circle positions given a contact graph.
        contact_pairs: list of (i, j) pairs that are tangent
        boundary_contacts: dict mapping circle index to boundary ('L','R','B','T' or combinations)
        
        We fix radii and solve for positions, or fix positions and solve for radii.
        Here we use a hybrid: start from initial guess and iterate.
        """
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        r = r0.copy().astype(float)
        
        # Iterate: for each contact pair, enforce tangency
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            changed = False
            # Enforce boundary contacts
            for i, bounds in boundary_contacts.items():
                if 'L' in bounds:
                    new_r = x[i]
                    if abs(new_r - r[i]) > 1e-10:
                        r[i] = new_r
                        changed = True
                if 'R' in bounds:
                    new_r = 1.0 - x[i]
                    if abs(new_r - r[i]) > 1e-10:
                        r[i] = new_r
                        changed = True
                if 'B' in bounds:
                    new_r = y[i]
                    if abs(new_r - r[i]) > 1e-10:
                        r[i] = new_r
                        changed = True
                if 'T' in bounds:
                    new_r = 1.0 - y[i]
                    if abs(new_r - r[i]) > 1e-10:
                        r[i] = new_r
                        changed = True
            
            # Enforce contact pairs
            for (i, j) in contact_pairs:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-15)
                target_dist = r[i] + r[j]
                
                if dist < 1e-10:
                    continue
                    
                # Adjust positions to make them tangent
                # Move each circle halfway along the line connecting centers
                error = target_dist - dist
                if abs(error) > 1e-10:
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    half_err = error / 2.0
                    x[i] += half_err * dx_norm
                    y[i] += half_err * dy_norm
                    x[j] -= half_err * dx_norm
                    y[j] -= half_err * dy_norm
                    changed = True
            
            if not changed:
                break
        
        # Check validity
        score = evaluate(x, y, r)
        if score > -np.inf:
            return x, y, r, score
        return None, None, None, -np.inf
    
    def generate_contact_graphs(rng, num_graphs=20):
        """Generate random contact graph structures."""
        graphs = []
        
        # Hexagonal-like structure
        for _ in range(num_graphs):
            # Assign each circle a "layer" based on position
            # Create a random spanning tree plus some extra edges
            parent = np.zeros(N, dtype=int)
            for i in range(1, N):
                # Each circle connects to a previous one (tree structure)
                parent[i] = rng.integers(0, i)
            
            contact_pairs = set()
            for i in range(1, N):
                contact_pairs.add((min(i, parent[i]), max(i, parent[i])))
            
            # Add some extra contacts for rigidity
            num_extra = rng.integers(3, 8)
            for _ in range(num_extra):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i != j:
                    contact_pairs.add((min(i, j), max(i, j)))
            
            # Boundary contacts: assign each circle to touch 1-2 boundaries
            boundary_contacts = {}
            boundaries = ['L', 'R', 'B', 'T']
            for i in range(N):
                num_bounds = rng.integers(1, 3)
                chosen = rng.choice(boundaries, size=num_bounds, replace=False)
                boundary_contacts[i] = list(chosen)
            
            graphs.append((list(contact_pairs), boundary_contacts))
        
        return graphs
    
    def make_hexagonal_init(rng):
        """Create a hexagonal packing initial guess."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Hexagonal grid
        spacing = 0.12
        row = 0
        col = 0
        idx = 0
        while idx < N:
            cx = spacing + col * 2 * spacing * np.cos(np.pi/6)
            cy = spacing + row * 2 * spacing * np.sin(np.pi/6)
            if col % 2 == 1:
                cy += spacing * np.sin(np.pi/6)
            
            if cx < 0.95 and cy < 0.95:
                x[idx] = cx + (rng.random() - 0.5) * 0.02
                y[idx] = cy + (rng.random() - 0.5) * 0.02
                r[idx] = spacing * 0.95 + (rng.random() - 0.5) * 0.01
                idx += 1
            
            col += 1
            if col > 8:
                col = 0
                row += 1
        
        # Clip to valid range
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        r = np.clip(r, 0.02, 0.45)
        
        return x, y, r
    
    # Try multiple contact graph structures
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        
        # Generate initial positions
        x0, y0, r0 = make_hexagonal_init(rng)
        
        # Generate contact graph
        contact_pairs, boundary_contacts = generate_contact_graphs(rng, num_graphs=1)[0]
        
        # Solve the contact system
        x_sol, y_sol, r_sol, score = solve_contact_system(x0, y0, r0, contact_pairs, boundary_contacts)
        
        if x_sol is not None and score > best_score:
            best_score = score
            best_result = np.column_stack([x_sol, y_sol, r_sol])
    
    # Final polish: use gradient-based optimization from best result
    if best_result is not None and time.time() < deadline - 20:
        x, y, r = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        
        def objective(z):
            xi = z[0::3]
            yi = z[1::3]
            ri = z[2::3]
            
            score = np.sum(ri)
            penalty = 0.0
            
            for i in range(N):
                viol = ri[i] - xi[i]
                if viol > 0:
                    penalty += viol * viol
                viol = ri[i] - (1.0 - xi[i])
                if viol > 0:
                    penalty += viol * viol
                viol = ri[i] - yi[i]
                if viol > 0:
                    penalty += viol * viol
                viol = ri[i] - (1.0 - yi[i])
                if viol > 0:
                    penalty += viol * viol
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                    overlap = ri[i] + ri[j] - dist
                    if overlap > 0:
                        penalty += overlap * overlap
            
            return -score + penalty
        
        z0 = np.zeros(3 * N)
        z0[0::3] = x
        z0[1::3] = y
        z0[2::3] = r
        
        try:
            result = minimize(
                objective,
                z0,
                method='L-BFGS-B',
                bounds=[(0.001, 0.499)] * (3 * N),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < 0:  # valid
                xp = result.x[0::3]
                yp = result.x[1::3]
                rp = result.x[2::3]
                score = evaluate(xp, yp, rp)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([xp, yp, rp])
        except Exception:
            pass
    
    # If no valid result found, return the trivial seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
