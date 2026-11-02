"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def construct():
    """
    Optimizes circle packing using a contact-graph rigid-body assembly approach.
    Builds a sparse contact graph, solves for equilibrium using sparse linear solvers,
    and iteratively adjusts radius to find optimal packing.
    """
    
    def hex_initial(n):
        """Hexagonal lattice, centered and scaled to fit in [-0.5,0.5]^2."""
        pts = []
        for row in range(12):
            for col in range(12):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        if max_abs > 0:
            pts *= (0.48 / max_abs)
        return pts
    
    def grid_initial(n):
        """Grid layout."""
        m = int(np.ceil(np.sqrt(n)))
        xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, m), np.linspace(-0.45, 0.45, m))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts

    def compute_radius(pts):
        """Compute the packing radius for a configuration."""
        dists = np.sqrt(((pts[:, np.newaxis] - pts[np.newaxis]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)

    def build_contact_graph(pts, r, k_neighbors=6):
        """
        Build a contact graph: for each circle, find its k_nearest neighbors
        and those at distance close to 2r (contact distance).
        Returns list of (i, j, weight) for edges in the graph.
        """
        n = len(pts)
        # Compute pairwise distances
        dists = np.sqrt(((pts[:, np.newaxis] - pts[np.newaxis]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        
        edges = set()
        for i in range(n):
            # Find k nearest neighbors
            idx = np.argsort(dists[i])[:k_neighbors]
            for j in idx:
                if i < j:
                    edges.add((i, j))
            # Also add contacts at distance close to 2r
            close_contacts = np.where(dists[i] < 2.0 * r * 1.05)[0]
            for j in close_contacts:
                if i < j:
                    edges.add((i, j))
        
        return list(edges)

    def solve_equilibrium(pts, r, edges, max_iter=200, tol=1e-10):
        """
        Solve for equilibrium positions using iterative sparse linear solves.
        At equilibrium, forces from all contacts balance.
        
        For each edge (i,j), if dist < 2r, there's a repulsive force.
        We solve the linear system: A * delta_x = b, where A is the stiffness matrix
        and b is the net force.
        """
        n = len(pts)
        pos = pts.copy()
        
        for iteration in range(max_iter):
            # Compute forces
            forces = np.zeros((n, 2))
            
            # Pairwise forces from contact graph
            for (i, j) in edges:
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    # Perturb if coincident
                    d_vec = np.array([1e-8, 1e-8])
                    d = np.linalg.norm(d_vec)
                
                if d < 2 * r:
                    # Repulsive force proportional to overlap
                    overlap = 2 * r - d
                    force_mag = overlap / d
                    forces[i] += force_mag * d_vec
                    forces[j] -= force_mag * d_vec
            
            # Boundary forces: push circles back if they're too close to boundary
            bound_limit = 0.5 - r
            for i in range(n):
                # x-boundary
                if pos[i, 0] > bound_limit:
                    forces[i, 0] -= (pos[i, 0] - bound_limit) * 10.0
                elif pos[i, 0] < -bound_limit:
                    forces[i, 0] -= (pos[i, 0] + bound_limit) * 10.0
                # y-boundary
                if pos[i, 1] > bound_limit:
                    forces[i, 1] -= (pos[i, 1] - bound_limit) * 10.0
                elif pos[i, 1] < -bound_limit:
                    forces[i, 1] -= (pos[i, 1] + bound_limit) * 10.0
            
            # Build stiffness matrix (sparse)
            # A is 2n x 2n, symmetric positive definite
            A = lil_matrix((2 * n, 2 * n))
            
            # Self-stiffness from all contacts
            for (i, j) in edges:
                # Add to diagonal for i and j
                A[2*i, 2*i] += 1.0
                A[2*i, 2*i+1] += 0.0
                A[2*i+1, 2*i] += 0.0
                A[2*i+1, 2*i+1] += 1.0
                
                A[2*j, 2*j] += 1.0
                A[2*j, 2*j+1] += 0.0
                A[2*j+1, 2*j] += 0.0
                A[2*j+1, 2*j+1] += 1.0
                
                # Off-diagonal coupling
                A[2*i, 2*j] -= 1.0
                A[2*i, 2*j+1] -= 0.0
                A[2*i+1, 2*j] -= 0.0
                A[2*i+1, 2*j+1] -= 1.0
                
                A[2*j, 2*i] -= 1.0
                A[2*j, 2*i+1] -= 0.0
                A[2*j+1, 2*i] -= 0.0
                A[2*j+1, 2*i+1] -= 1.0
            
            # Add boundary stiffness to diagonal
            for i in range(n):
                A[2*i, 2*i] += 10.0
                A[2*i+1, 2*i+1] += 10.0
            
            A_csr = A.tocsr()
            
            # Solve A * delta = forces
            try:
                delta = spsolve(A_csr, forces.flatten())
                delta = delta.reshape(n, 2)
            except Exception:
                # Fallback to simple gradient descent step
                delta = forces * 0.01
            
            # Limit step size for stability
            max_step = np.max(np.abs(delta))
            if max_step > 0.01:
                delta *= 0.01 / max_step
            
            pos += delta
            
            # Check convergence
            if np.max(np.abs(delta)) < tol:
                break
        
        return pos

    def solve_for_radius(pts_init, r, max_iter=50):
        """
        Try to find a configuration with radius r using the contact-graph solver.
        Returns (pos, success).
        """
        pos = pts_init.copy()
        
        for _ in range(max_iter):
            # Build contact graph based on current positions
            edges = build_contact_graph(pos, r, k_neighbors=6)
            
            # Solve for equilibrium
            pos = solve_equilibrium(pos, r, edges, max_iter=100)
            
            # Check feasibility
            dists = np.sqrt(((pos[:, np.newaxis] - pos[np.newaxis]) ** 2).sum(-1))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            # Also check boundaries
            max_abs = np.max(np.abs(pos), axis=1)
            min_bound_dist = 0.5 - np.max(max_abs)
            
            if min_dist >= 2 * r - 1e-8 and min_bound_dist >= r - 1e-8:
                return pos, True
            else:
                # Not feasible, continue iterating
                continue
        
        return pos, False

    # Initial configurations
    configs = [
        hex_initial(N),
        grid_initial(N)
    ]
    
    # Add jittered versions
    rng = np.random.default_rng(42)
    for _ in range(4):
        base = hex_initial(N)
        jitter = rng.normal(0, 0.04, size=(N, 2))
        cfg = base + jitter
        cfg = np.clip(cfg, -0.49, 0.49)
        configs.append(cfg)
    
    best_pos = None
    best_r = 0.0
    
    for cfg in configs:
        # Estimate initial radius
        r0 = compute_radius(cfg)
        
        # Binary search for max radius
        r_lo = r0 * 0.98
        r_hi = 0.08  # Safe upper bound
        
        # Ensure r_lo is feasible
        pos_lo, feas_lo = solve_for_radius(cfg, r_lo)
        if not feas_lo:
            continue
        
        # Binary search
        for _ in range(25):
            r_mid = (r_lo + r_hi) / 2
            pos_mid, feas_mid = solve_for_radius(pos_lo, r_mid)
            
            if feas_mid:
                r_lo = r_mid
                pos_lo = pos_mid
                if r_mid > best_r:
                    best_r = r_mid
                    best_pos = pos_mid.copy()
            else:
                r_hi = r_mid
            
            if r_hi - r_lo < 1e-7:
                break
    
    # Safety fallback
    if best_pos is None or np.any(~np.isfinite(best_pos)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    # Final check: no coincident centers
    dists = np.sqrt(((best_pos[:, np.newaxis] - best_pos[np.newaxis]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
