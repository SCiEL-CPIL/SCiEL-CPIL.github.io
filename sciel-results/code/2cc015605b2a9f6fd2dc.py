"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def construct():
    """
    Sparse linear solver approach for rigid-body circle packing.
    
    Formulates the packing as a system of distance constraints on a contact graph.
    Uses iterative sparse linear solves to adjust node positions, bypassing
    local minima traps of smooth gradient-based methods.
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
    
    def jittered_hex(n, seed):
        """Hex with random jitter."""
        rng = np.random.RandomState(seed)
        pts = hex_initial(n)
        pts += rng.randn(n, 2) * 0.03
        return np.clip(pts, -0.49, 0.49)
    
    def pairwise_dists(pts):
        """Compute pairwise distance matrix."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        return dist
    
    def compute_radius(pts):
        """Compute the packing radius for a configuration."""
        dist = pairwise_dists(pts)
        r_pair = np.min(dist) / 2.0
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)
    
    def build_contact_graph(pts, k=6, margin=0.0):
        """
        Build contact graph: pairs within (2r + margin) distance.
        Returns list of (i, j) pairs.
        """
        dist = pairwise_dists(pts)
        n = len(pts)
        contacts = set()
        for i in range(n):
            nn = np.argsort(dist[i])[:k]
            for j in nn:
                a, b = min(i, j), max(i, j)
                contacts.add((a, b))
        return list(contacts)
    
    def solve_sparse_system(pts, r, contacts, n_iter=500, damping=0.5):
        """
        Solve for position adjustments using sparse linear system.
        
        For each contact (i,j) with dist < 2r, we need to push them apart.
        We build a sparse linear system A * delta = b where:
        - A encodes the direction constraints
        - b encodes the required separation
        
        Uses iterative refinement with sparse solves.
        """
        pos = pts.copy().astype(float)
        n = len(pos)
        dim = 2 * n
        
        for it in range(n_iter):
            # Build sparse system for this iteration
            A = lil_matrix((dim, dim))
            b = np.zeros(dim)
            
            # Contact constraints: for each pair too close, add separation force
            for (i, j) in contacts:
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < 2 * r:
                    if dist > 1e-12:
                        # Direction unit vector
                        u = d / dist
                        # Required separation
                        needed = 2 * r - dist
                        # Force: push i in -u direction, j in +u direction
                        # With damping to prevent oscillation
                        force = needed * damping
                        A[2*i, 2*i] += 1.0
                        A[2*i, 2*i+1] += 0.0
                        A[2*i+1, 2*i] += 0.0
                        A[2*i+1, 2*i+1] += 1.0
                        A[2*j, 2*j] += 1.0
                        A[2*j, 2*j+1] += 0.0
                        A[2*j+1, 2*j] += 0.0
                        A[2*j+1, 2*j+1] += 1.0
                        b[2*i] -= force * u[0]
                        b[2*i+1] -= force * u[1]
                        b[2*j] += force * u[0]
                        b[2*j+1] += force * u[1]
                    else:
                        # Coincident: random separation
                        angle = np.random.uniform(0, 2 * np.pi)
                        u = np.array([np.cos(angle), np.sin(angle)])
                        force = r * damping
                        A[2*i, 2*i] += 1.0
                        A[2*i+1, 2*i+1] += 1.0
                        A[2*j, 2*j] += 1.0
                        A[2*j+1, 2*j+1] += 1.0
                        b[2*i] -= force * u[0]
                        b[2*i+1] -= force * u[1]
                        b[2*j] += force * u[0]
                        b[2*j+1] += force * u[1]
            
            # Boundary constraints: nodes outside get pushed in
            for i in range(n):
                for c in range(2):
                    if pos[i, c] < -0.5:
                        A[2*i+c, 2*i+c] += 1.0
                        b[2*i+c] = -0.5 - pos[i, c]
                    elif pos[i, c] > 0.5:
                        A[2*i+c, 2*i+c] += 1.0
                        b[2*i+c] = 0.5 - pos[i, c]
            
            # Add small regularization to ensure invertibility
            for i in range(dim):
                A[i, i] += 1e-6
            
            # Solve sparse system
            A_csr = A.tocsr()
            try:
                delta = spsolve(A_csr, b)
                if np.any(~np.isfinite(delta)):
                    delta = np.zeros(dim)
            except:
                delta = np.zeros(dim)
            
            # Apply update
            pos += delta.reshape(n, 2)
            pos = np.clip(pos, -0.5, 0.5)
            
            # Check convergence
            if np.max(np.abs(delta)) < 1e-10:
                break
        
        return pos
    
    def refine_with_linear_solver(pts, r, n_iter=300):
        """
        Refine configuration using sparse linear solver.
        Rebuilds contact graph each iteration for robustness.
        """
        pos = pts.copy().astype(float)
        
        for outer in range(n_iter):
            # Rebuild contact graph based on current positions
            contacts = build_contact_graph(pos, k=8, margin=0.0)
            
            # Solve sparse system
            pos = solve_sparse_system(pos, r, contacts, n_iter=50, damping=0.3)
            
            # Check if we've achieved feasible radius
            dist = pairwise_dists(pos)
            min_dist = np.min(dist)
            if min_dist >= 2 * r - 1e-9:
                break
        
        return pos
    
    def try_radius_linear(pts, r, n_iter=200):
        """Try to find feasible configuration for radius r using linear solver."""
        pos = refine_with_linear_solver(pts, r, n_iter=n_iter)
        
        # Verify feasibility
        dist = pairwise_dists(pos)
        if np.min(dist) < 2 * r - 1e-8:
            return None, False
        if np.any(np.abs(pos) > 0.5 + 1e-8):
            return None, False
        return pos, True
    
    # Try multiple initial configurations
    best_pos = None
    best_r = 0.0
    
    configs = [
        hex_initial(N),
        grid_initial(N),
        jittered_hex(N, 42),
        jittered_hex(N, 123),
        jittered_hex(N, 456),
        jittered_hex(N, 789),
    ]
    
    for cfg in configs:
        r0 = compute_radius(cfg)
        
        # Binary search for max radius
        r_lo = r0 * 0.95
        r_hi = r0 * 1.20
        
        # First, refine at r0
        pos_imp = refine_with_linear_solver(cfg, r0, n_iter=150)
        r_imp = compute_radius(pos_imp)
        r_lo = max(r_lo, r_imp)
        if r_imp > best_r:
            best_r = r_imp
            best_pos = pos_imp.copy()
        
        # Binary search
        for _ in range(20):
            r_mid = (r_lo + r_hi) / 2
            pos_try, feasible = try_radius_linear(cfg, r_mid, n_iter=150)
            if feasible:
                r_lo = r_mid
                if r_mid > best_r:
                    best_r = r_mid
                    best_pos = pos_try.copy()
            else:
                r_hi = r_mid
            
            if r_hi - r_lo < 1e-9:
                break
    
    # Final refinement on best configuration
    if best_pos is not None:
        r_cur = compute_radius(best_pos)
        pos_final = refine_with_linear_solver(best_pos, r_cur, n_iter=300)
        r_final = compute_radius(pos_final)
        if r_final > best_r:
            best_r = r_final
            best_pos = pos_final
    
    # Safety: ensure valid output
    if best_pos is None or np.any(~np.isfinite(best_pos)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    # Final check: no coincident centers
    dist = pairwise_dists(best_pos)
    if np.min(dist) < 1e-10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
