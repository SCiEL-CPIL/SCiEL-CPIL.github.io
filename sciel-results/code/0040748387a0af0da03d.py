"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import sys

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)

def _init_hexagonal():
    """Hexagonal lattice of ~50 points inside the square."""
    pts = []
    s = 0.14
    xs = np.arange(-0.5 + s / 2, 0.5, s)
    for row, y in enumerate(np.arange(-0.5 + s / 2, 0.5, s * np.sqrt(3) / 2)):
        offset = (s / 2) if row % 2 else 0.0
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
    pts = np.array(pts)
    if len(pts) >= N:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)[:N]
        return pts[idx].copy()
    else:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        g = np.array([(x, y) for y in ticks for x in ticks])
        return g[:N].copy()

def _expand_radius(c, R0, R1, step, deadline, t0):
    """Incrementally grow radius, resolving collisions by pushing apart."""
    c = c.copy()
    R = R0
    n = len(c)
    while R < R1:
        if (time.time() - t0) > deadline - 10:
            break
        for _ in range(200):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    diff = c[i] - c[j]
                    d = np.linalg.norm(diff)
                    if d < 2 * R - 1e-12:
                        if d < 1e-12:
                            diff = np.array([1e-6, 1e-6])
                            d = np.linalg.norm(diff)
                        push = (2 * R - d) / d * 0.5
                        c[i] += push * diff
                        c[j] -= push * diff
                        moved = True
            for i in range(n):
                c[i, 0] = np.clip(c[i, 0], -0.5 + R, 0.5 - R)
                c[i, 1] = np.clip(c[i, 1], -0.5 + R, 0.5 - R)
            if not moved:
                break
        R += step
    return c

def _sqp_polish(c, t0, deadline):
    """
    SQP-style polish using Delaunay triangulation to identify active contacts.
    We solve a least-squares problem to adjust positions to satisfy:
    1. Active pair distances == 2*r
    2. Wall constraints satisfied (distance >= r)
    
    We use an iterative approach:
    1. Compute Delaunay triangulation.
    2. Identify edges with length close to 2*r (active contacts).
    3. Build a linear system to move points to enforce these distances.
    4. Clip to walls.
    5. Repeat.
    """
    n = c.shape[0]
    c = c.copy()
    
    for _iter in range(50):
        if (time.time() - t0) > deadline - 5:
            break
            
        r = _radius(c)
        if r <= 0:
            break
            
        # 1. Delaunay Triangulation
        try:
            tri = Delaunay(c)
        except Exception:
            break
            
        # 2. Identify active contacts (edges in triangulation)
        # We consider an edge active if its length is very close to 2*r
        # or if it's a boundary edge (wall contact)
        
        # Get all edges from simplices
        simplices = tri.simplices
        edges = set()
        for simplex in simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
            
        edges = list(edges)
        
        # Filter edges that are "close" to contact
        # Threshold: within 1% of 2*r
        threshold = 2 * r * 1.01 
        active_edges = []
        for e in edges:
            i, j = e
            d = np.linalg.norm(c[i] - c[j])
            if d < threshold:
                active_edges.append((i, j, d))
        
        # Also check wall contacts
        # We'll handle walls by clipping after the linear solve.
        
        if not active_edges:
            break
            
        # 3. Build linear system
        # We want to find displacements dx such that for each active edge (i,j):
        # (c_i + dx_i - c_j - dx_j) dot n_ij = 0 (where n_ij is unit vector)
        # This is a linear constraint on dx.
        # We solve for dx that minimizes ||dx||^2 subject to A dx = b
        
        m = len(active_edges)
        A = np.zeros((m, 2 * n))
        b = np.zeros(m)
        
        for k, (i, j, d) in enumerate(active_edges):
            if d < 1e-12:
                continue
            n_ij = (c[i] - c[j]) / d
            # Constraint: (c_i - c_j + dx_i - dx_j) dot n_ij = 0
            # => dx_i dot n_ij - dx_j dot n_ij = 0
            A[k, 2*i] = n_ij[0]
            A[k, 2*i+1] = n_ij[1]
            A[k, 2*j] = -n_ij[0]
            A[k, 2*j+1] = -n_ij[1]
            b[k] = 0.0
            
        # Solve least squares: min ||A dx - b||^2
        # Using np.linalg.lstsq
        if m > 0:
            try:
                dx, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                c_new = c + dx.reshape(n, 2)
            except Exception:
                break
        else:
            c_new = c.copy()
            
        # 4. Clip to walls
        # We need to ensure c_new is inside [-0.5+r, 0.5-r]
        # Simple clipping might break the distance constraints, but it's a good start
        # We can iterate.
        
        # To improve wall handling, we can add wall constraints to the linear system
        # For now, let's just clip and see if it helps.
        
        # Actually, let's do a more robust wall handling:
        # If a point is too close to a wall, push it in.
        for i in range(n):
            # Left wall
            if c_new[i, 0] < -0.5 + r:
                c_new[i, 0] = -0.5 + r
            # Right wall
            if c_new[i, 0] > 0.5 - r:
                c_new[i, 0] = 0.5 - r
            # Bottom wall
            if c_new[i, 1] < -0.5 + r:
                c_new[i, 1] = -0.5 + r
            # Top wall
            if c_new[i, 1] > 0.5 - r:
                c_new[i, 1] = 0.5 - r
                
        # Check if we improved
        r_new = _radius(c_new)
        if r_new > r:
            c = c_new
        else:
            # If it got worse, try a smaller step
            c_trial = c + 0.5 * dx.reshape(n, 2)
            for i in range(n):
                c_trial[i, 0] = np.clip(c_trial[i, 0], -0.5 + r, 0.5 - r)
                c_trial[i, 1] = np.clip(c_trial[i, 1], -0.5 + r, 0.5 - r)
            r_trial = _radius(c_trial)
            if r_trial > r:
                c = c_trial
            else:
                break
                
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin

    # Start with hexagonal initialization
    c = _init_hexagonal()
    
    # Expand radius incrementally
    R0 = 0.05
    R1 = 0.072
    step = 0.001
    c = _expand_radius(c, R0, R1, step, deadline, t0)
    
    # Polish with SQP
    if (time.time() - t0) < deadline - 20:
        c = _sqp_polish(c, t0, deadline)
    
    # Ensure validity
    c = np.clip(c, -0.5 + 1e-15, 0.5 - 1e-15)
    if not np.all(np.isfinite(c)) or _radius(c) <= 0:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return c
# EVOLVE-BLOCK-END
