"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)

def _gradient(points):
    """Compute gradient of log energy with respect to points."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    return grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _spectral_embed(points):
    """
    Compute spectral embedding using the graph Laplacian of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        edge_set = set()
        for simplex in simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edge_set.add((a, b))
        
        if not edge_set:
            return points
            
        rows = []
        cols = []
        vals = []
        for a, b in edge_set:
            rows.append(a)
            cols.append(b)
            vals.append(1.0)
            rows.append(b)
            cols.append(a)
            vals.append(1.0)
            
        A = sp.coo_matrix((vals, (rows, cols)), shape=(len(points), len(points)))
        A = A.tocsr()
        
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - A
        L = L.tocsr()
        
        try:
            eigvals, eigvecs = spla.eigsh(L, k=4, sigma=0, which='LM')
            idx = np.argsort(np.abs(eigvals))
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            if abs(eigvals[0]) > 1e-5:
                 return points
                 
            V = eigvecs[:, 1:4].real
            V_centered = V - np.mean(V, axis=0)
            new_points = _project_to_sphere(V_centered)
            return new_points
        except Exception:
            return points
            
    except Exception:
        return points

def construct():
    best_points = None
    best_energy = float('inf')
    
    for seed in range(10):
        np.random.seed(seed)
        pts = np.random.randn(N, 3)
        pts = _project_to_sphere(pts)
        
        spectral_pts = _spectral_embed(pts)
        
        current_points = spectral_pts
        current_energy = _log_energy(current_points)
        
        lr = 0.05
        for it in range(200):
            grad = _gradient(current_points)
            grad_t = grad - np.sum(grad * current_points, axis=1, keepdims=True) * current_points
            new_points = current_points - lr * grad_t
            new_points = _project_to_sphere(new_points)
            new_energy = _log_energy(new_points)
            
            if new_energy < current_energy:
                current_energy = new_energy
                current_points = new_points
                lr = max(0.001, lr * 0.99)
            else:
                lr = max(0.001, lr * 0.9)
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current_points
            
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_points = _project_to_sphere(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
