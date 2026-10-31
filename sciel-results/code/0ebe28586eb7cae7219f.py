import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2

def pairwise_distances(pts):
    # pts is (N, D)
    # Returns (N, N) distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

def objective_flat(x):
    pts = x.reshape(N, D)
    d = pairwise_distances(pts)
    # Mask diagonal
    np.fill_diagonal(d, np.inf)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_min < 1e-9:
        return 1e10
    return (d_max / d_min) ** 2

def hexagonal_lattice(n):
    # Generate a hexagonal lattice of n points
    # Simple approach: generate a large grid and pick the n closest to origin
    # Or construct row by row
    pts = []
    # Try to make it roughly circular
    # Rows of points
    # For 16 points:
    # 3
    # 4
    # 5
    # 4
    # Total 16? No, 3+4+5+4 = 16.
    # Let's try this pattern
    rows = [3, 4, 5, 4]
    y = 0
    for count in rows:
        x_start = -(count - 1) / 2.0
        for i in range(count):
            x = x_start + i
            # Shift x by 0.5 for alternating rows to make it hexagonal
            if y % 2 != 0:
                x += 0.5
            pts.append([x, y * np.sqrt(3) / 2])
        y += 1
    pts = np.array(pts)
    # Center it
    pts -= np.mean(pts, axis=0)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = np.inf
    best_pts = None
    
    # Initial guesses
    guesses = []
    
    # 1. Hexagonal lattice
    guesses.append(hexagonal_lattice(N))
    
    # 2. Random points
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.rand(N, D) - 0.5
        guesses.append(pts)
        
    # 3. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    guesses.append(pts)
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        x0 = guess.flatten()
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_pts = res.x.reshape(N, D)
        except Exception:
            pass
            
    if best_pts is None:
        # Fallback to hexagonal lattice
        best_pts = hexagonal_lattice(N)
        
    return best_pts
