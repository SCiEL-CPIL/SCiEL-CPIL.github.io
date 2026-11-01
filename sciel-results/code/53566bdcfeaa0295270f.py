import numpy as np
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def construct():
    # Function to compute the ratio
    def ratio(x):
        # x is flattened (N*D,)
        X = x.reshape(N, D)
        # Center
        X = X - np.mean(X, axis=0)
        # Scale to unit norm (sum of squared norms = 1)
        norm_sq = np.sum(X**2)
        if norm_sq < 1e-12:
            return 1e10
        X = X / np.sqrt(norm_sq)
        
        # Compute pairwise squared distances
        # Using broadcasting
        dists_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists_sq = dists_sq[mask]
        
        min_d_sq = np.min(dists_sq)
        max_d_sq = np.max(dists_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    # Initial guesses
    best_x = None
    best_score = float('inf')
    
    # Try a few random initializations on the sphere
    # Also try icosahedron-based initialization
    # Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize ico_verts
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # We need 14 points. 
    # Strategy 1: 12 ico points + 2 random points on sphere
    # Strategy 2: 14 random points on sphere
    # Strategy 3: Fibonacci sphere
    
    inits = []
    
    # 1. Icosahedron + 2 poles? (The ico is symmetric, adding points might break symmetry)
    # Let's just add 2 random points near the ico vertices to perturb them?
    # Better: Start with 14 random points on the sphere.
    
    for _ in range(5):
        # Random points on sphere
        theta = np.random.uniform(0, 2 * np.pi, N)
        phi_ang = np.arccos(np.random.uniform(-1, 1, N))
        x0 = np.zeros(N * D)
        for i in range(N):
            x0[i*D] = np.sin(phi_ang[i]) * np.cos(theta[i])
            x0[i*D+1] = np.sin(phi_ang[i]) * np.sin(theta[i])
            x0[i*D+2] = np.cos(phi_ang[i])
        inits.append(x0)
        
    # 2. Fibonacci sphere
    # Golden angle
    golden_angle = np.pi * (3 - np.sqrt(5))
    x0 = np.zeros(N * D)
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x0[i*D] = r * np.cos(theta)
        x0[i*D+1] = r * np.sin(theta)
        x0[i*D+2] = z
    inits.append(x0)
    
    # 3. Icosahedron based: 12 ico points, and 2 points placed to maximize min dist from ico
    # This is complex to construct exactly, so let's just perturb the ico points and add 2 random ones
    # Actually, let's just use the random and fibonacci inits. They are usually good enough for N=14.
    
    # Run optimization for each init
    for x0 in inits:
        # Use Powell method as it handles non-differentiable objectives well
        res = minimize(ratio, x0, method='Powell', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
            
    # Reconstruct X from best_x
    X = best_x.reshape(N, D)
    # Center
    X = X - np.mean(X, axis=0)
    # Scale back to a reasonable magnitude (doesn't matter for ratio, but good for return)
    # The ratio is scale invariant.
    return X

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    # EDM-based optimization
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    def edm_objective(v):
        d_sq = np.maximum(v, 1e-10)
        D_mat = np.zeros((N, N))
        D_mat[idx_i, idx_j] = d_sq
        D_mat[idx_j, idx_i] = d_sq
        
        J = np.eye(N) - np.ones((N, N)) / N
        G = -0.5 * (J @ D_mat @ J)
        G = (G + G.T) / 2
        
        eigs = np.sort(np.linalg.eigvalsh(G))[::-1]
        
        psd_violation = np.sum(np.maximum(0, -eigs)**2)
        rank_violation = np.sum(eigs[3:]**2)
        penalty = psd_violation + rank_violation
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-10:
            return 1e10 + penalty
        ratio = max_d_sq / min_d_sq
            
        weight = 1e6
        return ratio + weight * penalty

    def reconstruct_X_from_G(G):
        eigs, evecs = np.linalg.eigh(G)
        idx = np.argsort(eigs)[::-1]
        eigs = eigs[idx]
        evecs = evecs[:, idx]
        
        X = evecs[:, :3] * np.sqrt(np.maximum(eigs[:3], 0))
        X = X - np.mean(X, axis=0)
        return X

    best_x = None
    best_score = float('inf')
    
    inits = []
    
    # 1. Fibonacci Sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    X_fib = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        X_fib[i, 0] = r * np.cos(theta)
        X_fib[i, 1] = r * np.sin(theta)
        X_fib[i, 2] = z
    
    d_sq_fib = np.sum((X_fib[:, np.newaxis, :] - X_fib[np.newaxis, :, :])**2, axis=2)
    d_sq_fib = d_sq_fib[idx_i, idx_j]
    inits.append(d_sq_fib)
    
    # 2. Icosahedron + 2 Random
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rand_pts = np.random.randn(2, D)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    X_ico = np.vstack([ico_verts, rand_pts])
    d_sq_ico = np.sum((X_ico[:, np.newaxis, :] - X_ico[np.newaxis, :, :])**2, axis=2)
    d_sq_ico = d_sq_ico[idx_i, idx_j]
    inits.append(d_sq_ico)
    
    # 3. Random
    for _ in range(3):
        X_rand = np.random.randn(N, D)
        d_sq_rand = np.sum((X_rand[:, np.newaxis, :] - X_rand[np.newaxis, :, :])**2, axis=2)
        d_sq_rand = d_sq_rand[idx_i, idx_j]
        inits.append(d_sq_rand)

    for v0 in inits:
        try:
            res = minimize(edm_objective, v0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            d_sq_res = np.maximum(res.x, 1e-10)
            D_res = np.zeros((N, N))
            D_res[idx_i, idx_j] = d_sq_res
            D_res[idx_j, idx_i] = d_sq_res
            J = np.eye(N) - np.ones((N, N)) / N
            G_res = -0.5 * (J @ D_res @ J)
            G_res = (G_res + G_res.T) / 2
            eigs_res = np.sort(np.linalg.eigvalsh(G_res))[::-1]
            penalty_res = np.sum(np.maximum(0, -eigs_res)**2) + np.sum(eigs_res[3:]**2)
            
            if penalty_res < 1e-4:
                min_d = np.min(d_sq_res)
                max_d = np.max(d_sq_res)
                if min_d > 1e-10:
                    score = max_d / min_d
                    if score < best_score:
                        best_score = score
                        best_x = reconstruct_X_from_G(G_res)
        except Exception:
            continue

    if best_x is None:
        # Fallback to coordinate optimization
        def ratio(x):
            X = x.reshape(N, D)
            X = X - np.mean(X, axis=0)
            norm_sq = np.sum(X**2)
            if norm_sq < 1e-12:
                return 1e10
            X = X / np.sqrt(norm_sq)
            dists_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
            mask = ~np.eye(N, dtype=bool)
            dists_sq = dists_sq[mask]
            min_d_sq = np.min(dists_sq)
            max_d_sq = np.max(dists_sq)
            if min_d_sq < 1e-12:
                return 1e10
            return max_d_sq / min_d_sq

        x0 = np.random.randn(N * D)
        res = minimize(ratio, x0, method='Powell', options={'maxiter': 1000})
        best_x = res.x.reshape(N, D)
        best_x = best_x - np.mean(best_x, axis=0)

    return best_x

# EVOLVE-BLOCK-END
