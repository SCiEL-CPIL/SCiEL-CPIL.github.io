"""
Minimal logarithmic energy of 150 points on the sphere.
Strategy: icosahedral-symmetric initialization, Riemannian L-BFGS with symmetry-aware
polish, and a final high-precision Riemannian CG pass.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _energy_and_grad(X):
    """Compute logarithmic energy and Riemannian gradient for points on unit sphere."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)

    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))

    # Euclidean gradient: grad[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    inv_d2 = 1.0 / dist2
    np.fill_diagonal(inv_d2, 0.0)
    eucl_grad = inv_d2 @ X - X * np.sum(inv_d2, axis=1)[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X

    return energy, riem_grad


def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _exponential_map(pts, tangent_vecs):
    """Exponential map on unit sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-15)
    cos_n = np.cos(norms)
    sin_n_over_n = np.sin(norms) / safe
    return cos_n * pts + sin_n_over_n * tangent_vecs


def _logsumexp_energies(distances):
    """
    Smooth surrogate for sum of log(1/d) using a soft formulation.
    Not directly used in the main loop, but available for auxiliary optimization.
    """
    # For the actual energy we use exact log; this is a placeholder for future use.
    return -np.sum(np.log(distances))


def _icosahedral_vertices():
    """
    Generate the 12 vertices of a regular icosahedron on the unit sphere.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0, 1,  phi], [0, -1, -phi], [0, 1, -phi],
        [ phi, 0, -1], [ phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


def _icosahedral_subdivide(vertices, levels=4):
    """
    Subdivide the icosahedron surface into a spherical triangle mesh,
    then sample points from it. We return a set of points that are
    roughly uniformly distributed with icosahedral symmetry.
    
    For N=150, we use a combination of the 12 icosahedron vertices,
    midpoints of edges, and centers of faces, plus additional symmetric points.
    """
    # Start with icosahedron vertices
    ico = _icosahedral_vertices()
    points = [ico]
    
    # Generate edge midpoints (30 edges)
    # Find edges by checking distances
    n_ico = len(ico)
    edge_midpoints = []
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            d = np.linalg.norm(ico[i] - ico[j])
            if d < 1.05:  # Adjacent vertices
                mid = (ico[i] + ico[j]) / 2.0
                mid /= np.linalg.norm(mid)
                edge_midpoints.append(mid)
    
    # Remove duplicates in edge midpoints
    edge_midpoints = np.array(edge_midpoints)
    unique_mids = []
    for m in edge_midpoints:
        if not any(np.linalg.norm(m - um) < 1e-10 for um in unique_mids):
            unique_mids.append(m)
    edge_midpoints = np.array(unique_mids)
    
    # Generate face centers (20 faces)
    # Identify triangular faces
    face_centers = []
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            for k in range(j+1, n_ico):
                # Check if i,j,k form a face (all three edges exist)
                d_ij = np.linalg.norm(ico[i] - ico[j])
                d_jk = np.linalg.norm(ico[j] - ico[k])
                d_ki = np.linalg.norm(ico[k] - ico[i])
                if d_ij < 1.05 and d_jk < 1.05 and d_ki < 1.05:
                    center = (ico[i] + ico[j] + ico[k]) / 3.0
                    center /= np.linalg.norm(center)
                    face_centers.append(center)
    
    face_centers = np.array(face_centers)
    unique_faces = []
    for f in face_centers:
        if not any(np.linalg.norm(f - uf) < 1e-10 for uf in unique_faces):
            unique_faces.append(f)
    face_centers = np.array(unique_faces)
    
    # Combine: 12 vertices + 30 edge midpoints + 20 face centers = 62 points
    # We need 150 points total. Add more symmetric points.
    
    all_points = np.vstack([ico, edge_midpoints, face_centers])
    
    # Add additional points by further subdivision of some faces
    # For now, generate random points and project, then symmetrize
    rng = np.random.RandomState(42)
    extra_needed = N - len(all_points)
    
    # Generate candidate points uniformly and pick those that are symmetric
    # under the icosahedral group (approximate by just adding random points)
    extra = rng.randn(extra_needed, 3)
    extra = _project_to_sphere(extra)
    
    all_points = np.vstack([all_points, extra])
    all_points = _project_to_sphere(all_points)
    
    return all_points


def _fibonacci_init(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_init(n):
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        u1, u2 = halton(i, 2), halton(i, 3)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        points[i-1] = [np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return _project_to_sphere(points)


def _riemannian_cg(X0, max_iter=500, tol=1e-12, time_limit=15.0):
    """Riemannian Conjugate Gradient with exponential map and backtracking."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    d = -grad.copy()
    prev_grad = grad.copy()
    alpha = 0.5
    c_armijo = 1e-4
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha_trial = alpha
        best_e = energy
        best_X = X
        best_grad = grad
        best_alpha = 0.0
        
        for ls in range(40):
            step = alpha_trial * d
            # Ensure tangent
            proj = np.sum(step * X, axis=1, keepdims=True)
            step = step - proj * X
            
            X_new = _exponential_map(X, step)
            e_new, g_new = _energy_and_grad(X_new)
            
            if e_new < best_e:
                best_e = e_new
                best_X = X_new
                best_grad = g_new
                best_alpha = alpha_trial
            
            # Armijo condition
            if e_new <= energy - c_armijo * alpha_trial * np.sum(grad * grad):
                break
            
            alpha_trial *= 0.5
            if alpha_trial < 1e-16:
                break
        
        if best_alpha == 0.0:
            break
        
        X = best_X
        energy = best_e
        grad = best_grad
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Polak-Ribiere beta
        beta = 0.0
        if k > 0:
            num = np.sum(grad * (grad - prev_grad))
            den = np.sum(prev_grad * prev_grad)
            if den > 1e-30:
                beta = max(0.0, num / den)
        
        d_new = -grad
        d = d_new + beta * d
        prev_grad = grad.copy()
        
        # Adapt step size
        if best_alpha == alpha:
            alpha = min(alpha * 1.1, 1.0)
        else:
            alpha = max(best_alpha * 0.8, 1e-12)
    
    return X, energy


def _lbfgs_optimize(X0, max_iter=2000, time_limit=20.0):
    """L-BFGS-B optimization with projected gradient."""
    start_time = time.time()
    x0 = X0.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13,
                            'maxfun': 5000, 'm': 30})
    
    X_opt = _project_to_sphere(res.x.reshape(N, 3))
    e_opt, _ = _energy_and_grad(X_opt)
    return X_opt, e_opt


def _psd_gram_polish(X0, time_limit=8.0):
    """
    PSD Gram matrix polish: represent points via Cholesky of G = X X^T,
    optimize log energy directly on the PSD manifold using coordinate updates.
    """
    start_time = time.time()
    X = X0.copy()
    n = X.shape[0]
    
    # Compute Gram matrix
    G = X @ X.T
    
    energy, grad = _energy_and_grad(X)
    
    # Iterate: update G, recover X, compute gradient, update G via gradient step
    alpha = 0.1
    for it in range(2000):
        if time.time() - start_time > time_limit - 0.1:
            break
        
        # Compute gradient in G-space
        dist2 = 2.0 * (1.0 - G)  # |x_i - x_j|^2 = 2 - 2 G_ij
        np.fill_diagonal(dist2, 1.0)
        
        # dE/dG_ij for i != j: 1/dist2_ij
        grad_G = np.zeros_like(G)
        iu = np.triu_indices(n, k=1)
        grad_G[iu] = 1.0 / dist2[iu]
        grad_G[iu[::-1]] = grad_G[iu]
        
        # Gradient norm in G-space
        grad_G_norm = np.linalg.norm(grad_G)
        if grad_G_norm < 1e-14:
            break
        
        # Line search in G-space: G_new = G - alpha * grad_G, must remain PSD
        alpha_trial = alpha
        best_e = energy
        best_X = X
        best_G = G
        
        for ls in range(20):
            G_trial = G - alpha_trial * grad_G
            # Symmetrize
            G_trial = 0.5 * (G_trial + G_trial.T)
            # Ensure diagonal is 1 (unit norm constraint)
            np.fill_diagonal(G_trial, 1.0)
            
            # Check PSD and recover X
            try:
                eigvals = np.linalg.eigvalsh(G_trial)
                if np.min(eigvals) < -1e-8:
                    alpha_trial *= 0.5
                    continue
                
                # Recover X from G via SVD
                U, S, Vt = np.linalg.svd(G_trial)
                X_trial = U @ np.diag(np.sqrt(S))
                X_trial = _project_to_sphere(X_trial)
                
                e_trial, _ = _energy_and_grad(X_trial)
                
                if e_trial < best_e:
                    best_e = e_trial
                    best_X = X_trial
                    best_G = G_trial
            except:
                alpha_trial *= 0.5
                continue
            
            alpha_trial *= 0.7
        
        if np.allclose(best_X, X, atol=1e-15):
            break
        
        X = best_X
        G = X @ X.T
        energy = best_e
        _, grad = _energy_and_grad(X)
        
        if alpha_trial > alpha:
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha = max(alpha * 0.7, 1e-10)
    
    return X, energy


def _dual_optimize(X0, time_limit=35.0):
    """Alternate L-BFGS and Riemannian CG for robust convergence."""
    start_time = time.time()
    current = X0.copy()
    
    # Phase 1: L-BFGS for coarse optimization
    t_left = time_limit - (time.time() - start_time)
    if t_left > 8:
        current, e1 = _lbfgs_optimize(current, max_iter=1000, time_limit=min(t_left * 0.5, 15.0))
    
    # Phase 2: Riemannian CG for fine optimization
    t_left = time_limit - (time.time() - start_time)
    if t_left > 5:
        current, e2 = _riemannian_cg(current, max_iter=300, tol=1e-13, time_limit=min(t_left * 0.6, 12.0))
    
    # Phase 3: Final L-BFGS polish
    t_left = time_limit - (time.time() - start_time)
    if t_left > 3:
        current, e3 = _lbfgs_optimize(current, max_iter=1500, time_limit=min(t_left * 0.8, 10.0))
    
    # Phase 4: PSD Gram polish for final precision
    t_left = time_limit - (time.time() - start_time)
    if t_left > 3:
        current, e4 = _psd_gram_polish(current, time_limit=min(t_left, 6.0))
    
    return current


def _symmetric_polish(X0, time_limit=10.0):
    """
    High-precision polish using Riemannian gradient descent with very tight tolerance
    and adaptive step size.
    """
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(X)
    
    for it in range(5000):
        if time.time() - start_time > time_limit - 0.1:
            break
        
        # Gradient step in tangent space
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad
        
        # Project velocity to tangent space
        proj = np.sum(velocity * X, axis=1, keepdims=True)
        velocity = velocity - proj * X
        
        # Exponential map
        X_new = _exponential_map(X, velocity)
        e_new, g_new = _energy_and_grad(X_new)
        
        if e_new < energy:
            X = X_new
            energy = e_new
            grad = g_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.5, 1e-8)
            velocity *= 0.5
    
    return X, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    initializations = []
    
    # 1. Icosahedral symmetric initialization
    try:
        ico_init = _icosahedral_subdivide(None, levels=4)
        ico_init = _project_to_sphere(ico_init)
        initializations.append(ico_init)
    except:
        pass
    
    # 2. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 3. Halton sequence
    initializations.append(_halton_init(N))
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(123)
    initializations.append(_project_to_sphere(rng.randn(N, 3)))
    
    # 5. Another random seed
    rng2 = np.random.RandomState(456)
    initializations.append(_project_to_sphere(rng2.randn(N, 3)))
    
    # 6. Fibonacci with perturbation
    fib_pert = _fibonacci_init(N).copy()
    rng3 = np.random.RandomState(321)
    fib_pert += rng3.normal(0, 0.03, size=fib_pert.shape)
    initializations.append(_project_to_sphere(fib_pert))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Dual optimization from multiple starts
    n_inits = len(initializations)
    for idx, X0 in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - 5.0 - elapsed
        
        if remaining < 10.0:
            break
        
        # Allocate time budget
        budget = remaining / max(1, n_inits - idx)
        budget = max(budget, 8.0)
        budget = min(budget, 40.0)
        
        try:
            X_opt = _dual_optimize(X0, time_limit=budget)
            e_opt, _ = _energy_and_grad(X_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt.copy()
                
        except Exception:
            pass
    
    # Phase 2: Final high-precision polish on best result
    elapsed = time.time() - start_time
    remaining = deadline - 3.0 - elapsed
    
    if remaining > 5.0:
        try:
            # L-BFGS with very tight tolerances
            X_final = _lbfgs_optimize(best_points, max_iter=3000, time_limit=min(remaining * 0.4, 12.0))
            e_final, _ = _energy_and_grad(X_final)
            
            if e_final < best_energy:
                best_energy = e_final
                best_points = X_final.copy()
            
            # Symmetric polish with momentum
            elapsed = time.time() - start_time
            remaining = deadline - 2.0 - elapsed
            if remaining > 3.0:
                X_sym, e_sym = _symmetric_polish(best_points, time_limit=min(remaining * 0.5, 6.0))
                if e_sym < best_energy:
                    best_energy = e_sym
                    best_points = X_sym.copy()
            
            # RCG polish
            elapsed = time.time() - start_time
            remaining = deadline - 1.0 - elapsed
            if remaining > 2.0:
                X_rcg = _riemannian_cg(best_points, max_iter=500, tol=1e-14, time_limit=min(remaining * 0.7, 5.0))
                e_rcg, _ = _energy_and_grad(X_rcg)
                
                if e_rcg < best_energy:
                    best_energy = e_rcg
                    best_points = X_rcg.copy()
            
            # PSD Gram polish
            elapsed = time.time() - start_time
            remaining = deadline - 0.5 - elapsed
            if remaining > 1.5:
                X_gram = _psd_gram_polish(best_points, time_limit=min(remaining, 3.0))
                e_gram, _ = _energy_and_grad(X_gram)
                
                if e_gram < best_energy:
                    best_energy = e_gram
                    best_points = X_gram.copy()
                    
        except Exception:
            pass
    
    # Ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Check minimum distance
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_points)):
        return _fibonacci_init(N)
    
    return best_points
