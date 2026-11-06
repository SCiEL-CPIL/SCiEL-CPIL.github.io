import numpy as np
import math
import time
from scipy.linalg import eigh

N = 100

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances vectorized
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and diagonal
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    d = np.sqrt(d2)
    d_safe = np.where(mask, d, 1.0)
    d_safe = np.maximum(d_safe, 1e-12)
    
    terms = np.where(mask, -np.log(d_safe), 0.0)
    
    # Sum only i < j
    iu = np.triu_indices(n, k=1)
    return math.fsum(terms[iu[0], iu[1]].tolist())

def _energy_and_hessian(pts):
    """
    Compute energy, gradient, and Hessian of log energy on the sphere.
    E = sum_{i<j} -log(|x_i - x_j|)
    
    Gradient:
    grad_i E = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    
    Hessian (in R^3):
    H_{ij} = sum_{k!=i,j} 3 (x_i - x_k)(x_j - x_k)^T / |x_i - x_k|^4
    Diagonal H_ii = sum_{k!=i} [3 (x_i - x_k)(x_i - x_k)^T / |x_i - x_k|^4 - 1 / |x_i - x_k|^2]
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero
    d2_safe = np.where(d2 < 1e-20, 1e-20, d2)
    
    # Gradient computation
    W = np.zeros((n, n))
    np.fill_diagonal(W, 0)
    W = 1.0 / d2_safe
    np.fill_diagonal(W, 0)
    
    row_sums = W.sum(axis=1)
    W_pts = W @ pts
    raw_grad = row_sums[:, np.newaxis] * pts - W_pts
    grad = -raw_grad
    
    # Hessian computation
    # H_{ij} = sum_{k!=i,j} 3 (x_i - x_k)(x_j - x_k)^T / d_ik^4
    # Let A = 3 / d2_safe^2, with 0 on diagonal
    A = np.zeros((n, n))
    np.fill_diagonal(A, 0)
    A = 3.0 / (d2_safe ** 2)
    np.fill_diagonal(A, 0)
    
    # H_{ij} = sum_k A_{ik} (x_i - x_k)(x_j - x_k)^T
    # This is complex to vectorize directly as a 4-tensor.
    # Let's compute H as a (n, n, 3, 3) tensor.
    H = np.zeros((n, n, 3, 3))
    
    # Iterate over pairs for Hessian? N=100 -> 10000 pairs, 3x3 matrices.
    # Vectorized approach:
    # H_{ij} = sum_k A_{ik} diff_{ik} diff_{jk}^T
    # Let's use einsum or manual loops if slow.
    # diff shape (n, n, 3)
    # A shape (n, n)
    # We want H[i, j, :, :] = sum_k A[i, k] * diff[i, k, :] * diff[j, k, :].T
    
    # This is a bit heavy. Let's try to optimize.
    # H_{ij} = sum_k A_{ik} diff_{ik} diff_{jk}^T
    # We can compute this by looping over i and j? Or using einsum.
    # np.einsum('ik,ikl,jkl->ijl', A, diff, diff) ? No, indices are tricky.
    # A[i, k] * diff[i, k, l] * diff[j, k, m] -> H[i, j, l, m]
    # np.einsum('ik,ikl,jkm->ijlm', A, diff, diff)
    
    try:
        H_offdiag = np.einsum('ik,ikl,jkm->ijlm', A, diff, diff)
    except Exception:
        # Fallback to loop if einsum fails or is too slow/complex
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # sum_k A[i, k] * diff[i, k, :] * diff[j, k, :].T
                # A[i, k] is (n,), diff[i, k, :] is (n, 3), diff[j, k, :] is (n, 3)
                # We want sum_k A[i, k] * outer(diff[i, k], diff[j, k])
                # This is (A[i, :, None, None] * diff[i, :, None, :] * diff[j, None, :, :]).sum(axis=1)
                # Actually:
                # term_k = A[i, k] * np.outer(diff[i, k], diff[j, k])
                # H[i, j] = sum(term_k)
                # Vectorized:
                # weights = A[i, :] (n,)
                # d_i = diff[i, :, :] (n, 3)
                # d_j = diff[j, :, :] (n, 3)
                # H[i, j] = np.sum(weights[:, None, None] * d_i[:, None, :] * d_j[None, :, :], axis=0)
                H[i, j] = np.sum(A[i, :, None, None] * diff[i, :, None, :] * diff[j, None, :, :], axis=0)
    
    # Diagonal H_ii = sum_{k!=i} [3 (x_i - x_k)(x_i - x_k)^T / d_ik^4 - 1 / d_ik^2]
    # H_ii = sum_{k!=i} [ A_ik * outer(diff_ik, diff_ik) - W_ik * I ]
    # Note A_ik = 3/d_ik^4, W_ik = 1/d_ik^2
    # Let's compute diagonal separately.
    for i in range(n):
        # Sum over k != i
        # A[i, k] * outer(diff[i, k], diff[i, k])
        # - W[i, k] * I
        
        # Vectorized sum of outer products weighted by A
        # S1 = sum_k A[i, k] * outer(diff[i, k], diff[i, k])
        # S2 = sum_k W[i, k] * I = (sum_k W[i, k]) * I
        
        # S1 = (diff[i] * A[i, :, None]).T @ diff[i] ? No.
        # S1 = diff[i].T @ diag(A[i]) @ diff[i]
        # Let's just use the loop for diagonal too if N is small enough, or vectorize.
        # N=100, 100 loops for diagonal is fine.
        
        # S1
        # diff[i] is (n, 3). We want sum_k A[i, k] * diff[i, k, l] * diff[i, k, m]
        # = (diff[i] * A[i, :, None]).T @ diff[i]
        S1 = (diff[i] * A[i, :, None]).T @ diff[i]
        S2 = W.sum(axis=1)[i] * np.eye(3)
        H[i, i] = S1 - S2
        
    # Compute energy
    d = np.sqrt(d2_safe)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    terms = np.where(mask, -np.log(d), 0.0)
    iu = np.triu_indices(n, k=1)
    energy = math.fsum(terms[iu[0], iu[1]].tolist())
    
    return energy, grad, H

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _spectral_init(n, deadline):
    """
    Generate initial points using eigenvectors of a Gaussian RBF kernel matrix.
    The low-energy eigenvectors of the RBF kernel correspond to smooth functions,
    and their extrema or nodes can provide well-separated points on the sphere.
    """
    # Generate a dense set of candidate points
    n_cand = 2000
    pts_cand = _random_sphere(n_cand, seed=123)
    
    # Compute RBF kernel matrix
    # K_ij = exp(-sigma^2 |x_i - x_j|^2)
    # Choose sigma such that points are well-separated.
    # For N=100, typical distance ~ 0.3. sigma ~ 1.0
    sigma = 1.0
    diff = pts_cand[:, np.newaxis, :] - pts_cand[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    K = np.exp(-sigma * sigma * d2)
    
    # Solve eigenvalue problem
    # We want the lowest non-trivial eigenvectors.
    # The first eigenvector is constant. We skip it.
    try:
        vals, vecs = eigh(K)
    except Exception:
        return _fibonacci_sphere(n)
    
    # Take the next few eigenvectors
    # Let's take the 10 lowest non-trivial eigenvectors
    # Project them to the sphere? Or use their values to weight points?
    # A common technique is to take the eigenvectors, find their local maxima, or just use the components.
    # Actually, a better spectral method for point placement is to use the eigenvectors of the Laplacian or similar.
    # Let's try a simpler spectral approach: 
    # Use the eigenvectors to define a map from R^m to S^2?
    # Or just take the extrema of the eigenvectors on the candidate set?
    
    # Alternative: Use the eigenvectors to generate a new set of points.
    # Let's take the top 3 eigenvectors (excluding constant) and use them as coordinates?
    # No, they are on the sphere.
    
    # Let's stick to a robust heuristic: 
    # The "eigenpoint" method often refers to taking the eigenvectors of the kernel matrix 
    # and using their components to define a new embedding.
    # But for simplicity and robustness, let's just return the Fibonacci sphere 
    # if spectral init is too complex or slow.
    # However, the prompt asks for a MATERIAL difference.
    # Let's implement a simple spectral clustering-based init:
    # 1. Compute K
    # 2. Compute eigenvectors
    # 3. Normalize rows of eigenvector matrix
    # 4. K-means on the normalized eigenvectors
    # 5. Return centroids
    
    # Take the first 10 eigenvectors (excluding the trivial constant one)
    # Actually, the constant eigenvector is the largest. We want the lowest non-trivial?
    # For RBF, the largest eigenvalue corresponds to the smoothest function (constant).
    # The next ones are the "modes".
    # Let's take the first 10 non-trivial eigenvectors.
    
    # eigh returns ascending order. So vals[0] is smallest.
    # For RBF, the smallest eigenvalues correspond to high-frequency oscillations?
    # No, RBF kernel is positive definite. The largest eigenvalue is the smoothest.
    # We want the smoothest non-constant functions.
    # So we should take the LARGEST eigenvalues, excluding the first (constant).
    
    vals, vecs = eigh(K)
    # Reverse to get largest first
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    
    # Take the first 10 eigenvectors (excluding the constant one, which is index 0)
    # Actually, index 0 is the largest. Is it constant?
    # Let's check. If it's constant, we skip it.
    # Let's take eigenvectors 1 to 10 (indices 1 to 10)
    
    n_modes = 10
    if n_cand > n_modes:
        eigvecs = vecs[:, 1:n_modes+1]  # (n_cand, n_modes)
        
        # Normalize rows
        norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        eigvecs_norm = eigvecs / norms
        
        # K-means on eigvecs_norm
        # Simple K-means implementation
        k = n
        # Initialize centroids randomly from eigvecs_norm
        np.random.seed(42)
        indices = np.random.choice(n_cand, k, replace=False)
        centroids = eigvecs_norm[indices].copy()
        
        for _ in range(10):  # 10 iterations of K-means
            # Assign points to nearest centroid
            dists = np.linalg.norm(eigvecs_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = np.mean(eigvecs_norm[mask], axis=0)
                    counts[i] = np.sum(mask)
                else:
                    # Re-initialize empty centroid
                    new_centroids[i] = eigvecs_norm[np.random.choice(n_cand)]
                    counts[i] = 1
            
            centroids = new_centroids
        
        # Map centroids back to the original points?
        # Or just use the centroids as directions?
        # The centroids are in the eigenvector space.
        # We can project them back to the sphere by taking the corresponding original points?
        # Or just normalize the centroids?
        # Let's try normalizing the centroids and using them as points.
        # This might not be on the sphere in the original space, but it's a direction.
        
        # Better: For each centroid, find the original point closest to it in eigenvector space,
        # and use that original point.
        
        final_pts = []
        for i in range(k):
            dists = np.linalg.norm(eigvecs_norm - centroids[i], axis=1)
            closest_idx = np.argmin(dists)
            final_pts.append(pts_cand[closest_idx])
            
        final_pts = np.array(final_pts)
        return _project_to_sphere(final_pts)
    else:
        return _fibonacci_sphere(n)

def _optimize_newton(pts_init, deadline):
    """
    Optimize using Newton's method on the sphere.
    We solve the linear system for the update in the tangent space.
    """
    pts = pts_init.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    max_iter = 50
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        e, grad, H = _energy_and_hessian(pts)
        
        # Project gradient to tangent space
        dots = np.sum(grad * pts, axis=1)
        grad_tan = grad - dots[:, np.newaxis] * pts
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < 1e-12:
            break
            
        # Newton step: Solve H_tan @ delta = -grad_tan
        # H_tan is the Hessian projected to tangent space.
        # This is complex. A simpler approach is to use the full Hessian 
        # and solve for the update in R^3, then project.
        # But the Hessian is singular on the sphere (radial direction).
        # We can add a small regularization to the radial direction.
        
        # Flatten H and grad
        H_flat = H.reshape(n*3, n*3)
        grad_flat = grad_tan.reshape(n*3)
        
        # Add regularization to the radial direction
        # The radial direction for point i is pts[i].
        # We want to penalize movement in the radial direction.
        # Let's add a large penalty to the diagonal elements corresponding to the radial direction.
        # Or simply use the pseudo-inverse?
        # Let's try adding a small lambda * I to H_flat.
        # But this might not fix the singularity.
        
        # A better approach: Use the projected Hessian.
        # H_tan[i, j] = P_i H[i, j] P_j, where P_i = I - pts[i] pts[i]^T
        # This is a (n*3, n*3) matrix.
        
        # Let's construct H_tan.
        H_tan = np.zeros((n*3, n*3))
        for i in range(n):
            P_i = np.eye(3) - np.outer(pts[i], pts[i])
            for j in range(n):
                P_j = np.eye(3) - np.outer(pts[j], pts[j])
                H_tan[i*3:(i+1)*3, j*3:(j+1)*3] = P_i @ H[i, j] @ P_j
        
        # Solve H_tan @ delta_flat = -grad_flat
        try:
            delta_flat = np.linalg.solve(H_tan, -grad_flat)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            delta_flat, residuals, rank, s = np.linalg.lstsq(H_tan, -grad_flat, rcond=None)
        
        # Unflatten delta
        delta = delta_flat.reshape(n, 3)
        
        # Line search
        alpha = 1.0
        for _ in range(20):
            pts_new = pts + alpha * delta
            pts_new = _project_to_sphere(pts_new)
            e_new = _log_energy(pts_new)
            
            if e_new < e:
                break
            alpha *= 0.5
            
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            break
            
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Multi-start strategy
    # 1. Spectral init
    # 2. Fibonacci
    # 3. Random with different seeds
    
    starts = []
    
    # Try spectral init
    try:
        starts.append(_spectral_init(N, deadline - 10.0))
    except Exception:
        pass
        
    starts.append(_fibonacci_sphere(N))
    
    # Add random starts
    for seed in range(5):
        starts.append(_random_sphere(N, seed=seed))
        
    for i, init_pts in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        # Optimize
        pts_opt, e_opt = _optimize_newton(init_pts, deadline - 2.0)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
            
    # Final polish on the best solution found
    if best_pts is not None and time.time() < deadline - 5.0:
        # Run a few more iterations with very small lr
        pts_polished, e_polished = _optimize_newton(best_pts, deadline - 1.0)
        if e_polished < best_energy:
            best_pts = pts_polished
            best_energy = e_polished
            
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
