import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse import csr_matrix

N = 300

def _hex_init(N):
    """Generate hexagonal lattice initialization."""
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]
    
    return points.copy()

def _compute_radius(points, i_idx, j_idx):
    """Compute the packing radius."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists[i_idx, j_idx]) / 2.0
    rad = np.linalg.norm(points, axis=1)
    bound_min = np.min(1.0 - rad)
    return min(pair_min, bound_min)

def _build_sparse_graph(points, max_neighbors=12):
    """Build a sparse neighbor graph using Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
        
        # Limit to max_neighbors per node
        from collections import defaultdict
        adj = defaultdict(list)
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Keep only closest neighbors
        final_edges = set()
        for i in range(len(points)):
            if adj[i]:
                neighbors = sorted(adj[i], key=lambda j: np.linalg.norm(points[i] - points[j]))
                for j in neighbors[:max_neighbors]:
                    final_edges.add((min(i, j), max(i, j)))
        
        if not final_edges:
            # Fallback: use all Delaunay edges
            final_edges = edges
        
        edge_list = list(final_edges)
        if not edge_list:
            return None, None, None
        
        src = np.array([e[0] for e in edge_list])
        dst = np.array([e[1] for e in edge_list])
        
        return src, dst, edge_list
    except Exception:
        return None, None, None

def _spectral_regularization(points, src, dst):
    """Compute gradient of spectral regularizer (algebraic connectivity)."""
    if src is None or len(src) == 0:
        return np.zeros_like(points)
    
    # Build adjacency matrix
    n = len(points)
    adj = np.zeros((n, n))
    for i, j in zip(src, dst):
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    
    # Compute Laplacian
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Find smallest non-zero eigenvalue (algebraic connectivity)
    # Use power iteration on L - lambda*I to find smallest eigenvalue
    # For efficiency, use scipy
    from scipy.sparse.linalg import eigsh
    
    L_sparse = csr_matrix(L)
    
    try:
        # Find smallest eigenvalues
        vals, vecs = eigsh(L_sparse, k=2, which='SM')
        # Algebraic connectivity is the second smallest eigenvalue
        idx = np.argsort(vals)
        if len(idx) >= 2:
            lambda_2 = vals[idx[1]]
            v_2 = vecs[:, idx[1]]
        else:
            return np.zeros_like(points)
    except Exception:
        return np.zeros_like(points)
    
    # Gradient of lambda_2 w.r.t. positions
    # This is complex; use a simpler approximation
    # Force particles to spread uniformly based on eigenvector
    # Particles with similar v_2 values should be pushed apart
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                # Push apart if eigenvector values are similar
                diff_v = v_2[i] - v_2[j]
                if abs(diff_v) < 0.1:
                    vec = points[i] - points[j]
                    dist = np.linalg.norm(vec)
                    if dist > 1e-9:
                        grad[i] += vec / dist * 0.1
                        grad[j] -= vec / dist * 0.1
    
    return grad

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initialization
    points = _hex_init(N)
    best_points = points.copy()
    
    # Precompute all pair indices for scoring
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_score = _compute_radius(points, i_idx, j_idx)
    
    # Build initial sparse graph
    src, dst, edge_list = _build_sparse_graph(points)
    
    # Optimization parameters
    alpha = 0.015
    cooling = 0.99995
    k_soft_start = 8.0
    k_soft_end = 300.0
    max_iter = 8000
    iter_count = 0
    stall_count = 0
    stall_threshold = 80
    graph_rebuild_interval = 200
    
    k_soft = k_soft_start
    
    # Phase 1: Main optimization with sparse graph
    while time.time() - start_time < time_limit * 0.75 and iter_count < max_iter:
        # Rebuild graph periodically
        if iter_count % graph_rebuild_interval == 0:
            src, dst, edge_list = _build_sparse_graph(points)
        
        # Compute pairwise distances for active edges only
        if src is not None and len(src) > 0:
            active_diff = points[src] - points[dst]
            active_dists = np.linalg.norm(active_diff, axis=1)
        else:
            # Fallback: use all pairs
            active_diff = points[i_idx] - points[j_idx]
            active_dists = np.linalg.norm(active_diff, axis=1)
            src = i_idx
            dst = j_idx
        
        # Boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # Constraint values
        pair_vals = active_dists / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Soft-min weights
        shifted_vals = all_vals - current_min
        threshold = 50.0 / k_soft
        mask_close = shifted_vals < threshold
        
        if np.any(mask_close):
            close_vals = shifted_vals[mask_close]
            weights = np.zeros_like(all_vals)
            close_weights = np.exp(-k_soft * close_vals)
            weights[mask_close] = close_weights
            sum_w = np.sum(weights)
            if sum_w > 0:
                weights /= sum_w
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        # Compute gradients
        grads = np.zeros_like(points)
        
        # Pair contributions (only active edges)
        pair_weights = weights[:len(src)]
        mask_nonzero = active_dists > 1e-9
        if np.any(mask_nonzero):
            vecs = (active_diff[mask_nonzero] / active_dists[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            np.add.at(grads, src[mask_nonzero], vecs)
            np.add.at(grads, dst[mask_nonzero], -vecs)
        
        # Boundary contributions
        bound_weights = weights[len(src):]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Add spectral regularization (small weight)
        if src is not None and len(src) > 0:
            spec_grad = _spectral_regularization(points, src, dst)
            grads += 0.05 * spec_grad
        
        # Normalize and update
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += alpha * grads
        
        # Project inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        alpha *= cooling
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 2)
        
        if stall_count > stall_threshold:
            noise_scale = alpha * 3.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha *= 0.85
        
        iter_count += 1

    # Phase 2: Polish with full pairwise check
    points = best_points.copy()
    polish_alpha = 0.0008
    polish_k = 500.0
    polish_time_limit = time_limit * 0.95
    
    while time.time() - start_time < polish_time_limit:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
        
        shifted_vals = all_vals - current_min
        mask_close = shifted_vals < 20.0
        weights = np.zeros_like(all_vals)
        if np.any(mask_close):
            close_vals = shifted_vals[mask_close]
            close_weights = np.exp(-polish_k * close_vals)
            weights[mask_close] = close_weights
            sum_w = np.sum(weights)
            if sum_w > 0:
                weights /= sum_w
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += polish_alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        polish_alpha *= 0.9995
        polish_k *= 1.0005

    return best_points
