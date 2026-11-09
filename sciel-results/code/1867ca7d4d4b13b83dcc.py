import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2
    start_time = time.time()
    max_time = 600  # Safety margin from 630s limit
    
    def compute_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize(points):
        # Center at origin
        points = points - np.mean(points, axis=0)
        # Scale so max distance is 1
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max
        return points

    def optimize(points, max_iter=5000):
        # Start with small beta to make objective smooth
        beta = 10.0
        lr = 0.1
        best_points = points.copy()
        best_score = compute_score(points)
        
        for it in range(max_iter):
            if time.time() - start_time > max_time - 10:
                break
                
            # Compute pairwise squared distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            # Set diagonal to 0 to avoid self-interaction issues in sums, though we mask them
            np.fill_diagonal(dists_sq, 0)
            
            # Mask to only consider i < j for symmetry, but the gradient formula 
            # derived uses full matrix W where W_ij = w_ij.
            # Let's stick to the derived gradient:
            # w_ij = (1/u) * exp(-beta * D_ij) / S_min - (1/v) * exp(beta * D_ij) / S_max
            
            # Calculate S_min and S_max
            # We need to be careful with overflow/underflow.
            # D_ij is >= 0.
            # exp(-beta * D) is in (0, 1].
            # exp(beta * D) can be large.
            
            # To improve stability, we can subtract max/min from exponents?
            # For S_min = sum exp(-beta D), max term is exp(0)=1. So no overflow.
            # For S_max = sum exp(beta D), max term is exp(beta D_max). 
            # If D_max is large, this overflows.
            # We can factor out exp(beta D_max):
            # S_max = exp(beta D_max) * sum exp(beta (D - D_max))
            # Then w_ij term 2: (1/v) * exp(beta D_ij) / S_max
            # = (1/v) * exp(beta D_ij) / (exp(beta D_max) * sum exp(beta (D - D_max)))
            # = (1/v) * exp(-beta (D_max - D_ij)) / sum exp(beta (D - D_max))
            
            # Let's implement this stable version.
            D_flat = dists_sq[np.triu_indices(n, k=1)]
            
            # Soft-min part
            # u = -1/beta * log(sum exp(-beta D))
            # S_min = sum exp(-beta D)
            # Since D >= 0, exp(-beta D) <= 1. Sum is <= n(n-1)/2.
            log_S_min = np.log(np.sum(np.exp(-beta * D_flat)))
            u = -1.0 / beta * log_S_min
            if u <= 0: u = 1e-10 # Safety
            
            # Soft-max part
            # v = 1/beta * log(sum exp(beta D))
            D_max_val = np.max(D_flat)
            # Shift to prevent overflow
            shifted_D = D_flat - D_max_val
            log_S_max_shifted = np.log(np.sum(np.exp(beta * shifted_D)))
            log_S_max = beta * D_max_val + log_S_max_shifted
            v = 1.0 / beta * log_S_max
            if v <= 0: v = 1e-10 # Safety
            
            # Weights w_ij for all pairs i,j
            # We need W matrix of size n x n
            # w_ij = (1/u) * exp(-beta D_ij) / S_min - (1/v) * exp(-beta (D_max - D_ij)) / S_max_shifted_sum
            # Note: S_min = exp(log_S_min), S_max_shifted_sum = exp(log_S_max_shifted)
            
            W = np.zeros((n, n))
            
            # Term 1: min part
            # exp(-beta D_ij) / S_min = exp(-beta D_ij - log_S_min)
            term1 = np.exp(-beta * dists_sq - log_S_min)
            
            # Term 2: max part
            # exp(-beta (D_max - D_ij)) / S_max_shifted_sum = exp(-beta (D_max - D_ij) - log_S_max_shifted)
            # D_max here is the global max distance in the set.
            # Note: D_ij is 0 on diagonal.
            term2 = np.exp(-beta * (D_max_val - dists_sq) - log_S_max_shifted)
            
            # Combine
            # w_ij = (1/u) * term1 - (1/v) * term2
            W = (1.0 / u) * term1 - (1.0 / v) * term2
            
            # Diagonal is 0 in W? 
            # D_ii = 0.
            # term1_ii = exp(0 - log_S_min) = 1/S_min
            # term2_ii = exp(-beta D_max - log_S_max_shifted) = exp(-log_S_max) = 1/S_max
            # w_ii = 1/(u S_min) - 1/(v S_max)
            # But D_ii does not depend on x_i in the same way? 
            # D_ii = 0 always. Gradient of D_ii is 0.
            # So we should set W_ii = 0.
            np.fill_diagonal(W, 0)
            
            # Compute Gradient
            # grad_x_k = 2 * ( (sum_j W_kj) * x_k - (W x)_k )
            S_k = np.sum(W, axis=1) # shape (n,)
            Wx = W @ points # shape (n, 2)
            grad = 2.0 * (S_k[:, np.newaxis] * points - Wx)
            
            # Update
            points = points + lr * grad
            
            # Normalize
            points = normalize(points)
            
            # Check score
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Schedule
            # Increase beta to sharpen
            if it % 100 == 0:
                beta *= 1.5
            # Decay learning rate
            if it % 100 == 0:
                lr *= 0.95
                
        return best_points, best_score

    # Initializations
    inits = []
    
    # 1. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Grid (4x4)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(x, y)
    grid_pts = np.column_stack([gx.flatten(), gy.flatten()])
    inits.append(grid_pts)
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.randn(n, 2))
    
    # 4. Hexagonal-ish
    # 3 rows: 5, 6, 5
    pts = []
    for i in range(3):
        count = 5 if i != 1 else 6
        offset = 0.5 if i == 1 else 0
        for j in range(count):
            x = j - (count-1)/2.0
            y = i * np.sqrt(3)/2
            pts.append([x + offset, y])
    hex_pts = np.array(pts)
    inits.append(hex_pts)
    
    best_overall_points = inits[0]
    best_overall_score = -1
    
    for init in inits:
        if time.time() - start_time > max_time - 30:
            break
        pts, score = optimize(init, max_iter=2000)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = pts
            
    return best_overall_points

# EVOLVE-BLOCK-END
