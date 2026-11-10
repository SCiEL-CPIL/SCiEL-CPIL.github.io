# EVOLVE-BLOCK-START
import time
import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    num_tubes = N * N
    
    # Initialize with the Keich binary construction as a starting point
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # x_k depends on i, y_k depends on j in the separable case, 
    # but here we allow all 64 to move independently to break separability
    init_x = keich[i_flat.astype(int)]
    init_y = keich[j_flat.astype(int)]
    
    params = np.concatenate([init_x, init_y])
    
    # Fixed sample points for the differentiable proxy (smaller than 500k for speed)
    rng = np.random.default_rng(42)
    # We sample in a normalized space roughly covering the expected range
    # The box size will be determined by the current params, but for the proxy 
    # we can use a fixed set of points in [0, 1]x[0, 1]x[0, 1] and map them,
    # or just use the actual box bounds. Using actual box bounds is better.
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Optimization parameters
    lr = 0.05
    sigma = 0.05  # Smoothness parameter for sigmoid
    num_proxy_points = 20000
    
    # Pre-generate proxy points in a unit cube, we will scale them to the box
    proxy_a = rng.uniform(0, 1, num_proxy_points)
    proxy_b = rng.uniform(0, 1, num_proxy_points)
    proxy_c = rng.uniform(0, 1, num_proxy_points)
    
    best_score = -np.inf
    best_params = params.copy()
    
    while time.time() < deadline:
        x = params[:num_tubes]
        y = params[num_tubes:]
        
        # Compute box bounds
        X0 = np.min(x)
        X1 = np.max(x + (i_flat + 1) / N)
        Y0 = np.min(y)
        Y1 = np.max(y + (j_flat + 1) / N)
        
        box_width = X1 - X0
        box_height = Y1 - Y0
        
        if box_width <= 1e-9 or box_height <= 1e-9:
            # Perturb to escape degenerate state
            params += rng.normal(0, 0.1, num_tubes * 2)
            continue
            
        # Map proxy points to the actual box
        a = X0 + proxy_a * box_width
        b = Y0 + proxy_b * box_height
        c = proxy_c
        
        # Compute differentiable proxy volume
        # For each tube, compute the "soft" indicator of points inside
        # Tube k: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        
        # Vectorized computation of distances to tube boundaries
        # This is 64 tubes * 20000 points = 1.28M operations, which is fast in NumPy
        
        # We want sum_k sigmoid( (min_dist_to_boundary_k) / sigma )
        # min_dist_to_boundary is negative if inside, positive if outside.
        # Actually, let's use: dist_x = min(a - (x_k + c*i_k/N), (x_k + c*i_k/N + 1/N) - a)
        # If dist_x > 0 and dist_y > 0, point is inside.
        # We can use a smooth approximation of the AND operation: sigmoid(dist_x/sigma) * sigmoid(dist_y/sigma)
        # Or sum of sigmoids for the union: 1 - prod(1 - sigmoid(...))
        # For optimization, sum of sigmoids is easier and a good proxy for small overlap.
        
        # Let's compute the "inside-ness" for each tube
        # To save memory, we process in batches of tubes
        batch_size = 16
        proxy_volume = 0.0
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(y)
        
        for start_idx in range(0, num_tubes, batch_size):
            end_idx = min(start_idx + batch_size, num_tubes)
            batch_x = x[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            batch_i = i_flat[start_idx:end_idx]
            batch_j = j_flat[start_idx:end_idx]
            
            # Batch shapes: (batch_size, 1)
            bx = batch_x[:, np.newaxis]
            by = batch_y[:, np.newaxis]
            bi = batch_i[:, np.newaxis]
            bj = batch_j[:, np.newaxis]
            
            # Point shapes: (1, num_proxy_points)
            pa = a[np.newaxis, :]
            pb = b[np.newaxis, :]
            pc = c[np.newaxis, :]
            
            # Tube left/right at height c
            left = bx + pc * bi / N
            right = left + 1.0 / N
            bottom = by + pc * bj / N
            top = bottom + 1.0 / N
            
            # Distances to boundaries (positive if outside, negative if inside)
            # We want a value that is large when inside.
            # dist_x = min(pa - left, right - pa). If inside, both positive.
            # Let's use: d_x = min(pa - left, right - pa)
            # If d_x > 0, inside in x.
            d_x = np.minimum(pa - left, right - pa)
            d_y = np.minimum(pb - bottom, top - pb)
            
            # Smooth indicator: sigmoid(d_x / sigma) * sigmoid(d_y / sigma)
            # This is 1 when deep inside, 0 when deep outside.
            # Gradient with respect to x_k (which shifts left and right):
            # d_x depends on x_k. 
            # This is getting complicated for analytic gradients. 
            # Let's use a simpler proxy: Sum of sigmoid(d_x/sigma) + sigmoid(d_y/sigma) is not right.
            
            # Alternative: Just use the raw sum of sigmoid(d_x/sigma) * sigmoid(d_y/sigma)
            # and use finite differences for gradients? Too slow.
            
            # Let's stick to a simpler differentiable metric that is easy to differentiate.
            # Metric: Sum over points of max(0, min(pa - left, right - pa, pb - bottom, top - pb))
            # This is the "depth" of the point inside the tube.
            # The volume is roughly the integral of this depth? No.
            
            # Let's go back to the sum of sigmoid indicators.
            # Indicator I_k(p) = sigmoid(d_x_k(p)/sigma) * sigmoid(d_y_k(p)/sigma)
            # Volume_proxy = sum_p I_k(p)
            
            # Gradient of I_k with respect to x_k:
            # d_x = min(pa - (x_k + ...), (x_k + ... + 1/N) - pa)
            # Let u = pa - (x_k + c*i/N), v = (x_k + c*i/N + 1/N) - pa
            # d_x = min(u, v)
            # d(d_x)/dx_k = -1 if u < v, +1 if v < u
            # d(I)/dx_k = sigmoid'(d_x/sigma) * (d(d_x)/dx_k)/sigma * sigmoid(d_y/sigma)
            
            # This is implementable!
            
            s_x = 1.0 / (1.0 + np.exp(-d_x / sigma))
            s_y = 1.0 / (1.0 + np.exp(-d_y / sigma))
            indicator = s_x * s_y
            
            proxy_volume += np.sum(indicator)
            
            # Gradients
            # s'_x = sigmoid(d_x/sigma) * (1 - sigmoid(d_x/sigma)) * (1/sigma)
            ds_x = s_x * (1.0 - s_x) / sigma
            ds_y = s_y * (1.0 - s_y) / sigma
            
            # d(d_x)/dx_k
            # u = pa - x_k - c*i/N
            # v = x_k + c*i/N + 1/N - pa
            # u < v  <=>  pa - x_k - ... < x_k + ... - pa  <=>  2*pa < 2*x_k + ... + 1/N
            # center_x = x_k + c*i/N + 1/(2N)
            # if pa < center_x, then u < v? 
            # pa - left < right - pa  <=> 2pa < left + right <=> pa < (left+right)/2
            # So if pa < center, d_x = pa - left, d(d_x)/dx_k = -1
            # if pa > center, d_x = right - pa, d(d_x)/dx_k = +1
            
            center_x = bx + pc * bi / N + 1.0 / (2 * N)
            center_y = by + pc * bj / N + 1.0 / (2 * N)
            
            sign_x = np.where(pa < center_x, -1.0, 1.0)
            sign_y = np.where(pb < center_y, -1.0, 1.0)
            
            # d(I)/dx_k = ds_x * sign_x * s_y
            # d(I)/dy_k = s_x * ds_y * sign_y
            
            grad_batch_x = np.sum(ds_x * sign_x * s_y, axis=1)
            grad_batch_y = np.sum(s_x * ds_y * sign_y, axis=1)
            
            grad_x[start_idx:end_idx] += grad_batch_x
            grad_y[start_idx:end_idx] += grad_batch_y
            
        # The proxy volume is a sum over points. The actual volume is proportional to this sum * box_area.
        # We want to minimize the actual volume.
        # V = (proxy_volume / num_proxy_points) * box_width * box_height
        # Gradient of V with respect to x_k involves the gradient of proxy_volume AND the gradient of box_width.
        # box_width = max(x + ...) - min(x). The gradient of max/min is 1 for the argmax/argmin, 0 otherwise.
        
        # For simplicity, let's just minimize the proxy_volume itself, assuming box size is stable.
        # Or, we can include the box size in the objective.
        
        # Let's use the full gradient for V.
        # dV/dx_k = (1/num_points) * ( d(proxy_vol)/dx_k * box_area + proxy_vol * d(box_area)/dx_k )
        
        d_box_width_dx = np.zeros_like(x)
        d_box_height_dy = np.zeros_like(y)
        
        # Find indices that define the box bounds
        idx_max_x = np.argmax(x + (i_flat + 1) / N)
        idx_min_x = np.argmin(x)
        idx_max_y = np.argmax(y + (j_flat + 1) / N)
        idx_min_y = np.argmin(y)
        
        d_box_width_dx[idx_max_x] += 1.0
        d_box_width_dx[idx_min_x] -= 1.0
        d_box_height_dy[idx_max_y] += 1.0
        d_box_height_dy[idx_min_y] -= 1.0
        
        proxy_vol_mean = proxy_volume / num_proxy_points
        box_area = box_width * box_height
        
        full_grad_x = (1.0 / num_proxy_points) * (grad_x * box_area + proxy_vol_mean * d_box_width_dx * box_height)
        full_grad_y = (1.0 / num_proxy_points) * (grad_y * box_area + proxy_vol_mean * d_box_height_dy * box_width)
        
        # Normalize gradients
        norm = np.sqrt(np.sum(full_grad_x**2) + np.sum(full_grad_y**2))
        if norm > 1e-8:
            full_grad_x /= norm
            full_grad_y /= norm
            
        # Update parameters
        params -= lr * np.concatenate([full_grad_x, full_grad_y])
        
        # Evaluate actual Monte Carlo volume for this configuration (expensive, do sparingly)
        # We can do it every 10 iterations or if we have time
        if time.time() > start_time + 2: # At least 2 seconds in
            rng_eval = np.random.default_rng(12345)
            a_eval = rng_eval.uniform(X0, X1, 500000)
            b_eval = rng_eval.uniform(Y0, Y1, 500000)
            c_eval = rng_eval.uniform(0, 1, 500000)
            
            # Check if points are inside any tube
            # This is the same logic as the proxy, but with 500k points and exact check
            # We can use the same vectorized batch approach
            
            inside_any = np.zeros(500000, dtype=bool)
            for start_idx in range(0, num_tubes, 32):
                end_idx = min(start_idx + 32, num_tubes)
                batch_x = x[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                batch_i = i_flat[start_idx:end_idx]
                batch_j = j_flat[start_idx:end_idx]
                
                bx = batch_x[:, np.newaxis]
                by = batch_y[:, np.newaxis]
                bi = batch_i[:, np.newaxis]
                bj = batch_j[:, np.newaxis]
                
                pa = a_eval[np.newaxis, :]
                pb = b_eval[np.newaxis, :]
                pc = c_eval[np.newaxis, :]
                
                left = bx + pc * bi / N
                right = left + 1.0 / N
                bottom = by + pc * bj / N
                top = bottom + 1.0 / N
                
                in_x = (pa >= left) & (pa <= right)
                in_y = (pb >= bottom) & (pb <= top)
                in_tube = in_x & in_y
                
                inside_any |= np.any(in_tube, axis=0)
                
            fraction = np.mean(inside_any)
            V = fraction * box_area
            
            if np.isfinite(V) and V > 0:
                score = 1.0 / V
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            # Reset timer for this evaluation phase? No, just check deadline.
            if time.time() > deadline - 10:
                break

    # Return the best params found
    best_x = best_params[:num_tubes]
    best_y = best_params[num_tubes:]
    positions = np.stack([best_x, best_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
