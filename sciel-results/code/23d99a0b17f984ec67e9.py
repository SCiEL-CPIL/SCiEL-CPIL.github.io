"""Combined program: MLP parameterization with Soft-Min objective and Violation Energy polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = HALF - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_obj_grad(pos, beta):
    """
    Computes the negative soft-min objective and its gradient.
    We minimize -soft_min, which is equivalent to maximizing soft_min.
    """
    pos = pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq + 1e-15)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # Effective distances: pair_dists/2 and wall_dists
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min: -logsumexp(-beta * x) / beta
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. positions
    # weights = softmax(-beta * all_eff_dists)
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient
    n_pairs = len(iu[0])
    pair_weights = weights[:n_pairs]
    
    i_idx = iu[0]
    j_idx = iu[1]
    d_vals = pair_dists
    
    valid = d_vals > 1e-14
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        d_v = d_vals[valid]
        w_v = pair_weights[valid]
        
        vecs = pos[i_v] - pos[j_v]
        coeffs = w_v / (2.0 * d_v)
        
        np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
        np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    wall_weights = weights[n_pairs:]
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    
    grad_wall = -wall_weights.reshape(N, 3) * signs
    grad_pos += grad_wall
    
    # We want to MAXIMIZE soft_min, so minimize -soft_min
    return -soft_min_val, -grad_pos.flatten()

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, seed=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.rng = np.random.default_rng(seed)
        
        # Initialize weights
        self.W = []
        self.b = []
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # He initialization
            W = self.rng.normal(0, np.sqrt(2.0 / prev_dim), size=(prev_dim, h_dim))
            b = np.zeros(h_dim)
            self.W.append(W)
            self.b.append(b)
            prev_dim = h_dim
            
        # Output layer
        W_out = self.rng.normal(0, np.sqrt(2.0 / prev_dim), size=(prev_dim, output_dim))
        b_out = np.zeros(output_dim)
        self.W.append(W_out)
        self.b.append(b_out)
        
        self.num_layers = len(hidden_dims) + 1

    def forward(self, z):
        """Forward pass, saving activations for backprop."""
        self.activations = [z]
        self.z = z
        
        h = z
        for i in range(self.num_layers - 1):
            z_linear = h @ self.W[i] + self.b[i]
            # ReLU activation
            h = np.maximum(0, z_linear)
            self.activations.append(h)
            
        # Output layer: Tanh
        z_linear_out = h @ self.W[-1] + self.b[-1]
        out = np.tanh(z_linear_out) * HALF
        
        self.z_linear_out = z_linear_out
        self.out = out
        return out

    def backward(self, grad_out):
        """
        Backward pass.
        grad_out: gradient of loss w.r.t. output (N*3,)
        Returns gradient w.r.t. input z (input_dim,)
        """
        grad_out = grad_out.reshape(N, 3)
        
        # Gradient of Tanh
        # d(tanh(x))/dx = 1 - tanh(x)^2
        # out = tanh(z_linear) * 0.5
        # d(out)/d(z_linear) = 0.5 * (1 - tanh(z_linear)^2)
        # tanh(z_linear) = out / 0.5 = 2 * out
        tanh_val = 2.0 * self.out
        grad_z_linear_out = grad_out * (0.5 * (1.0 - tanh_val**2))
        
        # Backprop through output layer
        h_prev = self.activations[-1] # Last hidden layer output
        grad_W_out = np.outer(h_prev, grad_z_linear_out.flatten())
        grad_b_out = grad_z_linear_out.flatten()
        
        grad_h = grad_z_linear_out @ self.W[-1].T
        
        # Backprop through hidden layers
        grads_W = [grad_W_out]
        grads_b = [grad_b_out]
        
        for i in range(self.num_layers - 2, -1, -1):
            # ReLU gradient
            grad_z_linear = grad_h * (self.activations[i+1] > 0)
            
            grad_W_i = np.outer(self.activations[i], grad_z_linear.flatten())
            grad_b_i = grad_z_linear.flatten()
            
            grads_W.insert(0, grad_W_i)
            grads_b.insert(0, grad_b_i)
            
            if i > 0:
                grad_h = grad_z_linear @ self.W[i].T
            else:
                # Gradient w.r.t input z
                grad_z = grad_z_linear @ self.W[0].T
                
        return grad_z.flatten()

def _mlp_loss_and_grad(z, beta):
    """
    Loss function for L-BFGS-B.
    z: latent vector
    beta: soft-min parameter
    """
    # We need a global or passed MLP instance
    # Since minimize passes only z, we use a closure or global
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Initialize MLP
    # Latent dim 150, Hidden 200, 200, Output 300
    mlp = MLP(150, [200, 200], N * 3, seed=42)
    
    # 2. Initialize latent vector z
    # We want the initial output to be close to a good packing (FCC)
    # Generate FCC lattice
    positions = []
    # 5x4x4 grid = 80 points. Need 20 more.
    # Let's just use a random z first, but scale weights so output is small?
    # Better: Initialize biases to FCC coords, weights to 0.
    # Then output = biases (approx) if weights are 0? No, ReLU(0) = 0.
    # If we set W=0, b=FCC_coords, then h=0, out = tanh(b_out)*0.5.
    # So we need to set the final bias to FCC_coords * 2 (since out = tanh(b)*0.5, we want out ~ FCC, so tanh(b) ~ 2*FCC, b ~ atanh(2*FCC)).
    
    # Generate FCC-like init
    nx, ny, nz = 5, 4, 4
    spacing = 1.0 / 4.0
    fcc_coords = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = -0.5 + (ix + 0.5) * spacing
                y = -0.5 + (iy + 0.5) * spacing
                z = -0.5 + (iz + 0.5) * spacing
                fcc_coords.append([x, y, z])
                if len(fcc_coords) < N:
                    bx = x + spacing/2
                    by = y + spacing/2
                    bz = z + spacing/2
                    if abs(bx) < 0.49 and abs(by) < 0.49 and abs(bz) < 0.49:
                        fcc_coords.append([bx, by, bz])
    
    fcc_coords = np.array(fcc_coords[:N]).flatten()
    
    # Set final bias to atanh(fcc_coords / 0.5) = atanh(2 * fcc_coords)
    # Clip to avoid inf
    target_tanh = np.clip(2.0 * fcc_coords, -0.99, 0.99)
    mlp.b[-1] = np.arctanh(target_tanh)
    
    # Set other weights to small random values to allow variation
    # Set final weights to small values so bias dominates initially
    mlp.W[-1] = mlp.rng.normal(0, 0.01, size=mlp.W[-1].shape)
    
    # Initial z
    z0 = np.zeros(150)
    
    # 3. Optimization Loop with Annealing
    best_r = _compute_radius(mlp.forward(z0))
    best_z = z0.copy()
    
    betas = [10, 20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        # Define objective for this beta
        def obj_and_grad(z):
            pos = mlp.forward(z)
            val, grad_pos = _soft_min_obj_grad(pos, beta)
            grad_z = mlp.backward(grad_pos)
            return val, grad_z
        
        # Optimize
        try:
            res = minimize(
                obj_and_grad,
                best_z,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            new_z = res.x
            new_pos = mlp.forward(new_z)
            new_r = _compute_radius(new_pos)
            
            if new_r > best_r:
                best_r = new_r
                best_z = new_z
        except Exception:
            pass

    # 4. Polish with direct coordinate optimization
    best_pos = mlp.forward(best_z)
    
    # Use violation energy minimization for final polish
    def _energy_and_gradient(x, R):
        centers = x.reshape(-1, 3)
        n = len(centers)
        energy = 0.0
        grad = np.zeros_like(centers)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

        abs_centers = np.abs(centers)
        max_abs = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_abs
        mask_wall = wall_dists < R
        if np.any(mask_wall):
            violations_wall = R - wall_dists[mask_wall]
            i_valid_wall = np.where(mask_wall)[0]
            energy += np.sum(violations_wall**2)
            
            max_k = np.argmax(abs_centers[i_valid_wall], axis=1)
            signs = np.sign(centers[i_valid_wall, max_k])
            grad_vals = 2.0 * violations_wall * signs
            np.add.at(grad, (i_valid_wall, max_k), grad_vals)
            
        return energy, grad.flatten()

    # Polish steps
    for step in range(5):
        if time.time() > deadline - 5.0:
            break
        target_r = best_r + 1e-5 * (step + 1)
        
        def obj_polish(x):
            val, g = _energy_and_gradient(x, target_r)
            return val, g
            
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                obj_polish,
                best_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            new_pos = res.x.reshape(N, 3)
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos
        except Exception:
            break

    return best_pos

# EVOLVE-BLOCK-END
