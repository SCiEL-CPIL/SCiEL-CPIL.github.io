import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.sparse import LinearOperator

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # The evaluator kills at 230s. We set a hard stop at 200s to be safe.
    deadline = start_time + 200.0

    # 1. Setup Fast Band Solver
    # Use 15x15 = 225 plane waves for internal optimization.
    # This is a good balance between accuracy and speed.
    # The evaluator uses 289 (17x17). 225 should give a very similar ranking.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a dense grid in the irreducible BZ.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    # We sample a grid and filter for kx <= ky.
    n_k_grid = 6
    k_points = []
    for i in range(n_k_grid + 1):
        for j in range(i, n_k_grid + 1):
            s = i / n_k_grid
            t = j / n_k_grid
            k_points.append((s * np.pi, t * np.pi))
    
    # Add critical high-symmetry points explicitly to ensure they are covered
    # M = (pi, 0), X = (pi/2, 0) -- wait, irreducible is X-M-G?
    # Square BZ: Gamma(0,0), X(pi/2, 0), M(pi, 0) is not a corner?
    # Standard square BZ:
    # Gamma (0,0)
    # X (pi/a, 0) -> in our units (pi, 0) if a=1? No, usually BZ is -pi/a to pi/a.
    # If lattice constant is 1, BZ is [-pi, pi].
    # Irreducible sector: 0 <= kx <= ky <= pi.
    # Points:
    # Gamma: (0,0)
    # X: (pi/2, 0) is on the edge? No, X is usually (pi, 0) in some conventions or (pi/2, 0) if BZ is 0..pi?
    # Let's stick to the grid 0..pi.
    # The grid 0..pi covers (0,0), (pi,0), (pi/2, pi/2), (pi, pi).
    # (pi, 0) is M? (pi, pi) is the corner?
    # For a square lattice with a=1, reciprocal lattice vectors are 2pi.
    # BZ boundaries are at k=pi.
    # So (pi, 0) is the midpoint of the BZ edge (M point).
    # (pi, pi) is the corner (X point? or M?).
    # Anyway, the grid covers the sector well.
    
    k_points = np.array(k_points)

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        Returns the gap-midgap ratio.
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        # We need the Fourier coefficients of 1/epsilon for TM modes?
        # For TM modes, the eigenvalue equation involves (k+G)^2 / epsilon(G).
        # Wait, the standard PWE formulation for TM is:
        # sum_G' (1/epsilon_{G-G'}) (k+G)^2 E_{G'} = omega^2/c^2 E_G
        # So the matrix elements involve 1/epsilon.
        
        # Let's compute 1/eps_F directly.
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        # Handle zero division (should not happen if eps_low > 0)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                # A * v = sum_j A_ij v_j
                # A_ij = (k+G_i)^2 / eps_{G_i - G_j}
                # This is a convolution.
                # Let w_i = (k+G_i)^2 v_i
                # Then result_i = sum_j (1/eps_{G_i-G_j}) w_j
                # This is circular convolution of w and 1/eps_F.
                
                w = kG2 * v
                # Reshape w to grid for FFT
                # Our G_vecs are ordered m, n from -7 to 7.
                # FFT expects 0..N-1.
                # We need to map the indices.
                # Actually, np.fft.fft2 operates on the array indices 0..N-1.
                # Our G_vecs are just a list. We need to ensure the convolution matches.
                
                # Let's construct the matrix explicitly for small N? 
                # N=225. 225x225 matrix is small.
                # But we can use FFT if we arrange G_vecs correctly.
                
                # Alternative: Explicit matrix construction is faster for N=225 in Python loops?
                # No, FFT is better.
                
                # To use FFT for convolution, the array must be arranged such that 
                # index difference corresponds to G difference.
                # Let's just build the sparse matrix or use a dense matrix multiplication 
                # since 225 is small.
                
                # Dense matrix approach for internal loop:
                # Precompute A? No, A depends on k.
                # A_ij = kG2[i] * inv_eps_{G_i - G_j}
                
                # Let's stick to the LinearOperator with FFT if possible, 
                # or just dense matmul.
                # 225x225 dense matmul is very fast.
                
                # We need inv_eps for all differences G_i - G_j.
                # G_i - G_j ranges from -14 to 14.
                # We have inv_eps_F for -7 to 7.
                # Wait, the convolution sum is over all G'.
                # A_ij = kG2_i * (1/eps)_{G_i - G_j}
                
                # Let's precompute the full 1/eps grid for differences up to 14.
                # But we only have eps_grid of size 32.
                # The Fourier coefficients for G > 15 are not computed in our 15x15 basis?
                # No, the basis is 15x15 G vectors.
                # The matrix element A_ij requires 1/eps_{G_i - G_j}.
                # G_i and G_j are in [-7, 7].
                # G_i - G_j is in [-14, 14].
                # We need 1/eps_F for these differences.
                # We can compute the FFT of the 32x32 grid to get these.
                
                pass # This logic is getting complicated. Let's use a simpler, robust method.
                
            # Robust Method: Dense Matrix
            # Precompute the 1/eps Fourier coefficients for all needed differences.
            # We need 1/eps_G for G in [-14, 14] x [-14, 14].
            # We can compute the FFT of a larger grid or just extract from the 32x32 FFT.
            # The 32x32 FFT gives coefficients for G in [-16, 15].
            # So we have all required coefficients.
            
            # Let's precompute a lookup table for inv_eps_F for G in [-14, 14].
            # Map G to FFT index:
            # G_x = -14 -> index 18 (32-14)
            # G_x = 0 -> index 0
            # G_x = 14 -> index 14
            
            # We will create a dense matrix A for each k.
            
            # Precompute G differences and their inv_eps values
            # G_vecs is (nG, 2).
            # We need inv_eps for all pairs (i, j).
            
            # Let's vectorize the matrix construction.
            # G_diff = G_i - G_j
            # This is a (nG, nG, 2) array.
            # We can compute this once.
            
            G_diffs = G_vecs[:, np.newaxis, :] - G_vecs[np.newaxis, :, :] # (nG, nG, 2)
            G_diff_x = G_diffs[:, :, 0]
            G_diff_y = G_diffs[:, :, 1]
            
            # Map to FFT indices
            # idx_x = G_diff_x % size
            # idx_y = G_diff_y % size
            idx_x = G_diff_x.astype(int) % size
            idx_y = G_diff_y.astype(int) % size
            
            # Extract inv_eps_F values
            # inv_eps_F is (32, 32).
            # We need inv_eps_F[idx_y, idx_x] ? 
            # FFT convention: F[kx, ky] corresponds to G = (2pi * kx, 2pi * ky).
            # My G_vecs are (2pi * m, 2pi * n).
            # So G_diff_x / (2pi) = m_diff.
            # The FFT index is m_diff % 32.
            
            # Note: np.fft.fft2 returns F[u, v] where u is x-index, v is y-index?
            # Usually F[u, v] = sum_{x,y} f[x, y] exp(-i 2pi (u x + v y) / N).
            # So u corresponds to x-frequency, v to y-frequency.
            # So we need inv_eps_F[idx_x, idx_y].
            
            inv_eps_matrix = inv_eps_F[idx_x, idx_y] # (nG, nG)
            
            for kx, ky in k_points:
                # A_ij = (kx + G_i_x)^2 + (ky + G_i_y)^2) * inv_eps_{G_i - G_j}
                kG_x = kx + G_vecs[:, 0]
                kG_y = ky + G_vecs[:, 1]
                kG2 = kG_x**2 + kG_y**2 # (nG,)
                
                # A = diag(kG2) @ inv_eps_matrix
                # We only need the smallest eigenvalues.
                # A is not symmetric? 
                # inv_eps_{G_i - G_j} is symmetric if eps is real and symmetric?
                # eps_{-G} = conj(eps_G). For real eps, eps_{-G} = eps_G.
                # So inv_eps is symmetric.
                # Thus A is symmetric.
                
                A = kG2[:, np.newaxis] * inv_eps_matrix
                
                try:
                    # Use dense eigsh? No, eigsh is for sparse.
                    # Use np.linalg.eigvalsh for small dense symmetric matrices.
                    vals = np.linalg.eigvalsh(A)
                    vals = np.sort(vals)
                    
                    # We need bands 4 and 5 (indices 3 and 4)
                    # Note: The lowest modes might be spurious or just the first bands.
                    # The problem asks for the gap between 4th and 5th TM bands.
                    
                    if len(vals) < 5:
                        return -1.0
                    
                    omega4 = np.sqrt(max(0, vals[3]))
                    omega5 = np.sqrt(max(0, vals[4]))
                    
                    if omega5 < min_omega5:
                        min_omega5 = omega5
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                        
                except Exception:
                    return -1.0

        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def generate_cell_from_params(params):
        """
        Generate a symmetric 32x32 cell from continuous parameters.
        Params:
        p[0]: inner radius (0 to 16)
        p[1]: outer radius (0 to 16)
        p[2]: superellipse exponent n (2 to 10)
        p[3]: rotation? No, fixed symmetry.
        
        We use a super-ellipse ring centered at the corners?
        Or a single super-ellipse in the center?
        The seed used 4 squares.
        Let's try a 4-fold symmetric ring of super-ellipses.
        
        Actually, let's use a single super-ellipse in the center, 
        and enforce the symmetry by replicating it to the 4 quadrants?
        No, the symmetry is about the center of the 32x32 cell.
        
        Let's define the shape by a distance function in the fundamental domain.
        Fundamental domain: x in [0, 16], y in [0, 16].
        
        Let's try a "hole" in the center.
        Dielectric is 1. Air is 0.
        Seed had 4 squares of dielectric.
        
        Let's parameterize a super-ellipse ring.
        Center of cell is (15.5, 15.5).
        
        We will define the cell based on a level set:
        dist(x, y) < r_inner -> Air (0)
        r_inner < dist(x, y) < r_outer -> Dielectric (1)
        dist(x, y) > r_outer -> Air (0)
        
        Where dist is a super-ellipse distance.
        
        This creates a ring.
        To break the degeneracy or enhance the gap, we might need to distort it.
        Let's add a "bump" parameter.
        
        Simpler: Just optimize the radius and exponent of a central super-ellipse hole.
        """
        r_inner = params[0]
        r_outer = params[1]
        n_exp = params[2]
        
        cell = np.zeros((size, size), dtype=int)
        
        # Coordinates
        x = np.arange(size) - 15.5
        y = np.arange(size) - 15.5
        X, Y = np.meshgrid(x, y)
        
        # Super-ellipse distance
        # (|X|/a)^n + (|Y|/a)^n = 1
        # Here a is the "radius".
        # dist = (|X|^n + |Y|^n)^(1/n)
        
        # To avoid 0^0 issues, use a small epsilon
        abs_X = np.abs(X) + 1e-6
        abs_Y = np.abs(Y) + 1e-6
        
        # For n=2, this is Euclidean distance.
        # For n=4, it's squarer.
        
        # We want to handle different n.
        # dist = (abs_X**n + abs_Y**n)**(1/n)
        
        # This can be numerically unstable for large n.
        # Use log-sum-exp trick?
        # log(dist) = (1/n) * log(exp(n log abs_X) + exp(n log abs_Y))
        
        log_abs_X = np.log(abs_X)
        log_abs_Y = np.log(abs_Y)
        
        max_log = np.maximum(n_exp * log_abs_X, n_exp * log_abs_Y)
        sum_exp = np.exp(n_exp * log_abs_X - max_log) + np.exp(n_exp * log_abs_Y - max_log)
        log_dist = (1/n_exp) * (max_log + np.log(sum_exp))
        dist = np.exp(log_dist)
        
        # Apply mask
        mask_inner = dist < r_inner
        mask_outer = dist < r_outer
        
        # Dielectric is 1 where r_inner < dist < r_outer
        cell[mask_outer] = 1
        cell[mask_inner] = 0
        
        # Enforce symmetry explicitly (though the function is symmetric)
        # The meshgrid is centered, so it should be symmetric.
        # However, pixel quantization might break it slightly.
        # We will enforce C4v symmetry by averaging/majority vote or just copying.
        
        # Since the function is symmetric, the result should be symmetric.
        # But let's ensure strict symmetry.
        
        # Return the cell
        return cell

    # Optimization Strategy
    # We want to maximize the gap.
    # Minimize -gap.
    
    def objective(params):
        cell = generate_cell_from_params(params)
        score = compute_score(cell)
        return -score

    # Initial guess
    # Try a ring with inner radius 5, outer radius 10, exponent 2 (circle)
    x0 = np.array([5.0, 10.0, 2.0])
    
    # Bounds
    # r_inner: 0 to 15
    # r_outer: 0 to 15
    # n_exp: 2 to 8
    bounds = [(0, 15), (0, 15), (2, 8)]
    
    # Nelder-Mead
    # Set a time limit for the optimizer
    # We have ~190s.
    # Each function eval takes ~0.1s? 
    # 225x225 eigvalsh is fast.
    # Let's run Nelder-Mead with maxiter=1000.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                       options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
        
        best_params = res.x
        best_cell = generate_cell_from_params(best_params)
        best_score = -res.fun
        
        # If the result is bad, try a few other starting points
        # e.g., square-like (n=4), or just a solid block
        alternative_starts = [
            [3.0, 8.0, 4.0], # Square ring
            [0.0, 12.0, 2.0], # Solid circle
            [8.0, 12.0, 2.0], # Small ring
            [5.0, 12.0, 6.0]  # Super-ellipse ring
        ]
        
        for start in alternative_starts:
            if time.time() > deadline - 30.0:
                break
            res_alt = minimize(objective, start, method='Nelder-Mead', bounds=bounds, 
                               options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            if -res_alt.fun > best_score:
                best_score = -res_alt.fun
                best_params = res_alt.x
                best_cell = generate_cell_from_params(best_params)
                
    except Exception as e:
        # Fallback to seed if optimizer fails
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_cell = cell
        best_score = compute_score(best_cell)

    # Final check
    if time.time() < deadline - 10.0:
         # Maybe do a quick pixel-based refinement on the best cell?
         # No, let's keep it simple. The continuous optimization should be good.
         pass

    return best_cell.tolist()

