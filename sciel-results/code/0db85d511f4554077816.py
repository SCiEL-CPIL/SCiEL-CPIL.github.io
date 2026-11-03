import time
import math
import sys
from itertools import product

# Entry point
def construct():
    deadline = time.time() + 145.0  # Leave 5s safety margin for the 150s limit
    return find_best_decomposition(deadline)

def find_best_decomposition(deadline):
    """
    Find a low-rank decomposition for 3x4 * 4x5.
    Returns a dict with keys 'U', 'V', 'W'.
    """
    n, m, p = 3, 4, 5
    
    # The tensor T is in R^{p * n * (n*m)}?
    # Task: multiply (n x m) by (m x p).
    # Input 1: A (n x m). Flatten: a_{i,j} index i*m+j. (n*m = 12)
    # Input 2: B (m x p). Flatten: b_{j,k} index j*p+k. (m*p = 20)
    # Output: C (n x p). Flatten: c_{i,k}.
    # The task specifies output factor follows trace convention W[k*n+i, r].
    # This means C is treated as a vector in the index k*n+i. (p*n = 15)
    
    # Decomposition:
    # We want C_{i,k} = sum_{r} (sum_{j} U_{i*m+j, r} * B_{j*p+k, r}?) 
    # Wait, the standard bilinear form is:
    # C_{i,k} = sum_{r=1}^{R} ( \sum_{j=1}^m U_{i*m+j, r} A_{i,j} ) * ( \sum_{j'=1}^m V_{j'*p+k, r} B_{j',k} )
    # Or more commonly:
    # C_{i,k} = sum_{r} ( \sum_{j} U_{i,j,r} A_{i,j} ) ( \sum_{j} V_{j,k,r} B_{j,k} )
    # Here U is (n*m) x R, V is (m*p) x R, W is (p*n) x R.
    
    # The tensor T_{i, k, j1, j2} = delta_{j1, j2} * delta_{i,i} * delta_{k,k}?
    # No, the multiplication tensor is T_{i,k, (i,j), (j',k')} = delta_{j,j'} .
    # Since the inputs share the index j.
    
    # Flattened Input indices:
    # I1 = i*m + j. Range 0..11.
    # I2 = j*p + k. Range 0..19.
    # Out  = k*n + i. Range 0..14. (Following W[k*n+i])
    
    # Tensor T_{I1, I2, Out} is non-zero iff:
    # I1 corresponds to (i, j)
    # I2 corresponds to (j', k)
    # Out corresponds to (k, i)
    # And j == j'.
    
    # We can build a list of non-zero tuples (I1, I2, Out).
    tensors = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                I1 = i * m + j
                I2 = j * p + k
                Out = k * n + i
                tensors.append((I1, I2, Out))
                
    # Total number of non-zeros = n * p * m = 60.
    # This is a 3-tensor with shape (12, 20, 15).
    # Rank bound: max(12, 20, 15) is too high? No, rank is at most 12*20*15 but usually much lower.
    # Known bounds for 3x4x5 matrix multiplication:
    # Naive: 60.
    # Can we do better?
    
    # Since 120s is available, we can try to solve for a rank R decomposition using Linear Programming or least squares if we fix R.
    # However, finding the exact rank is an NP-hard problem in general.
    # But for small tensors, we can iterate R from lower bound upwards.
    
    # Lower bound:
    # R >= max(d1, d2, d3) / (log d1 + log d2 + log d3) ??? No.
    # R >= max(d1, d2, d3) is a loose bound.
    # R >= (d1*d2*d3) / (max(d1,d2,d3)^2) ?
    
    # Let's start with R=20? Or maybe lower.
    # The product of two rectangular matrices of size n x m and m x p can be done in less than n*m*p operations.
    # Specifically, if we view it as tensor contraction.
    
    # Strategy:
    # 1. Start with R = 15 (max dimension) ? No, usually higher.
    # Let's try to minimize R by solving a system of linear equations.
    # Let U, V, W be matrices of size D1 x R, D2 x R, D3 x R.
    # We require:
    # \sum_{r=1}^R U_{a,r} V_{b,r} W_{c,r} = T_{a,b,c} for all a,b,c.
    
    # This is equivalent to:
    # For all c, \sum_{r=1}^R (U^c_r V^c_r) ... no.
    # For all c, the tensor slice T_{:, :, c} is the outer product of the r-th columns of U and V weighted by W_{c,r}.
    # T_{a,b,c} = \sum_r U_{a,r} V_{b,r} W_{c,r}
    
    # This looks like a 3-way decomposition.
    # We can reformulate this as finding a rank-R decomposition.
    
    # Algorithm:
    # For a given R:
    # 1. Generate random U (D1 x R), V (D2 x R).
    # 2. For each c (0..D3-1), we have a linear system for W_c (R x 1):
    #    T_{:, :, c} (flattened to D1*D2) = \sum_r W_{c,r} (U[:,r] \otimes V[:,r])
    #    Let M_c be the D1*D2 x R matrix where the r-th column is U[:,r] \otimes V[:,r] (Kronecker product).
    #    Let t_c be the flattened T_{:, :, c} vector.
    #    We need W_c such that M_c W_c = t_c.
    #    If M_c is square (R = D1*D2), we can solve.
    #    If R < D1*D2, we check if t_c is in the column space of M_c.
    
    # However, U and V are random. This is the "Hyperdeterminant" approach or similar?
    # Actually, this is just verifying if a random initialization leads to a valid W.
    # This only works if R is large enough to span the space?
    # No, if R is the rank, then there *exist* U, V, W.
    # Random U, V might not be in the correct subspace.
    
    # Better Approach for small dimensions:
    # Use SVD / Tensor Decomposition?
    # The tensor is sparse.
    
    # Let's try iterative reduction from R=60 downwards.
    # Or start from a known lower bound.
    
    # Let's implement a checker: `is_valid(U, V, W)`
    # And a solver for a fixed R?
    # Solving for R exactly is hard.
    
    # Alternative: Greedy Construction?
    # We know the tensor has 60 non-zeros.
    
    # Let's look at the structure.
    # T_{i*m+j, j*p+k, k*n+i} = 1.
    
    # We can try to find a decomposition using the "Strassen-like" approach?
    # For 3x4x5, the tensor rank is likely close to 60 or slightly less.
    # Actually, 3x4 matrix multiplication can be done with fewer than 12 operations?
    # Strassen for 2x2 is 7.
    # For 3x3, it's 23 (Strassen-like) vs 27.
    
    # Let's estimate the rank for 3x4x5.
    # If we just scale the 3x3 case...
    
    # Given the time limit, maybe we can just return the naive decomposition?
    # The prompt says "score is the rank". Lower is better.
    # The seed program returns rank 60.
    # If I can find a valid decomposition with rank 59, I score better.
    # If I find 50, even better.
    
    # How to find a lower rank?
    # We can try to eliminate dependencies.
    
    # Let's try a simple optimization:
    # Initialize U, V with identity/one-hot basis (naive solution).
    # Then try to "merge" basis vectors using linear combinations?
    # This is essentially local search over the decomposition space.
    
    # Representation:
    # U: 12 x R
    # V: 20 x R
    # W: 15 x R
    
    # If we fix R=60, U and V are permutation matrices (subset).
    # We can start with the naive decomposition and try to perturb U and V to allow W to have zeros, effectively reducing rank?
    # No, if R is fixed, the rank is R. We want to *reduce* R.
    
    # To reduce R, we need to show that the 60th tensor slice is a linear combination of the others?
    # Or that we can express the whole tensor with fewer terms.
    
    # Let's try to solve for R from 40 to 60?
    # Actually, let's try to find the minimal R by checking if the tensor lies in the secant variety of order R.
    
    # For small R, we can use `numpy.linalg.lstsq` on the Kronecker expanded matrix if we fix two matrices?
    # If we fix U and V randomly, the probability of finding a valid W for a rank R tensor (if R is the true rank) is zero because U and V must be specifically related.
    
    # Correct Approach for small tensor:
    # The tensor can be viewed as a matrix of size (D1 * D2) x D3 or similar.
    # But it's a 3-way tensor.
    
    # Let's use a heuristic:
    # 1. Start with Naive (R=60).
    # 2. Try to remove terms one by one.
    #    - Assume current decomposition has rank R.
    #    - Try to remove term r.
    #    - This requires that term r is a linear combination of the other R-1 terms in the tensor product space.
    #    - We can check this by trying to re-solve for W given the other U, V?
    #    - This is complex.
    
    # Simpler:
    # Randomized search for R.
    # Pick R.
    # Generate random U, V.
    # Solve for W using `lstsq`.
    # Check if reconstruction error is 0.
    # If yes, return.
    # If no, retry with different random seeds or higher R.
    
    # Why might random U, V work?
    # If R is large enough to "span" the degrees of freedom?
    # No, the decomposition is non-linear in U, V.
    # However, if R is very large (e.g., D1*D2), U and V can be almost anything, and W adjusts.
    # But we want MINIMAL R.
    
    # Let's try a gradient descent approach?
    # Minimize sum_{i,j,k} ( \sum_r U_{i,r} V_{j,r} W_{k,r} - T_{i,j,k} )^2
    # Variables: U, V, W.
    # This is non-convex.
    
    # Given 150s, I can afford many random restarts of a solver.
    
    # What solver?
    # `scipy.optimize.minimize` might be too slow.
    # `numpy.linalg` tricks?
    
    # Let's look at the specific dimensions: 12, 20, 15.
    # The tensor is "linear" in each mode if the others are fixed.
    
    # Algorithm:
    # 1. Set R.
    # 2. Initialize U, V randomly (Gaussian).
    # 3. Iterate:
    #    a. For each r, update W_r? No.
    #    b. Fix U, V. Solve for W (linear least squares per mode? No, W is fully determined if U,V fixed? No).
    #       The system is:
    #       T = \sum_r u_r \otimes v_r \otimes w_r
    #       This is not linear in all variables simultaneously.
    #       
    #    However, if we fix U and V, the problem becomes linear in W?
    #       T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}
    #       For a fixed i,j: T_{i,j,k} = \sum_r (U_{i,r} V_{j,r}) W_{k,r}
    #       Let x_{r}^{(i,j)} = U_{i,r} V_{j,r}.
    #       Then T_{i,j,:} = X_{i,j} W^T ?
    #       Where X_{i,j} is a vector of length R.
    #       And T_{i,j,:} is a vector of length D3 (15).
    #       We have D1*D2 = 12*20 = 240 such equations.
    #       Unknowns: W (15 x R = 15*R values).
    #       If 240 > 15*R, we are over-determined.
    #       We can solve for W using least squares?
    #       If the tensor is exactly rank R, there exists a W that satisfies all equations.
    #       So `lstsq` should give a solution with residual 0 if R is sufficient and U,V are "generic" enough to not constrain the subspace incorrectly?
    #       Wait. If I pick random U and V, the columns of the operator (U_i V_j) might span a subspace that doesn't contain the target tensors T_{i,j,:}.
    #       In fact, for a fixed tensor of rank R, the set of valid (U, V) pairs is a specific variety. Random (U, V) will almost surely fail to support a valid W.
    
    # So random init + linear solve in W will NOT work for minimal R.
    
    # I need an optimization method that updates U, V, and W jointly.
    # Gradient Descent / Adam?
    # `torch` is not available. `scipy` is.
    
    # `scipy.optimize.minimize` with 'BFGS' or 'L-BFGS-B' or 'Powell'.
    # Dimensions:
    # U: 12*R
    # V: 20*R
    # W: 15*R
    # Total params: 47*R.
    # If R=60, params = 2820.
    # This is doable for `L-BFGS-B`?
    # But finding the GLOBAL minimum is the issue.
    
    # Strategy:
    # 1. Try R from a low bound (e.g. 30) up to 60.
    # 2. For each R, do a few random restarts of gradient descent.
    # 3. If we find a solution with error < 1e-9, return it.
    
    # Since the tensor entries are integers (0 or 1), we can use exact arithmetic if we get close?
    # Or just check residual.
    
    # Let's implement a gradient descent step manually or use `scipy`.
    # Using `scipy.optimize.minimize` on a vector of all parameters.
    
    import numpy as np
    from scipy.optimize import minimize
    
    def pack_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
    def unpack_params(p, R):
        size_U = 12 * R
        size_V = 20 * R
        size_W = 15 * R
        U = p[:size_U].reshape(12, R)
        V = p[size_U:size_U+size_V].reshape(20, R)
        W = p[size_U+size_V:].reshape(15, R)
        return U, V, W
        
    def compute_residual(p, R):
        U, V, W = unpack_params(p, R)
        # Reconstruct tensor T_rec
        # T_rec[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # Efficient computation:
        # Temp[i, j, r] = U[i, r] * V[j, r]  -> Shape (12, 20, R)
        # T_rec[i, j, k] = sum_r Temp[i, j, r] * W[k, r]
        # 
        # Use einsum:
        # Temp = np.einsum('ir,jr->ijr', U, V)  # (12, 20, R)
        # T_rec = np.einsum('ijr,kr->ijk', Temp, W) # (12, 20, 15)
        
        U, V, W = unpack_params(p, R)
        # We only need to check the non-zero entries of the tensor?
        # The tensor T is zero everywhere except the 60 specific indices.
        # So we need T_rec[i, j, k] = 1 for those 60, and 0 for others.
        # Checking all 12*20*15 = 3600 entries is fast.
        
        Temp = U[:, np.newaxis, :] * V[np.newaxis, :, :] # (12, 20, R)
        T_rec = np.tensordot(Temp, W, axes=([2], [1])) # (12, 20, 15)
        
        # Compute target T
        # We can pre-compute T_target as a dense array.
        T_target = np.zeros((12, 20, 15))
        for t in tensors:
            T_target[t[0], t[1], t[2]] = 1.0
            
        diff = T_rec - T_target
        # Return sum of squares
        return np.sum(diff * diff)

    # Optimization Loop
    # Try R from low to high.
    # What is the minimum R?
    # Let's guess R_min is around 50-60.
    # If we don't find it quickly, we must return the naive one.
    
    best_result = None
    best_rank = 60 # Naive
    
    # Precompute T_target
    T_target = np.zeros((12, 20, 15))
    for t in tensors:
        T_target[t[0], t[1], t[2]] = 1.0

    # Try to find a solution with R < 60
    # Iterate R backwards? No, R must be at least the rank.
    # Let's try a few values: 59, 58, ...
    # If we use random restarts, we might miss it.
    
    # Actually, there is a known result for matrix multiplication ranks.
    # 3x3x3 (cubing) has rank 7 (Strassen) vs 27.
    # 2x2x2 has rank 7? No, 2x2 matmul is rank 7 (Strassen) vs 8.
    # Wait, 2x2 matmul: A(2x2)*B(2x2). n=m=p=2.
    # Tensors indices: I1 (2*2=4), I2 (2*2=4), Out (2*2=4).
    # Naive rank 8. Strassen rank 7.
    #
    # For 3x4x5:
    # Naive: 60.
    # Can we use Strassen?
    # Strassen decomposes 2x2 into 7.
    # If we can map 3x4x5 into a block of 2x2s?
    # 3x4 = (2+1)x(2+2).
    # This is complicated.
    
    # Let's just run the optimizer.
    
    # Set up target array for speed
    T_flat_target = T_target.flatten()
    
    def loss_func(p, R):
        U, V, W = unpack_params(p, R)
        Temp = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        T_rec = np.tensordot(Temp, W, axes=([2], [1]))
        diff = T_rec.flatten() - T_flat_target
        return np.dot(diff, diff)

    def grad_func(p, R):
        # Analytic gradient?
        # L = ||T_rec - T_tgt||^2
        # dL/dU[i,r] = 2 * sum_{j,k} (T_rec[i,j,k] - T_tgt[i,j,k]) * V[j,r] * W[k,r]
        # dL/dV[j,r] = 2 * sum_{i,k} (T_rec[i,j,k] - T_tgt[i,j,k]) * U[i,r] * W[k,r]
        # dL/dW[k,r] = 2 * sum_{i,j} (T_rec[i,j,k] - T_tgt[i,j,k]) * U[i,r] * V[j,r]
        
        U, V, W = unpack_params(p, R)
        Temp = U[:, np.newaxis, :] * V[np.newaxis, :, :] # (12, 20, R)
        T_rec = np.tensordot(Temp, W, axes=([2], [1])) # (12, 20, 15)
        Diff = T_rec - T_target # (12, 20, 15)
        
        # dU
        # dU[i,r] = 2 * \sum_{j,k} Diff[i,j,k] * V[j,r] * W[k,r]
        # Let M1[j,r,k] = V[j,r] * W[k,r]
        # This is efficient enough?
        # 12*20*15*R operations.
        
        # We can use einsum.
        # dU[i,r] = 2 * einsum('ijk, jr, kr -> ir', Diff, V, W)
        # dV[j,r] = 2 * einsum('ijk, ir, kr -> jr', Diff, U, W)
        # dW[k,r] = 2 * einsum('ijk, ir, jr -> kr', Diff, U, V)
        
        dU = 2.0 * np.einsum('ijk,jr,kr->ir', Diff, V, W)
        dV = 2.0 * np.einsum('ijk,ir,kr->jr', Diff, U, W)
        dW = 2.0 * np.einsum('ijk,ir,jr->kr', Diff, U, V)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    # Search
    # Try ranks 60 down to 40?
    # Or start with 60 and see if we can reduce?
    # If we start with R=60, and initialize with naive solution, the loss is 0.
    # Gradient is 0. It won't move.
    # We need to initialize RANDOMLY to escape local minima?
    # If we initialize randomly for R=60, can we find a zero loss?
    # Probably yes, since rank 60 is sufficient.
    # Once we have a valid decomposition of rank 60, can we perturb it to reduce rank?
    # Reducing rank is not a smooth process.
    
    # Better: Random restarts for R < 60.
    # Try R = 59, 58, ...
    
    # Time budget: 145s.
    # Let's try R from 55 to 60.
    # If we find one, return it.
    
    # Actually, maybe the rank is 60?
    # If the tensor is "generic" enough, the rank might be full?
    # No, matrix multiplication tensors are special.
    # But 3x4x5 might have rank 60?
    # Let's check 2x2x2. Rank 7 < 8.
    # 3x3x3. Rank 23 < 27.
    # So it's likely < 60.
    
    # Let's try R=59.
    # If it fails, R=58...
    
    # Optimization params
    max_iter = 500
    tol = 1e-12
    
    # Try a few R values
    for R in range(59, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this R
        for attempt in range(5): # Limit restarts per R
            if time.time() > deadline:
                break
                
            # Random init
            np.random.seed(42 + attempt + R)
            scale = 1.0
            U_init = np.random.randn(12, R) * scale
            V_init = np.random.randn(20, R) * scale
            W_init = np.random.randn(15, R) * scale
            x0 = pack_params(U_init, V_init, W_init)
            
            res = minimize(
                loss_func, x0, 
                method='BFGS', 
                jac=grad_func, 
                options={'maxiter': max_iter, 'disp': False, 'gtol': tol, 'maxls': 40},
                args=(R,)
            )
            
            final_loss = loss_func(res.x, R)
            
            # Check for exactness
            # Allow small numerical error, but snap to integers?
            # If loss < 1e-9, it's likely exact.
            if final_loss < 1e-9:
                # Convert to integers/half-integers if possible?
                # The task allows integer/half-integer entries.
                # If we find a float solution, we can try to round?
                # Or just return floats? "integer/half-integer entries" is specified in the interface description.
                # "entries that exactly decompose...".
                # If my solution has 0.999999, I should probably return 1.0.
                
                U, V, W = unpack_params(res.x, R)
                
                # Snapping to 0.5 grid
                # U = np.round(U * 2) / 2
                # V = np.round(V * 2) / 2
                # W = np.round(W * 2) / 2
                
                # Verify snapped version
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                Temp_snap = U_snap[:, np.newaxis, :] * V_snap[np.newaxis, :, :]
                T_rec_snap = np.tensordot(Temp_snap, W_snap, axes=([2], [1]))
                err_snap = np.sum((T_rec_snap - T_target)**2)
                
                if err_snap < 1e-9:
                    # Success
                    best_rank = R
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    break
                else:
                    # Use floats if snapping fails but loss was low?
                    # Task says "integer/half-integer". I should stick to that.
                    # If snapping fails, this R/attempt might not yield a valid half-integer solution.
                    pass
                    
        if best_result is not None:
            break

    if best_result is None:
        # Return Naive
        best_rank = 60
        U = [[0] * 60 for _ in range(12)]
        V = [[0] * 60 for _ in range(20)]
        W = [[0] * 60 for _ in range(15)]
        col = 0
        for i in range(3):
            for k in range(5):
                for j in range(4):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        best_result = {
            "U": U,
            "V": V,
            "W": W
        }
        
    return best_result

# Ensure the function is available
# (No need for global variables in the snippet, but keeping them is fine)
