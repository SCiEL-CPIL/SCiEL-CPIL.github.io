# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        # Convert to list of lists as per interface if required? 
        # The prompt says "factor matrices of shapes... with integer/half-integer entries".
        # It doesn't strictly say list of lists, but the seed used lists. 
        # However, numpy arrays are usually accepted if they have the right shape.
        # Let's return numpy arrays for speed, or lists if the evaluator is strict.
        # The seed returned lists. I will return lists to be safe, but numpy is faster to generate.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_exact(U, V, W):
        """Check if the decomposition is exact with integer/half-integer entries."""
        rank = len(U[0])
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check tensor equation
        # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We want T[i, j, k] = 1 for all i, j, k? 
        # No, the tensor for matrix multiplication is 1 if the indices are consistent?
        # The standard definition: 
        # C_{ik} = sum_j A_{ij} B_{jk}
        # Tensor T has T_{i, j, k} = 1 if we are computing C_{ik} from A_{ij} and B_{jk}.
        # So for every i, j, k, the term is 1.
        # Yes, the tensor is the all-ones tensor of size n x m x p.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    # Vectorized dot product
                    u_vec = np.array(U[i*m+j])
                    v_vec = np.array(V[j*p+k])
                    w_vec = np.array(W[k*n+i])
                    total = np.dot(u_vec, v_vec * w_vec)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(r, num_restarts=10, max_iters=100):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline - 10:
            return None
        
        u_idx = np.array([(i, j) for i in range(n) for j in range(m)])
        v_idx = np.array([(j, k) for j in range(m) for k in range(p)])
        w_idx = np.array([(k, i) for k in range(p) for i in range(n)])
        
        # Precompute index maps for faster access in loops if needed, 
        # but vectorization is better.
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize to prevent divergence
            # U /= np.linalg.norm(U, axis=0, keepdims=True)
            # V /= np.linalg.norm(V, axis=0, keepdims=True)
            # W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                # For each row u_idx (i, j), we want to minimize:
                # sum_k (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
                # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # We want to solve M @ U_row = 1 (vector of ones of length p)
                # This is a least squares problem.
                
                # We can vectorize the construction of M for all (i, j)?
                # M depends on i, j.
                # Shape of M is (p, r).
                # There are n*m such matrices.
                
                for idx in range(n*m):
                    i, j = u_idx[idx]
                    # Construct M
                    # V[j*p : (j+1)*p, :] has shape (p, r)
                    # W[:, i] ? No, W index is k*n+i.
                    # W[k*n+i, c] for k in 0..p-1.
                    # This is not a simple slice of W because of the stride n.
                    # W is shape (p*n, r).
                    # Indices: 0*n+0, 1*n+0, ..., (p-1)*n+0 for i=0.
                    # So W[0:n*p:n, :] gives rows for i=0?
                    # Row index for (k, i) is k*n + i.
                    # For fixed i, k varies 0..p-1.
                    # Indices: i, n+i, 2n+i, ... (p-1)n+i.
                    # This is a slice W[i : p*n : n, :]
                    
                    V_slice = V[j*p : (j+1)*p, :] # (p, r)
                    W_slice = W[i : p*n : n, :]   # (p, r)
                    
                    M = V_slice * W_slice # (p, r)
                    b = np.ones(p)
                    
                    try:
                        U[idx, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        pass

                # Update V
                for idx in range(m*p):
                    j, k = v_idx[idx]
                    # Minimize sum_i (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
                    # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # i varies 0..n-1.
                    # U indices: 0*m+j, 1*m+j, ... (n-1)*m+j.
                    # U is shape (n*m, r).
                    # Slice: U[j : n*m : m, :]
                    # W indices: k*n+0, k*n+1, ..., k*n+(n-1).
                    # Slice: W[k*n : k*n+n, :]
                    
                    U_slice = U[j : n*m : m, :] # (n, r)
                    W_slice = W[k*n : k*n+n, :] # (n, r)
                    
                    M = U_slice * W_slice # (n, r)
                    b = np.ones(n)
                    
                    try:
                        V[idx, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        pass

                # Update W
                for idx in range(p*n):
                    k, i = w_idx[idx]
                    # Minimize sum_j (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
                    # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # j varies 0..m-1.
                    # U indices: i*m+0, i*m+1, ..., i*m+(m-1).
                    # Slice: U[i*m : i*m+m, :]
                    # V indices: 0*p+k, 1*p+k, ..., (m-1)*p+k.
                    # Slice: V[k : m*p : p, :]
                    
                    U_slice = U[i*m : i*m+m, :] # (m, r)
                    V_slice = V[k : m*p : p, :] # (m, r)
                    
                    M = U_slice * V_slice # (m, r)
                    b = np.ones(m)
                    
                    try:
                        W[idx, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        pass

                # Check convergence
                # Calculate error
                # We can check a few random points or the max error.
                # Max error is O(n*m*p*r). 2*5*5*50 = 2500 ops. Fast.
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_vec = U[i*m+j, :]
                            v_vec = V[j*p+k, :]
                            w_vec = W[k*n+i, :]
                            val = np.dot(u_vec, v_vec * w_vec)
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    # Found a solution, try to round
                    U_rounded = [[round(val * 2) / 2.0 for val in U[row]] for row in range(n*m)]
                    V_rounded = [[round(val * 2) / 2.0 for val in V[row]] for row in range(m*p)]
                    W_rounded = [[round(val * 2) / 2.0 for val in W[row]] for row in range(p*n)]
                    if verify_exact(U_rounded, V_rounded, W_rounded):
                        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    
                    # If rounding fails, try snapping to integers if close
                    U_int = np.rint(U)
                    V_int = np.rint(V)
                    W_int = np.rint(W)
                    if verify_exact(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

        return None

    best_result = None
    # Try ranks from 49 down to 25
    # If 50 is the minimum, we return 50.
    # If 49 is possible, we find it.
    for r in range(49, 24, -1):
        if time.time() > deadline - 20:
            break
        
        # Increase restarts for lower ranks? Or keep constant?
        # Lower ranks are harder.
        result = try_rank(r, num_restarts=5, max_iters=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
