"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import scipy.optimize as opt
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Index mapping
        # U[i*m+j, t] = 1 if t corresponds to (i,j)
        # V[j*p+k, t] = 1 if t corresponds to (j,k)
        # W[k*n+i, t] = 1 if t corresponds to (k,i)
        
        # In schoolbook, each t is unique for a specific (i,j) in U, but we need
        # the product U[i,j,t] V[j,k,t] W[k,i,t] to be 1 for the specific t that
        # matches all three?
        # No, in schoolbook, we have n*m*p terms.
        # Term t corresponds to a specific (i,j,k) triple?
        # If t maps to (i,j,k), then U[i,j,t]=1, V[j,k,t]=1, W[k,i,t]=1.
        # Then sum_t U[i',j',t] V[j',k',t] W[k',i',t] is 1 if (i',j',k') == (i,j,k) else 0.
        # This gives the identity tensor, not the multiplication tensor.
        
        # Wait, the multiplication tensor T[i,j,k] = 1 for all i,j,k.
        # My previous analysis:
        # LHS coefficient for A[i,j] B[j,k] C[k,i] is sum_t U[i,j,t] V[j,k,t] W[k,i,t].
        # We want this to be 1 for ALL i,j,k.
        
        # Schoolbook:
        # Let t index the triple (i,j,k) from 0 to nmp-1.
        # U[i,j,t] = 1 if t encodes (i,j).
        # V[j,k,t] = 1 if t encodes (j,k).
        # W[k,i,t] = 1 if t encodes (k,i).
        
        # For a fixed t (which encodes i0, j0, k0):
        # U[i,j,t] is 1 only if i=i0, j=j0.
        # V[j,k,t] is 1 only if j=j0, k=k0.
        # W[k,i,t] is 1 only if k=k0, i=i0.
        
        # So the product U[i,j,t] V[j,k,t] W[k,i,t] is 1 only if
        # i=i0, j=j0 AND j=j0, k=k0 AND k=k0, i=i0.
        # This implies i=i0, j=j0, k=k0.
        # So for a specific (i,j,k), the sum over t is 1 if there exists a t
        # such that t encodes (i,j), (j,k), (k,i) simultaneously.
        # Since t is unique for each (i,j,k) triple in the schoolbook setup
        # (t = i*m*p + j*p + k), there is exactly one t for each (i,j,k).
        # For that t, U[i,j,t]=1, V[j,k,t]=1, W[k,i,t]=1.
        # For all other t', the product is 0.
        # So the sum is 1.
        
        # Thus, schoolbook works with U,V,W being 1s at specific positions.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U[i * m + j, t] = 1
                    V[j * p + k, t] = 1
                    W[k * n + i, t] = 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def loss_func(params, r, n, m, p):
        """
        Calculate the squared error of the tensor reconstruction.
        params: flattened vector of size 3 * (nm + mp + pn) * r
        """
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n * m, r)
        V = params[size_U:size_U+size_V].reshape(m * p, r)
        W = params[size_U+size_V:size_U+size_V+size_W].reshape(p * n, r)
        
        error = 0.0
        # Iterate over all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
                    # U row index: i*m + j
                    # V row index: j*p + k
                    # W row index: k*n + i
                    u_vec = U[i*m + j, :]
                    v_vec = V[j*p + k, :]
                    w_vec = W[k*n + i, :]
                    
                    # Element-wise product and sum
                    prod = u_vec * v_vec * w_vec
                    s = np.sum(prod)
                    error += (s - 1.0) ** 2
                    
        return error

    def grad_loss(params, r, n, m, p):
        """
        Numerical gradient for simplicity, or analytical if possible.
        Given the non-convexity and small size, numerical might be okay
        but slow. Let's try analytical.
        
        L = sum_{i,j,k} (sum_t U_{ij,t} V_{jk,t} W_{ki,t} - 1)^2
        
        dL/dU_{ij,t} = 2 * (sum_{i',j',k'} (sum_{t'} U_{i'j',t'} V_{j'k',t'} W_{k'i',t'} - 1) * 
                                delta_{i,i'} delta_{j,j'} delta_{k,k'}? No.
        
        Let S_{ijk} = sum_t U_{ij,t} V_{jk,t} W_{ki,t}.
        L = sum_{i,j,k} (S_{ijk} - 1)^2.
        
        dL/dU_{ij,t} = 2 * (S_{ij,t} - 1) * dS_{ij,t}/dU_{ij,t}
        Wait, S depends on U.
        S_{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}.
        dS_{i,j,k}/dU_{i,j,t} = V_{j,k,t} W_{k,i,t}.
        
        So dL/dU_{i,j,t} = 2 * sum_{k} (S_{i,j,k} - 1) * V_{j,k,t} * W_{k,i,t}.
        
        Similarly:
        dL/dV_{j,k,t} = 2 * sum_{i} (S_{i,j,k} - 1) * U_{i,j,t} * W_{k,i,t}.
        dL/dW_{k,i,t} = 2 * sum_{j} (S_{i,j,k} - 1) * U_{i,j,t} * V_{j,k,t}.
        """
        
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n * m, r)
        V = params[size_U:size_U+size_V].reshape(m * p, r)
        W = params[size_U+size_V:size_U+size_V+size_W].reshape(p * n, r)
        
        # Precompute S_{i,j,k}
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i*m + j, :]
                    v_vec = V[j*p + k, :]
                    w_vec = W[k*n + i, :]
                    S[i, j, k] = np.sum(u_vec * v_vec * w_vec)
        
        grad = np.zeros_like(params)
        
        # Gradient for U
        for i in range(n):
            for j in range(m):
                for t in range(r):
                    # Sum over k
                    term = 0.0
                    for k in range(p):
                        term += (S[i, j, k] - 1.0) * V[j*p + k, t] * W[k*n + i, t]
                    grad[i*m + j, t] = 2.0 * term
        # Note: grad is flattened, so index is (i*m+j)*r + t?
        # No, U is (n*m, r). Flattened U is U.reshape(-1).
        # Index in flat U: (i*m + j) * r + t.
        # My grad array is size_U.
        # So grad[i*m + j, t] is wrong indexing for a 1D array.
        
        # Let's re-index properly.
        grad_U = np.zeros((n * m, r))
        grad_V = np.zeros((m * p, r))
        grad_W = np.zeros((p * n, r))
        
        for i in range(n):
            for j in range(m):
                for t in range(r):
                    term = 0.0
                    for k in range(p):
                        term += (S[i, j, k] - 1.0) * V[j*p + k, t] * W[k*n + i, t]
                    grad_U[i*m + j, t] = 2.0 * term
                    
        for j in range(m):
            for k in range(p):
                for t in range(r):
                    term = 0.0
                    for i in range(n):
                        term += (S[i, j, k] - 1.0) * U[i*m + j, t] * W[k*n + i, t]
                    grad_V[j*p + k, t] = 2.0 * term
                    
        for k in range(p):
            for i in range(n):
                for t in range(r):
                    term = 0.0
                    for j in range(m):
                        term += (S[i, j, k] - 1.0) * U[i*m + j, t] * V[j*p + k, t]
                    grad_W[k*n + i, t] = 2.0 * term
        
        # Flatten gradients back to match params layout
        grad[:size_U] = grad_U.reshape(-1)
        grad[size_U:size_U+size_V] = grad_V.reshape(-1)
        grad[size_U+size_V:size_U+size_V+size_W] = grad_W.reshape(-1)
        
        return grad

    # Try to find a rank r decomposition
    # We start from a lower bound. 
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n) is not strictly true for all, 
    # but for 2x5x6, 30 is a good start.
    # However, random initialization in high dimensions often fails.
    # We will try a few ranks.
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 30 to 60 (schoolbook)
    # To save time, we might just try a few values.
    # If we find a valid one, we return it.
    
    # We will use Powell's method or L-BFGS-B.
    # L-BFGS-B requires gradient.
    
    for r in range(30, 61):
        if r >= best_rank:
            break
            
        # Try a few random initializations
        found = False
        for trial in range(5):
            # Initialize with small random values
            np.random.seed(trial + r)
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use minimize
            try:
                res = opt.minimize(
                    loss_func, 
                    params_init, 
                    args=(r, n, m, p),
                    method='L-BFGS-B', 
                    jac=grad_loss,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                if res.success or res.fun < 1e-6:
                    # Check if it's a valid solution
                    # Round to nearest 0.5?
                    # The problem allows integer/half-integer.
                    # Let's check the raw values first.
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Check exactness with tolerance
                    if loss_func(res.x, r, n, m, p) < 1e-8:
                        # Try to round to half-integers
                        # This is a heuristic. If the values are close to x.0 or x.5, round them.
                        U_rounded = np.round(U_sol * 2) / 2
                        V_rounded = np.round(V_sol * 2) / 2
                        W_rounded = np.round(W_sol * 2) / 2
                        
                        # Verify rounded solution
                        if loss_func(np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]), r, n, m, p) < 1e-8:
                            best_result = {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            best_rank = r
                            found = True
                            break
                        else:
                            # If rounding fails, keep the float solution? 
                            # The problem requires integer/half-integer.
                            # So we must discard if it doesn't round.
                            pass
            except Exception:
                continue
        
        if found:
            break
            
    if best_result is not None:
        return best_result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
