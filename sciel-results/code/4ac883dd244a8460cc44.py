import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x12 matrix multiplication.
    
    We use a structured approach:
    1. Start with the known rank-24 schoolbook-like decomposition
    2. Try to reduce rank by exploiting structure
    3. Use numerical optimization to find minimal rank
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,11}
    Decomposition: T[i,j,k] = sum_r U[i*2+j, r] * V[j*12+k, r] * W[k*2+i, r]
    """
    n, m, p = 2, 2, 12
    
    # Start with rank-24 construction (from the seed)
    def construct_rank24():
        rank = 2 * p  # 24
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for k in range(p):
            r1 = 2 * k
            r2 = 2 * k + 1
            
            # Term r1: handles j=0
            U[0 * m + 0][r1] = 1
            U[1 * m + 0][r1] = 1
            V[0 * p + k][r1] = 1
            W[k * n + 0][r1] = 1
            W[k * n + 1][r1] = 1
            
            # Term r2: handles j=1
            U[0 * m + 1][r2] = 1
            U[1 * m + 1][r2] = 1
            V[1 * p + k][r2] = 1
            W[k * n + 0][r2] = 1
            W[k * n + 1][r2] = 1
        
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        # Check that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False, (i, j, k, val)
        return True, None
    
    # Check if decomposition is valid
    U24, V24, W24 = construct_rank24()
    valid, info = verify(U24, V24, W24)
    if valid:
        print("Rank-24 decomposition is valid")
    
    # Try to find a lower rank decomposition
    # The rank of <2,2,p> is known to be 2p for p >= 2 in the standard model
    # But let's try to verify if we can do better
    
    # Strategy: Try rank 23, 22, etc. using random initialization + optimization
    def try_rank(rank, max_iter=10000, timeout=30):
        """Try to find a rank-r decomposition"""
        start_time = time.time()
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Scale to help convergence
        U *= 2.0
        V *= 2.0
        W *= 2.0
        
        def objective(params):
            if time.time() - start_time > timeout:
                return 1e10
            
            # Unpack
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        error += (val - 1.0) ** 2
            
            return error
        
        def gradient(params):
            if time.time() - start_time > timeout:
                return np.zeros_like(params)
            
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            dU = np.zeros_like(U_flat)
            dV = np.zeros_like(V_flat)
            dW = np.zeros_like(W_flat)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        err = 2.0 * (val - 1.0)
                        
                        for r in range(rank):
                            dU[i*m+j, r] += err * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                            dV[j*p+k, r] += err * U_flat[i*m+j, r] * W_flat[k*n+i, r]
                            dW[k*n+i, r] += err * U_flat[i*m+j, r] * V_flat[j*p+k, r]
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        # Try multiple random restarts
        best_error = 1e10
        best_params = None
        
        for restart in range(5):
            if time.time() - start_time > timeout:
                break
            
            U = np.random.randn(n * m, rank) * 0.1 * 2.0
            V = np.random.randn(m * p, rank) * 0.1 * 2.0
            W = np.random.randn(p * n, rank) * 0.1 * 2.0
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or BFGS
            try:
                result = minimize(objective, params, method='BFGS', jac=gradient,
                                  options={'maxiter': max_iter, 'gtol': 1e-6})
                if result.fun < best_error:
                    best_error = result.fun
                    best_params = result.x
            except Exception as e:
                pass
        
        if best_params is not None and best_error < 1e-6:
            U_opt = best_params[:n*m*rank].reshape(n*m, rank)
            V_opt = best_params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = best_params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return U_opt, V_opt, W_opt, best_error
        return None, None, None, best_error
    
    # Try to find lower rank
    best_U, best_V, best_W = None, None, None
    
    for rank in range(23, 15, -1):
        print(f"Trying rank {rank}...")
        U_opt, V_opt, W_opt, error = try_rank(rank, max_iter=5000, timeout=20)
        if U_opt is not None:
            valid, info = verify(U_opt, V_opt, W_opt, tol=1e-6)
            if valid:
                print(f"Found valid rank-{rank} decomposition!")
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                break
        else:
            print(f"  Best error: {error:.6e}")
    
    # If no lower rank found, use rank-24
    if best_U is None:
        U, V, W = construct_rank24()
        # Convert to lists as required
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Round to nice fractions if possible
        def round_matrix(M):
            M_rounded = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    # Try to round to nearest 0.5
                    rounded = round(val * 2) / 2.0
                    if abs(val - rounded) < 1e-4:
                        M_rounded[i, j] = rounded
                    else:
                        M_rounded[i, j] = val
            return M_rounded
        
        U_r = round_matrix(best_U)
        V_r = round_matrix(best_V)
        W_r = round_matrix(best_W)
        
        # Verify rounded version
        valid, info = verify(U_r, V_r, W_r, tol=1e-4)
        if valid:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
