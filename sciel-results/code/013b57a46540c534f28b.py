"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # Safety margin
    
    # Precompute the index mapping for the tensor
    # Tensor entry T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Output index: k*n + i (trace convention W[k*n+i, r])
    # U index: i*m + j
    # V index: j*p + k
    
    # Build the linear system structure
    # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try ranks from a lower bound upward
    # Lower bound: max(nm, mp, pn) = 35, but likely higher
    # Try from 35 up, but start with a reasonable guess
    
    best_result = None
    best_rank = nm * mp  # 105 as fallback
    
    # Generate all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eq = len(triples)  # 105
    
    for r in range(35, min(106, 105 + 1)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for this rank
        found = False
        max_restarts = 50 if r < 60 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly with small values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Alternating minimization
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j,k): U[i*m+j, :] dot (V[j*p+k,:] * W[k*n+i,:]) = 1
                # This is: sum_k [U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:])] = 1
                # For fixed (i,j): sum_k U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:]) = 1
                # This gives p equations for each (i,j), but they're coupled through U[i*m+j,:]
                
                # Actually, let me reformulate. For each (i,j,k):
                # U[i*m+j,:] . (V[j*p+k,:] .* W[k*n+i,:]) = 1
                # where . is dot product and .* is element-wise multiply
                
                # For fixed (i,j), we have p equations:
                # U[i*m+j,:] . (V[j*p+k,:] .* W[k*n+i,:]) = 1 for k=0..p-1
                # This is a system of p equations in r unknowns (U[i*m+j,:])
                
                # Similarly for V and W.
                
                # Let's do a simpler alternating scheme:
                # Update one factor at a time using least squares
                
                # Update U: for each (i,j), solve the system
                # A_{k, c} = V[j*p+k, c] * W[k*n+i, c] for k=0..p-1, c=0..r-1
                # Then A @ U[i*m+j, :] = 1 (vector of ones of length p)
                
                for i in range(n):
                    for j in range(m):
                        # Build A: p x r
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A @ U[i*m+j, :] = ones(p)
                        # Least squares solution
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve
                # For fixed (j,k): sum_i U[i*m+j,:] * W[k*n+i,:] ... 
                # Actually: for (i,j,k): U[i*m+j,:] . (V[j*p+k,:] .* W[k*n+i,:]) = 1
                # For fixed (j,k): sum over i of U[i*m+j,:] . (V[j*p+k,:] .* W[k*n+i,:]) = ... 
                # No, each (i,j,k) is a separate equation.
                
                # Let me reconsider. The equation is:
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # 
                # For fixed (j,k), and varying i:
                # sum_c V[j*p+k,c] * W[k*n+i,c] * U[i*m+j,c] = 1
                # This is: V[j*p+k,:] . (W[k*n+i,:] .* U[i*m+j,:]) = 1 for each i
                # So for fixed (j,k): n equations in r unknowns V[j*p+k,:]
                # A_{i, c} = W[k*n+i, c] * U[i*m+j, c]
                # A @ V[j*p+k, :] = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = W[k * n + i, :] * U[i * m + j, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for fixed (k,i), varying j:
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed (k,i): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for each j
                # A_{j, c} = U[i*m+j, c] * V[j*p+k, c]
                # A @ W[k*n+i, :] = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 9:
                    # Compute residual
                    residual = 0.0
                    for (i, j, k) in triples:
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        residual += (val - 1.0) ** 2
                    if residual < 1e-10:
                        converged = True
                        break
            
            if converged:
                # Verify the solution
                max_err = 0.0
                for (i, j, k) in triples:
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
        if time.time() > deadline - 10:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition
    rank = nm * mp
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

