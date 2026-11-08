"""Optimized matrix multiplication tensor decomposition for 4x5 * 5x5."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] is the coefficient of (A[i,j] * B[j,k]) in the product.
    
    # We need to find rank-r decomposition where:
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Try to find a low-rank decomposition using random initialization
    # and gradient-free optimization (e.g., alternating minimization or greedy)
    
    # First, let's establish an upper bound with a known construction.
    # The trivial bound is n*m*p = 4*5*5 = 100.
    # We can do better by exploiting structure.
    
    # Let's try a systematic search for low-rank decompositions.
    # Start with a reasonable guess and try to refine.
    
    deadline = time.time() + 140  # Leave 10s margin before the 150s kill
    
    # Try various ranks, starting from a lower bound.
    # Theoretical lower bound for matrix multiplication: max(n, m, p) = 5
    # But actual bounds are higher. For 4x5 * 5x5, let's try ranks from 10 up.
    
    best_result = None
    best_rank = None
    
    # First, provide the schoolbook solution as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m*p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p*n, schoolbook_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    def to_list(arr):
        """Convert numpy array to list of lists with exact fractions if possible."""
        # Check if all values are close to integers or half-integers
        result = np.round(arr * 2) / 2  # Round to nearest half-integer
        # Verify this is exact
        if np.allclose(arr, result, atol=1e-10):
            # Convert to list of lists
            return [[float(x) for x in row] for row in result]
        else:
            return [[float(x) for x in row] for row in arr]
    
    # Try to find lower rank decompositions
    # Use a simple approach: random initialization + alternating projections
    
    for rank in range(10, min(60, schoolbook_rank + 1)):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization: fix two, optimize the third
            converged = False
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor from factors
                # S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Optimize W: for each k, i, we want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is a linear system for W given U, V
                
                # Let A_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r]
                # We want sum_r A_r[i,j,k] * W[k*n+i, r] = 1 for all i,j,k
                
                # For fixed (k, i), we have: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let B_{(k,i), (j,r)} = U[i*m+j, r] * V[j*p+k, r]
                # Then sum_{j,r} B_{(k,i), (j,r)} * W[k*n+i, r] = 1
                # But W is indexed by (k*n+i, r), so for each (k,i), we have one equation
                # with unknowns W[k*n+i, r] for r = 0..rank-1, but the sum is over j too.
                
                # Actually, let's think differently. The tensor T is the outer product structure.
                # Let's use a different approach: view it as a matrix completion problem.
                
                # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                
                # Compute S[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                # This is like a tensor contraction.
                
                # For fixing W and optimizing U,V, or fixing U,V and optimizing W:
                # If we fix U and V, then for each (k, i):
                # sum_{j,r} U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1
                # Let C_{(k,i), r} = sum_j U3[i,j,r] * V3[j,k,r]
                # Then sum_r C_{(k,i), r} * W3[k,i,r] = 1
                # This is: for each (k,i), a linear equation in W3[k,i,:]
                # But we have p*n = 20 equations, each with 'rank' unknowns.
                # If rank > 1, this is underdetermined. We can pick the minimum norm solution.
                
                # Let's solve for W given U, V:
                # For each k in range(p), i in range(n):
                # sum_r (sum_j U3[i,j,r] * V3[j,k,r]) * W3[k,i,r] = 1
                # Let D[k,i,r] = sum_j U3[i,j,r] * V3[j,k,r]
                # Then sum_r D[k,i,r] * W3[k,i,r] = 1 for all k,i
                
                D = np.zeros((p, n, rank))
                for k in range(p):
                    for i in range(n):
                        for r in range(rank):
                            D[k, i, r] = np.sum(U3[i, :, r] * V3[:, k, r])
                
                # Now solve: for each (k,i), find W3[k,i,:] such that dot(D[k,i,:], W3[k,i,:]) = 1
                # Minimum norm solution: W3[k,i,:] = D[k,i,:] / (D[k,i,:] @ D[k,i,:]) if norm > 0
                norm_sq = np.sum(D * D, axis=2)  # (p, n)
                norm_sq_safe = np.where(norm_sq > 1e-10, norm_sq, 1.0)
                W_new3 = D / norm_sq_safe[:, :, np.newaxis]
                
                # Check if this gives exactly 1
                residuals = np.sum(D * W_new3, axis=2)  # Should be 1 where norm_sq > 0
                if np.all(residuals[norm_sq > 1e-10] > 0.99):
                    W = W_new3.reshape(p*n, rank)
                
                # Similarly optimize U given V, W:
                # For each (i, j): sum_{k,r} V3[j,k,r] * W3[k,i,r] * U3[i,j,r] = 1
                # Let E[i,j,r] = sum_k V3[j,k,r] * W3[k,i,r]
                # Then sum_r E[i,j,r] * U3[i,j,r] = 1
                E = np.zeros((n, m, rank))
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            E[i, j, r] = np.sum(V3[j, :, r] * W3[:, i, r])
                norm_sq_E = np.sum(E * E, axis=2)
                norm_sq_E_safe = np.where(norm_sq_E > 1e-10, norm_sq_E, 1.0)
                U_new3 = E / norm_sq_E_safe[:, :, np.newaxis]
                U = U_new3.reshape(n*m, rank)
                
                # Optimize V given U, W:
                # For each (j, k): sum_{i,r} U3[i,j,r] * W3[k,i,r] * V3[j,k,r] = 1
                # Let F[j,k,r] = sum_i U3[i,j,r] * W3[k,i,r]
                F = np.zeros((m, p, rank))
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            F[j, k, r] = np.sum(U3[:, j, r] * W3[k, :, r])
                norm_sq_F = np.sum(F * F, axis=2)
                norm_sq_F_safe = np.where(norm_sq_F > 1e-10, norm_sq_F, 1.0)
                V_new3 = F / norm_sq_F_safe[:, :, np.newaxis]
                V = V_new3.reshape(m*p, rank)
                
                # Check convergence: compute the actual tensor and see if it's close to 1
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                S = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S[i, j, k] = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                
                if np.allclose(S, 1.0, atol=1e-6):
                    converged = True
                    break
            
            if converged:
                # Verify with integer/half-integer rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded version is exact
                U3 = U_rounded.reshape(n, m, rank)
                V3 = V_rounded.reshape(m, p, rank)
                W3 = W_rounded.reshape(p, n, rank)
                S_check = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S_check[i, j, k] = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                
                if np.allclose(S_check, 1.0, atol=1e-10):
                    best_rank = rank
                    best_result = {
                        "U": [[float(x) for x in row] for row in U_rounded],
                        "V": [[float(x) for x in row] for row in V_rounded],
                        "W": [[float(x) for x in row] for row in W_rounded]
                    }
                    break
            elif time.time() > deadline - 5:
                break
        
        if best_rank is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        return {
            "U": [[float(x) for x in row] for row in U_sb],
            "V": [[float(x) for x in row] for row in V_sb],
            "W": [[float(x) for x in row] for row in W_sb]
        }
    
    return best_result

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank found: {rank}")
