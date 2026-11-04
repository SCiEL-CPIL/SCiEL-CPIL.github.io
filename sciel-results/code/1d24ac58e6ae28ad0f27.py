"""Matrix multiplication tensor decomposition for 4x4 times 4x7."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # T[i,j,k] = e_{i*m+j} in U, e_{j*p+k} in V, e_{k*n+i} in W
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid, else 0
    
    # The tensor is: for each (i,j,k) triple, T[i,j,k] = 1
    # T[i,j,k] = delta_{row_i, j} * delta_{col_j, k} ... no, it's just the multiplication tensor
    # Actually T[i,j,k] = 1 for all valid (i,j,k) in the standard matmul tensor
    # But we need: prod_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for each (i,j,k)
    # and = 0 for "invalid" entries... but actually the tensor only has entries for valid (i,j,k)
    
    # The tensor T has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the standard matrix multiplication tensor where each entry is 1)
    # Wait no - the tensor for matrix mult C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
    # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Wait, that's not right either. Let me re-read the convention.
    
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor decomposition: C[i,k] = sum_c U_c[i] V_c[k] W_c
    # But here it's: sum_c U[i*m+j, c] * A[i,j] summed over j, times V[j*p+k, c] * B[j,k] summed over j
    # That doesn't work because j appears in both U and V.
    
    # Actually the correct form is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # No wait, that would give sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # = sum_c W[k*n+i,c] * (U_c^T * A[:,i]... no this is getting confused.
    
    # Let me think more carefully. The matrix multiplication tensor T is a 3-tensor with indices (i, j, k)
    # where i indexes rows of A and C, j indexes cols of A and rows of B, k indexes cols of B and C.
    # T[i,j,k] = 1 means the (i,k) entry of C gets contribution from A[i,j]*B[j,k].
    
    # A rank-r decomposition of this tensor means:
    # T[i,j,k] = sum_{c=1}^r U_c[i] * V_c[j] * W_c[k]
    # But wait, the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # So U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    # The condition is: for the multiplication to be correct,
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    # where T[i,j,k] = 1 if this is a valid (i,j,k) triple in the tensor.
    
    # Actually I think the convention is that the tensor T[i,j,k] = 1 for all valid (i,j,k),
    # and the decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    
    # But that can't be right because there are n*m*p = 112 such equations, and we want rank < 112.
    # The equations are: for each (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # This is a system of 112 equations in 3*r*... variables. We can solve this!
    # For a given rank r, we have 16*r + 28*r + 28*r = 72*r unknowns.
    # 112 equations. So r >= 112/72 ≈ 1.56, but we need integer solutions.
    
    # Actually, we can use the fact that this is a structured tensor. Let me try to find
    # a low-rank decomposition by solving the system directly.
    
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is bilinear in (U,V) for fixed W, etc.
    
    # Let me try a different approach: use the known result that the border rank of 4x4x4 matmul is 25 (Strassen)
    # and for 4x4x7, the rank might be related.
    
    # Actually, let me just try to find a decomposition with rank around 28-50 using optimization.
    
    deadline = time.time() + 100  # leave 50s margin for verification
    
    # Build the constraint matrix structure
    # For each (i,j,k), the equation involves U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    
    # Let me try random initialization + gradient descent on the non-convex problem
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    # Generate the schoolbook solution as fallback
    U_fb = np.zeros((n*m, best_rank))
    V_fb = np.zeros((m*p, best_rank))
    W_fb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W, tol=1e-10):
        """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from low to high
    np.random.seed(42)
    
    for target_rank in [17, 18, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 90, 100, 112]:
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            continue
        
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Scale to make initial residual closer to target
            # Each term U*V*W should be ~1/target_rank on average
            scale = (1.0 / target_rank) ** (1/3)
            U *= scale
            V *= scale
            W *= scale
            
            # Gradient descent
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual: for each (i,j,k), r_ijk = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Gradient wrt U[i*m+j,c]: -r_ijk * V[j*p+k,c] * W[k*n+i,c]
                # Gradient wrt V[j*p+k,c]: -r_ijk * U[i*m+j,c] * W[k*n+i,c]
                # Gradient wrt W[k*n+i,c]: -r_ijk * U[i*m+j,c] * V[j*p+k,c]
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = 0.0
                            for c in range(target_rank):
                                res += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            r = 1.0 - res
                            for c in range(target_rank):
                                gU[i*m+j, c] -= r * V[j*p+k, c] * W[k*n+i, c]
                                gV[j*p+k, c] -= r * U[i*m+j, c] * W[k*n+i, c]
                                gW[k*n+i, c] -= r * U[i*m+j, c] * V[j*p+k, c]
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Check convergence
                if it % 50 == 0:
                    max_res = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res = 0.0
                                for c in range(target_rank):
                                    res += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                max_res = max(max_res, abs(1.0 - res))
                    if max_res < 1e-10:
                        break
                
                if time.time() > deadline - 5:
                    break
            
            # Verify
            if verify(U, V, W):
                best_rank = target_rank
                best_solution = (U.copy(), V.copy(), W.copy())
                break
        if best_solution is not None and best_rank == target_rank:
            break
    
    # If no better solution found, use schoolbook
    if best_solution is None:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    U_sol, V_sol, W_sol = best_solution
    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
# EVOLVE-BLOCK-END
