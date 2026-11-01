import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] where i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i, j, k] = 1 if this term is part of the product?
    # The product C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]^* ?
    # The bilinear map is (A, B) -> C.
    # The tensor T has components T_{ijk} such that C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i, j, k?
    # No, the index j is the summation index.
    # The tensor is sum_{j} (sum_i e_i \otimes e_j) \otimes (sum_k e_j \otimes e_k)?
    # Let's stick to the definition:
    # We want U, V, W such that for all i, k:
    # C[i, k] = sum_j A[i, j] B[j, k] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_k V[j, k, r] B[j, k])? No.
    
    # Standard definition:
    # U is (n*m, R), V is (m*p, R), W is (p*n, R).
    # C[i, k] = sum_r W[k*n + i, r] * (sum_j U[i*m + j, r] * A[i, j]) * (sum_k V[j*p + k, r] * B[j, k])?
    # No, the structure is:
    # Term r contributes: W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k])?
    # This doesn't look right. The indices must match.
    
    # Correct structure:
    # The r-th term is:
    # M_r = (sum_{i,j} U[i*m+j, r] A[i,j]) * (sum_{j,k} V[j*p+k, r] B[j,k])
    # This is a scalar times a scalar.
    # Then C[i,k] += W[k*n+i, r] * M_r?
    # No, that would make C a scalar, not a matrix.
    
    # The correct decomposition is:
    # C[i, k] = sum_r W[k*n+i, r] * U[i, r] * V[k, r]? No.
    
    # Let's look at the seed program:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # This implies the term is:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]
    # We want this to equal A[i,j] B[j,k] for the specific i,j,k and 0 otherwise?
    # Actually, we want:
    # sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = delta_{i,k} ? No.
    # We want the bilinear form to reproduce the product.
    # The condition is:
    # For all i, j, k:
    # sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = 1 if this term is in the sum?
    # No. The term A[i,j] B[j,k] contributes to C[i,k].
    # So for a fixed i, k, the coefficient of A[i,j] B[j,k] in the expansion of C[i,k] must be 1.
    # And for other combinations, 0.
    
    # The expansion of C[i,k] using the decomposition is:
    # sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k])?
    # This is still confusing.
    
    # Let's look at the tensor definition again.
    # T_{ijk} = 1.
    # We want to write T as a sum of rank-1 tensors.
    # A rank-1 tensor is u \otimes v \otimes w.
    # (u \otimes v \otimes w)_{ijk} = u_i v_j w_k.
    # So we need:
    # sum_r U[i, r] V[j, r] W[k, r] = T_{ijk}
    # Where U is n x R, V is m x R, W is p x R?
    # But the seed program uses flattened indices.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This corresponds to:
    # sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c]
    # This sum is over c.
    # We want this sum to be 1 if (i,j,k) is a valid term?
    # Actually, the tensor T_{ijk} is 1 for all i,j,k?
    # No, the matrix multiplication tensor is not all ones.
    # The tensor is defined such that the bilinear map is matrix multiplication.
    # The map is (A, B) -> C.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as:
    # C_{ik} = sum_{i,j,k} T_{ijk} A_{ij} B_{jk}
    # where T_{ijk} = 1 if the indices are valid?
    # No, T_{ijk} is the coefficient.
    # For a fixed i, k, the term is sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i, j, k?
    # Yes, because every pair (A_{ij}, B_{jk}) contributes to C_{ik}.
    # So the tensor is indeed the all-ones tensor of shape (n*m, m*p, p*n)?
    # No, the dimensions are (n*m, m*p, p*n).
    # The indices are (ij, jk, ki).
    # The term is A_{ij} B_{jk} C_{ki}^*?
    # The condition is:
    # sum_c W_{ki, c} U_{ij, c} V_{jk, c} = 1
    # for all valid i, j, k?
    # Wait, if i, j, k are independent, there are n*m*p terms.
    # But they are not independent in the tensor?
    # The tensor has size (n*m) x (m*p) x (p*n).
    # The entry at (ij, jk, ki) is 1.
    # All other entries are 0?
    # No, the tensor is sparse.
    # Only entries where the indices match (j in ij and jk, i in ij and ki, k in jk and ki) are non-zero.
    # Specifically, T[ij, jk, ki] = 1.
    # There are n*m*p such entries.
    # So we need:
    # sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = 1
    # for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # And for other combinations of indices (which don't form a valid path i->j->k), the sum should be 0?
    # But the indices in U, V, W are flattened.
    # U has indices 0..n*m-1.
    # V has indices 0..m*p-1.
    # W has indices 0..p*n-1.
    # The equation is only required for the n*m*p valid triples (i,j,k).
    # Are there constraints for other triples?
    # The tensor is zero elsewhere.
    # So if we pick arbitrary indices a, b, c that don't correspond to a valid i,j,k, the sum should be 0.
    # But we only need to satisfy the n*m*p equations.
    
    # So the problem is:
    # Find U, V, W of rank R such that:
    # sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = 1
    # for all i, j, k.

    # This is a system of n*m*p equations.
    # Variables are U, V, W entries.
    
    # We can solve this using least squares.
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from 15 to 24
    for R in range(15, 25):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Define the equations
            # We want sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = 1
            # This is a trilinear equation.
            # We can linearize it by fixing two matrices and solving for the third?
            # Or use gradient descent on the error.
            
            # Error function:
            # E = sum_{i,j,k} (sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] - 1)^2
            
            # This is non-convex.
            # We can use a simple iterative method:
            # 1. Fix V, W, solve for U (linear least squares).
            # 2. Fix U, W, solve for V.
            # 3. Fix U, V, solve for W.
            
            # Let's implement this alternating minimization.
            
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Solve for U
                # For each i, j:
                # sum_c U[i*m+j, c] (W[k*n+i, c] V[j*p+k, c]) = 1
                # This is not quite right. The sum is over c, but i, j, k are fixed.
                # The equation is:
                # sum_c U[i*m+j, c] * (W[k*n+i, c] V[j*p+k, c]) = 1
                # This must hold for all k?
                # No, for a fixed i, j, the term U[i*m+j, c] is multiplied by V[j*p+k, c] and W[k*n+i, c].
                # So for fixed i, j, we have p equations (one for each k).
                # sum_c U[i*m+j, c] * (W[k*n+i, c] V[j*p+k, c]) = 1 for all k.
                
                # This is a linear system for U[i*m+j, :].
                # Let M_k be a vector where M_k[c] = W[k*n+i, c] * V[j*p+k, c].
                # Then sum_c U[i*m+j, c] M_k[c] = 1.
                # So U[i*m+j, :] M_k^T = 1.
                # We have p such vectors M_k.
                # We want to find U_row such that U_row dot M_k = 1 for all k.
                # This is an overdetermined system (p equations, R unknowns).
                # We can solve it using least squares.
                
                # Construct matrix A_u of shape (p, R)
                # A_u[k, c] = W[k*n+i, c] * V[j*p+k, c]
                # b_u = ones(p)
                # Solve A_u U_row^T = b_u
                
                for i in range(n):
                    for j in range(m):
                        A_u = np.zeros((p, R))
                        for k in range(p):
                            for c in range(R):
                                A_u[k, c] = W[k*n+i, c] * V[j*p+k, c]
                        b_u = np.ones(p)
                        # Solve A_u @ U_row = b_u
                        # Using np.linalg.lstsq
                        U_row, residuals, rank, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                        U[i*m+j, :] = U_row
                        
                # Similarly for V
                # For fixed j, k:
                # sum_c V[j*p+k, c] (U[i*m+j, c] W[k*n+i, c]) = 1 for all i.
                for j in range(m):
                    for k in range(p):
                        A_v = np.zeros((n, R))
                        for i in range(n):
                            for c in range(R):
                                A_v[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        b_v = np.ones(n)
                        V_row, residuals, rank, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                        V[j*p+k, :] = V_row
                        
                # Similarly for W
                # For fixed i, k:
                # sum_c W[k*n+i, c] (U[i*m+j, c] V[j*p+k, c]) = 1 for all j.
                for i in range(n):
                    for k in range(p):
                        A_w = np.zeros((m, R))
                        for j in range(m):
                            for c in range(R):
                                A_w[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        b_w = np.ones(m)
                        W_row, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                        W[k*n+i, :] = W_row
                        
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(R):
                            val += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                        error += (val - 1)**2
                        
            if error < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(R):
                                val += W_rounded[k*n+i, c] * U_rounded[i*m+j, c] * V_rounded[j*p+k, c]
                            if abs(val - 1) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                if exact:
                    # Convert to list of lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
