import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # Build the multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook solution (Rank 56)
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school), dtype=np.float64)
    V_school = np.zeros((dim_v, rank_school), dtype=np.float64)
    W_school = np.zeros((dim_w, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using a greedy approach
    # We look for rank-1 tensors (u, v, w) such that T - sum(u_i v_i w_i) is close to 0.
    # Since we need EXACT decomposition, we will use the greedy method to find 
    # a set of candidate vectors, then solve for exact coefficients if possible.
    
    # However, finding the *minimal* rank exactly is hard. 
    # A known upper bound for 2x4x7 is 25. 
    # Let's try to construct rank 25 or 27 explicitly if possible, 
    # or use a fast heuristic that guarantees a valid (not necessarily minimal) 
    # decomposition better than 56.
    
    # Heuristic: Try to find a decomposition of rank R by random projection and SVD?
    # Or simply use the Karatsuba-based construction which gives Rank 42.
    # Rank 42 is a solid improvement over 56.
    
    # Let's implement the Rank 42 construction explicitly.
    # C = A_L B_L + A_R B_R
    # A_L is 2x2, B_L is 2x7.
    # For each column k in B_L:
    #   Let b0 = B[0,k], b1 = B[1,k]
    #   Let a0 = A[0,0], a1 = A[0,1], a2 = A[1,0], a3 = A[1,1]
    #   P1 = a0 * b0
    #   P2 = a3 * b1
    #   P3 = (a0+a3) * (b0+b1)
    #   C[0,k] += a1*b1 + (P3 - P1 - P2)  # Wait, Karatsuba: (a0+a3)(b0+b1) = a0b0 + a0b1 + a3b0 + a3b1.
    #   So a0b1 + a3b0 = P3 - P1 - P2.
    #   We need C[0,k] = a0b0 + a1b1. We have P1=a0b0. We need a1b1.
    #   We need C[1,k] = a2b0 + a3b1. We have P2=a3b1. We need a2b0.
    
    # The standard Karatsuba for 2x2 matrix mult computes:
    # M1 = a00*b00
    # M2 = a11*b11
    # M3 = (a00+a11)*(b00+b11)
    # M4 = (a01+a11)*(b10+b11) -- No, this is for full 2x2.
    
    # For 2x1 vector mult by 1x1? No.
    # We are doing 2x2 matrix times 2x7 matrix.
    # This is equivalent to 7 independent 2x2 matrix times 1x1 (column of B)?
    # No, B is 2x7. The multiplication is C[:,k] = A * B[:,k].
    # This is a 2x2 matrix times a 2x1 vector.
    # Rank of 2x2x1 is 4 (schoolbook) or 3 (Karatsuba-like)?
    # (a00 b00; a10 b00) * (x; y) -> a00x + a01y, a10x + a11y.
    # We can compute:
    # P1 = a00*x
    # P2 = a11*y
    # P3 = (a00+a11)*(x+y) = a00x + a00y + a11x + a11y
    # P4 = (a01+a10)*? No.
    
    # Actually, the rank of 2x2x1 is 4? No, it's a bilinear map from C^2 x C^2 to C^2.
    # T_{i,j,k} where i in {0,1} (A row), j in {0,1} (A col / B row), k in {0,1} (C row).
    # It is just matrix-vector product. The tensor rank of 2x2x1 is 4?
    # Let's check. We need to compute 2 outputs from 4 inputs.
    # It is known that R_{2,2,1} = 4.
    
    # So the Karatsuba trick doesn't reduce the rank for matrix-vector product 2x2x1.
    # It reduces 2x2x2 (matrix-matrix) from 8 to 7.
    
    # So my Rank 42 estimate is incorrect if R_{2,2,1}=4.
    # 7 columns * 4 mults = 28 for A_L B_L.
    # 28 for A_R B_R.
    # Total 56. Same as schoolbook.
    
    # We need to exploit the structure of the 4x4 block or the 2x7 shape.
    
    # Let's use the **Rank 25** construction from literature if possible, 
    # or fall back to a **Greedy Search** that might find a rank < 56.
    
    # Greedy Search Strategy:
    # 1. Start with T.
    # 2. Find u, v, w that maximize <T, u⊗v⊗w>.
    # 3. Subtract the projection.
    # 4. Repeat.
    # 5. After K steps, check if residual is 0.
    # 6. If not, try to fit the remaining tensor with a few more terms.
    
    # Since exactness is required, and greedy is approximate, 
    # this is risky. 
    # However, for small dimensions, we can use **integer/half-integer** constraints.
    
    # A safer improvement: Use **Strassen's algorithm** adapted.
    # Or simply return the schoolbook if we can't guarantee a valid lower rank.
    # But 56 is the seed. We MUST beat 56.
    
    # Let's try a **Rank 32** construction using 2x4x2 blocks?
    # No.
    
    # I will implement a **Randomized Search** for a low-rank decomposition 
    # using the fact that the entries are likely to be small integers.
    # We will try to find U, V, W with entries in {-1, 0, 1} or similar.
    
    # Given the time constraint and the need for a valid solution, 
    # I will provide a **Rank 42** construction based on a different grouping 
    # if I can find one, or stick to a **Greedy Approximation** 
    # that is then refined to be exact.
    
    # Actually, the simplest valid improvement is to use the 
    # **Winograd's F4F** algorithm which gives rank 24 for 2x4x4?
    # For 2x4x7, we can split 7 into 4+3.
    # R(2x4x4) <= 20 (known).
    # R(2x4x3) <= ?
    
    # Let's just use the **Schoolbook** but with **compressed sensing** 
    # to find a sparse exact solution?
    
    # I will output the Schoolbook solution to ensure validity, 
    # as I cannot guarantee a valid lower-rank construction in this thought process 
    # without the exact formulas for Rank 25.
    # Wait, the score is 1.0 for Rank 56. 
    # If I return Rank 56, I tie. 
    # I need to BEAT it.
    
    # Let's try **Rank 28** using the following trick:
    # C = A B.
    # Use the identity: 
    # (A_{i0} + A_{i1} + A_{i2} + A_{i3}) (B_{0k} + B_{1k} + B_{2k} + B_{3k})
    # This doesn't help directly.
    
    # I will use a **Greedy Rank-1 Decomposition** with exact refinement.
    # If it fails to find an exact decomposition of rank < 56, I return 56.
    
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Try to find a decomposition of rank R
    # We'll try R = 25, 26, ..., 55
    # For each R, we attempt to find U, V, W.
    
    # Since finding exact rank is NP-hard, we use a heuristic:
    # 1. Generate random rank-1 tensors.
    # 2. Use them as basis.
    # 3. Solve linear system for coefficients.
    
    # This is too slow for 150s if not optimized.
    
    # Alternative: Return Rank 56.
    
    # Let's try a greedy search for a lower rank
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try ranks from 25 to 55
    for target_rank in range(25, 56):
        if time.time() - start_time > deadline - 5:
            break
            
        # Initialize random U, V, W
        U_rand = np.random.randint(-2, 3, (dim_u, target_rank)).astype(np.float64)
        V_rand = np.random.randint(-2, 3, (dim_v, target_rank)).astype(np.float64)
        W_rand = np.random.randint(-2, 3, (dim_w, target_rank)).astype(np.float64)
        
        # Compute the tensor decomposition
        # T_decomp[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # This is equivalent to matmul(U, diag(V.T * W.T))? No.
        # T_decomp = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand)
        
        # Instead, let's solve for the coefficients.
        # We have T = sum_r u_r ⊗ v_r ⊗ w_r
        # This is a linear system in the coefficients if we fix the vectors.
        # But here we are fixing the vectors and checking if they span T.
        
        # A better approach: use the fact that we need EXACT decomposition.
        # Let's try to find a set of rank-1 tensors that sum to T.
        # We can do this by solving a linear system.
        
        # Flatten T
        T_flat = T.flatten()
        
        # Build the design matrix A where each column is a rank-1 tensor (u ⊗ v ⊗ w)
        # But we don't know u, v, w.
        
        # Alternative: use the greedy algorithm to find u, v, w.
        
        # Greedy step:
        # Maximize <T, u ⊗ v ⊗ w>
        # This is equivalent to finding the best rank-1 approximation.
        
        # We can use the SVD of the reshaped tensor.
        # T_reshaped1 = T.reshape(dim_u, dim_v * dim_w)
        # U_svd, S_svd, Vt_svd = np.linalg.svd(T_reshaped1, full_matrices=False)
        # u = U_svd[:, 0]
        # v_w = Vt_svd[0, :].reshape(dim_v, dim_w)
        # U_svd2, S_svd2, Vt_svd2 = np.linalg.svd(v_w, full_matrices=False)
        # v = U_svd2[:, 0]
        # w = Vt_svd2[0, :]
        
        # Let's implement this greedy approach.
        residual = T.copy()
        U_greedy = []
        V_greedy = []
        W_greedy = []
        
        for step in range(target_rank):
            if time.time() - start_time > deadline - 2:
                break
                
            # Find best rank-1 approximation of residual
            T_reshaped1 = residual.reshape(dim_u, dim_v * dim_w)
            U_svd, S_svd, Vt_svd = np.linalg.svd(T_reshaped1, full_matrices=False)
            u = U_svd[:, 0]
            v_w = Vt_svd[0, :].reshape(dim_v, dim_w)
            U_svd2, S_svd2, Vt_svd2 = np.linalg.svd(v_w, full_matrices=False)
            v = U_svd2[:, 0]
            w = Vt_svd2[0, :]
            
            # Compute the coefficient
            # We want to find alpha such that alpha * u ⊗ v ⊗ w is the best approximation.
            # This is equivalent to minimizing ||residual - alpha * u ⊗ v ⊗ w||_F^2
            # The optimal alpha is <residual, u ⊗ v ⊗ w> / ||u ⊗ v ⊗ w||_F^2
            
            uvw = np.einsum('i,j,k->ijk', u, v, w)
            alpha = np.sum(residual * uvw) / np.sum(uvw * uvw)
            
            if alpha == 0:
                break
                
            # Update residual
            residual -= alpha * uvw
            
            # Store the vectors
            U_greedy.append(u * np.sqrt(abs(alpha)))
            V_greedy.append(v * np.sign(alpha) * np.sqrt(abs(alpha)))
            W_greedy.append(w)
            
            # Check if residual is close to zero
            if np.allclose(residual, 0, atol=1e-10):
                break
        
        # Check if we have a valid decomposition
        if len(U_greedy) <= target_rank and np.allclose(residual, 0, atol=1e-10):
            # We have a valid decomposition of rank len(U_greedy)
            if len(U_greedy) < best_rank:
                best_rank = len(U_greedy)
                # Pad with zeros if necessary
                U_final = np.zeros((dim_u, best_rank))
                V_final = np.zeros((dim_v, best_rank))
                W_final = np.zeros((dim_w, best_rank))
                for i in range(best_rank):
                    U_final[:, i] = U_greedy[i]
                    V_final[:, i] = V_greedy[i]
                    W_final[:, i] = W_greedy[i]
                best_U = U_final
                best_V = V_final
                best_W = W_final
    
    # Return the best solution found
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
