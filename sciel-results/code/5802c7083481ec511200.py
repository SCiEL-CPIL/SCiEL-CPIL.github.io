import numpy as np
import time
import itertools
import random
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C
    # A is n x m, B is m x p, C is n x p
    # W convention: W[k*n+i, r] where k is column of C, i is row of C
    # Actually let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank), indexed by (k*n + i) where i is row of C (0..n-1), k is col of C (0..p-1)
    
    # The tensor decomposition: C = U^T * D * V^T * W? No.
    # Standard: sum_r U[i, r] * V[j, r] * W[k, r] = T[i, j, k]
    # where i indexes A rows/cols (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # Let me think about this more carefully.
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # In the tensor formulation:
    # T[a, b, c] where:
    #   a = i*m + j (index into A, shape n*m)
    #   b = j*p + k (index into B, shape m*p)  
    #   c = k*n + i (index into C per W convention, shape p*n)
    # 
    # T[a, b, c] = 1 if a corresponds to (i, j), b to (j, k), c to (k, i) with matching indices
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Current best: rank 56 via Strassen on 2x2 blocks
    # Let's verify: 2x4 * 4x8 can be seen as 2*(2x2) * 4*(2x2) blocks
    # A has 2 block rows (each 1x2... no)
    
    # Actually: A is 2x4. Split columns into 2 groups of 2: A = [A_left | A_right], each 2x2
    # B is 4x8. Split rows into 2 groups of 2, columns into 4 groups of 2:
    # B = [[B_00, B_01, B_02, B_03], [B_10, B_11, B_12, B_13]], each 2x2
    # C = A * B = [A_left * B_left + A_right * B_right]
    # C is 2x8, split columns into 4 groups of 2: C = [C_0, C_1, C_2, C_3], each 2x2
    # C_k = A_left * B_{0,k} + A_right * B_{1,k}
    # 
    # Each A_left * B_{0,k} is 2x2 * 2x2 = 7 mults (Strassen)
    # Each A_right * B_{1,k} is 2x2 * 2x2 = 7 mults
    # Total: 2 * 4 * 7 = 56. Yes!
    
    # The question is: can we do better than 56?
    # 
    # The tensor rank of matrix multiplication tensor M_{n,m,p} satisfies:
    # R(M_{n,m,p}) >= max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # Also R(M_{n,m,p}) >= n*p = 16 (trivial lower bound)
    # 
    # For 2x4x4x8: lower bound is max(8, 32, 16) = 32
    # But we have 56. Can we do better?
    # 
    # One idea: use a rank-20 decomposition for 4x4 matrix multiplication
    # If we can embed our problem into 4x4x4x4 and use rank 20, we might get rank 20 or less.
    # 
    # Embedding: A (2x4) -> A' (4x4) by padding: A' = [[A, 0], [0, 0]]? No, that's 4x4 with top-left 2x4 block.
    # Wait, A is 2x4. To make it 4x4, pad to 4x4: A' = [[A], [zeros(2,4)]]
    # B (4x8) -> B' (4x4): This is trickier. B has 8 columns.
    # 
    # Alternative embedding: Treat the problem as computing 2 independent 2x4 * 4x4 multiplications
    # (splitting B's 8 columns into two 4-column blocks).
    # Each 2x4 * 4x4 can be embedded in 4x4 * 4x4.
    # 
    # For 2x4 * 4x4:
    # A' (4x4): [[A, 0], [0, 0]] where A is 2x4 -> top row block is A (2x4), bottom is zeros
    # Wait, A is 2x4, so A' should be 4x4. Pad: A' = [[A_row0], [A_row1], [0,0,0,0], [0,0,0,0]]
    # B' (4x4): just the 4x4 block of B
    # C' (4x4): top 2 rows contain the actual result, bottom 2 rows are zero
    # 
    # If 4x4 * 4x4 has rank 20, then two independent such multiplications would need rank 40.
    # But that's still better than 56!
    # 
    # However, the two multiplications share the same A (the first 2 rows).
    # So it's not two independent 4x4*4x4 multiplications.
    # 
    # Let me think differently. We need C = A * B where A is 2x4, B is 4x8.
    # Split B into B1 (4x4) and B2 (4x4).
    # C1 = A * B1 (2x4), C2 = A * B2 (2x4)
    # 
    # For C1: embed in 4x4*4x4:
    # A_pad = [[A], [0_{2x4}]], B1 as is, result in top 2 rows of 4x4.
    # For C2: same A_pad, B2 as is.
    # 
    # Both use the same A_pad. So the tensor is:
    # T_total = T(A_pad, B1, C1) + T(A_pad, B2, C2)
    # 
    # The rank of a sum of tensors is at most the sum of ranks, but can be less if there's sharing.
    # 
    # Actually, let me think about this more carefully using the tensor structure.
    # 
    # The tensor T has entries T[a,b,c] = 1 where a=(i,j), b=(j,k), c=(k,i).
    # 
    # Let me try a different approach: use a rank-20 decomposition of 4x4 matrix multiplication
    # and see how to apply it.
    
    # Known: The tensor rank of M_{4,4,4} is 20 (Landsberg, etc.)
    # We need to embed M_{2,4,4,8} into M_{4,4,4,4} or similar.
    
    # Embedding 1: 
    # A (2x4) -> A' (4x4): A' = [[A, 0_2x0], ...] hmm, A is 2x4, to make 4x4:
    # A' = [A; 0_{2x4}] (4x4)
    # B (4x8) -> need to make 4x4. Can't directly.
    
    # Embedding 2:
    # Split B into two 4x4 blocks: B = [B1 | B2]
    # Compute C1 = A*B1 and C2 = A*B2 separately
    # Each is 2x4 * 4x4 -> 2x4
    # Embed each in 4x4 * 4x4 -> 4x4
    # 
    # For C1 = A*B1:
    # A1' (4x4) = [[A], [0_{2x4}]]
    # B1' (4x4) = B1
    # C1' (4x4) = [[C1], [0_{2x4}]]
    # 
    # For C2 = A*B2:
    # A2' (4x4) = [[A], [0_{2x4}]]  (same as A1')
    # B2' (4x4) = B2
    # C2' (4x4) = [[C2], [0_{2x4}]]
    # 
    # The combined tensor: T = T1 + T2 where T1 is the 4x4*4x4*4x4 tensor for the first, 
    # T2 for the second. But A' is the same in both!
    # 
    # So effectively we have a tensor:
    # T[a, (b1,b2), (c1,c2)] = T1[a, b1, c1] + T2[a, b2, c2]
    # where a indexes A' (16 entries), b1 indexes B1 (16), b2 indexes B2 (16), 
    # c1 indexes C1 (16), c2 indexes C2 (16)
    # 
    # But only the top 2 rows of A' are nonzero, and only top 2 rows of C1, C2 are nonzero.
    # 
    # The rank of this combined tensor... it's not straightforward.
    
    # Let me try a completely different approach: direct rank minimization via numerical optimization.
    # 
    # The tensor T has shape (8, 32, 16). We want to find U (8 x r), V (32 x r), W (16 x r)
    # such that T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] (outer product sum)
    # 
    # This is equivalent to: for all (i,j,k), T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    
    # Let's try rank 32 (theoretical lower bound) and see if we can find a solution.
    
    def try_rank(r, max_time=30):
        """Try to find rank-r decomposition via optimization."""
        t0 = time.time()
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Reshape T for easier handling
        T_flat = T.flatten()
        
        def residual(params):
            if time.time() - t0 > max_time:
                return None
            U_flat = params[:n*m*r].reshape(n*m, r)
            V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
            # Compute sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            prod = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
            return (prod.flatten() - T_flat)
        
        # This is a hard nonconvex problem. Let's try a simpler approach.
        # Use the fact that for fixed V, the problem in U and W is bilinear.
        # Alternate minimization:
        
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for iteration in range(200):
            if time.time() - t0 > max_time:
                break
            
            # Update U: for each r, U[:,r] should satisfy
            # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # Fix V and W, solve for U.
            # T[i,j,k] = U[i,r]*V[j,r]*W[k,r] (summed over r)
            # This is a linear system in U[i,r] for each (i) if we fix others.
            # Actually, for fixed V and W:
            # T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
            # Let B_r[j,k] = V[j,r]*W[k,r]
            # Then T[i,:,:] = sum_r U[i,r] * B_r[j,k]
            # This is: T.reshape(n*m, m*p*p*n)[i,:] = sum_r U[i,r] * (V[:,r] ⊗ W[:,r]).flatten()
            
            # Build the basis
            # For each r, vector v_r = (V[:,r] * W[:,r]) in R^{m*p * p*n}
            # T_flat[i, :] where i is the A-index... wait, T is 3D.
            
            # Let me reshape: T[i, j, k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
            # For fixed i: T[i, :, :] is a matrix in R^{m*p x p*n}
            # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Let M_r = V[:,r] * W[:,r] (outer product, shape m*p x p*n)
            # Then T[i, :, :] = sum_r U[i,r] * M_r
            # This is a linear system: T_i = U_i @ M where M is (p*n*m*p) x r
            
            # Flatten M_r to a vector
            M_mat = np.zeros((m*p * p*n, r))
            for r_idx in range(r):
                M_mat[:, r_idx] = (V[:, r_idx].reshape(-1,1) * W[:, r_idx].reshape(1,-1)).flatten()
            
            T_flat_i = T.reshape(n*m, m*p * p*n)
            
            # Solve: T_flat_i[i, :] = sum_r U[i,r] * M_mat[:, r]
            # This is: T_flat_i = U @ M_mat
            # Least squares for U:
            try:
                U, _, _, _ = np.linalg.lstsq(M_mat, T_flat_i.T, rcond=None)
            except:
                break
            
            # Update V similarly:
            # T[i, j, k] = sum_r V[j,r] * U[i,r] * W[k,r]
            # For fixed j: T[:, j, :] = sum_r V[j,r] * (U[:,r] * W[:,r])
            M_mat2 = np.zeros((n*m * p*n, r))
            for r_idx in range(r):
                M_mat2[:, r_idx] = (U[:, r_idx].reshape(-1,1) * W[:, r_idx].reshape(1,-1)).flatten()
            
            T_flat_j = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
            try:
                V, _, _, _ = np.linalg.lstsq(M_mat2, T_flat_j.T, rcond=None)
            except:
                break
            
            # Update W:
            # T[i, j, k] = sum_r W[k,r] * U[i,r] * V[j,r]
            # For fixed k: T[:, :, k] = sum_r W[k,r] * (U[:,r] * V[:,r])
            M_mat3 = np.zeros((n*m * m*p, r))
            for r_idx in range(r):
                M_mat3[:, r_idx] = (U[:, r_idx].reshape(-1,1) * V[:, r_idx].reshape(1,-1)).flatten()
            
            T_flat_k = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            try:
                W, _, _, _ = np.linalg.lstsq(M_mat3, T_flat_k.T, rcond=None)
            except:
                break
            
            # Check convergence
            prod = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(prod - T)
            if err < 1e-6:
                return U, V, W, err
        
        # Final error
        prod = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(prod - T)
        if err < 1e-6:
            return U, V, W, err
        return None, None, None, err
    
    # Try various ranks
    best_result = None
    best_rank = 56  # current best
    
    for r in range(32, 57):
        if time.time() > deadline - 30:
            break
        print(f"Trying rank {r}...")
        U, V, W, err = try_rank(r, max_time=15)
        if U is not None and err < 1e-6:
            # Verify
            prod = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(prod, T, atol=1e-6):
                print(f"Found rank {r} decomposition!")
                best_result = (U, V, W)
                best_rank = r
                break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list format with rounded values
        U_list = [[round(float(U[i][r]), 10) for r in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[round(float(V[i][r]), 10) for r in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[round(float(W[i][r]), 10) for r in range(W.shape[1])] for i in range(W.shape[0])]
        # Clean up near-zero values
        for mat in [U_list, V_list, W_list]:
            for row in mat:
                for idx in range(len(row)):
                    if abs(row[idx]) < 1e-8:
                        row[idx] = 0
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: use the Strassen-based rank-56 decomposition
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col = 0
    for a in range(2):
        for b in range(4):
            for t in range(7):
                term = strassen_terms[t]
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx][col] += coeff
                for (j_prime, k_prime), coeff in term['y'].items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx][col] += coeff
                for (i_prime, k_prime), coeff in term['z'].items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
