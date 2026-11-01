"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 125  # safety margin before 120s eval limit + buffer
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # First, try to find a rank-14 decomposition of M_{2,3,3}
    # If successful, we can use it to build a rank-42 decomposition of M_{2,3,9}
    
    def build_rank_42_from_233(U233, V233, W233):
        """
        Given rank-14 decomposition of M_{2,3,3} with U(6,14), V(9,14), W(6,14),
        build rank-42 decomposition of M_{2,3,9}.
        
        The idea: M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3}
        where M_{1,1,3} has rank 3 with:
        U1(1,3) = [[1,1,1]]
        V1(3,3) = identity 3x3
        W1(3,3) = [[0,1,0],[0,0,1],[1,0,0]]  # shift register
        
        Actually, for multiplying a 1x1 matrix by 1x3 matrix:
        C[0,0] * A[0,k] for k=0,1,2
        
        This is just scalar multiplication of each element.
        U1[0,k] = 1 for all k (since C[0,0]=1)
        V1[j,k] = 1 if j==0 else 0  # A[0,k]
        W1[k,0] = 1 if k==0 else 0  # C[0,0]
        
        Wait, let me think more carefully.
        
        M_{n,m,p}: multiply n×m by m×p to get n×p.
        Tensor T[i,j,k] = delta_{j,k} for the multiplication tensor.
        
        For M_{1,1,3}: multiply 1×1 by 1×3 to get 1×3.
        T[i,j,k] where i∈{0}, j∈{0}, k∈{0,1,2}
        T[0,0,k] = delta_{0,k}? No.
        
        Actually, the matrix multiplication tensor T_{M_{n,m,p}}[i,j,k] 
        represents the coefficient of A[i,j] * B[j,k] in C[i,k].
        
        For M_{1,1,3}: A is 1×1, B is 1×3, C is 1×3.
        C[0,k] = A[0,0] * B[0,k]
        So T[0,0,k] = 1 for all k=0,1,2.
        
        Rank-3 decomposition:
        U1[0,k] = 1 for k=0,1,2  (size 1×3)
        V1[j,k] = 1 if j==0 else 0  (size 3×3, only row 0 is all 1s)
        W1[k,0] = 1 for all k  (size 3×1)
        
        Check: sum_k U1[0,k] * V1[j,k] * W1[k,0]
        = sum_k 1 * (1 if j==0 else 0) * 1
        = 3 if j==0, 0 otherwise
        But we want T[0,j,k]... wait, the tensor is 3-indexed.
        
        Let me redo. The tensor T_{M_{n,m,p}} is in R^{n×m × m×p × p×n}.
        T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
        
        For M_{1,1,3}: n=1, m=1, p=3
        T[0*1+0, 0*3+k, k*1+0] = T[0, k, k] = 1 for k=0,1,2
        
        So the tensor has non-zero entries at:
        (0, 0, 0), (0, 1, 1), (0, 2, 2)
        
        Rank-3 decomposition:
        For each k, we need a rank-1 tensor that is 1 at position (0, k, k).
        U1[0, k] = 1
        V1[k, k] = 1  
        W1[k, k] = 1
        
        So U1 = [[1,1,1]] (1×3)
        V1 = identity 3×3
        W1 = identity 3×3
        
        Check: sum_r U1[0,r] * V1[j,r] * W1[k,r]
        For r=k: U1[0,k]=1, V1[j,k]=delta_{j,k}, W1[k,k]=1
        = delta_{j,k}
        But we want T[0, j, k] = delta_{j,k}? 
        T[0, j, k] should be 1 iff j==k (from T[i*m+j, j*p+k, k*n+i] = 1)
        With i=0: T[j, j*3+k, k] = 1 iff the first index equals j and... 
        
        I'm getting confused. Let me just use the Kronecker product approach.
        
        If M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3}, then:
        U_{239} = U_{233} ⊗ U_{113}  (but these are not standard Kronecker)
        
        Actually, for tensor products of decompositions:
        If T1 = sum_r u1_r ⊗ v1_r ⊗ w1_r (rank R1)
        and T2 = sum_s u2_s ⊗ v2_s ⊗ w2_s (rank R2)
        then T1 ⊗ T2 = sum_{r,s} (u1_r ⊗ u2_s) ⊗ (v1_r ⊗ v2_s) ⊗ (w1_r ⊗ w2_s)
        with rank R1 * R2.
        
        The index mapping for Kronecker:
        If T1 has indices (a,b,c) with a∈[0,d1a), b∈[0,d1b), c∈[0,d1c)
        and T2 has indices (d,e,f) with d∈[0,d2a), e∈[0,d2b), f∈[0,d2c)
        then T1⊗T2 has indices (a*d2a+d, b*d2b+e, c*d2c+f)
        
        For M_{2,3,3}: dimensions n1=2, m1=3, p1=3
        U1 size: (n1*m1) × R1 = 6 × 14
        V1 size: (m1*p1) × R1 = 9 × 14  
        W1 size: (p1*n1) × R1 = 6 × 14
        
        For M_{1,1,3}: dimensions n2=1, m2=1, p2=3
        U2 size: (n2*m2) × R2 = 1 × 3
        V2 size: (m2*p2) × R2 = 3 × 3
        W2 size: (p2*n2) × R2 = 3 × 3
        
        Product M_{2,3,9}: dimensions n=2, m=3, p=9
        n = n1 * n2 = 2*1 ✓
        m = m1 * m2 = 3*1 ✓
        p = p1 * p2 = 3*3 = 9 ✓
        
        Rank = R1 * R2 = 14 * 3 = 42
        
        U_combined size: (n*m) × 42 = 6 × 42
        For combined index (i*m+j) where i∈[0,n), j∈[0,m):
        i = i1 * n2 + i2 = i1 * 1 + i2 = i1 + i2, but n2=1 so i2=0
        j = j1 * m2 + j2 = j1 * 1 + j2 = j1 + j2, m2=1 so j2=0
        So i = i1, j = j1
        
        U_combined[i*m+j, r*s] = U1[i1*m1+j1, r] * U2[i2*m2+j2, s]
        = U1[i*m+j, r] * U2[0, s]  (since i2=j2=0)
        
        V_combined[j*p+k, r*s] = V1[j1*p1+k1, r] * V2[j2*p2+k2, s]
        j = j1, so j1 = j
        k = k1 * p2 + k2 = k1 * 3 + k2
        k1 = k // 3, k2 = k % 3
        j2 = 0
        V_combined[j*9+k, r*s] = V1[j*3+k1, r] * V2[0*3+k2, s] = V1[j*3+k//3, r] * V2[k%3, s]
        
        W_combined[k*n+i, r*s] = W1[k1*n1+i1, r] * W2[k2*n2+i2, s]
        i = i1, k1 = k // 3, k2 = k % 3
        i2 = 0
        W_combined[k*2+i, r*s] = W1[(k//3)*2+i, r] * W2[(k%3)*1+0, s] = W1[(k//3)*2+i, r] * W2[k%3, s]
        
        Where r*s is the combined rank index: rank_idx = r * 3 + s
        """
        R1 = U233.shape[1]
        R2 = 3  # rank of M_{1,1,3}
        R = R1 * R2
        
        # M_{1,1,3} rank-3 decomposition
        U2 = np.ones((1, 3))
        V2 = np.eye(3)
        W2 = np.eye(3)
        
        U_comb = np.zeros((n * m, R))
        V_comb = np.zeros((m * p, R))
        W_comb = np.zeros((p * n, R))
        
        for r in range(R1):
            for s in range(R2):
                rank_idx = r * R2 + s
                for i in range(n):
                    for j in range(m):
                        U_comb[i * m + j, rank_idx] = U233[i * m + j, r] * U2[0, s]
                for j in range(m):
                    for k in range(p):
                        k1 = k // 3
                        k2 = k % 3
                        V_comb[j * p + k, rank_idx] = V233[j * 3 + k1, r] * V2[k2, s]
                for k in range(p):
                    for i in range(n):
                        k1 = k // 3
                        k2 = k % 3
                        W_comb[k * n + i, rank_idx] = W233[k1 * 2 + i, r] * W2[k2, s]
        
        # Convert to lists with integer/half-integer entries if possible
        def to_list(mat):
            # Try to round to nearest integer or half-integer
            rounded = np.round(mat * 2) / 2
            if np.allclose(mat, rounded, atol=1e-8):
                # Convert to list of lists
                result = []
                for row in rounded:
                    row_list = []
                    for val in row:
                        if abs(val * 2 - round(val * 2)) < 1e-8:
                            row_list.append(int(round(val)))
                        else:
                            row_list.append(float(val))
                    result.append(row_list)
                return result
            else:
                # Just return as float list
                return [[float(x) for x in row] for row in mat]
        
        return {"U": to_list(U_comb), "V": to_list(V_comb), "W": to_list(W_comb)}
    
    # Try to find rank-14 decomposition of M_{2,3,3}
    def find_rank_14_233():
        """Find rank-14 decomposition of M_{2,3,3} via alternating least squares."""
        n1, m1, p1 = 2, 3, 3
        R = 14
        
        # Build target tensor
        # T[i*m1+j, j*p1+k, k*n1+i] = 1
        target = {}
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    target[(i * m1 + j, j * p1 + k, k * n1 + i)] = 1
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n1 * m1, R)
        V = np.random.randn(m1 * p1, R)
        W = np.random.randn(p1 * n1, R)
        
        # Normalize columns to prevent blowup
        for r in range(R):
            norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
            if norm > 1e-10:
                scale = norm ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Fix V, W, solve for U
            # For each (i,j): sum_r U[i,j,r] * (V * W)[i,j,r] = T[i*m+j, ?, ?]
            # Actually, the tensor is T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
            # For fixed (i,j), varying k: T[i*m+j, j*p+k, k*n+i]
            # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
            
            # This is not a simple linear solve because V and W depend on k.
            # Let's use a different approach: build the full linear system.
            
            # For each rank element r, the contribution to the tensor is:
            # T_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
            # Total: sum_r T_r[a,b,c] = T[a,b,c]
            
            # Fix V and W, solve for U:
            # For each (a,b,c) in support: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Group by a: for fixed a, we have equations over (b,c)
            # sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            
            U_new = U.copy()
            for a in range(n1 * m1):
                # Collect all (b,c) pairs where T[a,b,c] != 0
                equations = []
                for b in range(m1 * p1):
                    for c in range(p1 * n1):
                        if (a, b, c) in target:
                            coeff = np.array([V[b, r] * W[c, r] for r in range(R)])
                            equations.append((coeff, target[(a, b, c)]))
                
                if len(equations) >= R:
                    A_mat = np.array([eq[0] for eq in equations[:R * 2]])
                    b_vec = np.array([eq[1] for eq in equations[:R * 2]])
                    # Least squares
                    U_new[a, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Normalize
                norm = np.sum(U_new[a, :]**2)
                if norm > 1e-10:
                    U_new[a, :] /= np.sqrt(norm)
            
            U = U_new
            
            # Fix U, W, solve for V
            V_new = V.copy()
            for b in range(m1 * p1):
                equations = []
                for a in range(n1 * m1):
                    for c in range(p1 * n1):
                        if (a, b, c) in target:
                            coeff = np.array([U[a, r] * W[c, r] for r in range(R)])
                            equations.append((coeff, target[(a, b, c)]))
                
                if len(equations) >= R:
                    A_mat = np.array([eq[0] for eq in equations[:R * 2]])
                    b_vec = np.array([eq[1] for eq in equations[:R * 2]])
                    V_new[b, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                norm = np.sum(V_new[b, :]**2)
                if norm > 1e-10:
                    V_new[b, :] /= np.sqrt(norm)
            
            V = V_new
            
            # Fix U, V, solve for W
            W_new = W.copy()
            for c in range(p1 * n1):
                equations = []
                for a in range(n1 * m1):
                    for b in range(m1 * p1):
                        if (a, b, c) in target:
                            coeff = np.array([U[a, r] * V[b, r] for r in range(R)])
                            equations.append((coeff, target[(a, b, c)]))
                
                if len(equations) >= R:
                    A_mat = np.array([eq[0] for eq in equations[:R * 2]])
                    b_vec = np.array([eq[1] for eq in equations[:R * 2]])
                    W_new[c, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                norm = np.sum(W_new[c, :]**2)
                if norm > 1e-10:
                    W_new[c, :] /= np.sqrt(norm)
            
            W = W_new
            
            # Compute error
            error = 0.0
            for (a, b, c), val in target.items():
                computed = np.sum(U[a, :] * V[b, :] * W[c, :])
                error += (computed - val) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_error < 1e-8:
                break
        
        if best_error < 1e-6:
            # Try to round to half-integers
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            
            # Verify exactness
            error_r = 0.0
            for (a, b, c), val in target.items():
                computed = np.sum(U_r[a, :] * V_r[b, :] * W_r[c, :])
                error_r += (computed - val) ** 2
            
            if error_r < 1e-8:
                return U_r, V_r, W_r
            else:
                # Return the floating point version
                return best_U, best_V, best_W
        
        return None
    
    # Try to find the decomposition
    result = find_rank_14_233()
    if result is not None:
        U233, V233, W233 = result
        if time.time() < deadline - 5:
            return build_rank_42_from_233(U233, V233, W233)
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
