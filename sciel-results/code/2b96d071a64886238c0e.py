"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Try rank 16 first with structured construction
    # The idea: use a cyclic/structured approach for the 5 columns
    
    # Attempt 1: Direct rank-16 construction using bilinear structure
    # For 2x2 * 2x5, we have output C[i,j] = sum_k A[i,k] * B[k,j]
    # C has shape 2x5, so 10 output elements
    # Each output C[i,j] needs to be represented as sum of rank-1 terms
    
    # Let me try a different approach: exploit that we have 2x2 on left
    # and 2x5 on right. The tensor is in R^{4} x R^{10} x R^{10}
    
    # Rank-16 attempt: Use the fact that 2x2 multiplication has rank 7 (Strassen)
    # and try to share structure across the 5 columns
    
    # Actually, let me try a direct algebraic construction for rank 16
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i=0,1 and j=0,1,2,3,4
    
    # That's 10 outputs, each is a sum of 2 products = 20 "natural" terms
    # Strassen reduces 2x2*2x2 to 7 terms, saving 1 term per 2x2 block
    
    # For rank 16, I need to find 16 bilinear forms that span all 10 outputs
    
    # Let me try: treat columns 0-3 as two Strassen blocks (14 terms) 
    # and column 4 with 2 shared terms (total 16)
    
    # But that's still 16... let me verify if it works
    
    # Actually, the issue is that Strassen for 2x2*2x2 uses 7 terms,
    # but the cross terms between different column blocks might allow sharing
    
    # Let me try a rank-16 construction:
    # Use 7 Strassen terms for columns 0,1
    # Use 7 Strassen terms for columns 2,3  
    # Use 2 terms for column 4 (but share with existing structure)
    
    # Wait - the schoolbook for column 4 needs 4 terms (A[0,0]*B[0,4], A[0,1]*B[1,4], 
    # A[1,0]*B[0,4], A[1,1]*B[1,4]). Can I reduce this to 2?
    
    # No, because each of the 4 products is independent.
    
    # Alternative: Can I restructure to use fewer than 7+7+4=18?
    
    # Let me try: 7 terms for cols 0-1, and 9 terms for cols 2-4
    # For 2x2 * 2x3, what's the minimal rank?
    
    # 2x2 * 2x3: outputs are C[0,2],C[0,3],C[0,4],C[1,2],C[1,3],C[1,4] (6 outputs)
    # Natural: 12 terms. Can we do better?
    
    # Let me just try rank 17 first as a stepping stone, then 16
    
    # RANK 17 ATTEMPT: Share one term between the two Strassen blocks
    
    # In standard Strassen for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # For columns 0,1: use M1-M7 (7 terms)
    # For columns 2,3: use M1'-M7' (7 terms)
    # For column 4: use 3 terms (share one with existing)
    
    # Actually, let me try a completely different approach.
    
    # The multiplication tensor for 2x2 * 2x5 can be viewed as:
    # T in R^{4 x 10 x 10} where T[a,b,c] = 1 if the product A[a] * B[b] contributes to C[c]
    
    # Let me use numpy to verify and search for rank-16
    
    # Build the target tensor
    T = np.zeros((4, 10, 10))  # A index, B index, C index
    for i in range(2):
        for k in range(2):
            for j in range(5):
                a_idx = i * 2 + k  # A[i,k]
                b_idx = k * 5 + j  # B[k,j]
                c_idx = j * 2 + i  # C[j,i] (column-major: C[j,i] = row i, col j)
                # Wait, let me re-read the convention: W[k*n+i, r]
                # So C is indexed as [k*n+i] where k is column (0-4), i is row (0-1)
                c_idx = j * n + i  # j is col (0-4), i is row (0-1)
                T[a_idx, b_idx, c_idx] = 1
    
    # Verify: T has 4*2*5 = 40 nonzero entries
    assert np.count_nonzero(T) == 40
    
    # Now search for rank-16 decomposition
    # T = sum_{r=0}^{15} u_r ⊗ v_r ⊗ w_r
    
    # This is a hard NP-hard problem in general, but for small cases we can try
    
    # Strategy: Use the known rank-18 solution and try to merge two terms
    # Or use ALS (alternating least squares) to find a rank-16 decomposition
    
    # Let me try ALS for rank 16
    rank = 16
    np.random.seed(42)
    
    # Initialize with random factors
    U = np.random.randn(4, rank)
    V = np.random.randn(10, rank)
    W = np.random.randn(10, rank)
    
    # Normalize to avoid scale issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
            scale = (norm_u * norm_v * norm_w) ** (1/3)
            U[:, r] /= scale
            V[:, r] /= scale
            W[:, r] /= scale
    
    best_obj = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(500):
        if time.time() > deadline - 20:
            break
        
        # Update U: for each row i, U[i,r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] / (sum_r' V[j,r']^2 * W[k,r']^2)
        # Actually, the objective is ||T - sum_r u_r v_r w_r||_F^2
        # Fix V,W: U[i,r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r] / sum_r' V[j,r']^2 W[k,r']^2
        # This isn't quite right. Let me do it properly.
        
        # T ≈ sum_r U[:,r] V[:,r] W[:,r] (outer product sum)
        # Residual R = T - sum_r U[:,r] V[:,r] W[:,r]
        # For U[i,r]: minimize sum_{j,k} (T[i,j,k] - sum_r' U[i,r'] V[j,r'] W[k,r'])^2
        # = sum_{j,k} (T[i,j,k] - U[i,r]V[j,r]W[k,r] - sum_{r'≠r} U[i,r']V[j,r']W[k,r'])^2
        # Let S[i,j,k] = sum_{r'≠r} U[i,r']V[j,r']W[k,r']
        # Then U[i,r] = sum_{j,k} (T[i,j,k] - S[i,j,k]) V[j,r] W[k,r] / sum_{j,k} V[j,r]^2 W[k,r]^2
        
        # This is O(4*10*10*16) per update which is fine
        
        # Update U
        for i in range(4):
            for r in range(rank):
                # Compute residual without term r
                residual = T[i, :, :].copy()
                for r2 in range(rank):
                    if r2 != r:
                        residual -= U[i, r2] * np.outer(V[:, r2], W[:, r2])
                numerator = np.sum(residual * np.outer(V[:, r], W[:, r]))
                denominator = np.sum(V[:, r]**2 * W[:, r]**2)
                if denominator > 1e-10:
                    U[i, r] = numerator / denominator
        
        # Update V
        for j in range(10):
            for r in range(rank):
                residual = T[:, j, :].copy()
                for r2 in range(rank):
                    if r2 != r:
                        residual -= np.outer(U[:, r2], W[:, r2]) * V[j, r2]
                numerator = np.sum(residual * np.outer(U[:, r], W[:, r]))
                denominator = np.sum(U[:, r]**2 * W[:, r]**2)
                if denominator > 1e-10:
                    V[j, r] = numerator / denominator
        
        # Update W
        for k in range(10):
            for r in range(rank):
                residual = T[:, :, k].copy()
                for r2 in range(rank):
                    if r2 != r:
                        residual -= np.outer(U[:, r2], V[:, r2]) * W[k, r2]
                numerator = np.sum(residual * np.outer(U[:, r], V[:, r]))
                denominator = np.sum(U[:, r]**2 * V[:, r]**2)
                if denominator > 1e-10:
                    W[k, r] = numerator / denominator
        
        # Compute objective
        obj = 0
        for i in range(4):
            for j in range(10):
                for k in range(10):
                    approx = 0
                    for r in range(rank):
                        approx += U[i, r] * V[j, r] * W[k, r]
                    obj += (T[i, j, k] - approx)**2
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_obj < 1e-6:
            break
    
    # Check if we found a valid rank-16 decomposition
    if best_obj < 1e-6:
        # Convert to list format
        U_list = [[int(round(x * 2)) / 2 for x in best_U[i]] for i in range(4)]
        V_list = [[int(round(x * 2)) / 2 for x in best_V[i]] for i in range(10)]
        W_list = [[int(round(x * 2)) / 2 for x in best_W[i]] for i in range(10)]
        
        # Verify
        T_check = np.zeros((4, 10, 10))
        for r in range(rank):
            for i in range(4):
                for j in range(10):
                    for k in range(10):
                        T_check[i, j, k] += U_list[i][r] * V_list[j][r] * W_list[k][r]
        
        if np.allclose(T_check, T, atol=1e-9):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If rank 16 didn't work, try rank 17
    if time.time() < deadline - 30:
        rank = 17
        np.random.seed(43)
        U = np.random.randn(4, rank)
        V = np.random.randn(10, rank)
        W = np.random.randn(10, rank)
        
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        best_obj = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(500):
            if time.time() > deadline - 25:
                break
            
            for i in range(4):
                for r in range(rank):
                    residual = T[i, :, :].copy()
                    for r2 in range(rank):
                        if r2 != r:
                            residual -= U[i, r2] * np.outer(V[:, r2], W[:, r2])
                    numerator = np.sum(residual * np.outer(V[:, r], W[:, r]))
                    denominator = np.sum(V[:, r]**2 * W[:, r]**2)
                    if denominator > 1e-10:
                        U[i, r] = numerator / denominator
            
            for j in range(10):
                for r in range(rank):
                    residual = T[:, j, :].copy()
                    for r2 in range(rank):
                        if r2 != r:
                            residual -= np.outer(U[:, r2], W[:, r2]) * V[j, r2]
                    numerator = np.sum(residual * np.outer(U[:, r], W[:, r]))
                    denominator = np.sum(U[:, r]**2 * W[:, r]**2)
                    if denominator > 1e-10:
                        V[j, r] = numerator / denominator
            
            for k in range(10):
                for r in range(rank):
                    residual = T[:, :, k].copy()
                    for r2 in range(rank):
                        if r2 != r:
                            residual -= np.outer(U[:, r2], V[:, r2]) * W[k, r2]
                    numerator = np.sum(residual * np.outer(U[:, r], V[:, r]))
                    denominator = np.sum(U[:, r]**2 * V[:, r]**2)
                    if denominator > 1e-10:
                        W[k, r] = numerator / denominator
            
            obj = 0
            for i in range(4):
                for j in range(10):
                    for k in range(10):
                        approx = 0
                        for r in range(rank):
                            approx += U[i, r] * V[j, r] * W[k, r]
                        obj += (T[i, j, k] - approx)**2
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_obj < 1e-6:
                break
        
        if best_obj < 1e-6:
            U_list = [[int(round(x * 2)) / 2 for x in best_U[i]] for i in range(4)]
            V_list = [[int(round(x * 2)) / 2 for x in best_V[i]] for i in range(10)]
            W_list = [[int(round(x * 2)) / 2 for x in best_W[i]] for i in range(10)]
            
            T_check = np.zeros((4, 10, 10))
            for r in range(rank):
                for i in range(4):
                    for j in range(10):
                        for k in range(10):
                            T_check[i, j, k] += U_list[i][r] * V_list[j][r] * W_list[k][r]
            
            if np.allclose(T_check, T, atol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to rank 18 (known working solution)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
