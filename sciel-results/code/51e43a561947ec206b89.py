"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def get_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Uses the middle-product structure where V-factors for the second block
        share support patterns with the first block.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen with shared structure
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        return U, V, W
    
    def build_target_tensor():
        """Build the true multiplication tensor T[i,j,k] = 1 if valid."""
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    def tensor_from_factors(U, V, W):
        """Compute tensor from factor matrices."""
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def try_rank13_search():
        """
        Attempt to find a rank-13 decomposition using structured search.
        Key insight: The two Strassen blocks share the same U structure,
        so we can try to merge terms that have similar U patterns.
        """
        if time.time() > deadline - 20:
            return None
            
        U14, V14, W14 = get_strassen_rank14()
        U14_np = np.array(U14, dtype=float)
        V14_np = np.array(V14, dtype=float)
        W14_np = np.array(W14, dtype=float)
        T_target = build_target_tensor()
        
        best_result = None
        best_norm = np.inf
        
        # Strategy 1: Try dropping each term and optimizing
        for drop_idx in range(14):
            if time.time() > deadline - 5:
                break
                
            # Create initial guess for rank 13 by dropping one column
            U13 = np.delete(U14_np, drop_idx, axis=1).copy()
            V13 = np.delete(V14_np, drop_idx, axis=1).copy()
            W13 = np.delete(W14_np, drop_idx, axis=1).copy()
            
            # Try multiple random perturbations with different seeds
            for trial in range(5):
                if time.time() > deadline - 3:
                    break
                    
                np.random.seed(42 + drop_idx * 100 + trial)
                scale = 0.03
                U13_trial = U13 + scale * np.random.randn(4, 13)
                V13_trial = V13 + scale * np.random.randn(8, 13)
                W13_trial = W13 + scale * np.random.randn(8, 13)
                
                # Gradient descent with adaptive learning rate
                lr = 0.003
                for step in range(300):
                    if time.time() > deadline - 2:
                        break
                        
                    T_curr = tensor_from_factors(U13_trial, V13_trial, W13_trial)
                    residual = T_curr - T_target
                    norm = np.linalg.norm(residual)
                    
                    if norm < 1e-10:
                        break
                    
                    # Compute gradients
                    dU = np.einsum('ijk,jr,kr->ir', residual, V13_trial, W13_trial)
                    dV = np.einsum('ijk,ir,kr->jr', residual, U13_trial, W13_trial)
                    dW = np.einsum('ijk,ir,jr->kr', residual, U13_trial, V13_trial)
                    
                    # Update with adaptive learning rate
                    grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                    if grad_norm > 1e-15:
                        effective_lr = lr * min(1.0, 0.05 / grad_norm)
                        U13_trial -= effective_lr * dU
                        V13_trial -= effective_lr * dV
                        W13_trial -= effective_lr * dW
                    
                    # Reduce learning rate
                    if step > 100:
                        lr *= 0.98
                    elif step > 200:
                        lr *= 0.995
                
                # Final check with high precision
                T_final = tensor_from_factors(U13_trial, V13_trial, W13_trial)
                final_norm = np.linalg.norm(T_final - T_target)
                
                if final_norm < 1e-8 and final_norm < best_norm:
                    best_norm = final_norm
                    U_list = [[float(U13_trial[i,r]) for r in range(13)] for i in range(4)]
                    V_list = [[float(V13_trial[j,r]) for r in range(13)] for j in range(8)]
                    W_list = [[float(W13_trial[k,r]) for r in range(13)] for k in range(8)]
                    
                    if verify(U_list, V_list, W_list, tol=1e-8):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # Strategy 2: Try merging pairs of terms with similar U patterns
        # The left and right blocks have identical U structures for corresponding terms
        # Try to find a rank-13 by combining M1 and N1 (both have U[0]=1, U[3]=1)
        if time.time() > deadline - 10:
            return None
            
        # Create a merged initialization: use 13 terms where term 0 and 7 are merged
        U13_merge = np.zeros((4, 13))
        V13_merge = np.zeros((8, 13))
        W13_merge = np.zeros((8, 13))
        
        # Copy terms 1-6 from left block
        for r in range(1, 7):
            U13_merge[:, r-1] = U14_np[:, r]
            V13_merge[:, r-1] = V14_np[:, r]
            W13_merge[:, r-1] = W14_np[:, r]
        
        # Copy terms 8-13 from right block
        for r in range(8, 14):
            U13_merge[:, r-1] = U14_np[:, r]
            V13_merge[:, r-1] = V14_np[:, r]
            W13_merge[:, r-1] = W14_np[:, r]
        
        # Term 6 (index 6 in 13-term) is the merged M1+N1
        # M1: U=[1,0,0,1], V=[1,0,0,0,0,1,0,0], W=[1,0,1,0,0,0,0,0]
        # N1: U=[1,0,0,1], V=[0,0,1,0,0,0,0,1], W=[0,0,0,0,1,0,0,1]
        # Merged: U=[1,0,0,1], V=[1,0,1,0,0,1,0,1], W=[1,0,1,0,1,0,0,1]
        U13_merge[0, 6] = 1.0
        U13_merge[3, 6] = 1.0
        V13_merge[0, 6] = 1.0
        V13_merge[2, 6] = 1.0
        V13_merge[5, 6] = 1.0
        V13_merge[7, 6] = 1.0
        W13_merge[0, 6] = 1.0
        W13_merge[2, 6] = 1.0
        W13_merge[4, 6] = 1.0
        W13_merge[7, 6] = 1.0
        
        # Optimize this merged initialization
        for trial in range(3):
            if time.time() > deadline - 3:
                break
                
            np.random.seed(200 + trial)
            scale = 0.02
            U13_opt = U13_merge + scale * np.random.randn(4, 13)
            V13_opt = V13_merge + scale * np.random.randn(8, 13)
            W13_opt = W13_merge + scale * np.random.randn(8, 13)
            
            lr = 0.002
            for step in range(400):
                if time.time() > deadline - 2:
                    break
                    
                T_curr = tensor_from_factors(U13_opt, V13_opt, W13_opt)
                residual = T_curr - T_target
                norm = np.linalg.norm(residual)
                
                if norm < 1e-10:
                    break
                
                dU = np.einsum('ijk,jr,kr->ir', residual, V13_opt, W13_opt)
                dV = np.einsum('ijk,ir,kr->jr', residual, U13_opt, W13_opt)
                dW = np.einsum('ijk,ir,jr->kr', residual, U13_opt, V13_opt)
                
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-15:
                    effective_lr = lr * min(1.0, 0.03 / grad_norm)
                    U13_opt -= effective_lr * dU
                    V13_opt -= effective_lr * dV
                    W13_opt -= effective_lr * dW
                
                if step > 150:
                    lr *= 0.98
                elif step > 300:
                    lr *= 0.995
            
            T_final = tensor_from_factors(U13_opt, V13_opt, W13_opt)
            final_norm = np.linalg.norm(T_final - T_target)
            
            if final_norm < 1e-8 and final_norm < best_norm:
                best_norm = final_norm
                U_list = [[float(U13_opt[i,r]) for r in range(13)] for i in range(4)]
                V_list = [[float(V13_opt[j,r]) for r in range(13)] for j in range(8)]
                W_list = [[float(W13_opt[k,r]) for r in range(13)] for k in range(8)]
                
                if verify(U_list, V_list, W_list, tol=1e-8):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try to find rank-13 first
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Fall back to rank-14
    U, V, W = get_strassen_rank14()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
