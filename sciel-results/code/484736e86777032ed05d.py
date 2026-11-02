"""Improved program for the suite task."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Start with the known rank-26 decomposition
    # U: 9x26, V: 9x26, W: 9x26
    # Indexing: U[i*n+j, k] for A[i,j], V[j*p+k, l] for B[j,k], W[k*n+i, m] for C[k,i]
    
    def build_rank26():
        rank = 26
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        
        col = 0
        def add_term(u_idx, v_idx, w_idx, coeff=1):
            nonlocal col
            U[u_idx][col] += coeff
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
        
        # Strassen on top-left 2x2 block
        # A indices: 0=A00, 1=A01, 3=A10, 4=A11
        # B indices: 0=B00, 1=B01, 3=B10, 4=B11
        # C indices: 0=C00, 3=C01, 1=C10, 4=C11
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0][col]=1; U[4][col]=1; V[0][col]=1; V[4][col]=1; W[0][col]=1; W[4][col]=1
        col+=1
        # M2: (A10+A11)B00 -> C10, -C11
        U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1
        col+=1
        # M3: A00(B01-B11) -> C01, C11
        U[0][col]=1; V[1][col]=1; V[4][col]=-1; W[3][col]=1; W[4][col]=1
        col+=1
        # M4: A11(B10-B00) -> C00, C10
        U[4][col]=1; V[3][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1
        col+=1
        # M5: (A00+A01)B11 -> -C00, C01
        U[0][col]=1; U[1][col]=1; V[4][col]=1; W[0][col]=-1; W[3][col]=1
        col+=1
        # M6: (A10-A00)(B00+B01) -> C11
        U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1
        col+=1
        # M7: (A01-A11)(B10+B11) -> C00
        U[1][col]=1; U[4][col]=-1; V[3][col]=1; V[4][col]=1; W[0][col]=1
        col+=1
        
        # Cross terms
        # 2a: A[0:2,2]*B[2,0:2] -> C[0:2,0:2]  (U:2,5 V:6,7 W:0,3,1,4)
        add_term(2,6,0); add_term(2,7,3); add_term(5,6,1); add_term(5,7,4)
        # 2b: A[0:2,0:2]*B[0:2,2] -> C[0:2,2]  (U:0,1,3,4 V:2,5 W:6,7)
        add_term(0,2,6); add_term(1,5,6); add_term(3,2,7); add_term(4,5,7)
        # 2c: A[0:2,2]*B[2,2] -> C[0:2,2]  (U:2,5 V:8 W:6,7)
        add_term(2,8,6); add_term(5,8,7)
        # 2d: A[2,0:2]*B[0:2,0:2] -> C[2,0:2]  (U:6,7 V:0,1,3,4 W:2,5)
        add_term(6,0,2); add_term(7,3,2); add_term(6,1,5); add_term(7,4,5)
        # 2e: A[2,2]*B[2,0:2] -> C[2,0:2]  (U:8 V:6,7 W:2,5)
        add_term(8,6,2); add_term(8,7,5)
        # 2f: A[2,0:2]*B[0:2,2] -> C[2,2]  (U:6,7 V:2,5 W:8)
        add_term(6,2,8); add_term(7,5,8)
        # 2g: A[2,2]*B[2,2] -> C[2,2]
        add_term(8,8,8)
        
        return U, V, W
    
    U, V, W = build_rank26()
    
    # Convert to numpy for verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that sum_k U[:,k] outer V[:,k] outer W[:,k] = multiplication tensor"""
        # Build the multiplication tensor T[i,j,k,l] = delta_{i,k} * delta_{j,l} ... 
        # Actually T[A_idx, B_idx, C_idx] where C = A*B
        # T[i*n+j, j'*p+k, k'*n+i'] = 1 if i=k' and j=j' and k=i'
        # i.e., C[i',k'] = sum_j A[i,j]*B[j,k']
        # So T[(i,j),(j,k),(i',k')] = delta_{i,i'} * delta_{j,j} * delta_{k,k'}
        # More precisely: sum over j: A[i,j]*B[j,k] = C[i,k]
        # Tensor: T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k'
        
        # Check: for each k, U[:,k] @ V[:,k] @ W[:,k] summed should give T
        # T[a,b,c] where a=i*3+j, b=j'*3+k, c=i'*3+k'
        # T[a,b,c] = 1 if i==i' and j==j' and k==k'
        
        rank = U_mat.shape[1]
        T = np.zeros((9, 9, 9))
        for k in range(rank):
            T += np.einsum('a,b,c->abc', U_mat[:,k], V_mat[:,k], W_mat[:,k])
        
        # Build reference tensor
        T_ref = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3+j
                    b = j*3+k
                    c = i*3+k
                    T_ref[a,b,c] = 1
        
        return np.allclose(T, T_ref, atol=1e-10)
    
    if not verify(U_arr, V_arr, W_arr):
        # Fallback: return schoolbook rank 27
        rank = 27
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i*9 + j*3 + k
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by eliminating one term
    # Strategy: try to express term r as a linear combination of other terms
    # For a valid rank-25 decomposition, one of the 26 terms must be redundant
    
    best_result = {"U": U, "V": V, "W": W}
    current_rank = 26
    
    # Try each term as the one to eliminate
    for elim in range(26):
        if time.time() > deadline - 30:
            break
        
        # Try to find coefficients alpha_k such that
        # U[:,elim] = sum_{k != elim} alpha_k * U[:,k]
        # V[:,elim] = sum_{k != elim} alpha_k * V[:,k]
        # W[:,elim] = sum_{k != elim} alpha_k * W[:,k]
        
        # This is a system of 27 equations (9+9+9) in 25 unknowns (alpha_k for k != elim)
        # Generally overdetermined, but let's try least squares and check
        
        idx = [k for k in range(26) if k != elim]
        U_other = U_arr[:, idx]  # 9 x 25
        V_other = V_arr[:, idx]  # 9 x 25
        W_other = W_arr[:, idx]  # 9 x 25
        
        U_elim = U_arr[:, elim]
        V_elim = V_arr[:, elim]
        W_elim = W_arr[:, elim]
        
        # Stack all equations
        A_stack = np.vstack([U_other, V_other, W_other])  # 27 x 25
        b_stack = np.concatenate([U_elim, V_elim, W_elim])  # 27
        
        # Solve least squares
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        recon = A_stack @ alpha
        if np.allclose(recon, b_stack, atol=1e-10):
            # Found a valid elimination!
            new_rank = 25
            U_new = [U_arr[:, k].tolist() for k in idx]
            U_new = np.array(U_new).T.tolist()
            V_new = [V_arr[:, k].tolist() for k in idx]
            V_new = np.array(V_new).T.tolist()
            W_new = [W_arr[:, k].tolist() for k in idx]
            W_new = np.array(W_new).T.tolist()
            
            # Verify
            U_v = np.array(U_new, dtype=float)
            V_v = np.array(V_new, dtype=float)
            W_v = np.array(W_new, dtype=float)
            if verify(U_v, V_v, W_v):
                # Round to nearest integer/half-integer
                U_int = [[round(x*2)/2 for x in row] for row in U_new]
                V_int = [[round(x*2)/2 for x in row] for row in V_new]
                W_int = [[round(x*2)/2 for x in row] for row in W_new]
                
                # Final verification with rounded values
                U_r = np.array(U_int, dtype=float)
                V_r = np.array(V_int, dtype=float)
                W_r = np.array(W_int, dtype=float)
                if verify(U_r, V_r, W_r):
                    current_rank = new_rank
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
                    break
    
    # If we have time, try to reduce further from rank 25 to 24
    if current_rank == 25 and time.time() < deadline - 60:
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for elim in range(25):
            if time.time() > deadline - 30:
                break
            
            idx = [k for k in range(25) if k != elim]
            U_other = U_arr[:, idx]
            V_other = V_arr[:, idx]
            W_other = W_arr[:, idx]
            
            U_elim = U_arr[:, elim]
            V_elim = V_arr[:, elim]
            W_elim = W_arr[:, elim]
            
            A_stack = np.vstack([U_other, V_other, W_other])
            b_stack = np.concatenate([U_elim, V_elim, W_elim])
            
            try:
                alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
            except:
                continue
            
            recon = A_stack @ alpha
            if np.allclose(recon, b_stack, atol=1e-10):
                new_rank = 24
                U_new = np.array([U_arr[:, k] for k in idx]).T
                V_new = np.array([V_arr[:, k] for k in idx]).T
                W_new = np.array([W_arr[:, k] for k in idx]).T
                
                U_int = [[round(x*2)/2 for x in row] for row in U_new.tolist()]
                V_int = [[round(x*2)/2 for x in row] for row in V_new.tolist()]
                W_int = [[round(x*2)/2 for x in row] for row in W_new.tolist()]
                
                U_r = np.array(U_int, dtype=float)
                V_r = np.array(V_int, dtype=float)
                W_r = np.array(W_int, dtype=float)
                if verify(U_r, V_r, W_r):
                    current_rank = new_rank
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
                    break
    
    # If still have time, try rank 23 via more aggressive search
    # Try removing 2 terms at once from rank 25
    if current_rank == 25 and time.time() < deadline - 45:
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Try pairs of eliminations
        for e1 in range(25):
            if time.time() > deadline - 30:
                break
            for e2 in range(e1+1, 25):
                if time.time() > deadline - 30:
                    break
                
                idx = [k for k in range(25) if k != e1 and k != e2]
                U_other = U_arr[:, idx]
                V_other = V_arr[:, idx]
                W_other = W_arr[:, idx]
                
                U_elim1 = U_arr[:, e1]
                V_elim1 = V_arr[:, e1]
                W_elim1 = W_arr[:, e1]
                U_elim2 = U_arr[:, e2]
                V_elim2 = V_arr[:, e2]
                W_elim2 = W_arr[:, e2]
                
                # 54 equations in 23 unknowns - very overdetermined
                A_stack = np.vstack([U_other, V_other, W_other])
                b_stack = np.concatenate([U_elim1, V_elim1, W_elim1, U_elim2, V_elim2, W_elim2])
                
                try:
                    alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
                except:
                    continue
                
                recon = A_stack @ alpha
                if np.allclose(recon, b_stack, atol=1e-10):
                    new_rank = 23
                    U_new = np.array([U_arr[:, k] for k in idx]).T
                    V_new = np.array([V_arr[:, k] for k in idx]).T
                    W_new = np.array([W_arr[:, k] for k in idx]).T
                    
                    U_int = [[round(x*2)/2 for x in row] for row in U_new.tolist()]
                    V_int = [[round(x*2)/2 for x in row] for row in V_new.tolist()]
                    W_int = [[round(x*2)/2 for x in row] for row in W_new.tolist()]
                    
                    U_r = np.array(U_int, dtype=float)
                    V_r = np.array(V_int, dtype=float)
                    W_r = np.array(W_int, dtype=float)
                    if verify(U_r, V_r, W_r):
                        current_rank = new_rank
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        break
            if current_rank == 23:
                break
    
    return best_result
# EVOLVE-BLOCK-END
