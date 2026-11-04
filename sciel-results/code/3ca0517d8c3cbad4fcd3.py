import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find a better rank using structure
    # 2x3x6 = 2 * (2x3x3)
    # If we find rank R for 2x3x3, we can get rank 2R for 2x3x6.
    # Schoolbook for 2x3x3 is 18. 2*18=36.
    # Target rank for 2x3x3: 14 (if it exists) -> 28 total.
    
    def find_decomposition_233(target_rank):
        n2, m2, p2 = 2, 3, 3
        U = np.random.randn(n2 * m2, target_rank)
        V = np.random.randn(m2 * p2, target_rank)
        W = np.zeros((p2 * n2, target_rank))
        
        # Target tensor T for 2x3x3
        # T[i*m2+j, j*p2+k, k*n2+i] = 1
        # We want sum_t U[:,t] V[:,t] W[:,t] = T
        # This is equivalent to:
        # For each (i, j, k), sum_t U[i*m2+j, t] V[j*p2+k, t] W[k*n2+i, t] = 1 if (i,j,k) valid, 0 else.
        # Actually, the tensor is sparse.
        
        # We can solve for W given U, V.
        # The equation is linear in W entries.
        # Let A_t = U[:,t] (size 6), B_t = V[:,t] (size 9).
        # We want sum_t A_t B_t^T W_t = T? No.
        # The product is outer product of vectors?
        # The tensor is sum_t u_t v_t^T w_t^T? No, it's a 3-way sum.
        # The condition is:
        # sum_t U_{ij,t} V_{jk,t} W_{ki,t} = delta_{j,j'} ...
        # Let's reshape U, V, W to matrices.
        # U_t (2x3), V_t (3x3), W_t (3x2).
        # We want sum_t U_t V_t W_t = I? No.
        # The multiplication is C = A B.
        # The decomposition is C = sum_t (A U_t) (V_t^T B) W_t^T?
        # Let's check the trace: tr(A B C) = sum_t tr(A U_t) tr(V_t^T B) tr(W_t^T C)? No.
        
        # Correct form:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # = sum_t (sum_j A_{ij} U_{t,ij}) (sum_j B_{jk} V_{t,jk}) W_{t,ki}
        # Let x_t = A U_t (vector of size n? No, U_t is n x m, A is n x m. A U_t is not defined).
        # U_t is a matrix. A U_t is not a scalar.
        # The term is (A * U_t) where * is element-wise product sum?
        # Yes, <A, U_t> is a scalar.
        # So C_{ik} = sum_t <A, U_t> <B, V_t> W_{t,ki}.
        
        # We want this to equal sum_j A_{ij} B_{jk}.
        # This must hold for all A, B.
        # This implies that the tensor sum_t U_t \otimes V_t \otimes W_t must match the multiplication tensor.
        
        # To solve for W given U, V:
        # For each (i, k), we have:
        # sum_t W_{t,ki} <A, U_t> <B, V_t> = sum_j A_{ij} B_{jk}
        # This is a system of linear equations for W_{t,ki}.
        # The variables are W_{t,ki} for t=1..R, k=1..p, i=1..n.
        # The equations are indexed by (A, B). But A, B are continuous.
        # We can pick specific basis vectors for A and B.
        # Let A = E_{ij}, B = E_{jk}.
        # Then <A, U_t> = U_{t,ij}, <B, V_t> = V_{t,jk}.
        # Equation: sum_t W_{t,ki} U_{t,ij} V_{t,jk} = delta_{j,j'}? No.
        # If A=E_{i0,j0}, B=E_{j1,k0}, then RHS is 1 if j0=j1, else 0.
        # So for each i, k, and for each pair (j0, j1):
        # sum_t W_{t,ki} U_{t,i,j0} V_{t,j1,k} = delta_{j0,j1}
        
        # This is a linear system for W.
        # Number of unknowns: R * p * n.
        # Number of equations: n * p * m * m.
        
        def solve_W(U_mat, V_mat, R, n, m, p):
            # U_mat: (n*m, R), V_mat: (m*p, R)
            # W_mat: (p*n, R)
            num_eqs = n * p * m * m
            num_vars = R * p * n
            
            # We can solve this using least squares if num_vars < num_eqs?
            # Or we can just set up the matrix.
            
            # Instead of building a huge matrix, we can iterate.
            # But for R=14, n=2, m=3, p=3:
            # num_vars = 14 * 3 * 2 = 84.
            # num_eqs = 2 * 3 * 3 * 3 = 54.
            # Underdetermined. We can find a solution.
            
            # Let's construct the system A x = b.
            # x is a vector of W entries.
            # Index x by (t, k, i).
            # Equation index by (i, k, j0, j1).
            # LHS: sum_t W_{t,ki} U_{t,i,j0} V_{t,j1,k}
            # RHS: delta_{j0,j1}
            
            # We can use np.linalg.lstsq.
            
            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j0 in range(m):
                        for j1 in range(m):
                            # RHS
                            if j0 == j1:
                                b_vec[eq_idx] = 1
                            
                            # LHS
                            for t in range(R):
                                var_idx = t * (p * n) + k * n + i
                                # U_{t, i, j0} is U_mat[i*m + j0, t]
                                # V_{t, j1, k} is V_mat[j1*p + k, t]
                                A_mat[eq_idx, var_idx] = U_mat[i*m + j0, t] * V_mat[j1*p + k, t]
                            
                            eq_idx += 1
            
            # Solve
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_mat = sol.reshape((R, p, n)).transpose((1, 2, 0))
                W_mat = W_mat.reshape((p * n, R))
                return W_mat
            except:
                return None

        # ALS Loop
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Solve for W
            W = solve_W(U, V, target_rank, n2, m2, p2)
            if W is None:
                continue
            
            # Check error
            # Compute tensor and compare
            err = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = 0
                        for t in range(target_rank):
                            val += U[i*m2+j, t] * V[j*p2+k, t] * W[k*n2+i, t]
                        if i == i and j == j and k == k: # Always true
                            # The target is 1 for all (i,j,k) in the support?
                            # The multiplication tensor has 1 at (i,j,k) if the indices match the multiplication flow.
                            # Yes, for 2x3x3, all (i,j,k) are valid?
                            # A is 2x3, B is 3x3, C is 2x3.
                            # C_{ik} = sum_j A_{ij} B_{jk}.
                            # The tensor T_{i,j,k} is 1 if it contributes to C_{ik}.
                            # Yes, T_{i,j,k} = 1 for all i, j, k.
                            # Wait, is it?
                            # T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}.
                            # So yes, all entries are 1.
                            pass
                        
                        # Actually, we just need to check the reconstruction.
                        # Let's compute the max error over all (i,k) for random A,B?
                        # Or just check the tensor equality.
                        # T_{ijk} = 1.
                        # So we want sum_t ... = 1.
                        err += (val - 1)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-8:
                break
            
            # Update U, V similarly?
            # For simplicity, just re-randomize if stuck?
            # Or solve for U given V, W.
            # Let's just do a few iterations of re-randomization if it fails.
            U = np.random.randn(n2 * m2, target_rank)
            V = np.random.randn(m2 * p2, target_rank)
            
        if best_err < 1e-8:
            return best_U, best_V, best_W
        else:
            return None

    # Try to find rank 14 for 2x3x3
    U_233, V_233, W_233 = find_decomposition_233(14)
    
    if U_233 is not None:
        # Construct rank 28 for 2x3x6
        # We have two blocks of 3x3.
        # B = [B1, B2].
        # C = [A B1, A B2].
        # Use U_233, V_233, W_233 for both.
        
        U_new = np.zeros((n * m, 28))
        V_new = np.zeros((m * p, 28))
        W_new = np.zeros((p * n, 28))
        
        # Block 1 (k=0..2)
        # V indices: j*p + k. For k=0..2, it's j*3 + k.
        # W indices: k*n + i. For k=0..2, it's k*2 + i.
        
        U_new[:, 0:14] = U_233
        V_new[:, 0:14] = V_233
        W_new[:, 0:14] = W_233
        
        # Block 2 (k=3..5)
        # V indices: j*p + k. For k=3..5, it's j*3 + (3+r) = j*3 + 3 + r.
        # This is not contiguous in the same way?
        # V is size 3x6.
        # V[j, 0..2] corresponds to B1.
        # V[j, 3..5] corresponds to B2.
        # In the vector form V_new (size 18x28):
        # Row index for V is j*6 + k.
        # For B1 (k=0..2), row is j*6 + k.
        # For B2 (k=3..5), row is j*6 + 3 + r.
        
        # Wait, my V_233 is for 3x3.
        # V_233 row index is j*3 + k.
        # We need to map this to V_new.
        
        # Let's re-construct carefully.
        # V_new has shape (18, 28).
        # First 14 cols: correspond to B1.
        # For col t in 0..13:
        #   For j in 0..2, k in 0..2:
        #     V_new[j*6 + k, t] = V_233[j*3 + k, t]
        
        # Last 14 cols: correspond to B2.
        # For col t in 14..27:
        #   For j in 0..2, k in 0..2:
        #     V_new[j*6 + 3 + k, t+14] = V_233[j*3 + k, t]
        
        # U_new:
        # U is 2x3. Same for both blocks.
        # U_new[:, 0:14] = U_233
        # U_new[:, 14:28] = U_233
        
        # W_new:
        # W is 3x2 (p x n).
        # W_233 is 3x2.
        # W_new is 6x2 (p x n).
        # W_new row index is k*2 + i.
        # For B1 (k=0..2), W rows are 0..5.
        # For B2 (k=3..5), W rows are 6..11.
        
        # W_new[0:6, 0:14] = W_233
        # W_new[6:12, 14:28] = W_233
        
        # Let's fill it
        for t in range(14):
            for j in range(3):
                for k in range(3):
                    V_new[j*6 + k, t] = V_233[j*3 + k, t]
                    V_new[j*6 + 3 + k, t + 14] = V_233[j*3 + k, t]
            
            U_new[:, t] = U_233[:, t]
            U_new[:, t + 14] = U_233[:, t]
            
            W_new[0:6, t] = W_233[:, t]
            W_new[6:12, t + 14] = W_233[:, t]
            
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

