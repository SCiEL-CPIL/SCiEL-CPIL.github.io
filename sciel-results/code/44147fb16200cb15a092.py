import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Strassen matrices for M_{2,2,2}
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook matrices for M_{2,2,1}
    U_school = np.eye(4)
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    def build_solution(rank_blocks):
        # rank_blocks is a list of (U, V, W) tuples for each block
        U = np.zeros((n * m, sum(u.shape[1] for u, v, w in rank_blocks)))
        V = np.zeros((m * p, sum(u.shape[1] for u, v, w in rank_blocks)))
        W = np.zeros((p * n, sum(u.shape[1] for u, v, w in rank_blocks)))
        
        col_offset = 0
        for idx, (U_b, V_b, W_b) in enumerate(rank_blocks):
            r_b = U_b.shape[1]
            # Map block indices to global indices
            # Block idx covers k indices
            # For M_{2,2,2}, k covers 2 indices.
            # For M_{2,2,1}, k covers 1 index.
            
            # We need to know the k-range for this block
            # This function is generic, so we pass the k-range?
            # Let's refactor to pass k_start, k_end
            
        # This helper is tricky because of index mapping.
        # Better to build directly.
        return None

    # Direct construction for Rank 53
    def get_rank_53():
        U = np.zeros((n * m, 53))
        V = np.zeros((m * p, 53))
        W = np.zeros((p * n, 53))
        
        col_offset = 0
        for block in range(6):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][c] = U_strassen[i * 2 + j][r]
                for j in range(2):
                    V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                    V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk * n + i][c] = W_strassen[k * 2 + i][r]
            col_offset += 7
        
        k_last = 12
        for r in range(4):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_school[i * 2 + j][r]
            for j in range(2):
                V[j * p + k_last][c] = V_school[j][r]
            for i in range(2):
                W[k_last * n + i][c] = W_school[i][r]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find Rank 10 for M_{2,2,3}
    def find_m223_rank10():
        # U: 4x10, V: 6x10, W: 6x10
        # Indices: i,j in 0..1, k in 0..2
        # Equations: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        best_solution = None
        start_time = time.time()
        
        for attempt in range(1000):
            if time.time() - start_time > 10:
                break
            
            # Random initialization with integers in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(4, 10))
            V_rand = np.random.randint(-1, 2, size=(6, 10))
            
            # Compute A[i,k,r] = sum_j U[i,j,r] V[j,k,r]
            # i: 0..1, k: 0..2, r: 0..9
            # j: 0..1
            A = np.zeros((2, 3, 10))
            for i in range(2):
                for k in range(3):
                    for r in range(10):
                        A[i, k, r] = U_rand[i*2+0, r] * V_rand[0*3+k, r] + U_rand[i*2+1, r] * V_rand[1*3+k, r]
            
            # Solve for W
            # For each i, k: sum_r A[i,k,r] W[k,i,r] = 1
            # W is 6x10. W[k,i,r].
            # k: 0..2, i: 0..1.
            # We have 6 equations.
            # Unknowns are W[k,i,r].
            # We can set most W to 0 and solve for a few?
            # Or solve the underdetermined system.
            
            # Let's fix W to be sparse.
            # Choose 6 non-zero entries in W to satisfy the 6 equations.
            # This is a combinatorial search.
            
            # Alternative: Use np.linalg.lstsq on a constructed system.
            # We want W such that for all i,k: dot(A[i,k,:], W[k,i,:]) = 1.
            # Let's pick 6 random columns r for each (i,k) pair? No, r is shared.
            
            # Let's try to find W by setting W[k,i,r] = 0 for most r.
            # Pick 6 distinct r indices for the 6 equations.
            # This might not work if A is singular.
            
            # Simpler: Use scipy to solve the linear system for W entries.
            # Flatten W to 60 variables.
            # 6 equations.
            # We can add a regularization to keep values small.
            
            # Construct matrix M (6x60)
            M = np.zeros((6, 60))
            b = np.ones(6)
            
            idx = 0
            for i in range(2):
                for k in range(3):
                    for r in range(10):
                        M[idx, k*2 + i + r*6] = A[i, k, r] # W[k,i,r] is at index k*2+i + r*6?
                        # W shape (6, 10). Index (k*2+i, r).
                        # Flat index: (k*2+i)*10 + r.
                        M[idx, (k*2+i)*10 + r] = A[i, k, r]
                    idx += 1
            
            # Solve M W_flat = b
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                W_candidate = W_flat.reshape(6, 10)
                
                # Check if close to integers/half-integers
                # Multiply by 2 to check for integers
                W_scaled = W_candidate * 2
                W_rounded = np.round(W_scaled)
                
                # Check error
                err = np.sum((W_scaled - W_rounded)**2)
                if err < 1e-6:
                    W_int = W_rounded / 2.0
                    # Verify
                    valid = True
                    for i in range(2):
                        for k in range(3):
                            for j in range(2):
                                # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                                s = np.sum(U_rand[i*2+j, :] * V_rand[j*3+k, :] * W_int[k*2+i, :])
                                if abs(s - 1.0) > 1e-6:
                                    valid = False
                                    break
                        if not valid: break
                    if valid:
                        return U_rand.tolist(), V_rand.tolist(), W_int.tolist()
                        
            except Exception:
                pass

        return None

    # Attempt to find Rank 52 solution
    if time.time() < deadline - 20:
        m223_sol = find_m223_rank10()
        if m223_sol is not None:
            U_m, V_m, W_m = m223_sol
            U = np.zeros((n * m, 52))
            V = np.zeros((m * p, 52))
            W = np.zeros((p * n, 52))
            
            # Fill first 6 blocks (42 cols)
            col_offset = 0
            for block in range(6):
                k0 = 2 * block
                k1 = 2 * block + 1
                for r in range(7):
                    c = col_offset + r
                    for i in range(2):
                        for j in range(2):
                            U[i * m + j][c] = U_strassen[i * 2 + j][r]
                    for j in range(2):
                        V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                        V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
                    for k in range(2):
                        kk = k0 + k
                        for i in range(2):
                            W[kk * n + i][c] = W_strassen[k * 2 + i][r]
                col_offset += 7
            
            # Fill last block (10 cols)
            for r in range(10):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][c] = U_m[i * 2 + j][r]
                for j in range(2):
                    for k in range(3):
                        V[j * p + (12 + k)][c] = V_m[j * 3 + k][r]
                for k in range(3):
                    for i in range(2):
                        W[(12 + k) * n + i][c] = W_m[k * 2 + i][r]
                        
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback Rank 53
    return get_rank_53()
