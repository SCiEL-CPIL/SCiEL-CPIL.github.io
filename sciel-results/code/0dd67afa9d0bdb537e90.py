import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using 6 Strassen blocks
    rank = 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2 matrices
    # u_k, v_k are 4-vectors corresponding to A and B entries
    # A = [[a, b], [c, d]] -> index 0, 1, 2, 3
    # B = [[e, f], [g, h]] -> index 0, 1, 2, 3
    
    u_factors = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [0, 0, 1, -1],# 6
        [0, 1, 0, -1] # 7
    ]
    
    v_factors = [
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 1, 0, -1],# 3
        [0, 0, 1, -1],# 4
        [0, 0, 0, 1], # 5
        [1, 1, 0, 0], # 6
        [0, 0, 1, 1]  # 7
    ]
    
    # We need to find w_factors (7 vectors of size 4) such that
    # sum_k u_k \otimes v_k \otimes w_k = T_MM
    # T_MM[i, j, k] = 1 if i == k (in 2x2 sense) and j is the middle index?
    # Tensor T[a, b, c] where a is A index, b is B index, c is C index.
    # C = A B. C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Construct the tensor T of shape (4, 4, 4)
    T = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_idx = i * 2 + j
                b_idx = j * 2 + k
                c_idx = i * 2 + k
                T[a_idx, b_idx, c_idx] = 1.0
                
    # We have 7 equations for each c_idx? No.
    # The equation is: sum_k (u_k[a] * v_k[b]) * w_k[c] = T[a, b, c]
    # This is a linear system for w_k.
    # Let's build the matrix A_sys where each row corresponds to (a, b, c)
    # and each column corresponds to (k, c') where c' is the component of w_k.
    # Actually, for each c, we have 16 equations (a, b).
    # sum_k (u_k[a] * v_k[b]) * w_k[c] = T[a, b, c]
    
    # We can solve for w_k[c] for each c independently?
    # No, w_k is a vector. The sum is over k.
    # For a fixed c, let W_c be the vector of w_k[c] for k=1..7.
    # The equation is: sum_k M_{ab}^{(k)} W_{c,k} = T_{abc}
    # where M_{ab}^{(k)} = u_k[a] * v_k[b].
    
    # Let's form a 16x7 matrix M_c? No, M is the same for all c.
    # M is 16x7. Rows are (a,b), cols are k.
    # M[a*4+b, k] = u_k[a] * v_k[b]
    
    M = np.zeros((16, 7))
    for a in range(4):
        for b in range(4):
            for k in range(7):
                M[a * 4 + b, k] = u_factors[k][a] * v_factors[k][b]
                
    # We want to find W_mat (7x4) such that M @ W_mat[:, c] = T[:, :, c].ravel()
    # T is 4x4x4. T[:, :, c] is 4x4.
    
    W_factors = np.zeros((7, 4))
    for c in range(4):
        rhs = T[:, :, c].ravel()
        # Solve M @ w_vec = rhs
        # M is 16x7. Overdetermined. Use least squares.
        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
        W_factors[:, c] = w_vec
        
    # Verify the solution
    # Reconstruct T_recon
    T_recon = np.zeros((4, 4, 4))
    for k in range(7):
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    T_recon[a, b, c] += u_factors[k][a] * v_factors[k][b] * W_factors[k, c]
                    
    if not np.allclose(T_recon, T):
        # Fallback to schoolbook if Strassen decomposition fails
        rank = 48
        U = np.zeros((4, 48))
        V = np.zeros((24, 48))
        W = np.zeros((24, 48))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(12):
                    U[i*2+j, col] = 1
                    V[j*12+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct U, V, W for rank 42
    # 6 blocks
    for r in range(6):
        for k in range(7):
            col_idx = r * 7 + k
            
            # U: same for all blocks
            U[:, col_idx] = u_factors[k]
            
            # V: block r corresponds to columns 2r, 2r+1 of B
            # B indices in V are j*p + k.
            # j is row of B (0, 1). k is col of B (0..11).
            # For block r, cols are 2r, 2r+1.
            # v_factors[k] has 4 entries: b00, b01, b10, b11
            # b00 -> V[0*12 + 2r, col]
            # b01 -> V[0*12 + 2r+1, col]
            # b10 -> V[1*12 + 2r, col]
            # b11 -> V[1*12 + 2r+1, col]
            
            V[0 * 12 + 2 * r, col_idx] = v_factors[k][0]
            V[0 * 12 + 2 * r + 1, col_idx] = v_factors[k][1]
            V[1 * 12 + 2 * r, col_idx] = v_factors[k][2]
            V[1 * 12 + 2 * r + 1, col_idx] = v_factors[k][3]
            
            # W: block r corresponds to cols 2r, 2r+1 of C
            # C indices in W are k*n + i.
            # k is col of C (0..11). i is row of C (0, 1).
            # W_factors[k] has 4 entries: c00, c01, c10, c11
            # c00 -> W[2r*2 + 0, col]
            # c01 -> W[(2r+1)*2 + 0, col]
            # c10 -> W[2r*2 + 1, col]
            # c11 -> W[(2r+1)*2 + 1, col]
            
            W[2 * r * 2 + 0, col_idx] = W_factors[k, 0]
            W[(2 * r + 1) * 2 + 0, col_idx] = W_factors[k, 1]
            W[2 * r * 2 + 1, col_idx] = W_factors[k, 2]
            W[(2 * r + 1) * 2 + 1, col_idx] = W_factors[k, 3]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
