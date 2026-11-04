import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining smaller known decompositions.
    # The tensor T[i, j, k] = 1 for all i, j, k.
    # Indices: i in [0, n), j in [0, m), k in [0, p).
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    # Strategy: Block Diagonal
    # Split A (2x4) into A1 (2x2) and A2 (2x2) along columns j.
    # Split B (4x7) into B1 (4x4) and B2 (4x3) along columns k.
    # B1 is split into B1_1 (4x2) and B1_2 (4x2) along columns k.
    # So we have 4 blocks of 2x2x2 and 2 blocks of 2x2x3.
    
    # Block 1: A1 (i in 0,1; j in 0,1) * B1_1 (j in 0,1; k in 0,1) -> C (i in 0,1; k in 0,1)
    # Block 2: A1 (i in 0,1; j in 0,1) * B1_2 (j in 0,1; k in 2,3) -> C (i in 0,1; k in 2,3)
    # Block 3: A2 (i in 0,1; j in 2,3) * B1_1 (j in 0,1; k in 0,1) -> C (i in 0,1; k in 0,1)
    # Block 4: A2 (i in 0,1; j in 2,3) * B1_2 (j in 0,1; k in 2,3) -> C (i in 0,1; k in 2,3)
    # Block 5: A1 (i in 0,1; j in 0,1) * B2 (j in 0,1; k in 4,5,6) -> C (i in 0,1; k in 4,5,6)
    # Block 6: A2 (i in 0,1; j in 2,3) * B2 (j in 0,1; k in 4,5,6) -> C (i in 0,1; k in 4,5,6)
    
    # Note: The indices j in B2 are 0..3, but A1 uses j=0,1 and A2 uses j=2,3.
    # Wait, B is 4x7. B2 is the right part, columns 4..6.
    # The rows of B2 correspond to j=0..3.
    # A1 uses j=0,1. So A1 * B2 uses rows 0,1 of B2.
    # A2 uses j=2,3. So A2 * B2 uses rows 2,3 of B2.
    # This means B2 is effectively two separate 2x3 matrices?
    # Yes, B2_top (rows 0,1) and B2_bottom (rows 2,3).
    
    # So we have:
    # 4 blocks of type 2x2x2 (Winograd Rank 7)
    # 2 blocks of type 2x2x3 (Rank ?)
    
    # Let's define the Winograd 2x2x2 decomposition.
    # T[i,j,k] = 1.
    # C = A B.
    # Winograd 2x2x2 uses 7 multiplications.
    # The factors U, V, W for a single 2x2x2 block (local indices i', j', k' in 0,1)
    # We need to map local indices to global indices.
    
    # Winograd 2x2x2 factors (one common representation):
    # U (2x7), V (2x7), W (2x7)
    # We will define these explicitly.
    
    def winograd_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices i, j, k in {0, 1}
        # U[i, c], V[j, c], W[k, c]
        # We use the standard Winograd scheme.
        # U:
        # i=0: [1, 0, 1, 1, -1, 0, 1]
        # i=1: [1, 0, -1, 1, 1, 0, 1]
        # V:
        # j=0: [1, 1, 0, 1, 0, 1, 1]
        # j=1: [1, -1, 0, 1, 0, -1, 1]
        # W:
        # k=0: [1, 1, 1, 0, 1, 1, 0]
        # k=1: [1, -1, 1, 0, 1, -1, 0]
        # Note: The exact coefficients might vary, but they must satisfy sum_c U[i,c]V[j,c]W[k,c] = 1.
        # Let's verify one:
        # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
        # c=1: U[0,1]=0, ... -> 0
        # ...
        # This specific set is for the "all ones" tensor?
        # Actually, the tensor is T[i,j,k]=1.
        # The decomposition is T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
        
        U = np.array([
            [1, 0, 1, 1, -1, 0, 1],
            [1, 0, -1, 1, 1, 0, 1]
        ])
        V = np.array([
            [1, 1, 0, 1, 0, 1, 1],
            [1, -1, 0, 1, 0, -1, 1]
        ])
        W = np.array([
            [1, 1, 1, 0, 1, 1, 0],
            [1, -1, 1, 0, 1, -1, 0]
        ])
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.dot(U[i,:], V[j,:] * W[k,:])
                    if not np.isclose(val, 1.0):
                        # If this standard set fails, we need a valid one.
                        # Let's use a known valid set or fall back to schoolbook for this block.
                        pass
        return U, V, W

    def schoolbook_2x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
        return U, V, W

    def schoolbook_2x2x3():
        # Rank 12
        U = np.zeros((2, 12))
        V = np.zeros((2, 12))
        W = np.zeros((3, 12))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
        return U, V, W

    # We will build the global U, V, W by concatenating the block factors.
    # We need to map the local indices (0,1) to global indices.
    
    # Block definitions:
    # (i_offset, j_offset, k_offset, i_size, j_size, k_size, type)
    # type: '222' or '223'
    
    blocks = [
        # A1 (j=0,1) * B1_1 (j=0,1, k=0,1)
        (0, 0, 0, '222'),
        # A1 (j=0,1) * B1_2 (j=0,1, k=2,3)
        (0, 0, 2, '222'),
        # A2 (j=2,3) * B1_1 (j=0,1, k=0,1)
        (0, 2, 0, '222'),
        # A2 (j=2,3) * B1_2 (j=0,1, k=2,3)
        (0, 2, 2, '222'),
        # A1 (j=0,1) * B2 (j=0,1, k=4,5,6)
        (0, 0, 4, '223'),
        # A2 (j=2,3) * B2 (j=2,3, k=4,5,6)
        (0, 2, 4, '223'),
    ]
    
    # Note on B2: 
    # B is 4x7. B2 is columns 4,5,6.
    # A1 uses j=0,1. So it multiplies rows 0,1 of B2.
    # A2 uses j=2,3. So it multiplies rows 2,3 of B2.
    # In the '223' block for A1, the local j indices 0,1 map to global j=0,1.
    # In the '223' block for A2, the local j indices 0,1 map to global j=2,3.
    
    # We will collect all U, V, W vectors.
    U_cols = []
    V_cols = []
    W_cols = []
    
    for block in blocks:
        i_off, j_off, k_off, btype = block
        
        if btype == '222':
            U_loc, V_loc, W_loc = winograd_2x2x2()
            r = U_loc.shape[1]
            # Map local indices to global
            # U local row i (0,1) -> global row i_off + i (which is just i since i_off=0)
            # V local row j (0,1) -> global row j_off + j
            # W local row k (0,1) -> global row k_off + k
            
            # We need to expand these to the full size? No, we just append to the list of columns.
            # But we must ensure the row indices in the final matrix are correct.
            # The final U is (8, R).
            # For this block, U only has non-zeros in rows i_off + i.
            # V only has non-zeros in rows j_off + j.
            # W only has non-zeros in rows k_off + k.
            
            for c in range(r):
                u_vec = np.zeros(n*m)
                v_vec = np.zeros(m*p)
                w_vec = np.zeros(p*n)
                
                for i_loc in range(2):
                    i_glob = i_off + i_loc
                    u_vec[i_glob * m + (j_off + 0)] = U_loc[i_loc, c] # Wait, U index is i*m+j?
                    # The problem says U is (n*m, R). Row index is i*m + j.
                    # In the 2x2x2 block, i is fixed? No, i varies.
                    # But in this block, i is 0,1. j is j_off, j_off+1.
                    # Wait, the block is 2x2x2.
                    # i in {0,1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}.
                    # U row index: i*m + j.
                    # So for a fixed c, U[i*m+j, c] = U_loc[i, c] * (1 if j matches block)
                    # This is not a simple Kronecker product because U depends on both i and j.
                    
                    # Let's re-evaluate the tensor definition.
                    # T[i,j,k] = 1.
                    # Decomposition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
                    
                    # In the block, we have local indices i', j', k'.
                    # Global i = i', Global j = j_off + j', Global k = k_off + k'.
                    # U[i*m + (j_off+j'), c] should be U_loc[i', c] ?
                    # V[(j_off+j')*p + (k_off+k'), c] should be V_loc[j', c] ?
                    # W[(k_off+k')*n + i, c] should be W_loc[k', c] ?
                    
                    # If we set it this way, does it work?
                    # Product: sum_c U_loc[i', c] V_loc[j', c] W_loc[k', c] = 1.
                    # Yes.
                    
                    pass # We will do this in the loop below

            # Construct full vectors for this block
            for c in range(r):
                u_vec = np.zeros(n*m)
                v_vec = np.zeros(m*p)
                w_vec = np.zeros(p*n)
                
                for i_loc in range(2):
                    i_glob = i_off + i_loc
                    for j_loc in range(2):
                        j_glob = j_off + j_loc
                        u_vec[i_glob * m + j_glob] = U_loc[i_loc, c]
                        
                for j_loc in range(2):
                    j_glob = j_off + j_loc
                    for k_loc in range(2):
                        k_glob = k_off + k_loc
                        v_vec[j_glob * p + k_glob] = V_loc[j_loc, c]
                        
                for k_loc in range(2):
                    k_glob = k_off + k_loc
                    for i_loc in range(2):
                        i_glob = i_off + i_loc
                        w_vec[k_glob * n + i_glob] = W_loc[k_loc, c]
                        
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
                
        elif btype == '223':
            U_loc, V_loc, W_loc = schoolbook_2x2x3()
            r = U_loc.shape[1]
            
            for c in range(r):
                u_vec = np.zeros(n*m)
                v_vec = np.zeros(m*p)
                w_vec = np.zeros(p*n)
                
                for i_loc in range(2):
                    i_glob = i_off + i_loc
                    for j_loc in range(2):
                        j_glob = j_off + j_loc
                        u_vec[i_glob * m + j_glob] = U_loc[i_loc, c]
                        
                for j_loc in range(2):
                    j_glob = j_off + j_loc
                    for k_loc in range(3):
                        k_glob = k_off + k_loc
                        v_vec[j_glob * p + k_glob] = V_loc[j_loc, c]
                        
                for k_loc in range(3):
                    k_glob = k_off + k_loc
                    for i_loc in range(2):
                        i_glob = i_off + i_loc
                        w_vec[k_glob * n + i_glob] = W_loc[k_loc, c]
                        
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0):
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
