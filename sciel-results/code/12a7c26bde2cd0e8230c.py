"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # We want to check C_{i,k} = sum_j A_{i,j} B_{j,k} = 1 for all i,j,k?
        # No, the tensor is T_{i,j,k} = 1.
        # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
        
        # We can compute this efficiently.
        # For each c, we have u_c (n*m), v_c (m*p), w_c (p*n).
        # The contribution to T is u_c * v_c * w_c (outer product structure).
        # T = sum_c u_c \otimes v_c \otimes w_c.
        
        # We can check by iterating over c and adding to a 3D array.
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # T_{i,j,k} += u_c[i,j] * v_c[j,k] * w_c[k,i]
            # This is a bit tricky to vectorize directly as a sum.
            # T_{i,j,k} = sum_c u_c[i,j] v_c[j,k] w_c[k,i]
            
            # Let's just do the loop, it's fast enough for rank 44.
            pass
        
        # The original loop is fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    # Strassen-based block decomposition for 3x4 x 4x4
    # Split A (3x4) into row blocks (2,1) and col blocks (2,2)
    # Split B (4x4) into row blocks (2,2) and col blocks (2,2)
    # 2x2x2 blocks use Strassen (rank 7), 1x2x2 blocks use rank-4
    
    rank = 44
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))
        
    def add_strassen(a,b,c,d, e,f,g,h, w11,w12,w21,w22):
        """
        Adds 7 terms for 2x2x2 Strassen decomposition.
        A block: [[a,b],[c,d]], B block: [[e,f],[g,h]], C block: [[w11,w12],[w21,w22]]
        """
        # M1 = (a+d)(e+h) -> C11(+1), C22(+1)
        add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
        # M2 = (c+d)e -> C21(+1), C22(-1)
        add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
        # M3 = a(f-h) -> C12(+1), C22(+1)
        add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
        # M4 = d(g-e) -> C11(+1), C21(+1)
        add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
        # M5 = (a+b)h -> C12(+1), C11(-1)
        add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
        # M6 = (c-a)(e+f) -> C22(+1)
        add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
        # M7 = (b-d)(g+h) -> C11(+1)
        add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])
        
    def add_rank4(a,b, c,d,e,f, w1,w2):
        """
        Adds 4 terms for 1x2 x 2x2 multiplication.
        A row: [a,b], B: [[c,d],[e,f]], C row: [w1,w2]
        C1 = a*c + b*e, C2 = a*d + b*f
        """
        add_term([(a,1)], [(c,1)], [(w1,1)])
        add_term([(b,1)], [(e,1)], [(w1,1)])
        add_term([(a,1)], [(d,1)], [(w2,1)])
        add_term([(b,1)], [(f,1)], [(w2,1)])

    # A (3x4) indices: row i, col j -> i*4+j
    # B (4x4) indices: row j, col k -> j*4+k
    # C (3x4) -> W index: k*3+i (trace convention W[k*n+i])
    
    # Case 1: A[R0,1,C0,1] * B[R0,1,C0,1] -> C[R0,1,C0,1]
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    
    # Case 2: A[R0,1,C0,1] * B[R0,1,C2,3] -> C[R0,1,C2,3]
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    
    # Case 3: A[R2,C0,1] * B[R0,1,C0,1] -> C[R2,C0,1]
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    
    # Case 4: A[R2,C0,1] * B[R0,1,C2,3] -> C[R2,C2,3]
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    
    # Case 5: A[R0,1,C2,3] * B[R2,3,C0,1] -> C[R0,1,C0,1]
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    
    # Case 6: A[R0,1,C2,3] * B[R2,3,C2,3] -> C[R0,1,C2,3]
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    
    # Case 7: A[R2,C2,3] * B[R2,3,C0,1] -> C[R2,C0,1]
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    
    # Case 8: A[R2,C2,3] * B[R2,3,C2,3] -> C[R2,C2,3]
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)
    
    # Fill U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if verify_decomposition(U_np, V_np, W_np, rank):
        best_U, best_V, best_W = U_np, V_np, W_np
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
