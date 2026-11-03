"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term."""
        U_rows.append(list(u_vec))
        V_rows.append(list(v_vec))
        W_rows.append(list(w_vec))
    
    def zeros(length):
        return [0] * length
    
    # Strassen-like decomposition for 2x4 * 4x2 -> 2x2
    # A is 2x4 (rows i=0,1; cols j=0..3)
    # B is 4x2 (rows j=0..3; cols k=0,1)
    # C is 2x2 (rows i=0,1; cols k=0,1)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split A into left half A_L (2x2, cols 0,1) and right half A_R (2x2, cols 2,3)
    # Split B into top half B_T (2x2, rows 0,1) and bottom half B_B (2x2, rows 2,3)
    
    # C = A_L * B_T + A_R * B_B
    # Use Strassen on each 2x2*2x2 product, but we can share some computations
    
    # Actually, let's use a 7-term decomposition for the whole 2x4*4x2:
    # M1 = (A_L + A_R) * (B_T + B_B)
    # M2 = (A_L - A_R) * (B_T + B_B)
    # M3 = A_L * (B_T - B_B)
    # M4 = A_R * (B_T - B_B)
    # M5 = (A_L + A_R) * B_T
    # M6 = (A_L - A_R) * B_B
    # M7 = A_L * B_T  (or some other combination)
    
    # Let me use the standard Strassen on two separate 2x2 products:
    # C = A_L * B_T + A_R * B_B
    # For A_L * B_T (2x2 * 2x2): 7 terms
    # For A_R * B_B (2x2 * 2x2): 7 terms
    # Total: 14 terms
    
    # But we can do better by sharing. Let me use a different approach.
    
    # For 2x4 * 4x2, we can use the following 7-term decomposition:
    # Let A = [A11 A12 A13 A14] (row 0), [A21 A22 A23 A24] (row 1)
    # Let B = [B11 B12; B21 B22; B31 B32; B41 B42] (rows 0-3, cols 0-1)
    
    # C[0,0] = A11*B11 + A12*B21 + A13*B31 + A14*B41
    # C[0,1] = A11*B12 + A12*B22 + A13*B32 + A14*B42
    # C[1,0] = A21*B11 + A22*B21 + A23*B31 + A24*B41
    # C[1,1] = A21*B12 + A22*B22 + A23*B32 + A24*B42
    
    # This is just 4 dot products of length 4, so rank 4 is enough!
    # Each C[i,k] is a dot product, and they share the same A and B vectors.
    
    # So for 2x4 * 4x2, rank 4 suffices (one rank-1 term per output entry).
    
    # Let me reconsider the whole problem.
    
    # For 3x4 * 4x7:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # This is 3*7 = 21 dot products of length 4.
    # Naive rank: 21 (one per output entry)
    # But we can do better using structure.
    
    # Actually, the schoolbook rank is n*m*p = 3*4*7 = 84.
    # The "one per output entry" approach gives rank n*p = 3*7 = 21.
    # Wait, that's not right either. Let me think again.
    
    # In the tensor decomposition T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the convention W[k*n+i, r])
    
    # For a single output C[i,k] = sum_j A[i,j] * B[j,k]:
    # We need terms where U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # for each j. So for fixed (i,k), we need m=4 terms.
    # Total: n*p*m = 3*7*4 = 84 terms (schoolbook).
    
    # To reduce rank, we need to share terms across different (i,k) pairs.
    
    # Key insight: If we group columns of B, we can use Strassen-like tricks.
    
    # Let me try: Split the 7 columns of B into groups of 2,2,2,1.
    # For each group of 2 columns, and for each pair of rows of A (2 rows at a time),
    # we can use a 2x4 * 4x2 decomposition with rank 4 (as shown above).
    
    # But we have 3 rows of A. Let's split into (row 0, row 1) and (row 2).
    
    # Part 1: Rows 0,1 of A (2x4) times all 7 columns of B (4x7)
    #   Split B's 7 columns into 2+2+2+1
    #   For each 4x2 block of B: 2x4 * 4x2 -> rank 4
    #   3 such blocks: 3*4 = 12
    #   For the remaining 4x1 block: 2x4 * 4x1 -> rank 2 (two dot products)
    #   Total for Part 1: 12 + 2 = 14
    
    # Part 2: Row 2 of A (1x4) times all 7 columns of B (4x7)
    #   For each column of B: 1x4 * 4x1 -> rank 1 (one dot product)
    #   7 columns: rank 7
    #   Total for Part 2: 7
    
    # Grand total: 14 + 7 = 21
    
    # Let me implement this.
    
    # Part 1: A_top (rows 0,1) * B (4x7)
    # A_top[i,j] for i in {0,1}, j in {0,1,2,3}
    # B[j,k] for j in {0,1,2,3}, k in {0,...,6}
    
    # Split B's columns into groups: [0,1], [2,3], [4,5], [6]
    
    for group_start in range(0, 7, 2):
        group_end = min(group_start + 2, 7)
        group_size = group_end - group_start
        
        if group_size == 2:
            # 2x4 * 4x2, rank 4
            # C[0, group_start] = sum_j A[0,j] * B[j, group_start]
            # C[0, group_start+1] = sum_j A[0,j] * B[j, group_start+1]
            # C[1, group_start] = sum_j A[1,j] * B[j, group_start]
            # C[1, group_start+1] = sum_j A[1,j] * B[j, group_start+1]
            
            # 4 rank-1 terms, one per output entry
            for i in range(2):
                for k in range(group_start, group_end):
                    u = zeros(n * m)
                    v = zeros(m * p)
                    w = zeros(p * n)
                    
                    for j in range(m):
                        u[i * m + j] = 1
                        v[j * p + k] = 1
                    w[k * n + i] = 1
                    
                    add_term(u, v, w)
        else:
            # group_size == 1: 2x4 * 4x1, rank 2
            for i in range(2):
                k = group_start
                u = zeros(n * m)
                v = zeros(m * p)
                w = zeros(p * n)
                
                for j in range(m):
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                w[k * n + i] = 1
                
                add_term(u, v, w)
    
    # Part 2: A_bottom (row 2) * B (4x7)
    # 1x4 * 4x7, rank 7 (one per output entry)
    for k in range(p):
        i = 2
        u = zeros(n * m)
        v = zeros(m * p)
        w = zeros(p * n)
        
        for j in range(m):
            u[i * m + j] = 1
            v[j * p + k] = 1
        w[k * n + i] = 1
        
        add_term(u, v, w)
    
    rank = len(U_rows)
    
    # Convert to proper matrices: U_rows is (rank, n*m), we want (n*m, rank)
    U = [list(col) for col in zip(*U_rows)]
    V = [list(col) for col in zip(*V_rows)]
    W = [list(col) for col in zip(*W_rows)]
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct the tensor
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T_reconstructed = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Check against the true multiplication tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1
    
    if not np.allclose(T_reconstructed, T_true, atol=1e-10):
        # Fallback to schoolbook (Rank 84)
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
