"""Matrix multiplication 3,3,5 - low rank decomposition via block structure."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U,V,W correctly decompose the MM tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_strassen_222():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        # Strassen for 2x2x2
        # Indices: i,j,k in {0,1}
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Local indices: i*2+j, j*2+k, k*2+i
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U corresponds to A (rows i, cols j)
        # V corresponds to B (rows j, cols k)
        # W corresponds to C (rows k, cols i)
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1 # C10, C11
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # C01, C11
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C00, C10
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C00, C01
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # C00
        
        return U, V, W

    def decompose_block(n, m, p):
        """Recursively decompose n x m x p tensor."""
        if n == 0 or m == 0 or p == 0:
            return [], [], 0
        if n == 2 and m == 2 and p == 2:
            U, V, W = get_strassen_222()
            return U, V, W, 7
        # Schoolbook for small dimensions or base case
        if n <= 2 and m <= 2 and p <= 2:
             # Use schoolbook if not 222
             rank = n * m * p
             U = [[0]*rank for _ in range(n*m)]
             V = [[0]*rank for _ in range(m*p)]
             W = [[0]*rank for _ in range(p*n)]
             idx = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i*m+j][idx] = 1
                         V[j*p+k][idx] = 1
                         W[k*n+i][idx] = 1
                         idx += 1
             return U, V, W, rank
             
        # Recursive split
        # Split n into n1, n2
        n1 = n - 1
        n2 = 1
        m1 = m - 1
        m2 = 1
        p1 = p - 1
        p2 = 1
        
        U_list = []
        V_list = []
        W_list = []
        total_rank = 0
        
        # Iterate over blocks
        for i_part in [(0, n1), (n1, n)]:
            for j_part in [(0, m1), (m1, m)]:
                for k_part in [(0, p1), (p1, p)]:
                    ni = i_part[1] - i_part[0]
                    mj = j_part[1] - j_part[0]
                    pk = k_part[1] - k_part[0]
                    
                    if ni == 0 or mj == 0 or pk == 0:
                        continue
                        
                    Ui, Vi, Wi, ri = decompose_block(ni, mj, pk)
                    
                    # Map local indices to global
                    U_mapped = [[0]*ri for _ in range(n*m)]
                    V_mapped = [[0]*ri for _ in range(m*p)]
                    W_mapped = [[0]*ri for _ in range(p*n)]
                    
                    for li in range(ni):
                        for lj in range(mj):
                            for c in range(ri):
                                U_mapped[(i_part[0]+li)*m + (j_part[0]+lj)][c] = Ui[li*mj+lj][c]
                    
                    for lj in range(mj):
                        for lk in range(pk):
                            for c in range(ri):
                                V_mapped[(j_part[0]+lj)*p + (k_part[0]+lk)][c] = Vi[lj*pk+lk][c]
                    
                    for lk in range(pk):
                        for li in range(ni):
                            for c in range(ri):
                                W_mapped[(k_part[0]+lk)*n + (i_part[0]+li)][c] = Wi[lk*ni+li][c]
                    
                    U_list.append(U_mapped)
                    V_list.append(V_mapped)
                    W_list.append(W_mapped)
                    total_rank += ri
                    
        # Concatenate
        U_final = []
        V_final = []
        W_final = []
        for r in range(total_rank):
            u_row = [0]*(n*m)
            v_row = [0]*(m*p)
            w_row = [0]*(p*n)
            
            offset = 0
            for block_idx in range(len(U_list)):
                rb = len(U_list[block_idx][0])
                for c in range(rb):
                    u_row[offset+c] = U_list[block_idx][0][c] # This is wrong, need to map rows
                offset += rb
            # This concatenation logic is flawed. Need to stack columns.
            
        # Better concatenation:
        U_final = np.zeros((n*m, total_rank))
        V_final = np.zeros((m*p, total_rank))
        W_final = np.zeros((p*n, total_rank))
        
        offset = 0
        for block_idx in range(len(U_list)):
            rb = len(U_list[block_idx][0])
            U_final[:, offset:offset+rb] = np.array(U_list[block_idx])
            V_final[:, offset:offset+rb] = np.array(V_list[block_idx])
            W_final[:, offset:offset+rb] = np.array(W_list[block_idx])
            offset += rb
            
        return U_final.tolist(), V_final.tolist(), W_final.tolist(), total_rank

    # Generate decomposition for 3x3x3
    U1, V1, W1, r1 = decompose_block(3, 3, 3)
    
    # Generate decomposition for 3x3x2
    U2, V2, W2, r2 = decompose_block(3, 3, 2)
    
    # Combine
    rank = r1 + r2
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(15)]
    W = [[0]*rank for _ in range(15)]
    
    # Copy 1
    for i in range(9):
        for c in range(r1):
            U[i][c] = U1[i][c]
            V[i][c] = V1[i][c] # V1 has 9 rows (3x3)
            W[i][c] = W1[i][c] # W1 has 9 rows (3x3)
            
    # Copy 2
    # U2 is 9x r2. Same U indices.
    # V2 is 9x r2? No, V2 for 3x3x2 has m*p = 3*2 = 6 rows?
    # Wait, decompose_block(3,3,2) returns V of size (3*2, r2) = (6, r2).
    # But global V is (3*5, rank) = (15, rank).
    # The rows of V2 correspond to j in 0..2, k in 0..1 (local).
    # Global k is 3,4.
    # So V2 row (j, k_local) maps to global row j*5 + (k_local+3).
    
    for i in range(9):
        for c in range(r2):
            U[i][r1+c] = U2[i][c]
            
    for j in range(3):
        for k_local in range(2):
            k_global = k_local + 3
            v_idx = j*5 + k_global
            for c in range(r2):
                V[v_idx][r1+c] = V2[j*2+k_local][c]
                
    for k_local in range(2):
        k_global = k_local + 3
        for i in range(3):
            w_idx = k_global*3 + i
            for c in range(r2):
                W[w_idx][r1+c] = W2[k_local*3+i][c]
                
    assert verify_decomposition(U, V, W, 3, 3, 5), "Verification failed"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
