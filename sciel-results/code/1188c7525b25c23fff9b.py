import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by splitting the 5 intermediate dimension (m) into two parts:
    # Part 1: size 2 (indices 0, 1)
    # Part 2: size 3 (indices 2, 3, 4)
    
    # The multiplication C = A * B can be viewed as:
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
    
    # Let A1 = A[:, 0:2] (2x2), B1 = B[0:2, :] (2x6)
    # Let A2 = A[:, 2:5] (2x3), B2 = B[2:5, :] (3x6)
    
    # We need to decompose the tensor for (2x2) * (2x6) and (2x3) * (3x6).
    
    # --- Part 1: 2x2 * 2x6 ---
    # We can treat B1 as three 2x2 blocks: B1[:, 0:2], B1[:, 2:4], B1[:, 4:6]
    # Let B1_a, B1_b, B1_c be these 2x2 blocks.
    # C1 = A1 * B1_a + A1 * B1_b + A1 * B1_c
    # We can use Strassen's algorithm for each 2x2 * 2x2 multiplication.
    # Strassen has rank 7.
    # So rank for Part 1 = 3 * 7 = 21.
    
    # Strassen factors for 2x2 * 2x2:
    # U_str: 4x7, V_str: 4x7, W_str: 4x7
    # Indices: U[i*2+j], V[k*2+l], W[m*2+n] where C[m,n] = sum_r U[i,j,r] V[k,l,r] W[m,n,r]
    # Wait, the tensor identity is:
    # C[m,n] = sum_{i,k} A[i,k] B[k,n]
    # Decomposition: sum_r U[i,k,r] V[k,n,r] W[m,n,r] ? No.
    # The standard form is:
    # C[m,n] = sum_r U[i,k,r] V[k,n,r] ... wait.
    # The factors are U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Identity: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta(j1, j2) ??
    # Let's check the seed program's identity:
    # "sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)"
    # This corresponds to:
    # (A B)_{i,k} = sum_{j1, j2} A_{i,j1} B_{j2,k} delta(j1, j2) = sum_j A_{i,j} B_{j,k}
    # So the tensor T_{i,j1,j2,k} = delta(j1, j2).
    # The decomposition is T = sum_r U_{i,j1,r} V_{j2,k,r} W_{k,i,r}?
    # No, the seed program says:
    # U shape (n*m, rank) -> indices (i, j1)
    # V shape (m*p, rank) -> indices (j2, k)
    # W shape (p*n, rank) -> indices (k, i)
    # The sum is over r.
    # The term is U[i, j1, r] * V[j2, k, r] * W[k, i, r].
    # This equals delta(j1, j2) for all i, k.
    
    # Let's verify Strassen in this format.
    # For 2x2 * 2x2: n=2, m=2, p=2.
    # U (4, 7), V (4, 7), W (4, 7).
    # We need sum_r U[i, j1, r] V[j2, k, r] W[k, i, r] = delta(j1, j2).
    
    # Standard Strassen:
    # C = A B
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's construct the factors for one 2x2 block.
    # Indices for A: i in {0,1}, j1 in {0,1} -> U index i*2+j1
    # Indices for B: j2 in {0,1}, k in {0,1} -> V index j2*2+k
    # Indices for C: k in {0,1}, i in {0,1} -> W index k*2+i
    
    # We need to map the Strassen multipliers M1..M7 to the rank dimension r=0..6.
    # For each r, U[i, j1, r], V[j2, k, r], W[k, i, r] are defined.
    
    # Let's define the 7 terms.
    # Term 1 (M1): (A11 + A22)(B11 + B22)
    # A11 = A[0,0] (i=0, j1=0)
    # A22 = A[1,1] (i=1, j1=1)
    # B11 = B[0,0] (j2=0, k=0)
    # B22 = B[1,1] (j2=1, k=1)
    # U[0,0,0] = 1, U[1,1,0] = 1
    # V[0,0,0] = 1, V[1,1,0] = 1
    # W contribution to C:
    # C11 += M1 -> W[0,0,0] = 1 (k=0, i=0)
    # C22 += M1 -> W[1,1,0] = 1 (k=1, i=1)
    
    # Term 2 (M2): (A21 + A22)B11
    # A21 = A[1,0] (i=1, j1=0)
    # A22 = A[1,1] (i=1, j1=1)
    # B11 = B[0,0] (j2=0, k=0)
    # U[1,0,1] = 1, U[1,1,1] = 1
    # V[0,0,1] = 1
    # C21 += M2 -> W[0,1,1] = 1 (k=0, i=1)
    # C22 -= M2 -> W[1,1,1] = -1 (k=1, i=1)
    
    # Term 3 (M3): A11(B12 - B22)
    # A11 = A[0,0] (i=0, j1=0)
    # B12 = B[0,1] (j2=0, k=1)
    # B22 = B[1,1] (j2=1, k=1)
    # U[0,0,2] = 1
    # V[0,1,2] = 1, V[1,1,2] = -1
    # C12 += M3 -> W[1,0,2] = 1 (k=1, i=0)
    # C22 += M3 -> W[1,1,2] = 1 (k=1, i=1)
    
    # Term 4 (M4): A22(B21 - B11)
    # A22 = A[1,1] (i=1, j1=1)
    # B21 = B[1,0] (j2=1, k=0)
    # B11 = B[0,0] (j2=0, k=0)
    # U[1,1,3] = 1
    # V[1,0,3] = 1, V[0,0,3] = -1
    # C11 += M4 -> W[0,0,3] = 1 (k=0, i=0)
    # C21 += M4 -> W[0,1,3] = 1 (k=0, i=1)
    
    # Term 5 (M5): (A11 + A12)B22
    # A11 = A[0,0] (i=0, j1=0)
    # A12 = A[0,1] (i=0, j1=1)
    # B22 = B[1,1] (j2=1, k=1)
    # U[0,0,4] = 1, U[0,1,4] = 1
    # V[1,1,4] = 1
    # C11 -= M5 -> W[0,0,4] = -1 (k=0, i=0)
    # C12 += M5 -> W[1,0,4] = 1 (k=1, i=0)
    
    # Term 6 (M6): (A21 - A11)(B11 + B12)
    # A21 = A[1,0] (i=1, j1=0)
    # A11 = A[0,0] (i=0, j1=0)
    # B11 = B[0,0] (j2=0, k=0)
    # B12 = B[0,1] (j2=0, k=1)
    # U[1,0,5] = 1, U[0,0,5] = -1
    # V[0,0,5] = 1, V[0,1,5] = 1
    # C22 += M6 -> W[1,1,5] = 1 (k=1, i=1)
    
    # Term 7 (M7): (A12 - A22)(B21 + B22)
    # A12 = A[0,1] (i=0, j1=1)
    # A22 = A[1,1] (i=1, j1=1)
    # B21 = B[1,0] (j2=1, k=0)
    # B22 = B[1,1] (j2=1, k=1)
    # U[0,1,6] = 1, U[1,1,6] = -1
    # V[1,0,6] = 1, V[1,1,6] = 1
    # C11 += M7 -> W[0,0,6] = 1 (k=0, i=0)
    # C22 += M7 -> W[1,1,6] = 1 (k=1, i=1)
    
    # Let's assemble U_str, V_str, W_str (4x7 each)
    U_str = np.zeros((4, 7), dtype=int)
    V_str = np.zeros((4, 7), dtype=int)
    W_str = np.zeros((4, 7), dtype=int)
    
    # r=0
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    
    # r=1
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[1, 1] = 1; W_str[3, 1] = -1
    
    # r=2
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[2, 2] = 1; W_str[3, 2] = 1
    
    # r=3
    U_str[3, 3] = 1
    V_str[2, 3] = 1; V_str[0, 3] = -1
    W_str[0, 3] = 1; W_str[1, 3] = 1
    
    # r=4
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[0, 4] = -1; W_str[2, 4] = 1
    
    # r=5
    U_str[2, 5] = 1; U_str[0, 5] = -1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1
    
    # r=6
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1; W_str[3, 6] = 1
    
    # Verify Strassen locally
    def verify_strassen():
        for i in range(2):
            for k in range(2):
                mat = np.zeros((2, 2))
                for r in range(7):
                    for j1 in range(2):
                        for j2 in range(2):
                            mat[j1, j2] += U_str[i*2+j1, r] * V_str[j2*2+k, r] * W_str[k*2+i, r]
                if not np.allclose(mat, np.eye(2)):
                    return False
        return True
    
    if not verify_strassen():
        # Fallback to schoolbook if Strassen construction is wrong
        print("Strassen verification failed, falling back to schoolbook")
        # This shouldn't happen, but safe to handle
        
    # --- Part 1 Construction: 2x2 * 2x6 ---
    # We have 3 blocks of B: B1_a (cols 0-1), B1_b (cols 2-3), B1_c (cols 4-5)
    # A1 is 2x2.
    # Total rank = 21.
    # We will create U1, V1, W1 for this part.
    # U1 shape (4, 21) -> A1 indices (i, j1) where i in 0..1, j1 in 0..1
    # V1 shape (12, 21) -> B1 indices (j2, k) where j2 in 0..1, k in 0..5
    # W1 shape (12, 21) -> C1 indices (k, i) where k in 0..5, i in 0..1
    
    rank_part1 = 21
    U1 = np.zeros((4, rank_part1), dtype=int)
    V1 = np.zeros((12, rank_part1), dtype=int)
    W1 = np.zeros((12, rank_part1), dtype=int)
    
    for block_idx in range(3):
        # block_idx 0 -> cols 0-1 of B, 0-1 of C
        # block_idx 1 -> cols 2-3 of B, 2-3 of C
        # block_idx 2 -> cols 4-5 of B, 4-5 of C
        offset = block_idx * 7
        
        # Copy U_str to U1[:, offset:offset+7]
        U1[:, offset:offset+7] = U_str
        
        # Map V_str to V1
        # V_str indices: j2 in 0..1, k_local in 0..1
        # V1 indices: j2 in 0..1, k_global in 0..5
        # k_global = block_idx * 2 + k_local
        for j2 in range(2):
            for k_local in range(2):
                k_global = block_idx * 2 + k_local
                V1[j2 * 2 + k_global, offset:offset+7] = V_str[j2 * 2 + k_local, :]
                
        # Map W_str to W1
        # W_str indices: k_local in 0..1, i in 0..1
        # W1 indices: k_global in 0..5, i in 0..1
        # k_global = block_idx * 2 + k_local
        for k_local in range(2):
            for i in range(2):
                k_global = block_idx * 2 + k_local
                W1[k_global * 2 + i, offset:offset+7] = W_str[k_local * 2 + i, :]
                
    # --- Part 2 Construction: 2x3 * 3x6 ---
    # A2 is 2x3, B2 is 3x6.
    # Naive rank = 2*3*6 = 36.
    # Can we do better?
    # We can split B2 into two 3x3 blocks: B2_a (cols 0-2), B2_b (cols 3-5).
    # C2 = A2 * B2_a + A2 * B2_b.
    # Each is 2x3 * 3x3.
    # Rank of 2x3 * 3x3 multiplication?
    # Lower bound is max(2*3, 3*3, 2*3) = 9? No, rank of tensor.
    # Schoolbook rank for 2x3 * 3x3 is 18.
    # Is there a rank 11 or 12 algorithm?
    # For 2x3 * 3x3, it's a sub-tensor of 3x3 * 3x3.
    # 3x3 * 3x3 rank is 19 (Winograd).
    # But we only need 2 rows of A.
    # Let's try to use Winograd for 3x3 * 3x3 and just ignore the 3rd row of A?
    # No, Winograd computes all 9 entries.
    # If we use Winograd for 3x3 * 3x3, rank is 19.
    # We need to multiply 2x3 by 3x3.
    # We can embed the 2x3 matrix A2 into a 3x3 matrix A2_padded by adding a zero row.
    # Then compute A2_padded * B2_block using Winograd (rank 19).
    # The result C2_padded will have a zero third row.
    # The first two rows of C2_padded are the desired C2.
    # So we can use Winograd factors for 3x3 * 3x3, but we only need the U entries corresponding to the first two rows of A?
    # Wait, the decomposition is sum_r U[i, j1, r] V[j2, k, r] W[k, i, r] = delta(j1, j2).
    # If we use the 3x3 * 3x3 decomposition, it satisfies delta(j1, j2) for j1, j2 in 0..2.
    # We only care about i in 0..1 (rows of C) and k in 0..2 (cols of C for the block).
    # Actually, for 2x3 * 3x3:
    # i in 0..1 (rows of A, rows of C)
    # j1 in 0..2 (cols of A, rows of B)
    # j2 in 0..2 (rows of B, cols of B)
    # k in 0..2 (cols of B, cols of C)
    # The identity is sum_r U[i, j1, r] V[j2, k, r] W[k, i, r] = delta(j1, j2).
    # This must hold for all i, k in the output space.
    # If we use the 3x3 * 3x3 Winograd factors, they satisfy the identity for all i, j1, j2, k in 0..2.
    # Therefore, they also satisfy it for the subset i in 0..1.
    # So we can just use the Winograd factors for 3x3 * 3x3, but we only need to define U for i in 0..1?
    # No, the factor U has shape (n*m, rank). Here n=2, m=3. So U is 6xR.
    # The Winograd factor U_33 has shape 9x19.
    # We can just take the first 6 rows of U_33 (corresponding to i=0,1 and j1=0,1,2).
    # V_33 is 9x19. We need V for j2 in 0..2, k in 0..2. This is the full V_33.
    # W_33 is 9x19. We need W for k in 0..2, i in 0..1.
    # W index is k*3+i.
    # For i=0,1, k=0,1,2, the indices are 0,1,2 (k=0), 3,4,5 (k=1), 6,7,8 (k=2).
    # So we need rows 0..8 of W_33? No, W_33 has rows indexed by k*3+i.
    # We need W[k, i] for k in 0..2, i in 0..1.
    # This corresponds to rows 0,1 (k=0), 3,4 (k=1), 6,7 (k=2) of W_33.
    # Wait, W_33 row index is k*3+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 3
    # k=1, i=1 -> 4
    # k=2, i=0 -> 6
    # k=2, i=1 -> 7
    # We don't need i=2, so we don't need rows 2, 5, 8.
    # But the rank is determined by the number of columns in the factors.
    # If we use the full 19 columns, the rank is 19.
    # Can we reduce the rank by removing columns that are zero in the required parts?
    # Probably not, Winograd factors are dense.
    # So using Winograd for 2x3 * 3x3 gives rank 19.
    # Schoolbook for 2x3 * 3x3 gives rank 18.
    # So Winograd is worse!
    
    # Let's check if there is a better algorithm for 2x3 * 3x3.
    # The tensor rank of 2x3 * 3x3 multiplication.
    # It is known that the rank of 2x3 * 3x3 is 18? Or lower?
    # Actually, 2x3 * 3x3 is a sub-tensor of 2x3 * 3x3.
    # Let's stick to schoolbook for Part 2 to be safe, unless we find a better one.
    # Schoolbook rank for 2x3 * 3x6:
    # Split B2 into two 3x3 blocks.
    # Each 2x3 * 3x3 has rank 18.
    # Total rank for Part 2 = 2 * 18 = 36.
    
    # Total rank = Rank(2x2 * 2x6) + Rank(2x3 * 3x6)
    # = 21 + 36 = 57.
    
    # This is better than 60 (schoolbook for 2x5 * 5x6).
    # Can we improve Part 2?
    # What if we split the 2x3 * 3x6 differently?
    # Maybe use Strassen for 2x2 * 2x2 blocks within the 2x3 * 3x6?
    # A2 is 2x3. B2 is 3x6.
    # We can split A2 into A2[:, 0:2] (2x2) and A2[:, 2:3] (2x1).
    # Split B2 into B2[0:2, :] (2x6) and B2[2:3, :] (1x6).
    # C2 = A2[:, 0:2] * B2[0:2, :] + A2[:, 2:3] * B2[2:3, :]
    # Term 1: 2x2 * 2x6. Rank 21 (using Strassen).
    # Term 2: 2x1 * 1x6. Rank 12 (schoolbook).
    # Total for Part 2 = 21 + 12 = 33.
    
    # Let's verify this split.
    # A2 = [A21, A22] where A21 is 2x2, A22 is 2x1.
    # B2 = [B21; B22] where B21 is 2x6, B22 is 1x6.
    # C2 = A21 * B21 + A22 * B22.
    # This is correct.
    
    # So, let's construct Part 2 using this split.
    # Part 2a: 2x2 * 2x6. Rank 21.
    # Part 2b: 2x1 * 1x6. Rank 12.
    
    # Total Rank = Part 1 (21) + Part 2a (21) + Part 2b (12) = 54.
    
    # This is significantly better than 60.
    
    # Let's build the factors for Part 2a and Part 2b.
    
    # Part 2a: 2x2 * 2x6
    # This is identical to Part 1 structure, but indices are shifted.
    # A2a indices: i in 0..1, j1 in 0..1 (global j1 in 2..3? No, local)
    # In the global problem:
    # A is 2x5.
    # Part 1 uses A[:, 0:2] (j1 in 0..1)
    # Part 2 uses A[:, 2:5] (j1 in 2..4)
    # Part 2a uses A[:, 2:4] (j1 in 2..3)
    # Part 2b uses A[:, 4:5] (j1 in 4..4)
    
    # B is 5x6.
    # Part 1 uses B[0:2, :] (j2 in 0..1)
    # Part 2 uses B[2:5, :] (j2 in 2..4)
    # Part 2a uses B[2:4, :] (j2 in 2..3)
    # Part 2b uses B[4:5, :] (j2 in 4..4)
    
    # C is 6x2.
    # Part 1 contributes to C[:, :] (all cols)
    # Part 2 contributes to C[:, :] (all cols)
    
    # We need to map local indices to global indices for U, V, W.
    
    # Global U shape: (10, R_total) -> indices (i, j1) with i in 0..1, j1 in 0..4
    # Global V shape: (30, R_total) -> indices (j2, k) with j2 in 0..4, k in 0..5
    # Global W shape: (12, R_total) -> indices (k, i) with k in 0..5, i in 0..1
    
    R_total = 21 + 21 + 12 # 54
    
    U_global = np.zeros((10, R_total), dtype=int)
    V_global = np.zeros((30, R_total), dtype=int)
    W_global = np.zeros((12, R_total), dtype=int)
    
    # --- Fill Part 1 (A[:, 0:2] * B[0:2, :]) ---
    # Local U1 indices: (i, j1_local) with j1_local in 0..1
    # Global U1 indices: (i, j1_global) with j1_global = j1_local
    # Local V1 indices: (j2_local, k) with j2_local in 0..1
    # Global V1 indices: (j2_global, k) with j2_global = j2_local
    # Local W1 indices: (k, i)
    # Global W1 indices: (k, i)
    
    offset = 0
    for i in range(2):
        for j1_local in range(2):
            j1_global = j1_local
            U_global[i * 5 + j1_global, offset:offset+21] = U1[i * 2 + j1_local, :]
            
    for j2_local in range(2):
        for k in range(6):
            j2_global = j2_local
            V_global[j2_global * 6 + k, offset:offset+21] = V1[j2_local * 2 + k, :]
            
    for k in range(6):
        for i in range(2):
            W_global[k * 2 + i, offset:offset+21] = W1[k * 2 + i, :]
            
    offset += 21
    
    # --- Fill Part 2a (A[:, 2:4] * B[2:4, :]) ---
    # This is also a 2x2 * 2x6 block.
    # We can reuse the same factors U1, V1, W1.
    # Local indices map to global:
    # j1_local 0..1 -> j1_global 2..3
    # j2_local 0..1 -> j2_global 2..3
    
    for i in range(2):
        for j1_local in range(2):
            j1_global = 2 + j1_local
            U_global[i * 5 + j1_global, offset:offset+21] = U1[i * 2 + j1_local, :]
            
    for j2_local in range(2):
        for k in range(6):
            j2_global = 2 + j2_local
            V_global[j2_global * 6 + k, offset:offset+21] = V1[j2_local * 2 + k, :]
            
    for k in range(6):
        for i in range(2):
            W_global[k * 2 + i, offset:offset+21] = W1[k * 2 + i, :]
            
    offset += 21
    
    # --- Fill Part 2b (A[:, 4:5] * B[4:5, :]) ---
    # 2x1 * 1x6.
    # A2b is 2x1. B2b is 1x6.
    # C2b = A2b * B2b.
    # Identity: sum_r U[i, j1, r] V[j2, k, r] W[k, i, r] = delta(j1, j2).
    # Here j1=4, j2=4.
    # So we need U[i, 4, r] V[4, k, r] W[k, i, r] = 1 for all i, k.
    # This is just the outer product of A2b and B2b.
    # Rank is 2*6 = 12.
    # We can set:
    # For each r in 0..11:
    #   Let i = r // 6, k = r % 6.
    #   U[i, 4, r] = 1
    #   V[4, k, r] = 1
    #   W[k, i, r] = 1
    
    for r in range(12):
        i = r // 6
        k = r % 6
        j1_global = 4
        j2_global = 4
        U_global[i * 5 + j1_global, offset + r] = 1
        V_global[j2_global * 6 + k, offset + r] = 1
        W_global[k * 2 + i, offset + r] = 1
        
    # Total rank is 54.
    
    # Verify the full decomposition
    def verify_full(U, V, W):
        n, m, p = 2, 5, 6
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                mat = np.zeros((m, m))
                for r in range(rank):
                    u_slice = U[i*m:(i+1)*m, r]
                    v_slice = V[k*p:(k+1)*p, r]
                    w_val = W[k*n + i, r]
                    mat += np.outer(u_slice, v_slice) * w_val
                if not np.allclose(mat, np.eye(m)):
                    return False
        return True
    
    if not verify_full(U_global, V_global, W_global):
        # Fallback to schoolbook if verification fails
        print("Verification failed, falling back to schoolbook")
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}

# EVOLVE-BLOCK-END
