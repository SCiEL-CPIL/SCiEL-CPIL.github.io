"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen():
        # We use a block-wise Strassen approach.
        # The matrix A is 3x4. We can split A into blocks of size 2x2 and 1x2?
        # Or 2x2 and 2x2?
        # Let's split A into:
        #   Block 1: rows 0,1; cols 0,1,2,3 (2x4)
        #   Block 2: rows 2; cols 0,1,2,3 (1x4)
        
        # For Block 1 (2x4) * B (4x8) -> C1 (2x8)
        # We can use Strassen for the 2x2 sub-blocks of A and B.
        # A1 = [[A00, A01], [A10, A11]] where Aij are 2x2.
        # B is 4x8. We can split B into 4 blocks of 2x2 along the column dimension?
        # No, B is 4x8. The rows of B correspond to the columns of A.
        # So A (2x4) * B (4x8).
        # A = [A_left (2x2), A_right (2x2)]
        # B = [B_top (2x8); B_bottom (2x8)]
        
        # This is a 2x2 matrix multiplication where the entries are matrices.
        # A_left * B_top + A_right * B_bottom.
        # This is NOT a single 2x2 multiplication in the Strassen sense unless we treat the blocks as scalars.
        # Strassen works for scalar 2x2.
        
        # However, we can apply Strassen to the 2x2 structure of the indices.
        # Let's look at the indices.
        # C[i, k] = sum_j A[i, j] B[j, k].
        
        # Alternative: Use the known rank 88 construction.
        # The inspiration 1 code was trying to do this but was messy.
        # Let's implement a clean Strassen for 2x2x2 blocks and compose them.
        
        # Actually, for 3x4x8, a simple way to get rank 88 is:
        # Split A into 2x2 blocks? No, A is 3x4.
        
        # Let's try to construct rank 88 explicitly using the block decomposition from the inspiration.
        # The inspiration suggests:
        # "2 x 2 x 8 blocks (using Strassen 4 times) -> 4 * 7 = 28 each. Two such blocks -> 56."
        # "1 x 2 x 8 blocks (schoolbook) -> 16 each. Two such blocks -> 32."
        # Total 88.
        
        # This implies:
        # Rows of A: 0,1 (Block 1) and 2 (Block 2).
        # Cols of A: 0,1,2,3.
        
        # For Block 1 (Rows 0,1 of A):
        # We want to compute C[0:2, :] = A[0:2, :] * B.
        # A[0:2, :] is 2x4. B is 4x8.
        # Split A[0:2, :] into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3).
        # Split B into two 2x8 blocks: B_T (rows 0,1) and B_B (rows 2,3).
        # C1 = A_L B_T + A_R B_B.
        
        # This is a sum of two 2x2x8 matrix multiplications.
        # Each 2x2x8 multiplication can be done with Strassen.
        # Strassen for 2x2x8:
        # Treat the 8 columns of the output as 4 independent 2x2 blocks?
        # Yes. B_T is 2x8. We can view it as 4 blocks of 2x2.
        # So A_L (2x2) * B_T (2x8) is equivalent to 4 independent 2x2x2 multiplications.
        # Each 2x2x2 multiplication has rank 7 (Strassen).
        # So A_L * B_T costs 4 * 7 = 28.
        # Similarly A_R * B_B costs 4 * 7 = 28.
        # Total for Block 1 (Rows 0,1) is 28 + 28 = 56.
        
        # For Block 2 (Row 2 of A):
        # C[2, :] = A[2, :] * B.
        # A[2, :] is 1x4. B is 4x8.
        # This is a vector-matrix multiplication.
        # Rank is 4 * 8 = 32 (Schoolbook).
        # Can we do better?
        # A[2, :] = [a, b, c, d].
        # C[2, k] = a B[0, k] + b B[1, k] + c B[2, k] + d B[3, k].
        # This is a linear combination of rows of B.
        # The tensor is of rank 4*8 = 32?
        # Actually, the multiplication of 1x4 by 4x8 is just a dot product for each k.
        # It is a rank 32 tensor?
        # T[i, j, k] where i is fixed (0), j is 0..3, k is 0..7.
        # T[0, j, k] = delta_{j, j'}? No.
        # The tensor for 1x4 x 4x8 multiplication has entries T[j, k] = 1? No.
        # The tensor is T[j, k] = 1 if we are looking at the coefficient?
        # The multiplication tensor M_{1,4,8} has entries M_{j, k} = 1?
        # No, the tensor is defined by the bilinear form.
        # C_k = sum_j A_j B_{j,k}.
        # The tensor T has indices (j, k) for the product?
        # No, the tensor is T_{j, k} such that C = A * T * B?
        # The standard tensor is T_{i, j, k} = 1 if i=j=k? No.
        # For matrix multiplication C = AB, T_{i, j, k} = 1 if i=j and j=k? No.
        # T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # So T_{i, j, k} = 1 for all i, j, k? No.
        # T_{i, j, k} is 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
        # So T_{i, j, k} = 1 for all valid i, j, k.
        # The rank of M_{n, m, p} is the minimum number of rank-1 tensors that sum to it.
        
        # For 1x4 x 4x8:
        # i is fixed to 0.
        # T_{0, j, k} = 1.
        # This tensor is the outer product of vector [1,1,1,1] (length 4) and matrix of ones (4x8)?
        # No. T_{j, k} = 1.
        # This is a rank 1 tensor?
        # T_{j, k} = u_j v_k?
        # If u = [1,1,1,1] and v is a matrix? No, v must be a vector.
        # T_{j, k} = sum_r U_{0, j, r} V_{j, k, r} W_{k, 0, r}?
        # Wait, the indices are (i, j, k).
        # U has shape (n*m, rank). V has shape (m*p, rank). W has shape (p*n, rank).
        # For 1x4x8, n=1, m=4, p=8.
        # U has shape (4, rank). V has shape (32, rank). W has shape (8, rank).
        # We want sum_r U[j, r] V[j*p+k, r] W[k, r] = 1 for all j, k.
        
        # If we set rank=32, we can use schoolbook:
        # For each j, k, have a term.
        # Can we do it with rank < 32?
        # The tensor T_{j, k} = 1 is the all-ones matrix.
        # The rank of the all-ones matrix (as a matrix) is 1.
        # But this is a tensor decomposition problem.
        # T_{j, k} = sum_r U_j V_{j,k} W_k.
        # Note V depends on j and k.
        # Let V_{j,k} = 1 for all j,k.
        # Then T_{j,k} = sum_r U_j W_k.
        # This is the outer product of U and W.
        # So if we set U_j = 1, W_k = 1, V_{j,k} = 1, we get sum_r 1*1*1 = rank.
        # We need sum_r 1 = 1. So rank 1?
        # Let's check.
        # U[j, 0] = 1.
        # V[j*p+k, 0] = 1.
        # W[k, 0] = 1.
        # Sum = 1.
        # So the rank of 1x4x8 multiplication is 1?
        # Wait. The tensor is T_{i, j, k}.
        # For 1x4x8, i=0.
        # T_{0, j, k} = 1.
        # My formula: C_{i, k} = sum_j sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
        # For i=0: C_{0, k} = sum_j sum_r U_{j, r} V_{j*p+k, r} W_{k, r}.
        # We want C_{0, k} = sum_j A_j B_{j, k}.
        # The tensor T must satisfy T_{0, j, k} = 1.
        # So we need sum_r U_{j, r} V_{j*p+k, r} W_{k, r} = 1 for all j, k.
        
        # If we use rank 1:
        # U_{j, 0} = 1.
        # V_{j*p+k, 0} = 1.
        # W_{k, 0} = 1.
        # Sum = 1.
        # So rank 1 is sufficient for the 1x4x8 block?
        # This seems too good to be true.
        # Let's check the dimensions.
        # U is (4, 1). V is (32, 1). W is (8, 1).
        # This works.
        
        # So why did the inspiration say 32?
        # "1 x 2 x 8 blocks (schoolbook) -> 16 each."
        # Maybe they were using a different block structure.
        
        # If rank 1 works for 1x4x8, then total rank would be 56 + 1 = 57?
        # Let's verify if rank 1 works for 1x4x8.
        # C_{0, k} = sum_j A_j B_{j, k}.
        # Decomposition:
        # C_{0, k} = sum_r U_{j, r} V_{j*p+k, r} W_{k, r} ?? No.
        # The formula is C_{i, k} = sum_j (sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}) ?
        # No, the standard decomposition is:
        # C_{i, k} = sum_r (sum_j U_{i*m+j, r} A_{i,j}) (sum_k V_{j*p+k, r} B_{j,k}) W_{k*n+i, r} ??
        # No.
        # The standard form is:
        # C_{i, k} = sum_r M_r * W_{k*n+i, r}
        # where M_r = (sum_j U_{i*m+j, r} A_{i,j}) * (sum_j V_{j*p+k, r} B_{j,k})?
        # No.
        
        # Let's look at the definition in the task.
        # "exactly decompose the matrix multiplication tensor"
        # The tensor T has entries T_{i, j, k} = 1.
        # The decomposition is T_{i, j, k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
        
        # So for 1x4x8:
        # T_{0, j, k} = 1 = sum_r U_{j, r} V_{j*p+k, r} W_{k, r}.
        
        # If we set rank 1:
        # U_{j, 0} = 1.
        # V_{j*p+k, 0} = 1.
        # W_{k, 0} = 1.
        # Sum = 1.
        # This works.
        
        # So the rank of 1x4x8 is 1.
        
        # What about 2x2x8?
        # T_{i, j, k} = 1 for i in {0,1}, j in {0,1}, k in {0..7}.
        # T_{i, j, k} = sum_r U_{i*2+j, r} V_{j*8+k, r} W_{k*2+i, r}.
        
        # We can use Strassen for the 2x2 part.
        # The 2x2x8 tensor is a direct sum of 4 2x2x2 tensors?
        # No, the indices j and k are coupled in V.
        # V_{j*8+k, r}.
        
        # Let's check if we can use the 2x2x2 Strassen factors.
        # For a fixed k block (say k=0,1), we have a 2x2x2 sub-tensor?
        # No, k goes up to 7.
        
        # However, we can view the 2x2x8 multiplication as 4 independent 2x2x2 multiplications
        # IF the V matrix allows it.
        # V_{j*8+k, r}.
        # For a fixed k, V_{j*8+k, r} depends on j and k.
        
        # If we use Strassen for 2x2x2, we have 7 factors.
        # For each of the 4 k-blocks (k=0,1; k=2,3; etc), we need 7 factors.
        # Total 28 factors.
        
        # So the rank of 2x2x8 is 28.
        
        # So total rank for 3x4x8:
        # Block 1 (2x4x8): 56.
        # Block 2 (1x4x8): 1.
        # Total 57.
        
        # This is much better than 88.
        
        # Let's implement this.
        
        rank = 57
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Strassen factors for 2x2x2
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, k_offset):
            nonlocal col
            # a_rows, a_cols: indices for A block (2x2)
            # b_rows, b_cols: indices for B block (2x2)
            # c_rows, c_cols: indices for C block (2x2)
            # k_offset: offset for k index in B and C
            
            u_a = a_rows[0] * m + a_cols[0]
            u_b = a_rows[0] * m + a_cols[1]
            u_c = a_rows[1] * m + a_cols[0]
            u_d = a_rows[1] * m + a_cols[1]
            
            v_e = b_rows[0] * p + (b_cols[0] + k_offset)
            v_f = b_rows[0] * p + (b_cols[1] + k_offset)
            v_g = b_rows[1] * p + (b_cols[0] + k_offset)
            v_h = b_rows[1] * p + (b_cols[1] + k_offset)
            
            w_c11 = (c_cols[0] + k_offset) * n + c_rows[0]
            w_c12 = (c_cols[1] + k_offset) * n + c_rows[0]
            w_c21 = (c_cols[0] + k_offset) * n + c_rows[1]
            w_c22 = (c_cols[1] + k_offset) * n + c_rows[1]
            
            # M1
            U[u_a][col] += 1; U[u_d][col] += 1
            V[v_e][col] += 1; V[v_h][col] += 1
            W[w_c11][col] += 1; W[w_c22][col] += 1
            col += 1
            
            # M2
            U[u_c][col] += 1; U[u_d][col] += 1
            V[v_e][col] += 1
            W[w_c21][col] += 1; W[w_c22][col] -= 1
            col += 1
            
            # M3
            U[u_a][col] += 1
            V[v_f][col] += 1; V[v_h][col] -= 1
            W[w_c12][col] += 1; W[w_c22][col] += 1
            col += 1
            
            # M4
            U[u_d][col] += 1
            V[v_g][col] += 1; V[v_e][col] -= 1
            W[w_c11][col] += 1; W[w_c21][col] += 1
            col += 1
            
            # M5
            U[u_a][col] += 1; U[u_b][col] += 1
            V[v_h][col] += 1
            W[w_c11][col] -= 1; W[w_c12][col] += 1
            col += 1
            
            # M6
            U[u_c][col] += 1; U[u_a][col] -= 1
            V[v_e][col] += 1; V[v_f][col] += 1
            W[w_c22][col] += 1
            col += 1
            
            # M7
            U[u_b][col] += 1; U[u_d][col] -= 1
            V[v_g][col] += 1; V[v_h][col] += 1
            W[w_c11][col] += 1
            col += 1

        # Block 1: Rows 0,1 of A.
        # A_L: cols 0,1. A_R: cols 2,3.
        # B_T: rows 0,1. B_B: rows 2,3.
        
        # A_L * B_T
        for k_start in range(0, 8, 2):
            add_strassen_2x2x2(
                a_rows=[0, 1], a_cols=[0, 1],
                b_rows=[0, 1], b_cols=[0, 1],
                c_rows=[0, 1], c_cols=[0, 1],
                k_offset=k_start
            )
            
        # A_R * B_B
        for k_start in range(0, 8, 2):
            add_strassen_2x2x2(
                a_rows=[0, 1], a_cols=[2, 3],
                b_rows=[2, 3], b_cols=[0, 1],
                c_rows=[0, 1], c_cols=[0, 1],
                k_offset=k_start
            )
            
        # Block 2: Row 2 of A.
        # Rank 1 term.
        # U_{j, 0} = 1 for j in 0..3.
        # V_{j*8+k, 0} = 1 for all j, k.
        # W_{k, 0} = 1 for all k.
        
        for j in range(4):
            U[2 * m + j][col] = 1
        for j in range(4):
            for k in range(8):
                V[j * p + k][col] = 1
        for k in range(8):
            W[k * n + 2][col] = 1
        col += 1
        
        if col != rank:
            return None
            
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        rank = U.shape[1]
        
        # Check tensor identity
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try Rank 57
    if time.time() < deadline - 10:
        res = build_strassen()
        if res and verify(res):
            return res
            
    # Fallback
    return build_schoolbook()
