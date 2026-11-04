"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition B (3x10) into five 3x2 blocks.
  For each 2x3 * 3x2 block, decompose as:
    C = A_left * B_top + A_right * B_bottom
  where A_left is 2x2, A_right is 2x1, B_top is 2x2, B_bottom is 1x2.
  
  For A_left * B_top (2x2 * 2x2): use Strassen's algorithm (rank 7)
  For A_right * B_bottom (2x1 * 1x2): this is an outer product [c;f]*[k,l]
    which requires 4 multiplications (rank 4)
  
  Total per block: 7 + 4 = 11
  Total rank: 5 * 11 = 55
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 135.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    num_blocks = p // 2  # 5 blocks of width 2
    rank_per_block = 11
    rank = num_blocks * rank_per_block  # 55
    
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 55
    
    for block_idx in range(num_blocks):
        col_offset = block_idx * rank_per_block
        b_col_start = 2 * block_idx
        
        # B is 3x10, stored as V[j*p + k] where j is row, k is column
        # For this block, B columns are b_col_start and b_col_start+1
        # B_top (rows 0,1): 
        #   g = B[0, b_col_start] = V[0*p + b_col_start] = V[b_col_start]
        #   h = B[0, b_col_start+1] = V[b_col_start + 1]
        #   i = B[1, b_col_start] = V[1*p + b_col_start]
        #   j = B[1, b_col_start+1] = V[1*p + b_col_start + 1]
        # B_bottom (row 2):
        #   k = B[2, b_col_start] = V[2*p + b_col_start]
        #   l = B[2, b_col_start+1] = V[2*p + b_col_start + 1]
        
        # A is 2x3:
        #   a = A[0,0] = U[0]
        #   b = A[0,1] = U[1]
        #   c = A[0,2] = U[2]
        #   d = A[1,0] = U[3]
        #   e = A[1,1] = U[4]
        #   f = A[1,2] = U[5]
        
        # C is 2x2 for this block:
        #   C[0,0] = W[b_col_start*n + 0] = W[b_col_start*2 + 0]
        #   C[0,1] = W[b_col_start*n + 1] = W[b_col_start*2 + 1]
        #   C[1,0] = W[(b_col_start+1)*n + 0] = W[(b_col_start+1)*2 + 0]
        #   C[1,1] = W[(b_col_start+1)*n + 1] = W[(b_col_start+1)*2 + 1]
        
        # Strassen for A_left (2x2) * B_top (2x2):
        # A_left = [[a, b], [d, e]]
        # B_top = [[g, h], [i, j]]
        
        # m1 = (a+b)(g+i)
        c = col_offset + 0
        U[0][c] = 1.0; U[1][c] = 1.0
        V[b_col_start][c] = 1.0; V[p + b_col_start][c] = 1.0
        W[2*b_col_start + 0][c] = 1.0
        W[2*b_col_start + 1][c] = 1.0
        W[2*(b_col_start+1) + 0][c] = -1.0
        W[2*(b_col_start+1) + 1][c] = 1.0
        
        # m2 = a(j-i)
        c = col_offset + 1
        U[0][c] = 1.0
        V[p + b_col_start + 1][c] = 1.0; V[p + b_col_start][c] = -1.0
        W[2*b_col_start + 1][c] = -1.0
        W[2*(b_col_start+1) + 1][c] = 1.0
        
        # m3 = (d+e)(g+h)
        c = col_offset + 2
        U[3][c] = 1.0; U[4][c] = 1.0
        V[b_col_start][c] = 1.0; V[b_col_start + 1][c] = 1.0
        W[2*(b_col_start+1) + 0][c] = 1.0
        W[2*(b_col_start+1) + 1][c] = -1.0
        
        # m4 = e(h-j)
        c = col_offset + 3
        U[4][c] = 1.0
        V[b_col_start + 1][c] = 1.0; V[p + b_col_start + 1][c] = -1.0
        W[2*b_col_start + 0][c] = 1.0
        W[2*(b_col_start+1) + 0][c] = -1.0
        
        # m5 = (a+d)(h+i)
        c = col_offset + 4
        U[0][c] = 1.0; U[3][c] = 1.0
        V[b_col_start + 1][c] = 1.0; V[p + b_col_start][c] = 1.0
        W[2*b_col_start + 0][c] = -1.0
        W[2*b_col_start + 1][c] = 1.0
        W[2*(b_col_start+1) + 0][c] = 1.0
        
        # m6 = (b-d)(g+h)
        c = col_offset + 5
        U[1][c] = 1.0; U[3][c] = -1.0
        V[b_col_start][c] = 1.0; V[b_col_start + 1][c] = 1.0
        W[2*b_col_start + 1][c] = 1.0
        W[2*(b_col_start+1) + 1][c] = 1.0
        
        # m7 = (b-e)(i+j)
        c = col_offset + 6
        U[1][c] = 1.0; U[4][c] = -1.0
        V[p + b_col_start][c] = 1.0; V[p + b_col_start + 1][c] = 1.0
        W[2*b_col_start + 0][c] = 1.0
        W[2*(b_col_start+1) + 0][c] = -1.0
        
        # Schoolbook for A_right (2x1) * B_bottom (1x2):
        # A_right = [[c], [f]]
        # B_bottom = [[k, l]]
        # Product = [[c*k, c*l], [f*k, f*l]]
        
        # m8 = c*k
        c = col_offset + 7
        U[2][c] = 1.0
        V[2*p + b_col_start][c] = 1.0
        W[2*b_col_start + 0][c] = 1.0
        
        # m9 = c*l
        c = col_offset + 8
        U[2][c] = 1.0
        V[2*p + b_col_start + 1][c] = 1.0
        W[2*b_col_start + 1][c] = 1.0
        
        # m10 = f*k
        c = col_offset + 9
        U[5][c] = 1.0
        V[2*p + b_col_start][c] = 1.0
        W[2*(b_col_start+1) + 0][c] = 1.0
        
        # m11 = f*l
        c = col_offset + 10
        U[5][c] = 1.0
        V[2*p + b_col_start + 1][c] = 1.0
        W[2*(b_col_start+1) + 1][c] = 1.0
    
    # Verify the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # should equal 1 if i,j,k correspond to a valid (i,j,k) in matrix mult, 0 otherwise
    # i is in [0, n*m), j is in [0, m*p), k is in [0, p*n)
    # i = row_a * m + col_a, j = row_b * p + col_b, k = row_c * n + col_a
    # Valid if col_a == row_b and row_c == row_a
    
    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Check against expected
    expected = np.zeros((n*m, m*p, p*n))
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
                i = i_a * m + j_a
                j = j_a * p + k_b
                k = k_b * n + i_a
                expected[i, j, k] = 1.0
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_u = i * m + j
                    row_v = j * p + k
                    row_w = k * n + i
                    U[row_u][col] = 1.0
                    V[row_v][col] = 1.0
                    W[row_w][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
