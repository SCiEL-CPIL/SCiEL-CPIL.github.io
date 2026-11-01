"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
    
    # Build Strassen-based decomposition
    # Partition j: J1={0,1}, J2={2,3}, J3={4}
    # Partition k: K1={0,1}, K2={2,3}, K3={4,5}
    # 6 blocks of 2x2x2 use Strassen (rank 7 each)
    # 3 blocks of 2x1x2 use schoolbook (rank 4 each)
    # Total: 6*7 + 3*4 = 54
    
    U = []
    V = []
    W = []
    
    def add_strassen_block(j_offset, k_offset, U_mat, V_mat, W_mat, rank_so_far):
        """Add a Strassen 2x2x2 block for i in {0,1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}"""
        # Strassen's 7 products for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For tensor: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # i=0 corresponds to row 0, i=1 to row 1
        # j_local = j - j_offset (0 or 1)
        # k_local = k - k_offset (0 or 1)
        
        # We need to assign values such that for each (i, j_local, k_local),
        # the sum over the 7 Strassen terms equals 1.
        
        # Strassen decomposition for matrix mult:
        # C[i,j] = sum over 7 terms
        # Term t contributes to C[i,j] with coefficient coeff[i,j,t]
        
        # For 2x2 matrices:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A*B
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # In tensor form, for each term t, we have:
        # U[i, j_local, t] * V[j_local, k_local, t] * W[k_local, i, t]
        
        # Let's define the contributions:
        # For term 1: U[i, j, t1] * V[j, k, t1] * W[k, i, t1]
        # We need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for each (i,j,k)
        
        # This is equivalent to: the tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # should equal the matrix multiplication tensor.
        
        # For a 2x2x2 block, the tensor is:
        # T[i,j,k] = 1 if this (i,j,k) corresponds to a valid multiplication term, 0 otherwise
        # Actually no - the tensor for matrix mult is:
        # T[i,j,k] = 1 for all (i,j,k) in the 2x2x2 box, because
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has T[i,j,k] = 1 for all i,j,k in {0,1}^3
        
        # Wait, that's not right. The tensor for matrix multiplication C = A*B is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i, j, k (it's the all-ones tensor for 2x2x2)
        # No wait, that's the structure tensor. The multiplication tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1
        # Actually the matrix multiplication tensor m_{n} is:
        # m_n = sum_{i,j,k=1}^n e_i ⊗ e_j ⊗ e_k
        # No, that's not right either.
        
        # The correct tensor: For C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
        # The bilinear form is: C[i,k] = sum_{j,t} U[i,j,t] V[j,k,t] W[k,i,t] ... 
        # No, the standard form is: C[i,k] = sum_t U[i,j,t] ... 
        # I'm confusing myself. Let me restart.
        
        # The tensor decomposition: we want to compute C = A * B where A is n×m, B is m×p, C is n×p.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # We want: C[i,k] = sum_t U[i,j,t]... no.
        
        # The correct formulation:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_t (sum_j U[i,j,t] A[i,j]) (sum_k V[j,k,t] B[j,k]) ... no this doesn't work.
        
        # Standard tensor decomposition for matrix mult:
        # C[i,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] doesn't make sense because j is summed over.
        
        # Let me re-read the task. The task says:
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
        
        # So the tensor T[i,j,k] = 1 for all i,j,k, and we're decomposing the all-ones tensor?
        # No, that can't be right for matrix multiplication.
        
        # Actually, I think the formulation is:
        # The matrix multiplication tensor is defined such that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This can be written as a trilinear form:
        # C[i,k] = sum_{j,t} A[i,j] B[j,k] * delta_{j,t} ... no.
        
        # I think the correct interpretation is:
        # We have three tensors U, V, W and we want:
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This means the "tensor" we're decomposing is the all-ones tensor of shape (n*m) x (m*p) x (p*n).
        # But that's not matrix multiplication...
        
        # Wait, I think I misunderstand. Let me re-read:
        # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 6 matrix"
        # "The output factor follows the official trace convention W[k*n+i,r]"
        
        # I think the convention is:
        # U has shape (n*m, rank) - indexed by (i,j) where i is row of A, j is col of A / row of B
        # V has shape (m*p, rank) - indexed by (j,k) where j is col of A / row of B, k is col of B
        # W has shape (p*n, rank) - indexed by (k,i) where k is col of B, i is row of A
        
        # And the condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # This is indeed the all-ones tensor decomposition! But how does this relate to matrix multiplication?
        
        # Oh! I think the point is that this tensor decomposition, when applied, gives:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]
        # No, that's not right either.
        
        # I think the actual computation is:
        # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # No, that doesn't work dimensionally.
        
        # Let me just accept the formulation as given: we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # for all i in range(n), j in range(m), k in range(p).
        
        # This is the decomposition of the all-ones tensor of shape (n*m) x (m*p) x (p*n).
        # The rank of this tensor is what we're minimizing.
        
        # For the all-ones tensor, the schoolbook decomposition has rank n*m*p, with each term
        # corresponding to a single (i,j,k) triple.
        
        # Now, for Strassen on 2x2x2 all-ones tensor:
        # We need: sum_{t=1}^{7} U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k in {0,1}
        
        # This is a different problem than matrix multiplication! The all-ones tensor of shape 2x2x2
        # has rank... let me think. The all-ones tensor J of shape n×n×n has rank at most n^2 (trivially)
        # but can be lower.
        
        # Actually, I realize I've been overthinking this. Let me just implement a direct search
        # for the all-ones tensor decomposition with small integer/half-integer values.
        
        pass
    
    # Given the confusion, let me try a different approach:
    # Use a structured search with values in {-1, -1/2, 0, 1/2, 1}
    # and try to find a valid decomposition with rank < 60.
    
    start_time = time.time()
    time_limit = 110.0
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find better decompositions using random search with small values
    values = [-1, -0.5, 0, 0.5, 1]
    
    def gen_matrix(rows, cols, density=0.4):
        mat = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if random.random() < density:
                    mat[i][j] = random.choice(values)
        return mat
    
    for r in range(rank_sb - 1, max(rank_sb - 10, 40), -1):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            U_rand = gen_matrix(n * m, r)
            V_rand = gen_matrix(m * p, r)
            W_rand = gen_matrix(p * n, r)
            
            if is_valid_decomposition(U_rand, V_rand, W_rand, r):
                best_U, best_V, best_W, best_rank = U_rand, V_rand, W_rand, r
                break
        
        if best_rank < rank_sb:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
