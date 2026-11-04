"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 130.0
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build target entries
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.add((a, b, c))
    
    def validate_decomp(U, V, W, tol=1e-9):
        """Check if decomposition exactly reproduces the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all target entries
        for (a, b, c) in target_entries:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check off-diagonal entries
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j == jp:
                        continue
                    for k in range(p):
                        a = i * m + j
                        b = jp * p + k
                        c = k * n + i
                        val = np.dot(U[a, :] * V[b, :], W[c, :])
                        if abs(val) > tol:
                            return False
        return True
    
    # Strategy: Partition into 4x4x4 core + strips
    # Core: i in [0,4), j in [0,4), k in [0,4) - use DFT-based rank-15
    # Strip 1: i in [0,4), j in [4,5), k in [0,6) - schoolbook, rank 24
    # Strip 2: i in [0,4), j in [0,4), k in [4,6) - schoolbook, rank 32
    # Strip 3: i in [0,4), j in [4,5), k in [4,6) - schoolbook, rank 8
    # Total: 15 + 24 + 32 + 8 = 79
    
    # Actually let me reconsider the partition:
    # Core: i in [0,4), j in [0,4), k in [0,4) - rank 15 (DFT)
    # Strip A: i in [0,4), j in [4,5), k in [0,6) - 4*1*6 = 24
    # Strip B: i in [0,4), j in [0,4), k in [4,6) - 4*4*2 = 32
    # Strip C: i in [0,4), j in [4,5), k in [4,6) - 4*1*2 = 8
    # Total: 15 + 24 + 32 + 8 = 79
    
    # Build DFT-based 4x4x4 decomposition (rank 15)
    # Using 4-point DFT: F[i][j] = sum_j A[i][j] * omega^(ij) where omega = i (imaginary unit)
    # For 4-point DFT over complex numbers, we can use the structure:
    # F[0] = a0+a1+a2+a3
    # F[1] = a0+a1*i-a2-a3*i  
    # F[2] = a0-a1+a2-a3
    # F[3] = a0-a1*i-a2+a3*i
    
    # The key insight: for cyclic convolution of length 4, we need:
    # C[k] = sum_j A[j] * B[(k-j) mod 4]
    # In frequency domain: C_hat[k] = A_hat[k] * B_hat[k]
    # Then inverse DFT gives C[k]
    
    # But we need integer/half-integer entries. Let's use a different approach.
    # Use the fact that 4-point DFT can be decomposed into real operations.
    
    # Actually, let me use a simpler approach: Winograd-style with 4-point DFT
    # The 4-point DFT matrix is:
    # [1  1  1  1]
    # [1  i -1 -i]
    # [1 -1  1 -1]
    # [1 -i -1  i]
    
    # For matrix multiplication C = A*B where A is 4x4 and B is 4x4:
    # We can use the tensor rank decomposition based on DFT.
    
    # Let me construct the DFT-based decomposition for 4x4x4:
    # The tensor T[i,j,k] = delta_{j,k} for the multiplication tensor
    # We decompose it as sum_r u_r[i] * v_r[j] * w_r[k]
    
    # Using 4-point DFT basis:
    # For each frequency component f in {0,1,2,3}, and for each "shift" s in {0,1,2,3}:
    # This gives 4*4 = 16 terms, but we can reduce by 1 using symmetry.
    
    # Actually, the standard Winograd for 4x4x4 has rank 16 (same as schoolbook for 4x4x4=64... no wait)
    # Schoolbook for 4x4x4 is rank 64. Winograd reduces it.
    
    # Let me use a different approach. For 4x4x4 matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Using DFT: if we define for each row i of A: a_i = A[i,:]
    # and for each column k of B: b_k = B[:,k]
    # then C[i,k] = dot(a_i, b_k)
    
    # The DFT approach: 
    # Let F be the 4-point DFT matrix. Then:
    # a_i_hat = F * a_i
    # b_k_hat = F^H * b_k  (conjugate transpose)
    # C[i,k] = (1/4) * dot(a_i_hat, b_k_hat)
    
    # This doesn't directly help with tensor rank.
    
    # Let me use a simpler construction. For 4x4x4, I'll use the following:
    # Split into 2x2 blocks and use Strassen on each, but that gives 7*7=49 for the full thing.
    
    # Actually, let me just use schoolbook for the core too and focus on getting a valid decomposition.
    # Schoolbook for 4x4x4 core: rank 64
    # Strip A: 4*1*6 = 24
    # Strip B: 4*4*2 = 32  
    # Strip C: 4*1*2 = 8
    # Total: 64 + 24 + 32 + 8 = 128 (worse than schoolbook 120)
    
    # So I need a better core decomposition. Let me try Strassen for the 4x4x4 core.
    # 4x4x4 can be done with 4x(2x2x2) = 4*7 = 28 operations if we use Strassen on 2x2 blocks.
    # But actually, 4x4 matrix multiplication can be done with rank 49 (Strassen applied to 2x2 blocks).
    
    # Let me use Strassen for the 4x4x4 core:
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B into 2x2 blocks: B11, B12, B21, B22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen for each 2x2x2: rank 7
    # But we have 4 block multiplications, each of rank 7, so total rank = 4*7 = 28
    
    # Total: 28 + 24 + 32 + 8 = 92
    
    # Let me implement Strassen-based 4x4x4 decomposition.
    
    # Strassen's algorithm for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply this to each of the 4 block positions.
    
    # Let me build the full decomposition.
    # U has shape (20, R), V has shape (30, R), W has shape (24, R)
    
    # Core: i in [0,4), j in [0,4), k in [0,4)
    # Strip A: i in [0,4), j=4, k in [0,6)
    # Strip B: i in [0,4), j in [0,4), k in [4,6)
    # Strip C: i in [0,4), j=4, k in [4,6)
    
    # Let me use rank 15 for the core (DFT-based) and schoolbook for strips.
    # Total: 15 + 24 + 32 + 8 = 79
    
    # DFT-based 4x4x4 decomposition:
    # I'll use the following construction based on 4-point DFT.
    # The key is that for cyclic convolution, we can use the DFT to diagonalize.
    
    # For the 4x4x4 multiplication tensor, we need:
    # sum_r U[i,j,r] * V[j',k,r] * W[k',i,r] = delta_{j,j'} * delta_{k,k'}
    # Wait, that's not right. The tensor is T[i,j,k] = 1 if this is a valid multiplication term.
    
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where the "input" indices are (i,j) for A and (j,k) for B,
    # and the "output" index is (i,k) for C.
    # 
    # In the tensor decomposition: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i, r] = delta_{j,j'}
    # for the (i,j,j',k) -> (i,k) mapping.
    
    # Let me just use schoolbook for everything and verify it works, then try to improve.
    
    # Schoolbook decomposition:
    R_schoolbook = n * m * p  # 120
    U = np.zeros((nm, R_schoolbook))
    V = np.zeros((mp, R_schoolbook))
    W = np.zeros((pn, R_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    if validate_decomp(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If schoolbook doesn't work (shouldn't happen), return zeros
    return {"U": np.zeros((nm, 1)).tolist(), "V": np.zeros((mp, 1)).tolist(), "W": np.zeros((pn, 1)).tolist()}
# EVOLVE-BLOCK-END
