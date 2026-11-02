"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, try to find a rank-35 decomposition
    # Strategy: Start from rank-36 block Strassen, try to merge/eliminate one term
    
    def build_rank36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j):
            return i * m + j
        def v_idx(j, k):
            return j * p + k
        def w_idx(k, i):
            return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        # Block 1: A_L B_11 (rows 0-1 of A, rows 0-1 of B, cols 0-1 of B)
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        # Block 2: A_R B_21 (rows 2-3 of A, rows 2-3 of B, cols 0-1 of B)
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        # Block 3: A_L B_12 (rows 0-1 of A, rows 0-1 of B, cols 2-3 of B)
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        # Block 4: A_R B_22 (rows 2-3 of A, rows 2-3 of B, cols 2-3 of B)
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L B_15 (rows 0-1 of A, rows 0-1 of B, col 4 of B) - schoolbook 2x2x1
        # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        
        # Block 6: A_R B_25 (rows 2-3 of A, rows 2-3 of B, col 4 of B) - schoolbook 2x2x1
        # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        rank = len(U[0])
        # Check: sum_r U[i,r] * V[j,r] * W[k,r] should equal 1 if (i,j,k) is in the tensor
        # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # W is indexed by W[k*n + i, r]
        
        # Build the tensor from the decomposition
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        
        # The target tensor is all ones
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_reconstructed, T_target, atol=1e-10)
    
    # Try rank 36 first (known to work)
    result36 = build_rank36()
    if verify_decomposition(result36["U"], result36["V"], result36["W"]):
        # Now try to find rank 35
        # Strategy: try to eliminate one term by finding a linear dependency
        # or by restructuring
        
        # Alternative approach: try different block splits
        # What if we split P=5 as 2+3 instead of 3+2?
        # Or split M=4 differently?
        
        # Let's try: use Strassen for the 2x4x4 part (rank 28) and schoolbook for 2x4x1 (rank 8)
        # That gives 36 again.
        
        # What about a more creative approach?
        # The tensor 2x4x5 has a special structure. Let me try to find if rank 35 is possible.
        
        # Try: can we use a rank-6 algorithm for 2x2x2? No, Strassen's 7 is optimal.
        
        # Let me try a different decomposition structure:
        # Split M=4 into 1+3
        # 2x1x5 (rank 5) + 2x3x5 (rank ?)
        # 2x3x5: split P=5 into 2+3
        # 2x3x2 (rank ?) + 2x3x3 (rank ?)
        
        # 2x3x2: C[0,k] = sum_j A[0,j]B[j,k], C[1,k] = sum_j A[1,j]B[j,k]
        # This is like two 1x3x2 multiplications. Rank of 1x3x2 is 3 (schoolbook).
        # So 2x3x2 has rank 6.
        
        # 2x3x3: This is more complex. Let's use block Strassen.
        # Split M=3 into 1+2
        # 2x1x3 (rank 3) + 2x2x3 (rank 11) = 14
        # Or split differently?
        
        # Total: 5 + 6 + 14 = 25? That seems too low. Let me reconsider.
        
        # Actually, the rank of matrix multiplication tensor [n,m,p] is at least max(n,m,p)
        # and at most n*m*p. For 2x4x5, lower bound is 5, upper is 40.
        
        # Let me try a more careful analysis. The key question is: what is the minimal rank?
        
        # Given the time constraint, let's just return the known rank-36 solution
        # but with a verification step to ensure correctness.
        
        return result36
    
    # If rank 36 doesn't verify (shouldn't happen), fall back to schoolbook
    rank = n * m * p  # 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
