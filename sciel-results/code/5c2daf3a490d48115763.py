import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # Schoolbook rank = 64
    # Try to find rank-32 decomposition using block structure
    
    # Idea: Split the 4x8 matrix into two 4x4 blocks: B = [B1 | B2]
    # Then A*B = [A*B1 | A*B2]
    # Each A*Bi is 2x4, requiring rank 8 each, total 16
    # But we can share some computations
    # Actually this gives 16, but we need to verify the tensor structure
    
    # Let me think about this differently using the tensor formulation
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Try a structured decomposition with rank 32
    # Use the fact that we can factor through intermediate matrices
    
    # Approach: Use a 2-level decomposition
    # Level 1: Compute intermediate 2x4 * 4x4 products (two of them)
    # Level 2: Combine
    
    # Actually, let me try to use a known low-rank construction
    # For M_{2,4,8}, try rank = 2*4*4 + 2*4*4 - something
    
    # Let me try a direct construction:
    # Represent the computation as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j A[i,j] * B[j,k]
    
    # Split j into two groups of 2: j = 2*a + b where a in {0,1}, b in {0,1}
    # C[i,k] = sum_{a=0}^{1} sum_{b=0}^{1} A[i, 2a+b] * B[2a+b, k]
    
    # This doesn't immediately help.
    
    # Let me try a different approach: use the "border rank" or known algebraic geometry results
    # For now, let me try to find ANY rank < 64 by solving the system
    
    # The system is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of 2*4*8 = 64 equations in r*8 + r*32 + r*16 = 56r unknowns
    # For r=2, 56*2=112 > 64, so solutions might exist
    # But the equations are cubic (multilinear), so it's hard
    
    # Let me try a constructive approach with small integer entries
    
    # Try rank 32 with a specific structure:
    # Let U, V, W have entries in {0, 1/2, 1}
    
    # Actually, let me just return schoolbook for now and try to improve
    # The key insight: for 2x4x8, the rank might be as low as 32
    
    # Let me try a specific construction:
    # Use the identity: sum_j A[i,j]*B[j,k] can be computed with fewer multiplications
    # if we use algebraic identities
    
    # For 2x4x8, one approach:
    # Write B as a sum of 4 rank-1 matrices (since 4x8 has rank <= 4)
    # B = sum_{t=1}^{4} u_t * v_t^T
    # Then A*B = sum_t (A*u_t) * v_t^T
    # This requires 4*2 = 8 multiplications for A*u_t and 4*8 = 32 for the outer products
    # But in the tensor model, each "multiplication" corresponds to a rank-1 term
    
    # In the tensor decomposition, each rank-1 term corresponds to one scalar multiplication
    # So if B has rank r_B, then we can write:
    # C = A * B = sum_{t=1}^{r_B} (A * u_t) * v_t^T
    # The number of scalar multiplications is: for each t, we compute A*u_t (2*4 = 8 mults) 
    # and then (A*u_t)*v_t^T (2*8 = 16 mults)
    # Total: r_B * (8 + 16) = r_B * 24
    # If r_B = 4, total = 96, which is worse than 64
    
    # That's not helpful. Let me think again.
    
    # The tensor rank is about the number of scalar multiplications in an algebraic circuit
    # For standard matrix multiplication, it's n*m*p
    
    # For 2x4x8, let me try to find a decomposition with rank < 64
    # by using the fact that some entries can be shared
    
    # Let me try a brute-force search for small ranks with structured patterns
    
    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(nm)]
        V = [[0]*r for _ in range(mp)]
        W = [[0]*r for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W, r
    
    U, V, W, r = make_schoolbook()
    best_rank = r
    best_U, best_V, best_W = U, V, W
    
    # Try to find a rank-32 decomposition
    # Use a structured approach: split into two 2x4x4 sub-problems
    # and combine with shared terms
    
    # For now, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
