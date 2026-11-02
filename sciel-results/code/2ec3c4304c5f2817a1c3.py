import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _check_valid(U, V, W, n=4, m=4, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0])
    # Build tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = 1 if all indices valid, else 0.
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i, j, k] = 1 for all valid i, j, k.
    # W convention: W[k*n + i, c]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # Schoolbook rank = n*m*p = 112
    rank_school = n * m * p
    best_result = None
    best_rank = rank_school
    
    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    U_s, V_s, W_s = build_schoolbook()
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try to improve using structured approaches
    # For 4x4 * 4x7, we can try block-based Strassen-like decompositions
    
    # Approach: Try random search with small ranks below schoolbook
    # The tensor rank of matrix multiplication for 4x4*4x7 is at most 112.
    # Known lower bounds: For m x n * n x p, rank >= max(m*n, m*p, n*p) = max(16, 28, 28) = 28
    # But actual rank is likely much higher.
    
    # Let's try a structured approach: 
    # Split the 4x4 matrix into 2x2 blocks, giving a 2x2 block matrix
    # Each block is 2x2, so we have 4 blocks of A (top-left, top-right, bottom-left, bottom-right)
    # Similarly B is 4x7, split into 2x(2x4) or (2x2)x(2x4)...
    
    # Actually, let me think about this differently.
    # A is 4x4, B is 4x7, C is 4x7
    # Split A into 2x2 blocks of size 2x2: A = [[A11, A12], [A21, A22]]
    # Split B into 2x4 blocks of size 2x4: B = [[B11, B12], [B21, B22]] where B11 is 2x4, etc.
    # Wait, B is 4x7, so splitting rows: B1 = B[0:2, :] (2x7), B2 = B[2:4, :] (2x7)
    # C = A*B, so C[0:2, :] = A[0:2,:]*B, C[2:4,:] = A[2:4,:]*B
    
    # Let me try a simpler structured approach using the fact that we can use
    # precomputed products.
    
    # Try rank 56, 64, 72, ... down to some minimum
    # Actually, let's try to use a semi-structured search.
    
    # For now, let's try to find if rank < 112 is possible with a simple structured pattern.
    
    # One idea: For the 4x4 * 4x7 case, we can think of it as computing
    # 4 rows of C, each being a linear combination of the 4 rows of B.
    # C[i, :] = sum_j A[i,j] * B[j, :]
    # So C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor has 4*4*7 = 112 entries.
    
    # Let me try to see if we can do better than schoolbook with a specific structure.
    # Actually, for this problem, the best known bounds might keep us at 112 or slightly below.
    
    # Let me try a randomized approach: generate random matrices U, V, W of various ranks
    # and check if they work. This is unlikely to work for exact decomposition.
    
    # A better approach: use the fact that we can combine some schoolbook terms.
    # If two terms have the same (i,j) or (j,k) or (k,i) pattern, we might combine them.
    
    # Actually, let me just verify the schoolbook and return it, since finding
    # a better decomposition for 4x4*4x7 is non-trivial.
    
    # Let me try one structured idea: use Strassen-like for 2x2*2x2 subproblems.
    # A(4x4) = [[A00(2x2), A01(2x2)], [A10(2x2), A11(2x2)]]
    # B(4x7) - we can split B as B(2x7) stacked with B(2x7), or as four 1x7 rows...
    
    # Let me split B into two 2x7 matrices: B_top(2x7) = B[0:2,:], B_bot(2x7) = B[2:4,:]
    # Then C_top(2x7) = A00*B_top + A01*B_bot
    #      C_bot(2x7) = A10*B_top + A11*B_bot
    
    # Each 2x2 * 2x7 product can be done with 2*2*7 = 28 multiplications schoolbook.
    # Total: 4 * 28 = 112. Same as before.
    
    # To use Strassen: 2x2*2x2 uses 7 mults instead of 8.
    # But we have 2x2 * 2x7, not 2x2*2x2.
    # We can split 2x7 into seven 2x1 columns. Each 2x2*2x1 uses 4 mults schoolbook, 3 with... no, Strassen doesn't help for 2x2*2x1.
    
    # Alternative: treat 2x7 as 7 separate 2x1 vectors.
    # C_col = A(2x2) * B_col(2x1) = 4 mults per column, 7 columns = 28 per 2x2*2x7 block.
    # 4 blocks = 112 total.
    
    # I don't see an obvious way to beat 112 for this specific shape with simple structure.
    
    # Let me just return the schoolbook decomposition.
    
    elapsed = time.time() - start_time
    if elapsed > WALL_DEADLINE - SAFETY_MARGIN:
        return best_result
    
    # Try a few more structured ideas with time guard
    # Idea: Can we share computations between the 4 block products?
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    
    # If we compute P1 = A00*B_top, P2 = A01*B_bot, P3 = A10*B_top, P4 = A11*B_bot
    # Then C_top = P1+P2, C_bot = P3+P4
    # This is still 4 products of rank 28 each = 112.
    
    # What if we use: P1 = A00*B_top, P2 = A11*B_bot, P3 = (A00+A11)*(B_top+B_bot)
    # Then A00*B_top + A11*B_bot = P1+P2
    # And (A00+A11)*(B_top+B_bot) = A00*B_top + A00*B_bot + A11*B_top + A11*B_bot = P1 + P2 + A00*B_bot + A11*B_top
    # So A00*B_bot + A11*B_top = P3 - P1 - P2
    # But I need A01*B_bot and A10*B_top, not A00*B_bot and A11*B_top.
    # This doesn't directly help.
    
    # I'll return the schoolbook decomposition.
    return best_result
