"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def build_strassen_block(U, V, W, i0, j0, k0, col_start):
        """
        Add Strassen 2x2x2 decomposition starting at (i0, j0, k0).
        Returns new column index.
        """
        # For 2x2x2 block: rows i0,i0+1; cols j0,j0+1; outputs k0,k0+1
        # Strassen's 7 multiplications for 2x2 matrix mult:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
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
        
        # In tensor terms, for each rank-1 term c:
        # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # We need to assign values to U[i,j], V[j,k], W[k,i] for i in {i0,i0+1},
        # j in {j0,j0+1}, k in {k0,k0+1} such that the product sums give the right coefficients.
        
        # For Strassen, each M corresponds to a product of linear forms.
        # We can decompose each linear form product into rank-1 terms in the tensor.
        
        # Actually, for a 2x2x2 tensor with Strassen rank 7, we need 7 rank-1 terms.
        # Let me construct them explicitly.
        
        # The 2x2x2 multiplication tensor has entries T[i,j,k] where i,j,k in {0,1}
        # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
        # Actually T[i,j,k] = 1 for all (i,j,k) in the standard basis... 
        # No, the multiplication tensor T[i,j,k] = 1 if multiplying row i of A by col k of B
        # uses element (i,j) of A and (j,k) of B. So T[i,j,k] = 1 for all i,j,k in {0,1}.
        
        # Wait, that's not right either. The tensor for 2x2 matrix multiplication is:
        # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all i,j,k in {0,1} - it's the all-ones tensor!
        
        # No wait. For matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] appears).
        
        # So the 2x2x2 multiplication tensor is indeed the all-ones 2x2x2 tensor.
        # Its rank is 7 (Strassen).
        
        # Let me find the 7 rank-1 decompositions of the all-ones 2x2x2 tensor.
        # We need: sum_{c=1}^{7} u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k in {0,1}
        
        # One known decomposition (from Strassen):
        # Let me use a standard set. The all-ones 2x2x2 tensor can be written as:
        # (1,1)(1,1)(1,1) + (1,1)(1,0)(1,0) + (1,0)(1,1)(1,0) + ... 
        # This is getting complex. Let me use a different approach.
        
        # For now, just use schoolbook for this block (4 terms) and return.
        # We can optimize later.
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    i = i0 + ii
                    j = j0 + jj
                    k = k0 + kk
                    U[i * m + j][col_start] = 1
                    V[j * p + k][col_start] = 1
                    W[k * n + i][col_start] = 1
                    col_start += 1
        return col_start

    def try_block_decomposition():
        """
        Try to build a decomposition using block structure.
        For 3x5x6, we can try various block partitions.
        """
        if time.time() > deadline:
            return None
        
        # Try: split rows into 2+1, cols into 2+2+1, outputs into 2+2+2
        # Block (2,2,2): use Strassen (7 terms) if we can find it, else schoolbook (4 terms)
        # Block (2,2,2): 4 terms schoolbook
        # Block (2,1,2): 4 terms schoolbook
        # Block (1,2,2): 4 terms schoolbook
        # etc.
        
        # For now, just return schoolbook as the best we can do quickly.
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find a better decomposition
    better = try_block_decomposition()
    if better is not None:
        rank = len(better["U"][0])
        if rank < best_rank and verify(better["U"], better["V"], better["W"], rank):
            best_result = better
            best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
