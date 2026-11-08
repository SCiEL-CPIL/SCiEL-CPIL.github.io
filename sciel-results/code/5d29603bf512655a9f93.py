"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-55 tensor decomposition for 2x3 by 3x10 matrix multiplication.
    
    Strategy: Split the 10 columns of B into 5 pairs. For each pair (k, k+1),
    decompose the 2x3 * 3x2 sub-product into:
      - A 2x2 * 2x2 part using Strassen (rank 7)
      - A 2x1 * 1x2 tail part (rank 4)
    Total per pair: 11. Total rank: 5 * 11 = 55.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Each pair of columns contributes 11 rank-1 terms
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    column = 0
    for pair_idx in range(5):
        k0 = 2 * pair_idx      # first column of the pair
        k1 = 2 * pair_idx + 1  # second column of the pair
        
        # --- Strassen part: A[:, 0:2] * B[0:2, k0:k1+1] ---
        # A_sub is 2x2: rows i=0,1; cols j=0,1
        # B_sub is 2x2: rows j=0,1; cols k=k0,k1
        # Strassen products M1..M7
        
        # M1 = (a00 + a01) * (b00 + b01)
        # In our notation: a_i_j is A[i,j], b_j_k is B[j,k]
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {k0,k1}
        
        # M1: (A[0,0]+A[0,1]) * (B[0,k0]+B[1,k0]) -- wait, need to be careful
        # Standard Strassen for C = A*B where A is 2x2, B is 2x2:
        # M1 = (A00+A01)(B00+B10)
        # M2 = (A10+A11)(B00+B10)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A11)(B10+B11)
        # M6 = (A01-A10)(B00+B10)
        # M7 = (A01-A11)(B10+B11)
        
        # Here A is 2x2 with entries A[i,j] for i in {0,1}, j in {0,1}
        # B is 2x2 with entries B[j,k] for j in {0,1}, k in {k0,k1}
        # So B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        
        # M1 = (A[0,0]+A[0,1]) * (B[0,k0]+B[1,k0])
        U[0 * m + 0][column] = 1  # A[0,0]
        U[0 * m + 1][column] = 1  # A[0,1]
        V[0 * p + k0][column] = 1  # B[0,k0]
        V[1 * p + k0][column] = 1  # B[1,k0]
        W[k0 * n + 0][column] = 1  # C[0,k0] contribution
        W[k0 * n + 1][column] = 1  # C[1,k0] contribution
        column += 1
        
        # M2 = (A[1,0]+A[1,1]) * (B[0,k0]+B[1,k0])
        U[1 * m + 0][column] = 1  # A[1,0]
        U[1 * m + 1][column] = 1  # A[1,1]
        V[0 * p + k0][column] = 1
        V[1 * p + k0][column] = 1
        W[k0 * n + 1][column] = 1  # C[1,k0]
        column += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0 * m + 0][column] = 1  # A[0,0]
        V[0 * p + k1][column] = 1  # B[0,k1]
        V[1 * p + k1][column] = -1  # B[1,k1]
        W[k1 * n + 0][column] = 1  # C[0,k1]
        column += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[1 * m + 1][column] = 1  # A[1,1]
        V[1 * p + k0][column] = 1  # B[1,k0]
        V[0 * p + k0][column] = -1  # B[0,k0]
        W[k0 * n + 1][column] = 1  # C[1,k0]
        column += 1
        
        # M5 = (A[0,0]+A[1,1]) * (B[1,k0]+B[1,k1])
        U[0 * m + 0][column] = 1  # A[0,0]
        U[1 * m + 1][column] = 1  # A[1,1]
        V[1 * p + k0][column] = 1  # B[1,k0]
        V[1 * p + k1][column] = 1  # B[1,k1]
        W[k0 * n + 0][column] = 1  # C[0,k0]
        W[k1 * n + 0][column] = 1  # C[0,k1]
        column += 1
        
        # M6 = (A[0,1]-A[1,0]) * (B[0,k0]+B[1,k0])
        U[0 * m + 1][column] = 1  # A[0,1]
        U[1 * m + 0][column] = -1  # A[1,0]
        V[0 * p + k0][column] = 1  # B[0,k0]
        V[1 * p + k0][column] = 1  # B[1,k0]
        W[k0 * n + 0][column] = 1  # C[0,k0]
        column += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[0 * m + 1][column] = 1  # A[0,1]
        U[1 * m + 1][column] = -1  # A[1,1]
        V[1 * p + k0][column] = 1  # B[1,k0]
        V[1 * p + k1][column] = 1  # B[1,k1]
        W[k1 * n + 0][column] = 1  # C[0,k1]
        column += 1
        
        # --- Tail part: A[:, 2] * B[2, k0:k1+1] ---
        # This is 2x1 * 1x2 = 2x2, rank 4
        
        # P1 = A[0,2] * B[2,k0]
        U[0 * m + 2][column] = 1  # A[0,2]
        V[2 * p + k0][column] = 1  # B[2,k0]
        W[k0 * n + 0][column] = 1  # C[0,k0]
        column += 1
        
        # P2 = A[0,2] * B[2,k1]
        U[0 * m + 2][column] = 1  # A[0,2]
        V[2 * p + k1][column] = 1  # B[2,k1]
        W[k1 * n + 0][column] = 1  # C[0,k1]
        column += 1
        
        # P3 = A[1,2] * B[2,k0]
        U[1 * m + 2][column] = 1  # A[1,2]
        V[2 * p + k0][column] = 1  # B[2,k0]
        W[k0 * n + 1][column] = 1  # C[1,k0]
        column += 1
        
        # P4 = A[1,2] * B[2,k1]
        U[1 * m + 2][column] = 1  # A[1,2]
        V[2 * p + k1][column] = 1  # B[2,k1]
        W[k1 * n + 1][column] = 1  # C[1,k1]
        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
