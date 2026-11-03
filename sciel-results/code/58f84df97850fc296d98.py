"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 140  # seconds, with safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try Strassen-based decomposition: split 4-dim into two 2-dims
    # Each 3x2x2x5 sub-problem uses Strassen-like rank-7 method
    result = _strassen_3x4x5(n, m, p)
    if result is not None:
        return result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

def _strassen_3x4x5(n, m, p):
    """
    Decompose 3x4x5 matrix multiplication using two Strassen 2x2x2 blocks.
    Split the 4-dimension (j) into two halves of size 2.
    For each half, use a rank-7 Strassen-like decomposition.
    Total rank: 7 + 7 = 14? No, need to be more careful.
    
    Actually, for 3x2 x 2x5, the tensor structure requires more care.
    Let me use the known Strassen formulas adapted for rectangular cases.
    """
    # Strassen 2x2x2: rank 7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 3x2 x 2x5: A is 3x2, B is 2x5
    # Think of A as having 3 rows, 2 cols; B as 2 rows, 5 cols
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # We can use a Strassen-like decomposition with 7 multiplications
    # Define:
    # M1 = (A[:,0]+A[:,1]) * (B[0,:]+B[1,:])  -> 3x5 matrix
    # M2 = A[:,1] * B[0,:]  -> 3x5
    # M3 = A[:,0] * (B[1,:]-B[0,:])  -> 3x5
    # M4 = A[:,1] * (B[0,:]-B[1,:])  -> 3x5  [wait, this is -M3 + ...]
    
    # Let me use the standard Strassen mapping:
    # Treat the 2x2 core: A_core = [[A[i,0]], [A[i,1]]] for each i
    # B_core = [[B[0,k]], [B[1,k]]] for each k
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # In our case, for each (i,k):
    # A00 = A[i,0], A10 = A[i,1] (since A is 3x2, the "2x2" is across the j dimension)
    # B00 = B[0,k], B10 = B[1,k] (B is 2x5)
    # C00 = C[i,k] (only one output per i,k)
    
    # So: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Using Strassen with the 2x2 structure where:
    # "A00" = A[i,0], "A10" = A[i,1] (column 0 and 1 of A for row i)
    # "B00" = B[0,k], "B10" = B[1,k] (row 0 and 1 of B for column k)
    
    # The 7 multiplications become:
    # M1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    # M2 = (A[i,1]+A[i,1]) * B[0,k] = 2*A[i,1]*B[0,k]  [A10+A11 where A11=A10 in our case?]
    
    # I'm getting confused. Let me use a different approach.
    
    # The 3x2 x 2x5 multiplication can be done with rank 7 using:
    # For each i (3 values) and k (5 values), C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    
    # This is a bilinear map. The tensor T[i,j,k] for j in {0,1} is:
    # T[i,j,k] = 1 for all i,j,k
    
    # But as we discussed, the actual tensor to decompose is the structured one.
    
    # Let me just implement the schoolbook for now and verify correctness,
    # then try to find a lower-rank decomposition.
    
    # Actually, let me implement the known rank-58 decomposition.
    # The idea: split j into j=2a+b, a in {0,1}, b in {0,1}
    # For each a, we have a 3x2 x 2x5 sub-problem.
    # Each sub-problem can be done with rank 7 (Strassen-like).
    # Total rank: 7 + 7 = 14? That seems too low.
    
    # I think the issue is that the 3x2 x 2x5 tensor has higher rank than 7.
    # Let me compute: the tensor for 3x2 x 2x5 has shape (3*2) x (2*5) x (5*3) = 6x10x15.
    # The rank is at least max(6,10,15) = 15? No, that's not a valid lower bound.
    
    # Let me just implement schoolbook and return it. The Strassen improvement
    # requires more careful analysis.
    
    return None
# EVOLVE-BLOCK-END
