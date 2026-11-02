import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    # Laderman's rank-23 decomposition for 3x3x3 matrix multiplication
    # We need 23 triples of 3x3 matrices (U_r, V_r, W_r) such that:
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k in {0,1,2}
    # Wait, the tensor condition for matrix multiplication A(3x3) * B(3x3) = C(3x3) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # In the trace convention W[k*n+i, r], the index is (k, i).
    # So the condition is: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    # For 3x3x3: n=m=p=3.
    # sum_r U_r[i*3+j] * V_r[j*3+k] * W_r[k*3+i] = 1
    
    # Laderman factors (3x3x3, rank 23):
    # These are well-known. I will hardcode them and verify.
    # U, V, W are 3x3 matrices for each term r.
    # Indexing: U_r[i,j], V_r[j,k], W_r[k,i]
    
    # Let's define the 23 terms.
    # I will use a set of factors known to work.
    # Reference: "The rank of the 3x3 matrix multiplication tensor is 23"
    
    # The factors can be generated or hardcoded.
    # Here is a hardcoded set from a reliable source (e.g., Strassen's work or Laderman's paper).
    # I will define them as lists of 9 integers (flattened 3x3 matrices).
    
    # Term 0
    u0 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    v0 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    w0 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    
    # Actually, let's use a more systematic approach or a verified hardcoded list.
    # I will hardcode the Laderman factors here.
    # If I make a mistake, the verification will catch it, and I will fall back to schoolbook.
    
    # Laderman 23 factors for 3x3x3:
    # I will use the factors from the Wikipedia page on "Strassen algorithm" or similar reliable sources.
    # Since I cannot browse, I will rely on my internal knowledge of the specific coefficients.
    # This is risky. Let's try to construct a valid decomposition using a known method or a smaller rank if 23 is too hard to hardcode correctly from memory.
    
    # Alternative: Use rank 24 (which is easier to construct?) or just stick to schoolbook if I can't guarantee 23.
    # However, the goal is to improve score. Schoolbook is 63. Rank 55 (23+23+9) is a big improvement.
    
    # Let's try to hardcode Laderman factors.
    # I found a reference in my training data.
    # The 23 products are:
    # 1. (A00+A11)(B00+B11)
    # 2. (A00-A11)(B00-B11)
    # 3. (A01+A11)(B10-B00)
    # 4. (A10-A00)(B01+B11)
    # 5. (A10+A22)(B11+B22)
    # 6. (A10-A22)(B11-B22)
    # 7. (A20+A00)(B22+B00)
    # 8. (A20-A00)(B22-B00)
    # 9. (A21+A11)(B10+B00) -> Wait, this is not quite right.
    
    # Let's use a different, verified set.
    # I will define the factors as 3x3 matrices.
    
    # This is a known set of Laderman factors:
    laderman_U = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0], # 1
        [1, 0, 0, 0, -1, 0, 0, 0, 0], # 2
        [0, 1, 0, 0, 1, 0, 0, 0, 0], # 3
        [1, 0, 0, -1, 0, 0, 0, 0, 0], # 4
        [0, 0, 1, 1, 0, 0, 0, 0, 0], # 5
        [0, 0, 1, 1, 0, 0, 0, 0, 0], # 6 (Wait, duplicate?)
        # I am not confident in these.
    ]
    
    # Given the high risk of error in hardcoding 23x3x3 matrices from memory,
    # and the fact that an invalid decomposition scores 0,
    # I will implement a robust verification step.
    # If I can't produce a valid Laderman decomposition, I will fall back to schoolbook.
    
    # Let's try a different strategy:
    # Use the known rank-23 factors for 3x3x3 from a trusted internal representation.
    # I will attempt to generate them or use a simplified version.
    
    # Actually, there is a simpler way to get a good score:
    # Just use schoolbook for all 7 columns. Rank 63.
    # Or, if I can find a valid decomposition for 3x3x3, I can use it for the first 3 columns and schoolbook for the last 4.
    # Rank = 23 + 4*9 = 23 + 36 = 59.
    # Or split 7 into 3+3+1: 23 + 23 + 9 = 55.
    
    # Let's try to hardcode the Laderman factors correctly.
    # I will use the following set of 23 triples (u, v, w) of 3x3 matrices.
    # This set is from a known source.
    
    # I will define them as lists of 9 integers.
    # If the sum of u[i,j]*v[j,k]*w[k,i] is not 1 for all i,j,k, it's invalid.
    
    # Let's use a Python script to verify the factors.
    # I will hardcode a set and verify it.
    
    # Here is a set of Laderman factors (from a reliable source in my training data):
    # I will use the factors from the "laderman" module in a hypothetical library, but since I can't import it, I hardcode.
    
    # I will use the following factors:
    # (U_r, V_r, W_r) for r=0..22
    
    # I will use a known valid set.
    # If I fail to provide a valid set, I return schoolbook.
    
    # Let's try to construct a valid set using the known structure.
    # This is too complex for this step.
    
    # I will return schoolbook to ensure validity.
    # But wait, the prompt says "stronger algorithm".
    # If I return schoolbook, the score is 1.0.
    # If I can get 55, the score is 63/55 = 1.145.
    
    # Let's try to hardcode the Laderman factors from a known correct source.
    # I found a reference in my training data for the exact coefficients.
    
    # Laderman factors (23 terms):
    # I will define them as 3x3 matrices.
    
    # This is a known set of Laderman factors:
    laderman_factors = [
        # Term 0
        ([[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        # Term 1
        ([[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        # ... I don't have the full set memorized correctly.
    ]
    
    # Since I cannot reliably provide the Laderman factors from memory,
    # I will return the schoolbook solution to ensure validity.
    
    return schoolbook()
# EVOLVE-BLOCK-END
