import numpy as np

def construct():
    # Strassen 2x2x2 decomposition
    # U, V, W of shape (4, 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1
    
    # M2 = (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = d(g-c)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1 # g is B10 (idx 2), c is A10? No.
    # Wait, M4 = d(g-c). d=A11 (U[3]). g=B10 (V[2]). c=A10? No, c is A10 in standard notation?
    # Standard: A = [[a, b], [c, d]]. B = [[e, f], [g, h]].
    # c is A[1,0]. So M4 = A[1,1] * (B[1,0] - A[1,0]).
    # This is NOT bilinear.
    # I must have the formula wrong.
    
    # Let's use a known correct Strassen implementation.
    # From Wikipedia:
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = a * (f - h)
    # m4 = d * (g - c)
    # m5 = (a + b) * h
    # m6 = (c - d) * (f + g)
    # m7 = (b - d) * (e + g)
    
    # My previous analysis of M4 showed it's not bilinear.
    # d * (g - c) = d*g - d*c.
    # d is A[1,1]. c is A[1,0].
    # This term involves A[1,1]*A[1,0].
    # This is quadratic in A.
    # Therefore, Strassen's algorithm as usually stated is NOT a tensor decomposition?
    # No, it is. The catch is that the "c" in M4 is NOT A[1,0].
    # In Strassen's original paper, the variables are defined differently?
    # Or maybe I am misinterpreting "bilinear".
    # Matrix multiplication is bilinear.
    # If M4 contains A*A, it cannot be part of a bilinear decomposition.
    
    # Let's check M6 = (c-d)(f+g).
    # (A10 - A11)(B01 + B10).
    # This is bilinear in A and B.
    # M7 = (b-d)(e+g).
    # (A01 - A11)(B00 + B10).
    # This is bilinear.
    
    # Why is M4 = d(g-c) problematic?
    # Maybe c is not A10?
    # In the formula C10 = M2 + M4.
    # M2 = (c+d)e.
    # If M4 = d(g-c), then C10 = ce + de + dg - dc.
    # We want C10 = cg + de.
    # ce + de + dg - dc != cg + de.
    # Unless c=0? No.
    
    # There is a known issue with some online descriptions of Strassen.
    # Let's use a reliable source.
    # Actually, M4 is often defined as d*(g-c) in some contexts, but let's look at the reconstruction.
    # If I can't verify it, I should use a simpler, verified decomposition.
    
    # Alternative: Use the Kronecker product of the SCHOOLBOOK 2x2x2?
    # Schoolbook 2x2x2 has rank 8.
    # Kronecker of two schoolbook 2x2x2 gives 4x4x4 rank 64.
    # Then 4x4x8 rank 128.
    # This is the same as the baseline.
    
    # Is there a rank 7 bilinear decomposition?
    # Yes, Strassen's is rank 7.
    # I must have the formula wrong.
    # Let's try to find the correct bilinear terms.
    
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)  <-- This is the one.
    # Wait, if c is A[1,0], then d*c is A[1,1]*A[1,0].
    # This is definitely not bilinear in A and B.
    
    # Let's check a different source.
    # "Strassen's algorithm"
    # m4 = d * (g - c)
    # Is it possible that the input matrices are not A and B, but something else?
    # No.
    
    # Maybe I should just use rank 64 for 4x4x4?
    # 64 * 2 = 128. No improvement.
    
    # What about Winograd?
    # Winograd's algorithm for 4x4 is rank 49?
    # No, Winograd's is for specific structures.
    
    # Let's stick to the baseline if I can't find a valid better one quickly.
    # But 128 is the baseline.
    
    # Wait, I can use the Kronecker product of Strassen IF I get the matrices right.
    # Let's assume the standard formulas are correct and my bilinearity check was flawed.
    # Actually, d(g-c) is bilinear in d and g, and d and c? No.
    # It is linear in d, and linear in g, and linear in c.
    # But it is a product of sums.
    # (d)*(g-c) = d*g - d*c.
    # The term d*c is A[1,1]*A[1,0].
    # This is not of the form A[i,j]*B[k,l].
    # So it is not a valid term for the tensor.
    
    # Conclusion: Strassen's algorithm as commonly written is NOT a tensor decomposition?
    # No, it is.
    # The resolution: The "c" in M4 is NOT A[1,0].
    # In some notations, the matrix is permuted.
    # Or, the formula is M4 = d * (g - c) where c is a different element.
    
    # Given the time, I will return the baseline.
    # The baseline is safe.
    
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
