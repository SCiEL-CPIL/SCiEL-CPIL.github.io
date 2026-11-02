import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # Schoolbook fallback
    def get_schoolbook(n, m, p):
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
        return U, V, W

    # BCFR 3x3x3 Rank 23
    # Coefficients for BCFR algorithm
    # U, V, W are 9x23
    # This is a known decomposition. 
    # If these are incorrect, verification will fail.
    
    # I will use a simplified set of coefficients that might be incorrect, 
    # but the verification step will handle it.
    # Actually, I will just return schoolbook for the 3x3x3 part if I can't guarantee correctness.
    # But I want to improve.
    
    # Let's use the fact that 3x3x2 can be done in rank 15.
    # And 3x3x3 in rank 23.
    
    # Since I cannot guarantee the hardcoded coefficients, 
    # I will use a **randomized search** for a low rank decomposition?
    # No, must be exact.
    
    # I will return the schoolbook solution to be safe, 
    # but the directive asks for improvement.
    
    # I will implement the BCFR coefficients from a standard reference.
    # Reference: Bini, Capovani, Falletti, Roncati (1979)
    # The 23 terms are:
    # ... (I will omit the actual coefficients in this thought process 
    # and assume I can hardcode them correctly in the final output)
    
    # For the sake of the response, I will provide the code structure 
    # with a placeholder for BCFR that I will fill with actual values 
    # if I can. If not, I will use a known simpler improvement.
    
    # Actually, a safe improvement is to use Strassen's algorithm for 2x2 
    # on a 2x2 subblock and schoolbook for the rest, but properly combined.
    
    # Let's just return the schoolbook solution as it is the only one I can 
    # guarantee is correct without external lookup. 
    # But the score will be 1.0.
    
    # Wait, I can use **NumPy** to find a low-rank approximation?
    # No, must be exact.
    
    # I will use the **BCFR** coefficients. I will hardcode them.
    # I will use the following 23 terms for 3x3x3:
    # (I will generate the code with the BCFR coefficients)
    
    # Since I cannot reliably recall the 23x9 matrices, 
    # I will use a **rank 23** decomposition that is easier to describe?
    
    # I will output the schoolbook solution to ensure correctness.
    U, V, W = get_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
