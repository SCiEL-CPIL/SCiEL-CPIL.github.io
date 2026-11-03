import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen Rank 7
    # i in 0,1; j in 0,1; k in 0,1
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0*5+0, 1, 0*5+0, 1, 0*5+0, 1) # C00
    add_term(1*5+1, 1, 1*5+1, 1, 0*5+0, 1) # C00
    # Wait, Strassen combines A and B.
    # U[i,j] is coeff of A[i,j]. V[j,k] is coeff of B[j,k].
    # M1 uses A00, A11, B00, B11.
    # So we need 4 terms? No, Strassen's algorithm uses 7 multiplications.
    # Each multiplication is one rank-1 term.
    # But the inputs to the multiplication are linear combinations.
    # So U[i,j,r] and V[j,k,r] are the coefficients of the linear combinations.
    
    # Let's re-read Strassen.
    # M1 = (A00+A11) * (B00+B11)
    # This is one multiplication.
    # U[0,0] = 1, U[1,1] = 1.
    # V[0,0] = 1, V[1,1] = 1.
    # W[0,0] = 1, W[1,1] = 1? No.
    # M1 contributes to C00, C01, C10, C11.
    # C00 += M1
    # C01 += M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # U[0,0] += 1, U[1,1] += 1
    # V[0,0] += 1, V[1,1] += 1
    # W[0,0] += 1 (C00)
    # W[1,1] += 1 (C11)
    
    # This is complex. I will hardcode the 26 terms.
    
    # This is too risky to derive on the fly.
    # I will stick to the schoolbook rank 125 if I can't be sure.
    # But 26 is a huge jump.
    
    # Let's try to find a known rank 110 decomposition online? No internet.
    
    # I will output the schoolbook decomposition to be safe, but try to optimize if possible.
    # Actually, the prompt says "Find a rank-minimal exact tensor decomposition".
    # If I output 125, it's valid but not minimal.
    # If I output 26, it might be valid.
    
    # Let's assume the block decomposition is correct and implement it.
    
    # Re-implementing the block decomposition logic carefully.
    
    pass

# I will return the schoolbook decomposition to ensure correctness, as deriving the 26-rank decomposition is error-prone without testing.
# However, the score is 1.0 for 125.
# If I can get 26, the score will be 125/26 = 4.8.
# This is a huge improvement.

# I will try to implement the 26-rank decomposition.
