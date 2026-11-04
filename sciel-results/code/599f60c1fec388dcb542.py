import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    rank = 20
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank 1 term
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[i*m + j, col] = u_val
        V[j*p + k, col] = v_val
        W[k*n + i, col] = w_val
        col += 1

    # Part 1: A1 B1 (2x1 x 1)
    # A1 is A[:, 0], B1 is B[0, :]
    # C[i, k] += A[i, 0] * B[0, k]
    # Terms: i in 0..1, k in 0..4
    # This is 2 * 5 = 10 terms?
    # Wait, my previous analysis said R(2,1,5)=10.
    # But I split B into B1 (1x5) and B2 (2x5).
    # A1 is 2x1.
    # A1 B1 is 2x5.
    # Rank is 10.
    # My previous sum was 2 + 18 = 20.
    # Where did the 2 come from?
    # Ah, I split B rows into 1 and 2.
    # B1 is 1x5. A1 is 2x1.
    # A1 B1 is 2x5.
    # Rank 10.
    # Then A2 B2 is 2x2 x 5.
    # R(2,2,5) = 18.
    # Total 28.
    
    # My previous "2" was for 2x1 x 1.
    # That implies B1 was 1x1.
    # But B is 3x5.
    # If I split rows 1 and 2, B1 is 1x5.
    # So R(2,1,5) = 10.
    # So total is 10 + 18 = 28.
    
    # Can I do better?
    # Split B columns?
    # B is 3x5.
    # Split into 3x2 and 3x3.
    # A is 2x3.
    # A B1 (2x3 x 2). R(2,3,2).
    # A B2 (2x3 x 3). R(2,3,3).
    # R(2,3,2) = 11 (derived above).
    # R(2,3,3) = 18.
    # Total 29.
    
    # Split B rows into 1, 1, 1?
    # 3 * R(2,1,5) = 30.
    
    # Split A cols into 1, 2?
    # A1 (2x1) B (3x5) -> R(2,1,5)=10.
    # A2 (2x2) B (3x5) -> R(2,2,5)=18.
    # Total 28.
    
    # It seems 25 is the best from the 2x(2+3) split?
    # A B1 (2x3 x 2) = 11.
    # A B2 (2x3 x 3) = 18.
    # Total 29.
    
    # Wait, R(2,3,2) = 11.
    # R(2,3,3) = 18.
    # Sum 29.
    
    # What if R(2,3,3) is less than 18?
    # If R(2,3,3) = 17, then 28.
    
    # Let's stick to the schoolbook 30 if I can't verify a lower rank easily.
    # But 25 was derived from R(2,2,2)=7 and R(2,3,3)=18?
    # No, that was A B1 (2x3 x 2) and A B2 (2x3 x 3).
    # R(2,3,2) = 11.
    # R(2,3,3) = 18.
    # Sum 29.
    
    # Where did 25 come from?
    # "Split 5 into 2+3".
    # A (2x3).
    # B1 (3x2). R(2,3,2) = 11.
    # B2 (3x3). R(2,3,3) = 18.
    # Sum 29.
    
    # Maybe R(2,3,3) is 16?
    # If R(2,3,3) = 16, sum 27.
    
    # Given the uncertainty, I will output the schoolbook rank 30 to be safe, 
    # or try to implement the 28 solution (10+18).
    # 28 is better than 30.
    
    # Let's implement the 28 solution.
    # Part 1: A1 (2x1) * B (3x5) is not right.
    # Split A cols: A1 (2x1), A2 (2x2).
    # A1 * B: 2x1 x 3x5.
    # This is not a standard tensor.
    # A1 is 2x1. B is 3x5.
    # Product is 2x5.
    # C[i,k] = A[i,0] * B[0,k]? No.
    # A1 B = \sum_j A1[i,j] B[j,k].
    # A1 has only j=0.
    # C[i,k] = A[i,0] B[0,k].
    # This is 10 terms.
    
    # Part 2: A2 (2x2) * B (3x5).
    # A2 has j=1,2.
    # C[i,k] += A[i,1] B[1,k] + A[i,2] B[2,k].
    # This is 2x2 x 5.
    # R(2,2,5) = 18.
    
    # Total 28.
    
    pass
# EVOLVE-BLOCK-END
