import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into m1=1, m2=2
    m1 = 1
    m2 = 2
    
    # Block 1: 2x1x5
    # Rank 10
    U1 = []
    V1 = []
    W1 = []
    for i in range(n):
        for k in range(p):
            U1.append([1 if (i == i and 0 == 0) else 0 for _ in range(1)]) # U[i*m1+j, c]
            # Actually, let's build the matrices properly.
            
    # Let's just build the full U, V, W matrices of size rank x (n*m, m*p, p*n)
    # Rank 10 for 2x1x5
    rank1 = n * m1 * p
    U_block1 = np.zeros((n * m1, rank1))
    V_block1 = np.zeros((m1 * p, rank1))
    W_block1 = np.zeros((p * n, rank1))
    
    c = 0
    for i in range(n):
        for k in range(p):
            U_block1[i * m1 + 0, c] = 1
            V_block1[0 * p + k, c] = 1
            W_block1[k * n + i, c] = 1
            c += 1
            
    # Block 2: 2x2x5
    # Rank 18 (Strassen 7 + Lader 11)
    # I will use schoolbook for simplicity if I don't have Lader coeffs.
    # Schoolbook rank is 2*2*5 = 20.
    # Strassen + Schoolbook(2x2x3) = 7 + 12 = 19.
    # Strassen + Lader(2x2x3) = 7 + 11 = 18.
    # I will use rank 18.
    
    # I need Lader 11 for 2x2x3.
    # Lader 11 coefficients for 2x2x3:
    # This is getting complex. I will use rank 20 for block 2 to be safe?
    # Rank 20 + 10 = 30. No improvement.
    # I must use rank 18 or 17.
    # I will use rank 19 (Strassen 7 + Schoolbook 12).
    # 19 + 10 = 29.
    # Score 30/29 = 1.034.
    # Rank 28 was measured. So 28 is better than 29.
    # I will use rank 28.
    
    # Block 2 Rank 18 construction:
    # Split 2x2x5 into 2x2x2 (Strassen 7) and 2x2x3 (Lader 11).
    # I don't have Lader 11 coeffs.
    # I will use Schoolbook 12 for 2x2x3.
    # 7 + 12 = 19.
    # Total 29.
    
    # Wait, the knowledge says rank 28 was achieved.
    # "Strassen(7)+schoolbook(4) for the 2x2x5 block".
    # This implies the 2x2x5 block was rank 11?
    # No, 7+4=11.
    # Maybe the 2x2x5 block was split into 2x2x2 (Strassen 7) and 2x2x3 (Rank 4)?
    # Rank 4 for 2x2x3? No, rank is 11.
    # Maybe "schoolbook(4)" means something else.
    # I will just implement rank 28 using a known valid method.
    # If I can't find it, I will use rank 29.
    
    # Let's use rank 28.
    # I will assume the rank 28 construction is:
    # 2x1x5 (10) + 2x2x5 (18).
    # I will implement 2x2x5 with rank 18.
    # 2x2x5 = 2x2x2 (Strassen 7) + 2x2x3 (Lader 11).
    # I will use a placeholder for Lader 11.
    # If I can't, I will use schoolbook 12.
    # Then rank is 19. Total 29.
    
    # I will output rank 29.
    pass
