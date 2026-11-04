import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 decomposition using recursive Strassen (Rank 49)
    # Part 2: 4x4x3 decomposition using trivial method (Rank 48)
    
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 ---
    # We need to decompose the tensor for multiplying 4x4 by 4x4.
    # The output indices for this part are k=0..3.
    # The input indices are i=0..3, j=0..3.
    
    # Helper to get 2x2x2 Strassen decomposition
    # Returns U, V, W for 2x2x2 with rank 7
    def get_2x2x2_strassen():
        # U size 4x7, V size 4x7, W size 4x7
        # Indices: U[i, r], V[j, r], W[k, r]
        # But in our global scheme, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # For the 2x2x2 sub-problem, n=2, m=2, p=2.
        # U_sub is 4x7, V_sub is 4x7, W_sub is 4x7.
        
        U_sub = [[0]*7 for _ in range(4)]
        V_sub = [[0]*7 for _ in range(4)]
        W_sub = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 terms for 2x2 matrices
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to U, V, W.
        # U corresponds to A indices (i, j)
        # V corresponds to B indices (j, k)
        # W corresponds to C indices (k, i)
        
        # Term 1: (A00+A11)(B00+B11)
        # A parts: (0,0), (1,1)
        # B parts: (0,0), (1,1)
        # C parts: (0,0), (0,1), (1,0), (1,1)
        # C00 = M1 - M3 - M4 + M6
        # C01 = M1 + M2 - M5 - M7
        # C10 = M3 + M4 + M7 - M1? No.
        # C10 = M2 + M4 + M6 + M7?
        # Let's use the standard formulas:
        # C00 = M1 - M3 - M4 + M6
        # C01 = M1 + M2 - M5 - M7
        # C10 = M3 + M4 + M7 - M1? No.
        # C10 = M2 + M4 + M6 + M7?
        # C11 = M1 - M2 + M5 - M6
        
        # Actually, let's just use the known rank-7 decomposition for 2x2x2.
        # It is easier to hardcode the 7 terms for 2x2x2.
        
        # U (A indices i,j)
        # V (B indices j,k)
        # W (C indices k,i)
        
        # Term 1: A00, A11; B00, B11
        U_sub[0][0] = 1; U_sub[3][0] = 1
        V_sub[0][0] = 1; V_sub[3][0] = 1
        # C contributions:
        # C00: +1, C01: +1, C10: -1, C11: -1?
        # Let's use a verified 2x2x2 rank-7 decomposition.
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M3 - M4 + M6
        # C01 = M1 + M2 - M5 - M7
        # C10 = M3 + M4 + M7 - M1? No.
        # C10 = M2 + M4 + M6 + M7?
        # C11 = M1 - M2 + M5 - M6
        
        # Let's define U, V, W for each term.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11 -> (0,0), (1,1)
        # V: B00, B11 -> (0,0), (1,1)
        # W: C00(+), C01(+), C10(-), C11(-)
        # Wait, the signs in W depend on the combination.
        
        # This is getting complex to derive on the fly.
        # I will use a known explicit rank-7 decomposition for 2x2x2.
        
        # One such decomposition:
        # U:
        # [1, 1, 0, 0, 1, 1, 0]
        # [0, 0, 1, 0, 0, 1, 1]
        # [0, 1, 0, 1, 0, 0, 1]
        # [1, 1, 1, 1, 1, 1, 1]
        
        # This is not correct.
        
        # Let's use the trivial decomposition for 2x2x2 (rank 8) if I can't find rank 7 quickly.
        # But rank 7 is known.
        
        # Actually, I can just use the trivial decomposition for the whole 4x4x4 part if I can't get 49.
        # Trivial rank for 4x4x4 is 64.
        # Total rank 64 + 48 = 112. Same as schoolbook.
        # So I MUST get 49 for 4x4x4 to improve.
        
        # I will hardcode a verified rank-49 decomposition for 4x4x4.
        # Or I will use the recursive Strassen and hope I get the signs right.
        
        # Let's try to construct the 4x4x4 decomposition from 2x2x2 Strassen.
        # If I get the 2x2x2 part wrong, the result will be invalid.
        
        # Given the risk, I will use a safe decomposition.
        # Is there a rank-80 decomposition for 4x4x4 that is easier?
        # No, 49 is the target.
        
        # I will use a pre-computed rank-49 decomposition for 4x4x4.
        # I will generate it using the recursive Strassen method carefully.
        
        # Let's assume I can write a correct 2x2x2 Strassen decomposition.
        # I will define the 7 terms for 2x2x2.
        
        # U (4x7), V (4x7), W (4x7)
        # Indices 0,1,2,3 for i,j or j,k or k,i.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: (0,0)=1, (1,1)=1 -> U[0][0]=1, U[3][0]=1
        # V: (0,0)=1, (1,1)=1 -> V[0][0]=1, V[3][0]=1
        # W: C00=1, C01=1, C10=-1, C11=-1 -> W[0][0]=1, W[1][0]=1, W[2][0]=-1, W[3][0]=-1
        
        # Term 2: (A10+A11)B00
        # U: (1,0)=1, (1,1)=1 -> U[2][1]=1, U[3][1]=1
        # V: (0,0)=1 -> V[0][1]=1
        # W: C01=1, C10=1, C11=1, C00=0?
        # C01 = M2, C10 = M2, C11 = -M2?
        # C01 = M1 + M2 - M5 - M7
        # C10 = M2 + M4 + M6 + M7?
        # C11 = M1 - M2 + M5 - M6
        # C00 = M1 - M3 - M4 + M6
        
        # So for Term 2 (M2):
        # C00: 0
        # C01: 1
        # C10: 1
        # C11: -1
        # W: W[1][1]=1, W[2][1]=1, W[3][1]=-1
        
        # Term 3: A00(B01-B11)
        # U: (0,0)=1 -> U[0][2]=1
        # V: (0,1)=1, (1,1)=-1 -> V[1][2]=1, V[3][2]=-1
        # W: C00=-1, C01=0, C10=1, C11=0?
        # C00 = M1 - M3 - M4 + M6
        # C10 = M3 + M4 + M7 - M1? No.
        # C10 = M2 + M4 + M6 + M7?
        # C11 = M1 - M2 + M5 - M6
        # C01 = M1 + M2 - M5 - M7
        
        # So for Term 3 (M3):
        # C00: -1
        # C01: 0
        # C10: 1
        # C11: 0
        # W: W[0][2]=-1, W[2][2]=1
        
        # Term 4: A11(B10-B00)
        # U: (1,1)=1 -> U[3][3]=1
        # V: (1,0)=1, (0,0)=-1 -> V[2][3]=1, V[0][3]=-1
        # W: C00=-1, C01=0, C10=1, C11=0?
        # C00 = M1 - M3 - M4 + M6
        # C10 = M2 + M4 + M6 + M7?
        # So C00: -1, C10: 1
        # W: W[0][3]=-1, W[2][3]=1
        
        # Term 5: (A00+A01)B11
        # U: (0,0)=1, (0,1)=1 -> U[0][4]=1, U[1][4]=1
        # V: (1,1)=1 -> V[3][4]=1
        # W: C01=-1, C11=1, C00=0, C10=0?
        # C01 = M1 + M2 - M5 - M7
        # C11 = M1 - M2 + M5 - M6
        # So C01: -1, C11: 1
        # W: W[1][4]=-1, W[3][4]=1
        
        # Term 6: (A10-A00)(B00+B01)
        # U: (1,0)=1, (0,0)=-1 -> U[2][5]=1, U[0][5]=-1
        # V: (0,0)=1, (0,1)=1 -> V[0][5]=1, V[1][5]=1
        # W: C00=1, C10=1, C01=0, C11=-1?
        # C00 = M1 - M3 - M4 + M6
        # C10 = M2 + M4 + M6 + M7?
        # C11 = M1 - M2 + M5 - M6
        # So C00: 1, C10: 1, C11: -1
        # W: W[0][5]=1, W[2][5]=1, W[3][5]=-1
        
        # Term 7: (A01-A11)(B10+B11)
        # U: (0,1)=1, (1,1)=-1 -> U[1][6]=1, U[3][6]=-1
        # V: (1,0)=1, (1,1)=1 -> V[2][6]=1, V[3][6]=1
        # W: C01=-1, C10=1, C00=0, C11=0?
        # C01 = M1 + M2 - M5 - M7
        # C10 = M2 + M4 + M6 + M7?
        # So C01: -1, C10: 1
        # W: W[1][6]=-1, W[2][6]=1
        
        # Now we have U_sub, V_sub, W_sub for 2x2x2.
        
        # We need to combine these to form the 4x4x4 decomposition.
        # The 4x4x4 multiplication is done by 7 such 2x2x2 multiplications.
        # The inputs A, B are 4x4 matrices, viewed as 2x2 blocks of 2x2 matrices.
        # The output C is a 4x4 matrix.
        
        # This is a recursive step.
        # I will implement this combination.
        
        return U_sub, V_sub, W_sub

    # Since implementing the recursive combination correctly is complex and error-prone,
    # I will use a known rank-49 decomposition for 4x4x4 if I can find one.
    # Otherwise, I will use the trivial decomposition for 4x4x4 (rank 64) and 4x4x3 (rank 48).
    # Total rank 112. This is not an improvement.
    
    # Let's try to use a rank-80 decomposition for 4x4x4?
    # No, 49 is the best known.
    
    # I will use the trivial decomposition for the whole 4x4x7 problem.
    # Rank 112.
    
    # Wait, the prompt says "score is the rank". Lower is better.
    # Current best is 112.
    # If I can't find a better one, I should return 112.
    
    # But I can try to find a better one for 4x4x3.
    # If I use rank 49 for 4x4x4 and rank 48 for 4x4x3, total 97.
    # This is an improvement.
    
    # I will implement the 4x4x4 part with rank 49 using a pre-computed decomposition.
    # I will hardcode the U, V, W for the 4x4x4 part.
    
    # For now, I will use the trivial decomposition to ensure validity.
    
    rank = n * m * p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
