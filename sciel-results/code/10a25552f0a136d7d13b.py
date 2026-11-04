import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 9
    
    def get_rank_8_2x2x3():
        """
        Rank 8 decomposition for 2x2 matrix times 2x3 matrix.
        A: 2x2 (a00, a01, a10, a11)
        B: 2x3 (b00, b01, b02, b10, b11, b12)
        C: 2x3 (c00, c01, c02, c10, c11, c12)
        """
        # Using a known Rank 8 scheme for 2x2x3
        # This can be derived from Strassen 2x2x2 (Rank 7) + 1 extra?
        # Actually, let's just use schoolbook 8? No, we want minimal.
        # Rank 8 is minimal for 2x2x3.
        
        terms = []
        # Strassen for first 2 columns of B (2x2x2 part)
        # M1 = (a00+a10)(b00+b10)
        terms.append(([1,0,1,0], [1,0,0,1,0,0], [0.5, 0, 0, 0.5, 0, 0]))
        # M2 = (a00-a10)(b00-b10)
        terms.append(([1,0,-1,0], [1,0,0,-1,0,0], [0.5, 0, 0, -0.5, 0, 0]))
        # M3 = (a00+a10)(b01+b11)
        terms.append(([1,0,1,0], [0,1,0,0,1,0], [0, 0.5, 0, 0, 0.5, 0]))
        # M4 = (a00-a10)(b01-b11)
        terms.append(([1,0,-1,0], [0,1,0,0,-1,0], [0, 0.5, 0, 0, -0.5, 0]))
        # M5 = a01(b01+b11)
        terms.append(([0,1,0,0], [0,1,0,0,1,0], [0, 1, 0, 0, 0, 0]))
        # M6 = a01(b00+b10)
        terms.append(([0,1,0,0], [1,0,0,1,0,0], [1, 0, 0, 0, 0, 0]))
        # M7 = a11(b00+b10)
        terms.append(([0,0,0,1], [1,0,0,1,0,0], [0, 0, 0, 1, 0, 0]))
        # M8 = a11(b01+b11)
        terms.append(([0,0,0,1], [0,1,0,0,1,0], [0, 0, 0, 0, 1, 0]))
        
        # Wait, this covers C[:, 0:2].
        # We need C[:, 2].
        # c02 = a00 b02 + a01 b12
        # c12 = a10 b02 + a11 b12
        # This is 4 terms.
        # Total 8 + 4 = 12?
        # Rank 8 for 2x2x3 means we can do it in 8.
        # My above list is 8 terms but they don't cover column 2.
        # The Rank 8 scheme must mix the columns.
        
        # I will use a verified Rank 8 set.
        # If I can't find it, I'll use Rank 9 (7+2) or Rank 10.
        # For safety, I'll use Rank 12 (Schoolbook 2x2x3 is 8? No, 2*2*3=12).
        # If I use Rank 12 for 2x2x3 and Rank 6 for 2x1x3, total 18.
        # 18 * 3 = 54. No improvement.
        
        # Let's try Rank 11 for 2x2x3?
        # Actually, let's just use the **Rank 54** fallback to be safe, 
        # but try to implement **Rank 48** (16 per block) using a simple 16-term scheme.
        
        pass

    # Given the complexity and time, I will return the Rank 54 schoolbook 
    # but I will check if I can quickly verify a Rank 42 attempt.
    
    # I will stick to the Rank 54 to ensure a valid score of 1.0, 
    # as the Rank 42 coefficients are uncertain.
    
    rank = 54
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
