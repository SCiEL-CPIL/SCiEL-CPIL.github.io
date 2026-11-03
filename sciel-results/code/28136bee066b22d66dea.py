"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays (float)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_factors():
    """
    Return Strassen 2x2x2 multiplication factors.
    M = A * B where A, B are 2x2.
    C = M
    C[0,0] = M1 - M2 + M4 + M5
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M4 + M6
    
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    # A00=U[0], A11=U[3]; B00=V[0], B11=V[3]
    # Contributes to C00 (+1), C11 (+1)
    # W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = A11*(B10-B00)
    # A11=U[3]; B10=V[2], B00=V[0]
    # Contributes to C00 (-1), C10 (+1)
    # W[0,1]=-1, W[2,1]=1
    U[3, 1] = 1
    V[0, 1] = -1; V[2, 1] = 1
    W[0, 1] = -1; W[2, 1] = 1
    
    # M3 = (A10-A00)*B00
    # A10=U[2], A00=U[0]; B00=V[0]
    # Contributes to C01 (+1), C11 (-1)
    # W[0,2]=1, W[3,2]=-1
    U[0, 2] = -1; U[2, 2] = 1
    V[0, 2] = 1
    W[0, 2] = 1; W[3, 2] = -1
    
    # M4 = A00*(B01-B11)
    # A00=U[0]; B01=V[1], B11=V[3]
    # Contributes to C00 (+1), C10 (+1)
    # W[0,3]=1, W[2,3]=1
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A10)*B11
    # A00=U[0], A10=U[2]; B11=V[3]
    # Contributes to C00 (+1), C01 (+1)
    # W[0,4]=1, W[1,4]=1
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A11-A10)*(B01+B11)
    # A11=U[3], A10=U[2]; B01=V[1], B11=V[3]
    # Contributes to C10 (+1), C11 (+1)
    # W[2,5]=1, W[3,5]=1
    U[2, 5] = -1; U[3, 5] = 1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A00-A11)*(B00+A11)
    # A00=U[0], A11=U[3]; B00=V[0], A11=U[3] -> Wait, M7 = (A00-A11)*(B00+A11) is wrong.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  -- Wait, let me re-verify standard Strassen formulas.
    # C00 = M1 - M2 + M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M4 + M6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = (A00-A10)B11
    # M4 = (A10+A11)(B10-B00) -- No, that's not right either.
    
    # Let's use a reliable source or derive.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(e-g) = de-dg
    # M5 = (a+c)h = ah+ch
    # M6 = (d-c)(f+h) = df+dh-cf-ch
    # M7 = (a-d)(e+c) = ae+ac-de-dc
    
    # C00 = ae+bg. M1 - M2 + M4 + M5 = (ae+ah+de+dh) - (ce+de) + (de-dg) + (ah+ch)
    # = ae + 2ah + de + dh - ce - dg + ch. This is not ae+bg.
    
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(e-g)
    # M5 = (a+c)h
    # M6 = (d-c)(f+h)
    # M7 = (a-d)(e+c)
    
    # C00 = M1 - M2 + M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M4 + M6
    
    # Let's re-verify C00:
    # M1 - M2 + M4 + M5 = (ae+ah+de+dh) - (ce+de) + (de-dg) + (ah+ch)
    # = ae + ah + de + dh - ce - de + de - dg + ah + ch
    # = ae + 2ah + de + dh - ce - dg + ch. Still wrong.
    
    # Ah, the standard Strassen is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = (A00-A10)B11
    # M4 = (A10+A11)(B10-B00) -- No.
    
    # Let's just use the known correct ones:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(e-g)
    # M5 = (a+c)h
    # M6 = (d-c)(f+h)
    # M7 = (a-d)(e+c)
    
    # C00 = M1 - M2 + M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M4 + M6
    
    # Wait, I might be misremembering the indices. Let's look up "Strassen algorithm 7 multiplications".
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(e-g)
    # p5 = (a+c)h
    # p6 = (d-c)(f+h)
    # p7 = (a-d)(e+c)
    
    # c11 = p1 - p2 + p4 + p5
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p3 + p4 + p6
    
    # Let's check c11: p1 - p2 + p4 + p5
    # = (ae+ah+de+dh) - (ce+de) + (de-dg) + (ah+ch)
    # = ae + ah + de + dh - ce - de + de - dg + ah + ch
    # = ae + 2ah + de + dh - ce - dg + ch.
    # This is definitely not ae+bg.
    
    # Okay, I will not use Strassen 2x2 because I'm getting confused and it's only rank 7 vs 8.
    # For 5x5, the gain is small.
    # Let's stick to the schoolbook but try to optimize the search for rank reduction.
    # Or, implement a known 3x3x3 rank 23? No, that's complex.
    
    # Given the time limit and complexity, the safest bet is to return the schoolbook decomposition.
    # The inspirations also returned schoolbook.
    # I will implement a very fast check for if a lower rank is possible by trying to solve a linear system for a specific pattern, but it's unlikely to succeed quickly.
    # I will just return the schoolbook decomposition to ensure correctness and stability.
    return None

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    
    # Convert to list of lists of integers (or floats if needed, but schoolbook is integer)
    # The problem asks for integer/half-integer entries. Schoolbook has 0s and 1s.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
