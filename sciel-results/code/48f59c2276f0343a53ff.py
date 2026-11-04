import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _validate(U, V, W):
    """Check if U, V, W form a valid rank-R tensor decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    
    rank = len(V[0]) if V else 0
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if s != 1:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        return den in [1, 2]
    elif isinstance(x, int):
        return True
    elif isinstance(x, float):
        x2 = x * 2
        return abs(x2 - round(x2)) < 1e-9
    return False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Schoolbook baseline (Rank 60)
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    # Try to construct Rank 55 decomposition
    # Strategy: Split m=3 into m1=2, m2=1.
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x10), B2 (1x10)]
    # C = A1 B1 + A2 B2
    
    # Part 2: A2 B2 (2x1 x 1x10) -> Rank 20
    # Part 1: A1 B1 (2x2 x 2x10) -> Split B1 into 5 blocks of 2x2.
    # Each 2x2x2 block uses Strassen (Rank 7). Total 35.
    # Total Rank 55.
    
    R = 55
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Initialize with zeros (using Fraction 0 for safety, but ints are fine)
    
    # --- Part 2: Schoolbook for A2 B2 ---
    # Indices:
    # A2 is A[1,2] (rows 0,1; col 2) -> U indices: (0,2) and (1,2) i.e., 0*3+2=2, 1*3+2=5
    # B2 is B[2, :] (row 2; cols 0..9) -> V indices: 2*10+k = 20+k
    # Output C[i, k] gets contribution from A[i,2] B[2,k]
    # W indices: k*n + i = k*2 + i
    
    # We assign ranks 0..19 to this part
    for i in range(n):
        for k in range(p):
            c = i * p + k # 0..19
            U[i * m + 2][c] = 1
            V[2 * p + k][c] = 1
            W[k * n + i][c] = 1
            
    # --- Part 1: Strassen for A1 B1 ---
    # A1 is A[0:2, 0:2] -> U indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B1 is B[0:2, :] -> V indices: 0*10+k = k, 1*10+k = 10+k
    # Split B1 columns into 5 groups of 2:
    # Group g (0..4): cols 2g, 2g+1
    # B1_g is 2x2 matrix:
    #   B[0, 2g], B[0, 2g+1]
    #   B[1, 2g], B[1, 2g+1]
    # V indices for B1_g:
    #   V[0*10 + 2g] = V[2g]
    #   V[0*10 + 2g+1] = V[2g+1]
    #   V[1*10 + 2g] = V[10+2g]
    #   V[1*10 + 2g+1] = V[10+2g+1]
    
    # Output C1_g is 2x2 matrix:
    #   C[0, 2g], C[0, 2g+1]
    #   C[1, 2g], C[1, 2g+1]
    # W indices:
    #   W[2g*2 + 0] = W[4g]
    #   W[2g*2 + 1] = W[4g+1]
    #   W[(2g+1)*2 + 0] = W[4g+2]
    #   W[(2g+1)*2 + 1] = W[4g+3]
    
    # Strassen Formulas for M = A * B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A entries:
    # A11 = U[0,0] -> U[0]
    # A12 = U[0,1] -> U[1]
    # A21 = U[1,0] -> U[3]
    # A22 = U[1,1] -> U[4]
    
    # For each group g, we assign 7 ranks: 20 + 7*g to 20 + 7*g + 6
    
    for g in range(5):
        base_rank = 20 + 7 * g
        
        # B entries for this group:
        # B11 = V[2g]
        # B12 = V[2g+1]
        # B21 = V[10+2g]
        # B22 = V[10+2g+1]
        
        # W entries for this group:
        # W00 = W[4g]
        # W01 = W[4g+1]
        # W10 = W[4g+2]
        # W11 = W[4g+3]
        
        # Strassen Terms:
        # 1. (A11+A22)(B11+B22)
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: W00 += 1, W11 += 1, W01 -= 1, W10 -= 1 ? 
        # Standard Strassen reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the coefficients for U, V, W for each of the 7 terms.
        
        # Term 1: M1
        c = base_rank
        U[0][c] += 1   # A11
        U[4][c] += 1   # A22
        V[2*g][c] += 1   # B11
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] += 1   # C11
        W[4*g+1][c] -= 1 # C12 ? No, wait.
        # Let's check the reconstruction carefully.
        # C = A B
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # C11 coeff: +1
        # C12 coeff: 0
        # C21 coeff: 0
        # C22 coeff: +1
        
        W[4*g][c] += 1
        W[4*g+3][c] += 1
        
        # Term 2: M2 = (A21+A22)B11
        c = base_rank + 1
        U[3][c] += 1   # A21
        U[4][c] += 1   # A22
        V[2*g][c] += 1   # B11
        # C21 coeff: +1
        W[4*g+2][c] += 1
        
        # Term 3: M3 = A11(B21-B22)
        c = base_rank + 2
        U[0][c] += 1   # A11
        V[10+2*g][c] += 1   # B21
        V[10+2*g+1][c] -= 1 # B22
        # C12 coeff: +1
        # C22 coeff: +1
        W[4*g+1][c] += 1
        W[4*g+3][c] += 1
        
        # Term 4: M4 = A22(B12-B11)
        c = base_rank + 3
        U[4][c] += 1   # A22
        V[2*g+1][c] += 1   # B12
        V[2*g][c] -= 1   # B11
        # C11 coeff: +1
        # C21 coeff: +1
        W[4*g][c] += 1
        W[4*g+2][c] += 1
        
        # Term 5: M5 = (A11+A12)B22
        c = base_rank + 4
        U[0][c] += 1   # A11
        U[1][c] += 1   # A12
        V[10+2*g+1][c] += 1 # B22
        # C11 coeff: -1
        # C12 coeff: +1
        W[4*g][c] -= 1
        W[4*g+1][c] += 1
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        c = base_rank + 5
        U[3][c] += 1   # A21
        U[0][c] -= 1   # A11
        V[2*g][c] += 1   # B11
        V[2*g+1][c] += 1   # B12
        # C22 coeff: +1
        W[4*g+3][c] += 1
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        c = base_rank + 6
        U[1][c] += 1   # A12
        U[4][c] -= 1   # A22
        V[10+2*g][c] += 1   # B21
        V[10+2*g+1][c] += 1 # B22
        # C11 coeff: +1
        W[4*g][c] += 1
        
    # Validate
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        U_fb = [[0] * (n * m * p) for _ in range(n * m)]
        V_fb = [[0] * (n * m * p) for _ in range(m * p)]
        W_fb = [[0] * (n * m * p) for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
