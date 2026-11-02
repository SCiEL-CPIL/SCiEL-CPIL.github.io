import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    max_rank = n * m * p
    
    # Theoretical lower bound for 2x3x3 is 13
    min_rank = 13
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds margin
    
    # Try ranks from min_rank to max_rank
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Try several random attempts for this rank
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + r)
            
            # Initialize U and V with random small integers
            # U is n*m x r, V is m*p x r
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m * p)]
            
            # W will be constructed row by row
            # W is p*n x r
            W = [[0] * r for _ in range(p * n)]
            
            success = True
            # For each output index (k, i), solve for W[k*n + i, :]
            for k in range(p):
                for i in range(n):
                    row_w = k * n + i
                    # We need to solve for w such that:
                    # sum_c U[i*m + j, c] * V[j*p + k, c] * w[c] = 1 for all j in 0..m-1
                    # This is a system of m equations with r unknowns.
                    # B[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    # B * w = 1
                    
                    # Construct B (m x r)
                    B = [[0] * r for _ in range(m)]
                    for j in range(m):
                        for c in range(r):
                            B[j][c] = U[i * m + j][c] * V[j * p + k][c]
                    
                    # We want to find w (length r) such that B * w = [1, 1, ..., 1]
                    # To keep entries small, we select 3 columns that form an invertible 3x3 matrix
                    # and set the rest to 0.
                    
                    # Find 3 columns that are linearly independent
                    # Since m=3, we need a 3x3 submatrix with non-zero determinant.
                    cols_to_use = []
                    found = False
                    for c1 in range(r):
                        if found: break
                        for c2 in range(c1 + 1, r):
                            if found: break
                            for c3 in range(c2 + 1, r):
                                # Check determinant of B[:, [c1, c2, c3]]
                                det = (B[0][c1] * (B[1][c2] * B[2][c3] - B[1][c3] * B[2][c2]) -
                                       B[0][c2] * (B[1][c1] * B[2][c3] - B[1][c3] * B[2][c1]) +
                                       B[0][c3] * (B[1][c1] * B[2][c2] - B[1][c2] * B[2][c1]))
                                if det != 0:
                                    cols_to_use = [c1, c2, c3]
                                    found = True
                                    break
                            if found: break
                        if found: break
                    
                    if not found:
                        success = False
                        break
                    
                    # Solve the 3x3 system
                    # B_sub * w_sub = [1, 1, 1]
                    # w_sub is a vector of length 3
                    a00 = B[0][cols_to_use[0]]
                    a01 = B[0][cols_to_use[1]]
                    a02 = B[0][cols_to_use[2]]
                    a10 = B[1][cols_to_use[0]]
                    a11 = B[1][cols_to_use[1]]
                    a12 = B[1][cols_to_use[2]]
                    a20 = B[2][cols_to_use[0]]
                    a21 = B[2][cols_to_use[1]]
                    a22 = B[2][cols_to_use[2]]
                    
                    det = (a00 * (a11 * a22 - a12 * a21) -
                           a01 * (a10 * a22 - a12 * a20) +
                           a02 * (a10 * a21 - a11 * a20))
                    
                    # Cramer's rule
                    # b = [1, 1, 1]
                    b0 = 1
                    b1 = 1
                    b2 = 1
                    
                    w0 = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / det
                    w1 = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / det
                    w2 = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / det
                    
                    # Check if w0, w1, w2 are integers or half-integers
                    # They are Fractions if we used Fraction, but here we used integers so they are floats or fractions.
                    # Let's use Fraction for exactness.
                    # Re-calculate with Fraction
                    a00_f = Fraction(a00)
                    a01_f = Fraction(a01)
                    a02_f = Fraction(a02)
                    a10_f = Fraction(a10)
                    a11_f = Fraction(a11)
                    a12_f = Fraction(a12)
                    a20_f = Fraction(a20)
                    a21_f = Fraction(a21)
                    a22_f = Fraction(a22)
                    det_f = (a00_f * (a11_f * a22_f - a12_f * a21_f) -
                             a01_f * (a10_f * a22_f - a12_f * a20_f) +
                             a02_f * (a10_f * a21_f - a11_f * a20_f))
                    
                    w0_f = (Fraction(1) * (a11_f * a22_f - a12_f * a21_f) - a01_f * (Fraction(1) * a22_f - a12_f * Fraction(1)) + a02_f * (Fraction(1) * a21_f - a11_f * Fraction(1))) / det_f
                    w1_f = (a00_f * (Fraction(1) * a22_f - a12_f * Fraction(1)) - Fraction(1) * (a10_f * a22_f - a12_f * a20_f) + a02_f * (a10_f * Fraction(1) - Fraction(1) * a20_f)) / det_f
                    w2_f = (a00_f * (a11_f * Fraction(1) - Fraction(1) * a21_f) - a01_f * (a10_f * Fraction(1) - Fraction(1) * a20_f) + Fraction(1) * (a10_f * a21_f - a11_f * a20_f)) / det_f
                    
                    # Check if they are integers or half-integers
                    def is_half_int(f):
                        return f.denominator in [1, 2]
                        
                    if not (is_half_int(w0_f) and is_half_int(w1_f) and is_half_int(w2_f)):
                        success = False
                        break
                        
                    # Assign to W
                    W[row_w][cols_to_use[0]] = float(w0_f)
                    W[row_w][cols_to_use[1]] = float(w1_f)
                    W[row_w][cols_to_use[2]] = float(w2_f)
                    # Others are 0

            if not success:
                continue
                
            # Verify the solution
            # Check all 18 equations
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # If valid, return the solution
            # Convert to lists of floats (which are exactly representable for halves)
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook if no better solution found
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
# EVOLVE-BLOCK-END
