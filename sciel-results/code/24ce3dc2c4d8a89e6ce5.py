"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    def check_exact(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(r):
                        total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if total != 1:
                        return False
        return True
    
    def strassen_block_2x2x2():
        """Return a rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        Returns lists of (U_entries, V_entries, W_entries) where each is a dict mapping (row,col) to value.
        """
        # Strassen's 7 products for 2x2 matrices
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+b)(e+f)
        # M2 = (c+d)(e)
        # M3 = (a)(f-e)
        # M4 = (d)(g-h)
        # M5 = (a+d)(h)
        # M6 = (b+c)(g)
        # M7 = (a-c)(e+g)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        terms = []
        
        # M1 = (a+b)(e+f)
        U1 = {(0,0): 1, (0,1): 1}
        V1 = {(0,0): 1, (0,1): 1}
        W1 = {(0,0): 1, (1,1): 1}  # contributes to C[0,0] and C[1,1] with +1
        # Actually W should be such that W[k,i] gives the coefficient for C[i,k]
        # C[i,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
        # For M1: U[0,0]=1,U[0,1]=1; V[0,0]=1,V[0,1]=1
        # This computes (A[0,0]+A[0,1])(B[0,0]+B[0,1]) = (a+b)(e+f)
        # We want this to contribute to C[0,0] and C[1,1]
        # W[0,0] for C[0,0]: need W[0,0]=1
        # W[1,1] for C[1,1]: need W[1,1]=1
        terms.append((U1, V1, W1))
        
        # M2 = (c+d)(e)
        U2 = {(1,0): 1, (1,1): 1}
        V2 = {(0,0): 1}
        W2 = {(0,1): 1, (1,1): 1}  # contributes to C[1,0] and C[1,1]
        # Wait, C[1,0] = c*e + d*g, C[1,1] = c*f + d*h
        # M2 = (c+d)*e = c*e + d*e
        # This contributes c*e to C[1,0] and d*e to... hmm this doesn't work cleanly.
        # Let me reconsider.
        # Actually for Strassen, the W matrix entries are more complex.
        # Let me just use the known Strassen formulas directly.
        
        # Let me restart with a cleaner approach.
        # The 7 Strassen products:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B10)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B10)
        # P7 = (A01-A11)(B01+B11)
        # C00 = P1 - P3 - P4 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P5 - P6
        
        # For each product, define U (2x2), V (2x2), W (2x2) such that
        # U[i,j] * V[j,k] * W[k,i] sums correctly.
        
        # P1: U = [[1,0],[0,1]], V = [[1,1],[1,1]]
        # U[i,j]*V[j,k] = for i=0,k=0: U[0,0]*V[0,0]+U[0,1]*V[1,0] = 1*1+0*1=1
        # for i=0,k=1: 1*1+0*1=1
        # for i=1,k=0: 0*1+1*1=1  
        # for i=1,k=1: 0*1+1*1=1
        # So U*V = [[1,1],[1,1]], and we want W such that W[0,0]*1 + W[1,1]*1 contributes to C00 and C11
        # C00 gets P1 with coeff 1, C11 gets P1 with coeff 1
        # W[0,0]=1, W[1,1]=1
        terms1 = [
            ({(0,0):1, (1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            # P2: U = [[0,0],[1,1]], V = [[1,0],[0,0]]
            # U*V: i=1,k=0: 1*1+1*0=1, others 0
            # C10 gets P2 with coeff 1
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1}),
            # P3: U = [[1,0],[0,0]], V = [[0,1],[1,-1]]
            # U*V: i=0,k=1: 1*1+0*(-1)=1, i=0,k=0: 1*0+0*1=0
            # C01 gets P3 with coeff 1
            ({(0,0):1}, {(0,1):1,(1,0):1,(1,1):-1}, {(1,0):1}),
            # P4: U = [[0,0],[0,1]], V = [[1,-1],[0,0]]
            # U*V: i=1,k=0: 0*1+1*0=0... wait
            # U[1,0]=0, U[1,1]=1
            # (U*V)[1,0] = U[1,0]*V[0,0] + U[1,1]*V[1,0] = 0*1 + 1*0 = 0
            # (U*V)[1,1] = U[1,0]*V[0,1] + U[1,1]*V[1,1] = 0*(-1) + 1*0 = 0
            # That's wrong. Let me reconsider.
            # P4 = A11*(B01-B10)
            # U should extract A11: U = [[0,0],[0,1]]
            # V should compute B01-B10: V = [[0,1],[1,-1]]? No...
            # V[j,k] should be such that sum_j U[i,j]*V[j,k] gives the right thing.
            # For i=1: sum_j U[1,j]*V[j,k] = U[1,0]*V[0,k] + U[1,1]*V[1,k] = 0*V[0,k] + 1*V[1,k] = V[1,k]
            # So we need V[1,k] = B01-B10 when k... hmm, V is a fixed matrix.
            # V[1,0] should be -1 (coeff of B10), V[1,1] should be 1 (coeff of B01)
            # So V = [[0,0],[ -1, 1]]
            # Then (U*V)[1,0] = -1, (U*V)[1,1] = 1
            # W[0,1] = -1 (for C10), W[1,1] = 1 (for C11)? 
            # C10 gets P4 with coeff 1, C11 gets P4 with coeff -1
            # So W[0,1]=1, W[1,1]=-1
            ({(1,1):1}, {(1,0):-1,(1,1):1}, {(0,1):1,(1,1):-1}),
            # P5: U = [[1,1],[0,0]], V = [[0,0],[0,1]]
            # (U*V)[0,1] = 1*0+1*1=1, others 0
            # C01 gets P5 with coeff 1, C11 gets P5 with coeff 1
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(1,0):1,(1,1):1}),
            # P6: U = [[-1,0],[1,0]], V = [[1,1],[0,0]]
            # (U*V)[0,0]=-1, (U*V)[0,1]=-1, (U*V)[1,0]=1, (U*V)[1,1]=1
            # C11 gets P6 with coeff -1
            ({(0,0):-1,(1,0):1}, {(0,0):1,(0,1):1}, {(1,1):-1}),
            # P7: U = [[0,1],[0,-1]], V = [[0,0],[1,1]]
            # (U*V)[0,1]=1, (U*V)[1,1]=-1
            # C00 gets P7 with coeff 1
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
        ]
        return terms1
    
    def build_kronecker_decomposition():
        """Build a decomposition using Kronecker products of smaller decompositions."""
        # For 3x4x6, we can think of it as combinations of smaller blocks.
        # 3 = 2+1, 4 = 2+2, 6 = 3+3 or 2+2+2
        # This gets complex. Let's try a different structured approach.
        
        # Actually, let's try to find a lower rank by exploiting the fact that
        # 3x4x6 can be decomposed using rank-7 Strassen for 2x2x2 parts.
        
        # Split: 3x4 = (2+1)x(2+2), 4x6 = (2+2)x(3+3)
        # This is getting complicated. Let's try a simpler approach.
        
        # For now, return the schoolbook as fallback.
        return None
    
    def find_lower_rank_via_als(r_target, max_iter=5000):
        """Try to find a rank-r decomposition using ALS with exact rational arithmetic."""
        if r_target >= schoolbook_rank:
            return None
        
        # Initialize with random rationals (small integers)
        U = [[random.randint(1, 3) for _ in range(r_target)] for _ in range(n * m)]
        V = [[random.randint(1, 3) for _ in range(r_target)] for _ in range(m * p)]
        W = [[random.randint(1, 3) for _ in range(r_target)] for _ in range(p * n)]
        
        # ALS: fix two factors, solve for the third
        # For fixed V and W, solve for U:
        # For each (i,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is a system of p equations in r unknowns (U[i*m+j, 0..r-1])
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Solve for U given V, W
            U_new = [[0] * r_target for _ in range(n * m)]
            for i in range(n):
                for j in range(m):
                    # System: for each k, sum_c U_new[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    A_mat = np.zeros((p, r_target))
                    b_vec = np.ones(p)
                    for k in range(p):
                        for c in range(r_target):
                            A_mat[k, c] = V[j*p+k][c] * W[k*n+i][c]
                    # Solve A_mat * x = b_vec
                    try:
                        x = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for c in range(r_target):
                            U_new[i*m+j][c] = int(round(x[c]))
                    except:
                        U_new[i*m+j] = U[i*m+j]
            U = U_new
            
            # Solve for V given U, W
            V_new = [[0] * r_target for _ in range(m * p)]
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r_target))
                    b_vec = np.ones(n)
                    for i in range(n):
                        for c in range(r_target):
                            A_mat[i, c] = U[i*m+j][c] * W[k*n+i][c]
                    try:
                        x = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for c in range(r_target):
                            V_new[j*p+k][c] = int(round(x[c]))
                    except:
                        V_new[j*p+k] = V[j*p+k]
            V = V_new
            
            # Solve for W given U, V
            W_new = [[0] * r_target for _ in range(p * n)]
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r_target))
                    b_vec = np.ones(m)
                    for j in range(m):
                        for c in range(r_target):
                            A_mat[j, c] = U[i*m+j][c] * V[j*p+k][c]
                    try:
                        x = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for c in range(r_target):
                            W_new[k*n+i][c] = int(round(x[c]))
                    except:
                        W_new[k*n+i] = W[k*n+i]
            W = W_new
            
            # Check if we have an exact solution
            if check_exact(U, V, W, r_target):
                return U, V, W
        
        return None
    
    # Try to find a lower rank decomposition
    best_result = None
    
    # Try ranks from 50 up to 71
    for r in range(50, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        
        result = find_lower_rank_via_als(r, max_iter=1000)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, r)
            break  # Found a lower rank, stop
    
    # If no lower rank found, use schoolbook
    if best_result is None:
        rank = schoolbook_rank
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
    else:
        U, V, W, rank = best_result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
