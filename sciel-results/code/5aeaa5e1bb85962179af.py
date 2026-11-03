import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # Strassen 2x2x2 block for indices:
        # i in {0, 1}
        # j in {j_off, j_off+1}
        # k in {k_off, k_off+1}
        
        # m1: (a+c)(e+h)
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # m2: c(e+h)
        U[1*m + j_off][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # m3: (a+b)(e+f)
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 0][c] = -1
        c += 1
        
        # m4: b(e+f)
        U[0*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 0][c] = -1
        c += 1
        
        # m5: (a-d)(e-i) -> Wait, standard Strassen m5 is (a-d)(e-i) for 2x2x2?
        # Let's use the standard Strassen formulas carefully.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # m1 = (a+c)(e+h)
        # m2 = c(e+h)
        # m3 = (a+b)(e+f)
        # m4 = b(e+f)
        # m5 = (a-d)(g-i) -> No, g is bottom-left of B?
        # B = [[e, f], [g, h]]
        # m5 = (a-d)(g-i) is incorrect index mapping in my head.
        # Let's use the explicit indices from the previous working code which was verified.
        
        # The previous code used:
        # m1: U[0,j_off]=1, U[1,j_off+1]=1; V[j_off,k_off]=1, V[j_off+1,k_off+1]=1; W[k_off,0]=1, W[k_off+1,1]=1
        # m2: U[1,j_off]=1; V[j_off,k_off]=1, V[j_off+1,k_off+1]=1; W[k_off,1]=1, W[k_off+1,1]=-1
        # m3: U[0,j_off]=1; V[j_off,k_off+1]=1, V[j_off+1,k_off+1]=-1; W[k_off+1,0]=1, W[k_off+1,1]=1
        # m4: U[1,j_off+1]=1; V[j_off+1,k_off]=1, V[j_off,k_off]=-1; W[k_off,0]=1, W[k_off,1]=1
        # m5: U[0,j_off]=1, U[0,j_off+1]=1; V[j_off+1,k_off+1]=1; W[k_off,0]=-1, W[k_off+1,0]=1
        # m6: U[1,j_off]=1, U[0,j_off]=-1; V[j_off,k_off]=1, V[j_off,k_off+1]=1; W[k_off+1,1]=1
        # m7: U[0,j_off+1]=1, U[1,j_off+1]=-1; V[j_off+1,k_off]=1, V[j_off+1,k_off+1]=1; W[k_off,0]=1
        
        # I will replicate the EXACT logic from the current working program to ensure correctness.
        pass

    # Re-implementing the exact working logic from the prompt's current program
    def add_strassen_block_safe(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m2
        U[1*m + j_off][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        # m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        # m5
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        # m6
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    add_strassen_block_safe(0, 0)
    add_strassen_block_safe(0, 2)
    
    # Fill the rest with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    if c != rank:
        # Fallback if logic changes
        return _rank28_solution_fallback()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _rank28_solution_fallback():
    # This is identical to the logic above, provided as a safety net
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        U[1*m + j_off][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    assert c == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
