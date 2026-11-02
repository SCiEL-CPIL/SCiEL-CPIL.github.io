import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    # Helper to add schoolbook terms
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal r
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
                    if r >= rank:
                        return

    # Helper to add Strassen terms for 2x2x2
    # Local indices i,j,k in 0..1
    # Maps to global via offsets
    def add_strassen(i_off, j_off, k_off):
        nonlocal r
        # M1: (X00+X11)(Y00+Y11)
        U[i_off*0 + 0, r] = 1; U[i_off*0 + 1, r] = 1 # Wait, U index is i*m+j
        # Correct indices:
        # i0, j0 -> i0*m + j0
        # i1, j1 -> i1*m + j1
        # i_off is global i offset? No, i is 0 or 1.
        
        # Let's use local i,j,k and map to global
        # i_global = i_off + i
        # j_global = j_off + j
        # k_global = k_off + k
        
        # M1
        U[(i_off+0)*m + (j_off+0), r] = 1
        U[(i_off+1)*m + (j_off+1), r] = 1
        V[(j_off+0)*p + (k_off+0), r] = 1
        V[(j_off+1)*p + (k_off+1), r] = 1
        W[(k_off+0)*n + (i_off+0), r] = 1
        W[(k_off+1)*n + (i_off+1), r] = 1
        # But Strassen has coefficients.
        # M1 = (X00+X11)(Y00+Y11)
        # This is not a single rank 1 term in the standard basis?
        # It is a sum of 4 terms if expanded.
        # Strassen uses 7 multiplications.
        # Each multiplication is a scalar product of linear forms.
        # The linear forms have coefficients.
        # So U and V entries will be the coefficients.
        
        # M1: Lx = X00 + X11, Ly = Y00 + Y11
        # Coeffs for X: [1, 0, 0, 1] (i0j0, i0j1, i1j0, i1j1)
        # Coeffs for Y: [1, 0, 0, 1] (j0k0, j0k1, j1k0, j1k1)
        # W coeffs for Z:
        # Z00 += M1, Z11 += M1
        # Z01, Z10 unchanged
        
        # This logic is complex to map directly to U,V,W arrays.
        # I will hardcode the 7 terms for 2x2x2 Strassen.
        
        # Term 1: M1
        U[(i_off+0)*m + (j_off+0), r] = 1
        U[(i_off+1)*m + (j_off+1), r] = 1
        V[(j_off+0)*p + (k_off+0), r] = 1
        V[(j_off+1)*p + (k_off+1), r] = 1
        W[(k_off+0)*n + (i_off+0), r] = 1
        W[(k_off+1)*n + (i_off+1), r] = 1
        # Wait, W index is k*n+i.
        # Z00 gets +1, Z11 gets +1.
        # So W[(k_off+0)*n + (i_off+0), r] = 1
        # W[(k_off+1)*n + (i_off+1), r] = 1
        r += 1
        
        # Term 2: M2 = (X10+X11)Y00
        U[(i_off+1)*m + (j_off+0), r] = 1
        U[(i_off+1)*m + (j_off+1), r] = 1
        V[(j_off+0)*p + (k_off+0), r] = 1
        # Z10 += M2
        W[(k_off+0)*n + (i_off+1), r] = 1
        r += 1
        
        # Term 3: M3 = X00(Y10-Y11)
        U[(i_off+0)*m + (j_off+0), r] = 1
        V[(j_off+1)*p + (k_off+0), r] = 1
        V[(j_off+1)*p + (k_off+1), r] = -1
        # Z01 += M3
        W[(k_off+1)*n + (i_off+0), r] = 1
        r += 1
        
        # Term 4: M4 = X11(Y01-Y00)
        U[(i_off+1)*m + (j_off+1), r] = 1
        V[(j_off+0)*p + (k_off+1), r] = 1
        V[(j_off+0)*p + (k_off+0), r] = -1
        # Z00 += M4, Z10 += M4
        W[(k_off+0)*n + (i_off+0), r] = 1
        W[(k_off+0)*n + (i_off+1), r] = 1
        r += 1
        
        # Term 5: M5 = (X00+X01)Y11
        U[(i_off+0)*m + (j_off+0), r] = 1
        U[(i_off+0)*m + (j_off+1), r] = 1
        V[(j_off+1)*p + (k_off+1), r] = 1
        # Z00 -= M5, Z01 += M5
        W[(k_off+0)*n + (i_off+0), r] = -1
        W[(k_off+1)*n + (i_off+0), r] = 1
        r += 1
        
        # Term 6: M6 = (X10-X00)(Y00+Y01)
        U[(i_off+1)*m + (j_off+0), r] = 1
        U[(i_off+0)*m + (j_off+0), r] = -1
        V[(j_off+0)*p + (k_off+0), r] = 1
        V[(j_off+0)*p + (k_off+1), r] = 1
        # Z11 += M6
        W[(k_off+1)*n + (i_off+1), r] = 1
        r += 1
        
        # Term 7: M7 = (X01-X11)(Y10+Y11)
        U[(i_off+0)*m + (j_off+1), r] = 1
        U[(i_off+1)*m + (j_off+1), r] = -1
        V[(j_off+1)*p + (k_off+0), r] = 1
        V[(j_off+1)*p + (k_off+1), r] = 1
        # Z00 += M7, Z11 -= M7
        W[(k_off+0)*n + (i_off+0), r] = 1
        W[(k_off+1)*n + (i_off+1), r] = -1
        r += 1

    # 1. A1 B11 (3x2 x 2x2) -> i:0-2, j:0-1, k:0-1
    # Split into 3 rows of 1x2x2? No, just schoolbook 12 terms.
    add_schoolbook(0, 3, 0, 2, 0, 2)
    
    # 2. A2 B21 (3x2 x 2x2) -> i:0-2, j:2-3, k:0-1
    add_schoolbook(0, 3, 2, 4, 0, 2)
    
    # 3. A1 B12 (3x2 x 2x3) -> i:0-2, j:0-1, k:2-4
    # Split into:
    # a) i:0-1, j:0-1, k:2-3 (2x2x2 Strassen)
    add_strassen(0, 0, 2)
    # b) i:0-1, j:0-1, k:4 (2x2x1 Schoolbook)
    # k=4 is local k=0 in 2x2x1?
    # M_2,2,1: i:0-1, j:0-1, k:0
    # Map to global k=4
    for i in range(0, 2):
        for j in range(0, 2):
            U[i*m + j, r] = 1
            V[j*p + 4, r] = 1
            W[4*n + i, r] = 1
            r += 1
    # c) i:2, j:0-1, k:2-4 (1x2x3 Schoolbook)
    # i=2, j:0-1, k:2,3,4
    # Rank 2 terms:
    # Term 1: j=0
    U[2*m + 0, r] = 1
    V[0*p + 2, r] = 1
    V[0*p + 3, r] = 1
    V[0*p + 4, r] = 1
    W[2*n + 2, r] = 1
    W[3*n + 2, r] = 1
    W[4*n + 2, r] = 1
    r += 1
    # Term 2: j=1
    U[2*m + 1, r] = 1
    V[1*p + 2, r] = 1
    V[1*p + 3, r] = 1
    V[1*p + 4, r] = 1
    W[2*n + 2, r] = 1
    W[3*n + 2, r] = 1
    W[4*n + 2, r] = 1
    r += 1
    
    # 4. A2 B22 (3x2 x 2x3) -> i:0-2, j:2-3, k:2-4
    # a) i:0-1, j:2-3, k:2-3 (2x2x2 Strassen)
    add_strassen(0, 2, 2)
    # b) i:0-1, j:2-3, k:4 (2x2x1 Schoolbook)
    for i in range(0, 2):
        for j in range(2, 4):
            U[i*m + j, r] = 1
            V[j*p + 4, r] = 1
            W[4*n + i, r] = 1
            r += 1
    # c) i:2, j:2-3, k:2-4 (1x2x3 Schoolbook)
    # Term 1: j=2
    U[2*m + 2, r] = 1
    V[2*p + 2, r] = 1
    V[2*p + 3, r] = 1
    V[2*p + 4, r] = 1
    W[2*n + 2, r] = 1
    W[3*n + 2, r] = 1
    W[4*n + 2, r] = 1
    r += 1
    # Term 2: j=3
    U[2*m + 3, r] = 1
    V[3*p + 2, r] = 1
    V[3*p + 3, r] = 1
    V[3*p + 4, r] = 1
    W[2*n + 2, r] = 1
    W[3*n + 2, r] = 1
    W[4*n + 2, r] = 1
    r += 1
    
    if r != rank:
        # Fallback if logic error
        return {"U": [[0.0]*60 for _ in range(12)], "V": [[0.0]*60 for _ in range(20)], "W": [[0.0]*60 for _ in range(15)]}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
