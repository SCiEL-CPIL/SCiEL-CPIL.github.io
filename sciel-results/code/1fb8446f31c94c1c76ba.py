"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Return a rank-50 tensor decomposition for 2x3x9 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2 blocks (j={0,1}, k in pairs) plus schoolbook for remainder.
    U: (6, 50), V: (27, 50), W: (18, 50)
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Rank-50 construction:
    # j in {0,1} with k in pairs {0,1},{2,3},{4,5},{6,7}: Strassen 2x2x2 -> 7 each = 28
    # j in {0,1} with k=8: schoolbook 2x2x1 -> 4
    # j=2 with k in 0..8: schoolbook 2x1x9 -> 18
    # Total: 28 + 4 + 18 = 50
    
    rank = 50
    U = [[0]*rank for _ in range(n*m)]  # 6 x 50
    V = [[0]*rank for _ in range(m*p)]  # 27 x 50
    W = [[0]*rank for _ in range(p*n)]  # 18 x 50
    
    col = 0
    
    # Strassen 2x2x2: for A(2x2) * B(2x2) = C(2x2)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen_2x2x2(i_base, j0, j1, k0, k1):
        """Add 7 Strassen products for A[i_base:i_base+2, j0:j1+1] * B[j0:j1+1, k0:k1+1].
        i_base: starting row of A (0 or 1)
        j0, j1: the two j indices (0,1)
        k0, k1: the two k indices
        """
        global col
        # A indices: A[i, j] -> U[i*m + j]
        # B indices: B[j, k] -> V[j*p + k]
        # C indices: C[i, k] -> W[k*n + i]
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        u = [0]*6; v = [0]*27; w = [0]*18
        u[i_base*m + j0] = 1; u[(i_base+1)*m + j1] = 1
        v[j0*p + k0] = 1; v[j1*p + k1] = 1
        w[k0*n + i_base] = 1; w[k1*n + i_base] = 1
        w[k0*n + i_base+1] = 1; w[k1*n + i_base+1] = 1
        # Wait, this isn't right. Let me think again.
        
        # Actually, in the tensor decomposition, each "product" M_c corresponds to one column.
        # U[i,j,c] gives the coefficient for A[i,j] in product c
        # V[j,k,c] gives the coefficient for B[j,k] in product c
        # W[k,i,c] gives the coefficient for C[k,i] in product c
        # 
        # For Strassen M1 = (A00+A11)(B00+B11):
        # This product contributes to C00, C01, C10, C11 with coefficients from the reconstruction.
        # But in the tensor decomposition, each column c is ONE scalar product.
        # The product value P_c = (sum U[i,j,c]*A[i,j]) * (sum V[j,k,c]*B[j,k])
        # and C[i,k] += W[k,i,c] * P_c
        #
        # So for M1: U has 1 at (i0,j0) and (i1,j1), V has 1 at (j0,k0) and (j1,k1)
        # But that's a product of TWO sums, which gives 4 terms: A00*B00, A00*B11, A11*B00, A11*B11
        # That's NOT what we want. We want just the product (A00+A11)*(B00+B11) as ONE product.
        #
        # In tensor decomposition, one column = one scalar multiplication.
        # P_c = (linear combo of A) * (linear combo of B)
        # So U[i,j,c] can be 1 at multiple (i,j) positions - that's fine, it's one product of two sums.
        # Similarly V[j,k,c] can be 1 at multiple (j,k) positions.
        #
        # So for M1: U[i0,j0,c]=1, U[i1,j1,c]=1, V[j0,k0,c]=1, V[j1,k1,c]=1
        # P_c = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # Then W[k0,i0,c] += 1, W[k0,i1,c] += 1, W[k1,i0,c] += 1, W[k1,i1,c] += 1
        # (because M1 contributes +1 to all four C entries)
        
        # Let me redo this properly.
        # Clear and restart
        pass
    
    # Let me rebuild this more carefully.
    # Each column c defines: P_c = (Σ_{i,j} U[i,j,c] * A[i,j]) * (Σ_{j,k} V[j,k,c] * B[j,k])
    # C[i,k] = Σ_c W[k,i,c] * P_c
    
    # For the schoolbook term A[i,j]*B[j,k] contributing to C[i,k]:
    # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1
    
    # For Strassen M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]):
    # U[i0,j0,c]=1, U[i1,j1,c]=1
    # V[j0,k0,c]=1, V[j1,k1,c]=1
    # This product expands to A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j1,k1] + A[i1,j1]*B[j0,k0] + A[i1,j1]*B[j1,k1]
    # We need this to contribute to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
    # M1 contributes: +1 to C[i0,k0], +1 to C[i0,k1], +1 to C[i1,k0], +1 to C[i1,k1]
    # W[k0,i0,c]=1, W[k1,i0,c]=1, W[k0,i1,c]=1, W[k1,i1,c]=1
    
    # But wait - the cross terms A[i0,j0]*B[j1,k1] and A[i1,j1]*B[j0,k0] are "wrong" products
    # that don't appear in the original matrix multiplication! They'll be canceled by other Strassen terms.
    
    # This is correct - Strassen works by having the cross terms cancel in the reconstruction.
    
    # Let me implement the full Strassen 2x2x2 for i-block {i0,i1}, j-block {j0,j1}, k-block {k0,k1}
    
    def strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen columns for the 2x2x2 subproblem."""
        nonlocal col
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # Contributes +1 to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
        c = col
        U[i0*m+j0][c] = 1; U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1; V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1; W[k1*n+i0][c] = 1
        W[k0*n+i1][c] = 1; W[k1*n+i1][c] = 1
        col += 1
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # Contributes +1 to C[i1,k0]
        c = col
        U[i1*m+j0][c] = 1; U[i1*m+j1][c] = 1
        V[j0*p+k0][c] = 1
        W[k0*n+i1][c] = 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # Contributes +1 to C[i0,k1], -1 to C[i0,k0]
        # Wait: M3 = A[i0,j0]*(B[j1,k0]-B[j1,k1])
        # In C[i0,k0]: M3 contributes -1 (from -B[j1,k1]... no)
        # Let me recheck Strassen formulas:
        # C00 = M1 - M5 + M7 + M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # Using 0-indexed: i0=0,i1=1,j0=0,j1=1,k0=0,k1=1
        # A00=A[i0,j0], A01=A[i0,j1], A10=A[i1,j0], A11=A[i1,j1]
        # B00=B[j0,k0], B01=B[j0,k1], B10=B[j1,k0], B11=B[j1,k1]
        # C00=C[i0,k0], C01=C[i0,k1], C10=C[i1,k0], C11=C[i1,k1]
        
        # M3 = A00*(B10-B11) = A[i0,j0]*(B[j1,k0]-B[j1,k1])
        # C[i0,k0] += M3 * 0? No: C00 = M1-M5+M7+M4, so M3 doesn't appear in C00.
        # C01 = M3+M5, so M3 contributes +1 to C[i0,k1]
        # Wait, that doesn't seem right either. Let me verify:
        # C01 should be A00*B01 + A01*B11
        # M3+M5 = A00*(B10-B11) + (A00+A01)*B11 = A00*B10 - A00*B11 + A00*B11 + A01*B11 = A00*B10 + A01*B11
        # But C01 = A00*B01 + A01*B11. So A00*B10 ≠ A00*B01 in general!
        # I'm confusing myself. Let me use the standard Strassen carefully.
        
        # Standard Strassen for C = A*B where A,B are 2x2:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-g)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        # C11 = M1 - M5 + M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Mapping: a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
        #           e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
        # C[i0,k0] = ae+bg = M1-M5+M7+M4
        # C[i0,k1] = af+bh = M3+M5
        # C[i1,k0] = ce+dg = M2+M4
        # C[i1,k1] = cf+dh = M1-M2+M3+M6
        
        # M3 = a*(f-g) = A[i0,j0]*(B[j0,k1]-B[j1,k0])
        # Contributes +1 to C[i0,k1]
        c = col
        U[i0*m+j0][c] = 1
        V[j0*p+k1][c] = 1; V[j1*p+k0][c] = -1
        W[k1*n+i0][c] = 1
        col += 1
        
        # M4 = d*(h-g) = A[i1,j1]*(B[j1,k1]-B[j1,k0])
        # Contributes +1 to C[i0,k0], +1 to C[i1,k0]
        c = col
        U[i1*m+j1][c] = 1
        V[j1*p+k1][c] = 1; V[j1*p+k0][c] = -1
        W[k0*n+i0][c] = 1; W[k0*n+i1][c] = 1
        col += 1
        
        # M5 = (a+b)*h = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        # Contributes -1 to C[i0,k0], +1 to C[i0,k1]
        c = col
        U[i0*m+j0][c] = 1; U[i0*m+j1][c] = 1
        V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = -1; W[k1*n+i0][c] = 1
        col += 1
        
        # M6 = (c-a)*(e+g) = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j1,k0])
        # Contributes +1 to C[i1,k1]
        c = col
        U[i1*m+j0][c] = 1; U[i0*m+j0][c] = -1
        V[j0*p+k0][c] = 1; V[j1*p+k0][c] = 1
        W[k1*n+i1][c] = 1
        col += 1
        
        # M7 = (b-d)*(f+h) = (A[i0,j1]-A[i1,j1])*(B[j0,k1]+B[j1,k1])
        # Contributes +1 to C[i0,k0]
        c = col
        U[i0*m+j1][c] = 1; U[i1*m+j1][c] = -1
        V[j0*p+k1][c] = 1; V[j1*p+k1][c] = 1
        W[k0*n+i0][c] = 1
        col += 1
    
    # Apply Strassen for j-block {0,1} and k-pairs
    # i-block is {0,1} (both rows of A)
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    for k0, k1 in k_pairs:
        strassen_2x2x2(0, 1, 0, 1, k0, k1)
    
    # Schoolbook for j-block {0,1}, k=8 (2x2x1 = 4 products)
    for i in range(2):
        for j in range(2):
            c = col
            U[i*m+j][c] = 1
            V[j*p+8][c] = 1
            W[8*n+i][c] = 1
            col += 1
    
    # Schoolbook for j=2, all k (2x1x9 = 18 products)
    for i in range(2):
        for k in range(9):
            c = col
            U[i*m+2][c] = 1
            V[2*p+k][c] = 1
            W[k*n+i][c] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify: T[i,j,k] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] should be 1
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    verified = False
                    print(f"FAIL at i={i},j={j},k={k}: total={total}")
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
