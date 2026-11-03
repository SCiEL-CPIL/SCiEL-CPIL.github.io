"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def add_strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for a 2x2x2 block."""
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        add_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[j0*p+k1] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j1] = 1
        v[j1*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = -1; w[k1*n+i0] = 1
        add_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+k1] = 1
        w[k1*n+i1] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j1*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        add_term(u, v, w)
    
    def add_winograd_3x3x3(i0, j0, k0):
        """Add 23 Winograd terms for a 3x3x3 block.
        i0, j0, k0 are the base indices for i, j, k.
        Local indices 0,1,2 map to i0+local, j0+local, k0+local.
        """
        # Winograd 3x3 fast matrix multiplication
        # A is 3x3, B is 3x3, C = A*B is 3x3
        # Uses 23 multiplications
        
        # The Winograd algorithm for 3x3:
        # M1  = (A00+A11+A22)(B00+B11+B22)
        # M2  = (A01+A12+A20)(B01+B12+B20)
        # M3  = (A02+A10+A21)(B02+B10+B21)
        # M4  = (A00+A11-A22)(B00-B11+B22)
        # M5  = (A01-A12+A20)(B01-B12+B20)
        # M6  = (A02-A10+A21)(B02-B10+B21)
        # M7  = (A00-A11+A22)(B00+B11-B22)
        # M8  = (A01-A12-A20)(B01+B12-B20)
        # M9  = (A02-A10-A21)(B02+B10-B21)
        # M10 = (A00+A11+A22+A01+A12+A20)(B00+B11+B22+B01+B12+B20)
        # ... this is getting complex. Let me use a known correct set.
        
        # Actually, let me use a verified Winograd 3x3 decomposition.
        # The standard Winograd 3x3 uses these 23 products:
        
        # Let me define the coefficients explicitly.
        # Each term is (u_coeffs, v_coeffs, w_coeffs) where each is a dict mapping local (i,j), (j,k), (i,k) to coefficient.
        
        terms = []
        
        # I'll use the decomposition from Winograd's original work.
        # The 23 multiplications are:
        
        # Define helper to add a term
        def add_wterm(u_dict, v_dict, w_dict):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for (li, lj), coeff in u_dict.items():
                u[(i0+li)*m + (j0+lj)] += coeff
            for (lj, lk), coeff in v_dict.items():
                v[(j0+lj)*p + (k0+lk)] += coeff
            for (lk, li), coeff in w_dict.items():
                w[(k0+lk)*n + (i0+li)] += coeff
            add_term(u, v, w)
        
        # Winograd 3x3: 23 terms
        # M1 = (A00+A11+A22)(B00+B11+B22)
        add_wterm(
            {(0,0):1, (1,1):1, (2,2):1},
            {(0,0):1, (1,1):1, (2,2):1},
            {(0,0):1, (1,1):1, (2,2):1}
        )
        
        # M2 = (A01+A12+A20)(B01+B12+B20)
        add_wterm(
            {(0,1):1, (1,2):1, (2,0):1},
            {(0,1):1, (1,2):1, (2,0):1},
            {(1,0):1, (2,1):1, (0,2):1}
        )
        
        # M3 = (A02+A10+A21)(B02+B10+B21)
        add_wterm(
            {(0,2):1, (1,0):1, (2,1):1},
            {(0,2):1, (1,0):1, (2,1):1},
            {(2,0):1, (0,1):1, (1,2):1}
        )
        
        # M4 = (A00+A11-A22)(B00-B11+B22)
        add_wterm(
            {(0,0):1, (1,1):1, (2,2):-1},
            {(0,0):1, (1,1):-1, (2,2):1},
            {(0,0):1, (1,1):-1, (2,2):1}
        )
        
        # M5 = (A01-A12+A20)(B01-B12+B20)
        add_wterm(
            {(0,1):1, (1,2):-1, (2,0):1},
            {(0,1):1, (1,2):-1, (2,0):1},
            {(1,0):1, (2,1):-1, (0,2):1}
        )
        
        # M6 = (A02-A10+A21)(B02-B10+B21)
        add_wterm(
            {(0,2):1, (1,0):-1, (2,1):1},
            {(0,2):1, (1,0):-1, (2,1):1},
            {(2,0):1, (0,1):-1, (1,2):1}
        )
        
        # M7 = (A00-A11+A22)(B00+B11-B22)
        add_wterm(
            {(0,0):1, (1,1):-1, (2,2):1},
            {(0,0):1, (1,1):1, (2,2):-1},
            {(0,0):1, (1,1):1, (2,2):-1}
        )
        
        # M8 = (A01-A12-A20)(B01+B12-B20)
        add_wterm(
            {(0,1):1, (1,2):-1, (2,0):-1},
            {(0,1):1, (1,2):1, (2,0):-1},
            {(1,0):1, (2,1):1, (0,2):-1}
        )
        
        # M9 = (A02-A10-A21)(B02+B10-B21)
        add_wterm(
            {(0,2):1, (1,0):-1, (2,1):-1},
            {(0,2):1, (1,0):1, (2,1):-1},
            {(2,0):1, (0,1):1, (1,2):-1}
        )
        
        # M10 = (A00+A01+A02)(B00+B10+B20)
        add_wterm(
            {(0,0):1, (0,1):1, (0,2):1},
            {(0,0):1, (1,0):1, (2,0):1},
            {(0,0):1, (1,0):1, (2,0):1}
        )
        
        # M11 = (A00+A01+A02)(B01+B11+B21)
        add_wterm(
            {(0,0):1, (0,1):1, (0,2):1},
            {(0,1):1, (1,1):1, (2,1):1},
            {(1,0):1, (1,1):1, (1,2):1}
        )
        
        # M12 = (A00+A01+A02)(B02+B12+B22)
        add_wterm(
            {(0,0):1, (0,1):1, (0,2):1},
            {(0,2):1, (1,2):1, (2,2):1},
            {(2,0):1, (2,1):1, (2,2):1}
        )
        
        # M13 = (A10+A11+A12)(B00+B10+B20)
        add_wterm(
            {(1,0):1, (1,1):1, (1,2):1},
            {(0,0):1, (1,0):1, (2,0):1},
            {(0,1):1, (1,1):1, (2,1):1}
        )
        
        # M14 = (A10+A11+A12)(B01+B11+B21)
        add_wterm(
            {(1,0):1, (1,1):1, (1,2):1},
            {(0,1):1, (1,1):1, (2,1):1},
            {(1,1):1, (1,2):1, (2,2):1}
        )
        
        # M15 = (A10+A11+A12)(B02+B12+B22)
        add_wterm(
            {(1,0):1, (1,1):1, (1,2):1},
            {(0,2):1, (1,2):1, (2,2):1},
            {(2,1):1, (2,2):1, (0,0):1}
        )
        
        # M16 = (A20+A21+A22)(B00+B10+B20)
        add_wterm(
            {(2,0):1, (2,1):1, (2,2):1},
            {(0,0):1, (1,0):1, (2,0):1},
            {(0,2):1, (1,2):1, (2,0):1}
        )
        
        # M17 = (A20+A21+A22)(B01+B11+B21)
        add_wterm(
            {(2,0):1, (2,1):1, (2,2):1},
            {(0,1):1, (1,1):1, (2,1):1},
            {(1,2):1, (2,0):1, (0,1):1}
        )
        
        # M18 = (A20+A21+A22)(B02+B12+B22)
        add_wterm(
            {(2,0):1, (2,1):1, (2,2):1},
            {(0,2):1, (1,2):1, (2,2):1},
            {(2,2):1, (0,0):1, (1,1):1}
        )
        
        # M19 = (A00+A11+A22+A01+A12+A20)(B00+B11+B22+B01+B12+B20)
        # This is getting too complex and I'm not confident in the coefficients.
        # Let me abandon the Winograd approach and use a simpler strategy.
        
        # Actually, let me just verify what I have so far and use schoolbook for the rest.
        # The first 18 terms above are 18 of the 23. I'm not confident they're correct.
        # Let me take a different approach: use Strassen 2x2x2 blocks more aggressively.
        pass
    
    # Let me use a simpler, more reliable approach:
    # Split the 5-dim contraction into 2+2+1 and 7-dim output into 2+2+2+1
    # and 3-dim input into 2+1.
    # Use Strassen 2x2x2 for all 2x2x2 sub-blocks.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term_simple(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def add_strassen_2x2x2_simple(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for a 2x2x2 block with base indices."""
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        add_term_simple(u, v, w)
        
        # M2 = (A10+A11)B00
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        add_term_simple(u, v, w)
        
        # M3 = A00(B01-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[j0*p+k1] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        add_term_simple(u, v, w)
        
        # M4 = A11(B10-B00)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j1] = 1
        v[j1*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        add_term_simple(u, v, w)
        
        # M5 = (A00+A01)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = -1; w[k1*n+i0] = 1
        add_term_simple(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+k1] = 1
        w[k1*n+i1] = 1
        add_term_simple(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j1*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        add_term_simple(u, v, w)
    
    def add_schoolbook_term(i, j, k):
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i*m+j] = 1
        v[j*p+k] = 1
        w[k*n+i] = 1
        add_term_simple(u, v, w)
    
    # Split: i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    # i: 2+1 -> {0,1} and {2}
    # j: 2+2+1 -> {0,1}, {2,3}, {4}
    # k: 2+2+2+1 -> {0,1}, {2,3}, {4,5}, {6}
    
    i_sets = [{0, 1}, {2}]
    j_sets = [{0, 1}, {2, 3}, {4}]
    k_sets = [{0, 1}, {2, 3}, {4, 5}, {6}]
    
    for I in i_sets:
        for J in j_sets:
            for K in k_sets:
                if len(I) == 2 and len(J) == 2 and len(K) == 2:
                    i0, i1 = sorted(I)
                    j0, j1 = sorted(J)
                    k0, k1 = sorted(K)
                    add_strassen_2x2x2_simple(i0, i1, j0, j1, k0, k1)
                else:
                    for i in I:
                        for j in J:
                            for k in K:
                                add_schoolbook_term(i, j, k)
    
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    if verify(U_mat, V_mat, W_mat):
        # Try to remove redundant terms
        removed = True
        while removed and time.time() < deadline - 10:
            removed = False
            for idx in range(U_mat.shape[1]):
                if time.time() >= deadline - 10:
                    break
                U_new = np.delete(U_mat, idx, axis=1)
                V_new = np.delete(V_mat, idx, axis=1)
                W_new = np.delete(W_mat, idx, axis=1)
                if verify(U_new, V_new, W_new):
                    U_mat = U_new
                    V_mat = V_new
                    W_mat = W_new
                    removed = True
                    break
        
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
