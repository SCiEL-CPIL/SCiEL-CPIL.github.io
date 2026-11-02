import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, c):
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        
    # Helper to add Strassen block for j in [j0, j0+1] and k in [k0, k0+1]
    def add_strassen(j0, k0):
        nonlocal col
        # A indices: i in {0,1}, j in {j0, j0+1}
        # B indices: j in {j0, j0+1}, k in {k0, k0+1}
        # C indices: i in {0,1}, k in {k0, k0+1}
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00 is i=0, j=j0. A11 is i=1, j=j0+1.
        # B00 is j=j0, k=k0. B11 is j=j0+1, k=k0+1.
        # C00 is i=0, k=k0. C11 is i=1, k=k0+1.
        
        # We add 7 terms to U, V, W.
        
        # Term 1: M1
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1) ... wait, W coefficients depend on the combination.
        # Actually, Strassen defines 7 intermediate products.
        # We need to express the final C entries as linear combinations of these 7 products.
        # The tensor decomposition adds a column for each product.
        # For each product M_t, U, V, W have specific values.
        # Then W is summed? No, W is part of the decomposition.
        # The decomposition is $C = \sum_{t=1}^7 U_t \otimes V_t \otimes W_t$.
        # So for each t, we add one column to U, V, W.
        
        # Let's define the 7 Strassen products.
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We iterate t from 1 to 7.
        
        # t=1: P1
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1)
        add_term(0*m + j0, j0*p + k0, k0*n + 0, 1, 1, 1, col)
        add_term(1*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + 1, 1, 1, 1, col)
        # Wait, add_term adds to the SAME column.
        # So for column col, we add multiple (U,V,W) contributions?
        # No, the tensor decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is $u \otimes v \otimes w$.
        # So for one column (one rank-1 term), U, V, W are vectors.
        # But here, one Strassen product P1 involves multiple entries of A and B.
        # $P1 = (A00+A11)(B00+B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11$.
        # This is NOT a rank-1 tensor in the original A,B,C space.
        # It is a sum of 4 rank-1 tensors.
        # So Strassen's "7 products" are intermediate sums.
        # The decomposition of the multiplication tensor using Strassen has rank 7.
        # This means there are 7 columns in U, V, W.
        # For each column t, U_t, V_t, W_t are vectors.
        # $C = \sum_{t=1}^7 U_t \otimes V_t \otimes W_t$.
        
        # So for t=1 (corresponding to P1), what are U1, V1, W1?
        # P1 = (A00+A11)(B00+B11).
        # This product contributes to C00 and C11.
        # The term in the sum for column 1 should be such that:
        # $\sum_{i,j,k} U_{ij} V_{jk} W_{ki} = \dots$
        # Actually, the tensor $T$ has entries $T_{ijk} = 1$ if $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # The decomposition is $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
        # For Strassen, R=7.
        # The vectors $u_r, v_r, w_r$ have entries in $\{-1, 0, 1\}$.
        
        # Let's look up the explicit Strassen decomposition vectors.
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]].
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # The rank-1 terms correspond to the products P1..P7.
        # For P1:
        # u1 has 1 at (0,0) and (1,1).
        # v1 has 1 at (0,0) and (1,1).
        # w1 has 1 at (0,0) and (1,1).
        # Wait, if u1 = [1,0,0,1] (flattened A), v1 = [1,0,0,1] (flattened B), w1 = [1,0,0,1] (flattened C).
        # Then $u1 \otimes v1 \otimes w1$ gives terms like $A_{00} B_{00} C_{00}$, $A_{00} B_{11} C_{00}$, etc.
        # Let's check the contribution to C00.
        # $C_{00}$ gets $1 \cdot 1 \cdot 1 = 1$ from $A_{00} B_{00}$.
        # $C_{00}$ gets $1 \cdot 1 \cdot 1 = 1$ from $A_{00} B_{11}$? No, $v1$ has 1 at (1,1).
        # $A_{00} B_{11}$ is not a term in $C_{00}$. $C_{00} = A_{00} B_{00} + A_{01} B_{10}$.
        # So this simple outer product doesn't work directly if the supports don't align with the multiplication structure.
        
        # The Strassen decomposition is a change of basis.
        # The vectors $u_r, v_r, w_r$ are such that $\sum u_r \otimes v_r \otimes w_r = T_{mult}$.
        
        # Standard Strassen vectors for 2x2x2:
        # Indices: 00, 01, 10, 11.
        
        # r=1:
        # u: 1 at 00, 1 at 11.
        # v: 1 at 00, 1 at 11.
        # w: 1 at 00, 1 at 11.
        
        # r=2:
        # u: 1 at 10, 1 at 11.
        # v: 1 at 00.
        # w: 1 at 10.
        
        # r=3:
        # u: 1 at 00.
        # v: 1 at 01, -1 at 11.
        # w: 1 at 01.
        
        # r=4:
        # u: 1 at 11.
        # v: 1 at 10, -1 at 00.
        # w: 1 at 10, 1 at 00.
        
        # r=5:
        # u: 1 at 00, 1 at 01.
        # v: 1 at 11.
        # w: 1 at 01, -1 at 00.
        
        # r=6:
        # u: 1 at 10, -1 at 00.
        # v: 1 at 00, 1 at 01.
        # w: 1 at 11.
        
        # r=7:
        # u: 1 at 01, -1 at 11.
        # v: 1 at 10, 1 at 11.
        # w: 1 at 00, -1 at 11.
        
        # Let's verify r=1.
        # Contribution to C00: $u_{00} v_{00} w_{00} = 1 \cdot 1 \cdot 1 = 1$.
        # Contribution to C11: $u_{11} v_{11} w_{11} = 1 \cdot 1 \cdot 1 = 1$.
        # Contribution to C00 from $A_{00} B_{11}$? $u_{00}=1, v_{11}=1, w_{00}=1$.
        # The term is $A_{00} B_{11} C_{00}$.
        # But $C_{00}$ does not contain $A_{00} B_{11}$.
        # So the sum of these rank-1 tensors must cancel out the unwanted terms.
        
        # This is the correct way to implement it.
        
        # We need to map local indices (00,01,10,11) to global indices.
        # Local 00 -> i=0, j=j0. Global U: 5*0 + j0.
        # Local 11 -> i=1, j=j0+1. Global U: 5*1 + j0+1.
        
        # Let's write the function to add these 7 columns.
        
        def add_strassen_cols(j0, k0, start_col):
            # r=1
            # u: (0, j0), (1, j0+1)
            # v: (j0, k0), (j0+1, k0+1)
            # w: (k0, 0), (k0+1, 1)
            U[0*m + j0][start_col] = 1
            U[1*m + (j0+1)][start_col] = 1
            V[j0*p + k0][start_col] = 1
            V[(j0+1)*p + (k0+1)][start_col] = 1
            W[k0*n + 0][start_col] = 1
            W[(k0+1)*n + 1][start_col] = 1
            
            # r=2
            # u: (1, j0), (1, j0+1)
            # v: (j0, k0)
            # w: (k0, 1)
            U[1*m + j0][start_col+1] = 1
            U[1*m + (j0+1)][start_col+1] = 1
            V[j0*p + k0][start_col+1] = 1
            W[k0*n + 1][start_col+1] = 1
            
            # r=3
            # u: (0, j0)
            # v: (j0, k0+1), (j0+1, k0+1)
            # w: (k0+1, 0)
            U[0*m + j0][start_col+2] = 1
            V[j0*p + (k0+1)][start_col+2] = 1
            V[(j0+1)*p + (k0+1)][start_col+2] = -1
            W[(k0+1)*n + 0][start_col+2] = 1
            
            # r=4
            # u: (1, j0+1)
            # v: (j0+1, k0), (j0, k0)
            # w: (k0, 1), (k0, 0)
            U[1*m + (j0+1)][start_col+3] = 1
            V[(j0+1)*p + k0][start_col+3] = 1
            V[j0*p + k0][start_col+3] = -1
            W[k0*n + 1][start_col+3] = 1
            W[k0*n + 0][start_col+3] = 1
            
            # r=5
            # u: (0, j0), (0, j0+1)
            # v: (j0+1, k0+1)
            # w: (k0+1, 0), (k0, 0)
            U[0*m + j0][start_col+4] = 1
            U[0*m + (j0+1)][start_col+4] = 1
            V[(j0+1)*p + (k0+1)][start_col+4] = 1
            W[(k0+1)*n + 0][start_col+4] = 1
            W[k0*n + 0][start_col+4] = -1
            
            # r=6
            # u: (1, j0), (0, j0)
            # v: (j0, k0), (j0, k0+1)
            # w: (k0+1, 1)
            U[1*m + j0][start_col+5] = 1
            U[0*m + j0][start_col+5] = -1
            V[j0*p + k0][start_col+5] = 1
            V[j0*p + (k0+1)][start_col+5] = 1
            W[(k0+1)*n + 1][start_col+5] = 1
            
            # r=7
            # u: (0, j0+1), (1, j0+1)
            # v: (j0+1, k0), (j0+1, k0+1)
            # w: (k0, 0), (k0+1, 1)
            U[0*m + (j0+1)][start_col+6] = 1
            U[1*m + (j0+1)][start_col+6] = -1
            V[(j0+1)*p + k0][start_col+6] = 1
            V[(j0+1)*p + (k0+1)][start_col+6] = 1
            W[k0*n + 0][start_col+6] = 1
            W[(k0+1)*n + 1][start_col+6] = -1

        return start_col + 7

    # Helper to add schoolbook for 2x2x1
    # j in {j0, j0+1}, k = k0
    def add_schoolbook_2x2x1(j0, k0, start_col):
        # 4 terms
        # i=0, j=j0
        U[0*m + j0][start_col] = 1
        V[j0*p + k0][start_col] = 1
        W[k0*n + 0][start_col] = 1
        
        # i=0, j=j0+1
        U[0*m + (j0+1)][start_col+1] = 1
        V[(j0+1)*p + k0][start_col+1] = 1
        W[k0*n + 0][start_col+1] = 1
        
        # i=1, j=j0
        U[1*m + j0][start_col+2] = 1
        V[j0*p + k0][start_col+2] = 1
        W[k0*n + 1][start_col+2] = 1
        
        # i=1, j=j0+1
        U[1*m + (j0+1)][start_col+3] = 1
        V[(j0+1)*p + k0][start_col+3] = 1
        W[k0*n + 1][start_col+3] = 1
        
        return start_col + 4

    # Helper to add schoolbook for 2x1x5
    # j = j0, k in {0..4}
    def add_schoolbook_2x1x5(j0, start_col):
        # 10 terms
        # i=0, k=0..4
        for k in range(5):
            U[0*m + j0][start_col + k] = 1
            V[j0*p + k][start_col + k] = 1
            W[k*n + 0][start_col + k] = 1
            
        # i=1, k=0..4
        for k in range(5):
            U[1*m + j0][start_col + 5 + k] = 1
            V[j0*p + k][start_col + 5 + k] = 1
            W[k*n + 1][start_col + 5 + k] = 1
            
        return start_col + 10

    # Execute plan
    # Block 1: j in {0,1}
    col = add_strassen_cols(0, 0, col) # k in {0,1}
    col = add_strassen_cols(0, 2, col) # k in {2,3}
    col = add_schoolbook_2x2x1(0, 4, col) # k in {4}
    
    # Block 2: j in {2,3}
    col = add_strassen_cols(2, 0, col) # k in {0,1}
    col = add_strassen_cols(2, 2, col) # k in {2,3}
    col = add_schoolbook_2x2x1(2, 4, col) # k in {4}
    
    # Block 3: j in {4}
    col = add_schoolbook_2x1x5(4, col) # k in {0..4}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Verification
    import numpy as np
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 2, 5, 5
    rank = 46
    
    # Random test
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is not a simple matrix product.
    # C = sum_r (U[:,r] outer V[:,r]) reshaped?
    # U is (n*m, rank). V is (m*p, rank).
    # Let M_r = U[:,r] outer V[:,r]. This is (n*m) x (m*p).
    # M_r[i*m+j, j*p+k] = U[i*m+j, r] V[j*p+k, r].
    # We want C[i,k] = sum_j M_r[i*m+j, j*p+k] * W[k*n+i, r].
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        # u_r has non-zeros at i*m+j
        # v_r has non-zeros at j*p+k
        # w_r has non-zeros at k*n+i
        
        # Iterate over non-zeros? Or just do the sum.
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    total += u_r[i*m+j] * v_r[j*p+k]
                C_computed[i,k] += total * w_r[k*n+i]
                
    # Check
    if np.allclose(C_expected, C_computed):
        print("Verification passed")
    else:
        print("Verification failed")
        print(C_expected)
        print(C_computed)
