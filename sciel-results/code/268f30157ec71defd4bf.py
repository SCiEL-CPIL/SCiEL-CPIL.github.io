import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for rank 60.
    # Structure:
    # 1. Top 2 rows of C:
    #    - 2x2x2 Strassen x3 (for B cols 0-1, 2-3, 4-5) -> 3 * 7 = 21
    #    - 2x2x1 Schoolbook (for B col 6) -> 4
    #    - 2x1x7 Schoolbook (for A col 2, B row 2) -> 14
    #    Total top: 21 + 4 + 14 = 39
    # 2. Bottom 1 row of C:
    #    - 1x2x7 Schoolbook (for A cols 0-1, B rows 0-1) -> 14
    #    - 1x1x7 Schoolbook (for A col 2, B row 2) -> 7
    #    Total bottom: 14 + 7 = 21
    # Grand Total: 60
    
    rank = 60
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col_idx
        U[u_idx, col_idx] = val
        V[v_idx, col_idx] = val
        W[w_idx, col_idx] = val
        col_idx += 1

    # Strassen for 2x2 * 2x1 block
    # A is 2x2, B is 2x1. C is 2x1.
    # Indices for A: (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,2,3
    # Indices for B: (0,0), (1,0) -> V indices 0,1
    # Indices for C: (0,0), (1,0) -> W indices 0,1
    # Strassen for 2x2 * 2x2 is standard. For 2x2 * 2x1, it's just 4 mults?
    # Wait, Strassen reduces 8 to 7 for 2x2*2x2.
    # For 2x2*2x1, it is $A \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}$.
    # $c_1 = a_{11}b_1 + a_{12}b_2$
    # $c_2 = a_{21}b_1 + a_{22}b_2$
    # This is 4 multiplications. No Strassen benefit for rectangular?
    # Actually, if we treat it as 2x2 * 2x2 with the second column zero, we use 7 mults.
    # But 4 is better than 7.
    # So for 2x2 * 2x1, use schoolbook (4).
    # For 2x2 * 2x2, use Strassen (7).
    
    # Let's re-evaluate the cost.
    # B is 2x7. Split into 2x2, 2x2, 2x2, 2x1.
    # 3 * (2x2*2x2) + 1 * (2x2*2x1).
    # 3 * 7 + 1 * 4 = 25. Correct.
    
    # Strassen 2x2 * 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(a_base, b_base, c_base):
        # a_base: index in U for a11
        # b_base: index in V for b11
        # c_base: index in W for c11
        # A indices: a11, a12, a21, a22
        # B indices: b11, b12, b21, b22
        # C indices: c11, c12, c21, c22
        
        a11, a12, a21, a22 = a_base, a_base+1, a_base+2, a_base+3
        b11, b12, b21, b22 = b_base, b_base+1, b_base+2, b_base+3
        c11, c12, c21, c22 = c_base, c_base+1, c_base+2, c_base+3
        
        # M1
        U[a11, col_idx] = 1; U[a22, col_idx] = 1
        V[b11, col_idx] = 1; V[b22, col_idx] = 1
        W[c11, col_idx] = 1; W[c22, col_idx] = 1
        col_idx += 1
        
        # M2
        U[a21, col_idx] = 1; U[a22, col_idx] = 1
        V[b11, col_idx] = 1
        W[c21, col_idx] = 1; W[c22, col_idx] = -1
        col_idx += 1
        
        # M3
        U[a11, col_idx] = 1
        V[b12, col_idx] = 1; V[b22, col_idx] = -1
        W[c12, col_idx] = 1; W[c22, col_idx] = 1
        col_idx += 1
        
        # M4
        U[a22, col_idx] = 1
        V[b21, col_idx] = 1; V[b11, col_idx] = -1
        W[c11, col_idx] = 1; W[c21, col_idx] = 1
        col_idx += 1
        
        # M5
        U[a11, col_idx] = 1; U[a12, col_idx] = 1
        V[b22, col_idx] = 1
        W[c11, col_idx] = -1; W[c12, col_idx] = 1
        col_idx += 1
        
        # M6
        U[a21, col_idx] = 1; U[a11, col_idx] = -1
        V[b11, col_idx] = 1; V[b12, col_idx] = 1
        W[c22, col_idx] = 1
        col_idx += 1
        
        # M7
        U[a12, col_idx] = 1; U[a22, col_idx] = -1
        V[b21, col_idx] = 1; V[b22, col_idx] = 1
        W[c11, col_idx] = 1
        col_idx += 1

    def add_schoolbook_2x2(a_base, b_base, c_base):
        for i in range(2):
            for j in range(2):
                U[a_base + i*2 + j, col_idx] = 1
                V[b_base + j*2 + i, col_idx] = 1 # Note: B is 2x2, index j*2+i?
                # B indices: b11, b12, b21, b22.
                # V index for B[j,i]? No, V is m*p.
                # Here we are in a local 2x2 block.
                # Let's assume standard row-major.
                # A[i,j] -> U[a_base + i*2 + j]
                # B[j,k] -> V[b_base + j*2 + k]
                # C[i,k] -> W[c_base + i*2 + k]
                # In our call, we pass bases.
                pass

    # Let's rewrite add_schoolbook_2x2 correctly
    def add_schoolbook_2x2(a_base, b_base, c_base):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[a_base + i*2 + j, col_idx] = 1
                    V[b_base + j*2 + k, col_idx] = 1
                    W[c_base + i*2 + k, col_idx] = 1
                    col_idx += 1

    # We need to map the global indices to local bases.
    # Global U: n*m = 3*3 = 9.
    # Global V: m*p = 3*7 = 21.
    # Global W: p*n = 7*3 = 21.
    
    # Top part: A rows 0,1. A cols 0,1,2.
    # A[0:2, 0:2] is 2x2. Base in U: 0*3+0 = 0.
    # A[0:2, 2] is 2x1. Base in U: 0*3+2 = 2.
    
    # B rows 0,1. B cols 0..6.
    # B[0:2, 0:2] is 2x2. Base in V: 0*7+0 = 0.
    # B[0:2, 2:4] is 2x2. Base in V: 0*7+2 = 2.
    # B[0:2, 4:6] is 2x2. Base in V: 0*7+4 = 4.
    # B[0:2, 6:7] is 2x1. Base in V: 0*7+6 = 6.
    # B[2, 0:7] is 1x7. Base in V: 2*7+0 = 14.
    
    # C rows 0,1. C cols 0..6.
    # C[0:2, 0:2] is 2x2. Base in W: 0*3+0 = 0.
    # C[0:2, 2:4] is 2x2. Base in W: 0*3+2 = 2.
    # C[0:2, 4:6] is 2x2. Base in W: 0*3+4 = 4.
    # C[0:2, 6:7] is 2x1. Base in W: 0*3+6 = 6.
    
    # Bottom part: A row 2.
    # A[2, 0:2] is 1x2. Base in U: 2*3+0 = 6.
    # A[2, 2] is 1x1. Base in U: 2*3+2 = 8.
    
    # B rows 0,1.
    # B[0:2, 0:7] is 2x7. Base in V: 0.
    # B row 2.
    # B[2, 0:7] is 1x7. Base in V: 14.
    
    # C row 2.
    # C[2, 0:7] is 1x7. Base in W: 6*3+0 = 18.
    
    # 1. Top-Left 2x2 * B[0:2, 0:2] (Strassen)
    add_strassen_2x2(0, 0, 0)
    
    # 2. Top-Mid 2x2 * B[0:2, 2:4] (Strassen)
    add_strassen_2x2(0, 2, 2)
    
    # 3. Top-Right 2x2 * B[0:2, 4:6] (Strassen)
    add_strassen_2x2(0, 4, 4)
    
    # 4. Top-FarRight 2x2 * B[0:2, 6:7] (Schoolbook 2x2*2x1)
    # A[0:2, 0:2] (2x2) * B[0:2, 6:7] (2x1)
    # This is not 2x2*2x2. It is 2x2*2x1.
    # Cost 4.
    # U base 0. V base 6. W base 6.
    # A[i,j] B[j,k] -> C[i,k]. k=0 only.
    for i in range(2):
        for j in range(2):
            U[0 + i*2 + j, col_idx] = 1
            V[6 + j, col_idx] = 1 # V index: row j, col 6. 0*7+6=6, 1*7+6=13.
            # Wait, V base is 6. V[6] is B[0,6], V[13] is B[1,6].
            # In add_term, we need specific indices.
            # V index for B[j, 6] is j*7 + 6.
            W[6 + i, col_idx] = 1 # W index: row 6, col i. 6*3+i.
            col_idx += 1
            
    # 5. Top-Row2 A[0:2, 2] * B[2, 0:7] (Schoolbook 2x1*1x7)
    # A[0:2, 2] is 2x1. B[2, 0:7] is 1x7.
    # Cost 14.
    # U base 2 (A[0,2], A[1,2]).
    # V base 14 (B[2,0]..B[2,6]).
    # W base 0 (C[0,0]..C[1,6]).
    for i in range(2):
        for k in range(7):
            U[2 + i, col_idx] = 1 # A[i, 2]
            V[14 + k, col_idx] = 1 # B[2, k]
            W[k*3 + i, col_idx] = 1 # C[i, k]
            col_idx += 1
            
    # 6. Bottom-Left A[2, 0:2] * B[0:2, 0:7] (Schoolbook 1x2*2x7)
    # A[2, 0:2] is 1x2. B[0:2, 0:7] is 2x7.
    # Cost 14.
    # U base 6 (A[2,0], A[2,1]).
    # V base 0 (B[0,0]..B[1,6]).
    # W base 18 (C[2,0]..C[2,6]).
    for j in range(2):
        for k in range(7):
            U[6 + j, col_idx] = 1 # A[2, j]
            V[j*7 + k, col_idx] = 1 # B[j, k]
            W[18 + k, col_idx] = 1 # C[2, k]
            col_idx += 1
            
    # 7. Bottom-Right A[2, 2] * B[2, 0:7] (Schoolbook 1x1*1x7)
    # A[2, 2] is 1x1. B[2, 0:7] is 1x7.
    # Cost 7.
    # U base 8 (A[2,2]).
    # V base 14 (B[2,0]..B[2,6]).
    # W base 18 (C[2,0]..C[2,6]).
    for k in range(7):
        U[8, col_idx] = 1
        V[14 + k, col_idx] = 1
        W[18 + k, col_idx] = 1
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
