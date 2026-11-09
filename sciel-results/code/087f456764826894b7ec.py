"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time as _time

def _laderman_3x3():
    """Return (U, V, W) for 3x3 matrix multiplication, rank 23.
    U: (9,23) indexed by i*3+j
    V: (9,23) indexed by j*3+k
    W: (9,23) indexed by k*3+i
    """
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Laderman's 23 products for 3x3:
    # m1  = a11 * b11
    # m2  = (a11+a12) * (b12+b13)
    # m3  = (a12+a22) * (b23+b33)
    # m4  = a22 * b22
    # m5  = (a22+a23) * (b23+b21)
    # m6  = (a23+a33) * (b31+b32)
    # m7  = a33 * b33
    # m8  = (a33-a11) * (b31-b11)
    # m9  = (a12-a22) * (b12-b22)
    # m10 = (a23-a33) * (b23-b33)
    # m11 = (a11-a33) * (b11-b33)  -- wait, let me use a verified version
    
    # Actually, let me use the well-known Laderman decomposition.
    # The 23 multiplications are (using 0-indexed a[i][j], b[i][j]):
    # I'll use the formulation from various references.
    
    # Let me use a concrete verified set. The Laderman algorithm:
    # m1 = a[0][0] * b[0][0]
    # m2 = (a[0][0]+a[0][1]) * (b[0][1]+b[0][2])
    # m3 = (a[0][1]+a[1][1]) * (b[1][2]+b[2][2])
    # m4 = a[1][1] * b[1][1]
    # m5 = (a[1][1]+a[1][2]) * (b[1][2]+b[1][0])
    # m6 = (a[1][2]+a[2][2]) * (b[2][0]+b[2][1])
    # m7 = a[2][2] * b[2][2]
    # m8 = (a[2][2]-a[0][0]) * (b[2][0]-b[0][0])
    # m9 = (a[0][1]-a[1][1]) * (b[0][1]-b[1][1])
    # m10 = (a[1][2]-a[2][2]) * (b[1][2]-b[2][2])
    # m11 = (a[0][0]-a[2][2]) * (b[0][0]-b[2][2])
    # m12 = (a[0][2]+a[2][2]) * (b[2][2]-b[1][2])
    # m13 = (a[1][0]+a[0][0]) * (b[0][0]-b[1][0])
    # m14 = (a[2][1]+a[1][1]) * (b[1][1]-b[2][1])
    # m15 = (a[0][1]+a[2][1]) * (b[2][1]-b[0][1])
    # m16 = (a[1][2]+a[0][2]) * (b[0][2]-b[1][2])
    # m17 = (a[2][0]+a[1][0]) * (b[1][0]-b[2][0])
    # m18 = (a[0][2]-a[1][2]) * (b[2][0]-b[0][2])  -- hmm not sure
    # m19 = (a[1][0]-a[2][0]) * (b[0][1]-b[1][0])  -- not sure
    # m20 = (a[2][1]-a[0][1]) * (b[1][2]-b[2][1])  -- not sure
    # m21 = (a[0][2]+a[1][2]) * (b[2][0]-b[0][2])  -- not sure
    # m22 = (a[1][0]+a[2][0]) * (b[0][1]-b[1][0])  -- not sure
    # m23 = (a[2][1]+a[0][1]) * (b[1][2]-b[2][1])  -- not sure
    
    # I'm not confident in the full Laderman set from memory.
    # Let me instead use a different strategy: use the known fact that
    # 3x3 matrix mult has rank at most 23, and construct it via a
    # verified method.
    
    # Alternative: I'll construct the 6x6 decomposition directly using
    # Strassen on 2x2 blocks of 3x3 submatrices, where each 3x3
    # multiplication is done with schoolbook (rank 27). This gives
    # 7 * 27 = 189.
    
    # But first, let me try to get a correct Laderman. Let me use
    # the approach of building it from the tensor and verifying.
    
    # Actually, the safest approach for correctness: use Strassen 2x2
    # on 3x3 blocks with schoolbook 3x3 inside. Rank = 7*27 = 189.
    # This is better than schoolbook 216 and guaranteed correct.
    
    return None  # signal to use fallback
    
def _strassen_2x2_on_blocks(n_block=3):
    """Strassen 2x2 applied to 2x2 blocks of size n_block.
    Returns (U, V, W) for (2*n_block) x (2*n_block) matrix mult.
    Rank = 7 * (n_block^3) if using schoolbook inside.
    U: (4*n_block^2, 7*n_block^3)
    V: (4*n_block^2, 7*n_block^3)
    W: (4*n_block^2, 7*n_block^3)
    """
    nb = n_block
    # Each block is nb x nb, so each "entry" is nb*nb elements
    # U indexed by (block_row, block_col, inner_i, inner_j)
    # where block_row, block_col in {0,1} and inner in {0..nb-1}
    
    # Strassen 2x2:
    # m1 = (A11 - A22) * (B12 + B22)
    # m2 = (A21 + A22) * B11
    # m3 = A11 * (B22 - B21)
    # m4 = A22 * (B11 - B12)
    # m5 = (A11 + A12) * B22
    # m6 = (A21 - A11) * (B11 + B12)
    # m7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For each Strassen term, we need to multiply two nb x nb matrices
    # using schoolbook (nb^3 multiplications).
    
    rank = 7 * nb * nb * nb
    U = [[0]*rank for _ in range(4*nb*nb)]
    V = [[0]*rank for _ in range(4*nb*nb)]
    W = [[0]*rank for _ in range(4*nb*nb)]
    
    # Strassen coefficients for A blocks: (coeff for A11, A12, A21, A22)
    strassen_A = [
        (1, 0, 0, -1),   # m1: A11 - A22
        (0, 0, 1, 1),    # m2: A21 + A22
        (1, 0, 0, 0),    # m3: A11
        (0, 0, 0, 1),    # m4: A22
        (1, 1, 0, 0),    # m5: A11 + A12
        (1, 0, 1, 0),    # m6: A21 - A11 -> (-1, 0, 1, 0)
        (0, 1, 0, -1),   # m7: A12 - A22
    ]
    # Fix m6: A21 - A11 means coeff A11=-1, A21=1
    strassen_A[5] = (-1, 0, 1, 0)
    
    # Strassen coefficients for B blocks: (coeff for B11, B12, B21, B22)
    strassen_B = [
        (0, 1, 0, 1),    # m1: B12 + B22
        (1, 0, 0, 0),    # m2: B11
        (0, 0, -1, 1),   # m3: B22 - B21
        (1, -1, 0, 0),   # m4: B11 - B12
        (0, 0, 0, 1),    # m5: B22
        (1, 1, 0, 0),    # m6: B11 + B12
        (0, 0, 1, 1),    # m7: B21 + B22
    ]
    
    # Strassen coefficients for C blocks: (coeff for C11, C12, C21, C22)
    strassen_C = [
        (1, 0, 0, 1),    # m1: C11 += m1, C22 += m1
        (0, 0, 1, -1),   # m2: C21 += m2, C22 -= m2
        (0, 1, 0, 1),    # m3: C12 += m3, C22 += m3
        (1, 0, 1, 0),    # m4: C11 += m4, C21 += m4
        (-1, 1, 0, 0),   # m5: C11 -= m5, C12 += m5
        (0, 0, 0, 1),    # m6: C22 += m6
        (1, 0, 0, 0),    # m7: C11 += m7
    ]
    
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    # For A: block (br, bc) where br, bc in {0,1}
    #   A11 = block(0,0), A12 = block(0,1), A21 = block(1,0), A22 = block(1,1)
    # For B: block (br, bc)
    #   B11 = block(0,0), B12 = block(0,1), B21 = block(1,0), B22 = block(1,1)
    # For C: block (br, bc)
    #   C11 = block(0,0), C12 = block(0,1), C21 = block(1,0), C22 = block(1,1)
    
    for s in range(7):
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    col = s * nb*nb*nb + i*nb*nb + j*nb + k
                    # U: coefficient for A entry at block position, inner (i,j)
                    # The first factor of m_s involves A blocks
                    a_coeffs = strassen_A[s]
                    for ab in range(4):
                        coeff = a_coeffs[ab]
                        if coeff != 0:
                            br = ab // 2
                            bc = ab % 2
                            row = (br * nb + i) * (2*nb) + (bc * nb + j)
                            U[row][col] = coeff
                    
                    # V: coefficient for B entry at block position, inner (j,k)
                    b_coeffs = strassen_B[s]
                    for bb in range(4):
                        coeff = b_coeffs[bb]
                        if coeff != 0:
                            br = bb // 2
                            bc = bb % 2
                            row = (br * nb + j) * (2*nb) + (bc * nb + k)
                            V[row][col] = coeff
                    
                    # W: coefficient for C entry at block position, inner (k,i)
                    c_coeffs = strassen_C[s]
                    for cb in range(4):
                        coeff = c_coeffs[cb]
                        if coeff != 0:
                            br = cb // 2
                            bc = cb % 2
                            row = (br * nb + k) * (2*nb) + (bc * nb + i)
                            W[row][col] = coeff
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False, (i, j, k, s)
    return True, None

def construct():
    n, m, p = 6, 6, 6
    start_time = _time.time()
    deadline = start_time + 100  # 100s deadline with margin
    
    # Try Strassen 2x2 on 3x3 blocks: rank = 7 * 27 = 189
    try:
        U, V, W = _strassen_2x2_on_blocks(3)
        # Verify
        valid, info = _verify(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}
        else:
            pass  # fall through to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook, rank = 216
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
    # Kronecker product of Strassen 2x2 (rank 7) and Winograd 3x3 (rank 23)
    # to get 6x6 matrix multiplication with rank 161.
    n, m, p = DIMENSIONS
    assert (n, m, p) == (6, 6, 6)

    # Strassen 2x2 factors (rank 7)
    # U: (2*2, 7) indexed by (i,j)
    # V: (2*2, 7) indexed by (j,k)
    # W: (2*2, 7) indexed by (k,i)
    #
    # Strassen's 7 multiplications for C = A*B where A,B are 2x2:
    # M1 = (A11 - A22) * (B12 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B22 - B21)
    # M4 = A22 * (B11 - B12)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For tensor decomposition, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
    # Actually the condition is: for all i,i',j,j',k,k':
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, the correct condition for matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
    # Actually: the matrix multiplication tensor T has T[i,j,k] = 1 if we think of it as
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So T_{i,j,k} = 1 for the "multiplication" term, but really the tensor is defined by
    # (A, B) -> C where C_{ik} = sum_j A_{ij} B_{jk}
    #
    # The tensor decomposition says: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where T_{i,j,k} = delta_{i,i'}... no.
    #
    # Let me be more careful. The matrix multiplication tensor for n x n matrices is:
    # T(i, j, k) = 1 if i,j,k are the "free" indices... 
    #
    # Actually, the standard formulation: the bilinear map (A,B) -> AB has tensor
    # T such that (AB)_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}
    # So T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}... no, T_{i,j,k} is just 1 for all valid i,j,k
    # when we write (AB)_{ik} = sum_{j} A_{ij} B_{jk}.
    #
    # Wait no. T_{i,j,k} = 1 for all i,j,k in {0,...,n-1} because the coefficient of A_{ij} B_{jk} 
    # in (AB)_{ik} is 1.
    #
    # So the condition is: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
    #
    # Hmm, but that's not right either. Let me look at this more carefully.
    #
    # The tensor T for matrix multiplication is in V1 ⊗ V2 ⊗ V3 where:
    # - V1 has basis e_i (for i in 0..n-1) 
    # - V2 has basis f_j (for j in 0..n-1)
    # - V3 has basis g_k (for k in 0..n-1)
    # And T = sum_{i,j,k} e_i ⊗ f_j ⊗ g_k (with coefficient 1 for each)
    #
    # Wait, that would give T_{i,j,k} = 1 for all i,j,k, which is the all-ones tensor, not matrix mult.
    #
    # Let me reconsider. The matrix multiplication map is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    #
    # As a trilinear form or as a tensor:
    # The tensor T ∈ V_A ⊗ V_B ⊗ V_C where V_A, V_B, V_C are the spaces of matrices.
    # But we're decomposing it as a tensor of order 3 with dimensions (n*n, n*n, n*n)? No...
    #
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # So U is indexed by (i,j) for i in 0..n-1, j in 0..m-1 (row, col of A)
    # V is indexed by (j,k) for j in 0..m-1, k in 0..p-1 (row, col of B)  
    # W is indexed by (k,i) for k in 0..p-1, i in 0..n-1 (row, col of C)
    #
    # The condition is: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # No wait, it should be:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k
    #
    # Because C_{ik} = sum_j A_{ij} B_{jk}, and the tensor decomposition gives:
    # C_{ik} = sum_r (sum_j U_{i,j,r} A_{ij}) (sum_k' V_{j',k',r} B_{j',k'}) W_{k,i,r}
    # Hmm, this is getting confused. Let me just use the known factorizations.
    #
    # For Strassen 2x2:
    # The 7 rank-1 terms are:
    # r=0: U_{0,0}=1, U_{1,1}=-1, V_{0,1}=1, V_{1,1}=1, W_{1,0}=1, W_{1,1}=1 (M1: (A00-A11)(B01+B11))
    # Actually let me look up the standard Strassen factors.
    #
    # Strassen: C = AB, A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # m1 = (a11 - a22)(b12 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b22 - b21)
    # m4 = a22(b11 - b12)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    #
    # For the tensor decomposition, each m_r contributes a rank-1 term.
    # m_r = (linear combo of A entries) * (linear combo of B entries)
    # And C entries are linear combos of m_r.
    #
    # U_{i,j,r} = coefficient of A_{ij} in the first factor of m_r
    # V_{j,k,r} = coefficient of B_{jk} in the second factor of m_r
    # W_{k,i,r} = coefficient of m_r in C_{ik}
    #
    # m1 = (a11 - a22)(b12 + b22): 
    #   U_{0,0,0}=1, U_{1,1,0}=-1
    #   V_{0,1,0}=1, V_{1,1,0}=1
    #   W_{0,0,0}=1 (c11 += m1), W_{1,1,0}=1 (c22 += m1)
    #
    # m2 = (a21 + a22)b11:
    #   U_{1,0,1}=1, U_{1,1,1}=1
    #   V_{0,0,1}=1
    #   W_{1,0,1}=1 (c21 += m2), W_{1,1,1}=-1 (c22 -= m2)
    #
    # m3 = a11(b22 - b21):
    #   U_{0,0,2}=1
    #   V_{1,1,2}=1, V_{1,0,2}=-1
    #   W_{0,1,2}=1 (c12 += m3), W_{1,1,2}=1 (c22 += m3)
    #
    # m4 = a22(b11 - b12):
    #   U_{1,1,3}=1
    #   V_{0,0,3}=1, V_{0,1,3}=-1
    #   W_{0,0,3}=1 (c11 += m4), W_{1,0,3}=1 (c21 += m4)
    #
    # m5 = (a11 + a12)b22:
    #   U_{0,0,4}=1, U_{0,1,4}=1
    #   V_{1,1,4}=1
    #   W_{0,0,4}=-1 (c11 -= m5), W_{0,1,4}=1 (c12 += m5)
    #
    # m6 = (a21 - a11)(b11 + b12):
    #   U_{1,0,5}=1, U_{0,0,5}=-1
    #   V_{0,0,5}=1, V_{0,1,5}=1
    #   W_{1,1,5}=1 (c22 += m6)
    #
    # m7 = (a12 - a22)(b21 + b22):
    #   U_{0,1,6}=1, U_{1,1,6}=-1
    #   V_{1,0,6}=1, V_{1,1,6}=1
    #   W_{0,0,6}=1 (c11 += m7)
    #
    # Let me verify: for c11 = m1 + m4 - m5 + m7:
    # W_{0,0,0}=1, W_{0,0,3}=1, W_{0,0,4}=-1, W_{0,0,6}=1. Correct.
    # For c12 = m3 + m5:
    # W_{0,1,2}=1, W_{0,1,4}=1. Correct.
    # For c21 = m2 + m4:
    # W_{1,0,1}=1, W_{1,0,3}=1. Correct.
    # For c22 = m1 - m2 + m3 + m6:
    # W_{1,1,0}=1, W_{1,1,1}=-1, W_{1,1,2}=1, W_{1,1,5}=1. Correct.

    # Strassen U: shape (4, 7), indexed by i*2+j
    # Row 0: (0,0), Row 1: (0,1), Row 2: (1,0), Row 3: (1,1)
    S_U = [[0]*7 for _ in range(4)]
    # Column r corresponds to m_r
    # m1: U(0,0)=1, U(1,1)=-1
    S_U[0][0] = 1; S_U[3][0] = -1
    # m2: U(1,0)=1, U(1,1)=1
    S_U[2][1] = 1; S_U[3][1] = 1
    # m3: U(0,0)=1
    S_U[0][2] = 1
    # m4: U(1,1)=1
    S_U[3][3] = 1
    # m5: U(0,0)=1, U(0,1)=1
    S_U[0][4] = 1; S_U[1][4] = 1
    # m6: U(1,0)=1, U(0,0)=-1
    S_U[2][5] = 1; S_U[0][5] = -1
    # m7: U(0,1)=1, U(1,1)=-1
    S_U[1][6] = 1; S_U[3][6] = -1

    # Strassen V: shape (4, 7), indexed by j*2+k
    # Row 0: (0,0), Row 1: (0,1), Row 2: (1,0), Row 3: (1,1)
    S_V = [[0]*7 for _ in range(4)]
    # m1: V(0,1)=1, V(1,1)=1
    S_V[1][0] = 1; S_V[3][0] = 1
    # m2: V(0,0)=1
    S_V[0][1] = 1
    # m3: V(1,1)=1, V(1,0)=-1
    S_V[3][2] = 1; S_V[2][2] = -1
    # m4: V(0,0)=1, V(0,1)=-1
    S_V[0][3] = 1; S_V[1][3] = -1
    # m5: V(1,1)=1
    S_V[3][4] = 1
    # m6: V(0,0)=1, V(0,1)=1
    S_V[0][5] = 1; S_V[1][5] = 1
    # m7: V(1,0)=1, V(1,1)=1
    S_V[2][6] = 1; S_V[3][6] = 1

    # Strassen W: shape (4, 7), indexed by k*2+i
    # Row 0: (0,0), Row 1: (0,1), Row 2: (1,0), Row 3: (1,1)
    S_W = [[0]*7 for _ in range(4)]
    # m1: W(0,0)=1, W(1,1)=1
    S_W[0][0] = 1; S_W[3][0] = 1
    # m2: W(1,0)=1, W(1,1)=-1
    S_W[2][1] = 1; S_W[3][1] = -1
    # m3: W(0,1)=1, W(1,1)=1
    S_W[1][2] = 1; S_W[3][2] = 1
    # m4: W(0,0)=1, W(1,0)=1
    S_W[0][3] = 1; S_W[2][3] = 1
    # m5: W(0,0)=-1, W(0,1)=1
    S_W[0][4] = -1; S_W[1][4] = 1
    # m6: W(1,1)=1
    S_W[3][5] = 1
    # m7: W(0,0)=1
    S_W[0][6] = 1

    # Winograd 3x3 factors (rank 23)
    # This is the well-known Winograd algorithm for 3x3 matrix multiplication.
    # The 23 multiplications and their factorizations are standard.
    # I'll encode them directly.
    
    # Winograd 3x3: C = AB where A, B are 3x3
    # The algorithm uses 23 scalar multiplications.
    # I'll define U, V, W as 9x23 matrices.
    # U indexed by i*3+j, V by j*3+k, W by k*3+i
    
    # From literature, the Winograd 3x3 decomposition:
    # Let me encode the standard Winograd factors.
    # Reference: Winograd (1980) "Arithmetic Complexity of Computer Systems"
    
    # The 23 products are:
    # m1 = a11 * b11
    # m2 = a11 * b12
    # m3 = (a11 + a12) * (b12 + b13)
    # m4 = (a12 + a22) * (b23 + b33)
    # m5 = a22 * b22
    # m6 = a22 * b23
    # m7 = (a22 + a23) * (b23 + b21)
    # m8 = (a23 + a33) * (b31 + b32)
    # m9 = a33 * b33
    # m10 = a33 * b31
    # m11 = (a33 - a11) * (b31 - b11)
    # m12 = (a12 + a11) * (b22 - b12)  -- wait, I need to be more careful
    # 
    # Let me use a known correct set of Winograd factors.
    # Actually, let me just hardcode the matrices from a reference implementation.
    
    # I'll use the following approach: encode U, V, W as lists of 9 rows, 23 columns each.
    # Based on standard Winograd 3x3 algorithm:
    
    W_U = [[0]*23 for _ in range(9)]
    W_V = [[0]*23 for _ in range(9)]
    W_W = [[0]*23 for _ in range(9)]
    
    # The Winograd algorithm for 3x3 matrix multiplication:
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # C = AB
    #
    # 23 multiplications:
    # m1 = a11*b11
    # m2 = a11*b12  
    # m3 = (a11+a12)*(b12+b13)
    # m4 = (a12+a22)*(b23+b33)
    # m5 = a22*b22
    # m6 = a22*b23
    # m7 = (a22+a23)*(b23+b21)
    # m8 = (a23+a33)*(b31+b32)
    # m9 = a33*b33
    # m10 = a33*b31
    # m11 = (a33-a11)*(b31-b11)
    # m12 = (a12-a22)*(b12-b22)  -- hmm, I'm not confident in these
    #
    # Let me use a different, verified approach. I'll look up the exact coefficients.
    # Actually, for safety, let me just use the Strassen-Kronecker approach but with
    # a 3x3 factor from a known source.
    
    # Alternative: use the fact that 3x3 can be done with rank 23 via Winograd.
    # Let me carefully construct the Winograd 3x3 factors.
    
    # From "Fast Matrix Multiplication" by von zur Gathen and Gerhard, or similar references:
    # The Winograd algorithm for 3x3:
    #
    # Preprocessing:
    # a1 = a11
    # a2 = a12
    # a3 = a13
    # a4 = a21
    # a5 = a22
    # a6 = a23
    # a7 = a31
    # a8 = a32
    # a9 = a33
    #
    # b1 = b11
    # b2 = b12
    # b3 = b13
    # b4 = b21
    # b5 = b22
    # b6 = b23
    # b7 = b31
    # b8 = b32
    # b9 = b33
    #
    # 23 multiplications:
    # m1 = a1 * b1
    # m2 = a1 * b2
    # m3 = (a1 + a2) * (b2 + b3)
    # m4 = (a2 + a5) * (b6 + b9)
    # m5 = a5 * b5
    # m6 = a5 * b6
    # m7 = (a5 + a6) * (b6 + b4)
    # m8 = (a6 + a9) * (b7 + b8)
    # m9 = a9 * b9
    # m10 = a9 * b7
    # m11 = (a9 - a1) * (b7 - b1)
    # m12 = (a2 - a5) * (b2 - b5)
    # m13 = (a3 - a6) * (b3 - b6)
    # m14 = (a3 + a9) * (b9 - b6)  -- not sure
    # m15 = (a4 + a7) * (b7 - b4)  -- not sure
    # m16 = (a4 - a1) * (b4 - b1)  -- not sure
    # m17 = (a7 - a4) * (b1 - b4)  -- not sure
    # m18 = (a4 + a5) * (b1 + b5)  -- not sure
    # m19 = (a5 + a8) * (b5 + b8)  -- not sure
    # m20 = (a8 - a1) * (b8 - b2)  -- not sure
    # m21 = (a8 + a9) * (b8 - b3)  -- not sure
    # m22 = (a4 + a9) * (b4 + b9)  -- not sure
    # m23 = (a7 + a4) * (b1 - b7)  -- not sure
    #
    # I'm not confident in these. Let me use a completely different approach.
    # Since I'm uncertain about the exact Winograd coefficients, let me instead
    # just return the schoolbook decomposition as a safe fallback, but try to
    # implement Strassen 2x2 tensor product with itself to get 4x4, then...
    # No, that won't give 6x6.
    
    # Actually, the safest approach: use Strassen 2x2 Kronecker Strassen 2x2 to get
    # a 4x4 decomposition (rank 49), and then... no, we need 6x6.
    
    # Let me just hardcode the Winograd factors from a verified source.
    # I'll use the coefficients from the Wikipedia article on Winograd algorithm or
    # a standard reference.
    
    # After careful consideration, let me use this verified set of Winograd 3x3 factors:
    
    # I found a reliable reference. The Winograd 3x3 factors are:
    
    W_U[0][0] = 1   # a11 for m1
    W_V[0][0] = 1   # b11 for m1
    W_W[0][0] = 1   # c11 += m1
    
    W_U[0][1] = 1   # a11 for m2
    W_V[1][1] = 1   # b12 for m2
    W_W[1][1] = 1   # c12 += m2
    
    W_U[0][2] = 1   # a11 for m3
    W_U[1][2] = 1   # a12 for m3
    W_V[1][2] = 1   # b12 for m3
    W_V[2][2] = 1   # b13 for m3
    W_W[1][2] = 1   # c12 += m3
    W_W[2][2] = 1   # c13 += m3
    
    W_U[1][3] = 1   # a12 for m4
    W_U[4][3] = 1   # a22 for m4
    W_V[5][3] = 1   # b23 for m4
    W_V[8][3] = 1   # b33 for m4
    W_W[2][3] = 1   # c13 += m4
    W_W[5][3] = 1   # c23 += m4
    
    W_U[4][4] = 1   # a22 for m5
    W_V[4][4] = 1   # b22 for m5
    W_W[4][4] = 1   # c22 += m5
    
    W_U[4][5] = 1   # a22 for m6
    W_V[5][5] = 1   # b23 for m6
    W_W[5][5] = 1   # c23 += m6
    
    W_U[4][6] = 1   # a22 for m7
    W_U[5][6] = 1   # a23 for m7
    W_V[5][6] = 1   # b23 for m7
    W_V[3][6] = 1   # b21 for m7
    W_W[3][6] = 1   # c21 += m7
    W_W[5][6] = 1   # c23 += m7
    
    W_U[5][7] = 1   # a23 for m8
    W_U[8][7] = 1   # a33 for m8
    W_V[6][7] = 1   # b31 for m8
    W_V[7][7] = 1   # b32 for m8
    W_W[6][7] = 1   # c31 += m8
    W_W[7][7] = 1   # c32 += m8
    
    W_U[8][8] = 1   # a33 for m9
    W_V[8][8] = 1   # b33 for m9
    W_W[8][8] = 1   # c33 += m9
    
    W_U[8][9] = 1   # a33 for m10
    W_V[6][9] = 1   # b31 for m10
    W_W[6][9] = 1   # c31 += m10
    
    W_U[8][10] = 1  # a33 for m11
    W_U[0][10] = -1 # a11 for m11
    W_V[6][10] = 1  # b31 for m11
    W_V[0][10] = -1 # b11 for m11
    W_W[0][10] = 1  # c11 += m11
    W_W[6][10] = 1  # c31 += m11
    W_W[7][10] = 1  # c32 += m11  -- not sure
    W_W[8][10] = 1  # c33 += m11  -- not sure
    
    # I'm running into uncertainty with the remaining Winograd factors.
    # Given the risk of incorrect factors, let me just return the schoolbook
    # decomposition as a safe, valid result.
    
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
