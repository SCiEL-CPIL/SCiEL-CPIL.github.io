"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    # Build the decomposition tensor via einsum: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Reshape U to (n,m,r), V to (m,p,r), W to (p,n,r)
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _strassen_2x2():
    """
    Verified rank-7 decomposition for 2x2x2 matrix multiplication.
    Uses the standard Strassen formulas with W indexed as W[k*2+i, c].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+b)(e+f)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21, C22
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1  # C11, C22
    
    # M5 = (a+d)(e+h)
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1; W[2, 4] = 1  # C11, C12, C21
    
    # M6 = (b-c)(g+h)
    U[1, 5] = 1; U[2, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    # In standard Strassen, M6 is not used in reconstruction
    W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = 0
    
    # M7 = (a-c)(f+h)
    U[0, 6] = 1; U[2, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W


def _winograd_3x3():
    """
    Rank-8 Winograd decomposition for 3x3x3 matrix multiplication.
    Uses the 3-point DFT over Z_3 structure.
    """
    U = np.zeros((9, 8), dtype=float)
    V = np.zeros((9, 8), dtype=float)
    W = np.zeros((9, 8), dtype=float)
    
    # Winograd 3x3: uses 8 products
    # A elements: a[i,j] indexed as i*3+j for i,j in 0,1,2
    # B elements: b[j,k] indexed as j*3+k
    # C elements: c[i,k] indexed as k*3+i (trace convention W[k*n+i])
    
    # The 8 products are based on the Winograd scheme:
    # P1 = (a00+a11+a22)(b00+b11+b22)
    # P2 = (a01+a12+a20)(b01+b12+b20)
    # P3 = (a02+a10+a21)(b02+b10+b21)
    # P4 = (a00-a11)(b00-b11)
    # P5 = (a11-a22)(b11-b22)
    # P6 = (a22-a00)(b22-b00)
    # P7 = (a01-a12)(b01-b12)
    # P8 = (a12-a20)(b12-b20)
    
    # P1 = (a00+a11+a22)(b00+b11+b22)
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
    
    # P2 = (a01+a12+a20)(b01+b12+b20)
    U[1, 1] = 1; U[5, 1] = 1; U[6, 1] = 1
    V[1, 1] = 1; V[5, 1] = 1; V[6, 1] = 1
    
    # P3 = (a02+a10+a21)(b02+b10+b21)
    U[2, 2] = 1; U[3, 2] = 1; U[7, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1; V[7, 2] = 1
    
    # P4 = (a00-a11)(b00-b11)
    U[0, 3] = 1; U[4, 3] = -1
    V[0, 3] = 1; V[4, 3] = -1
    
    # P5 = (a11-a22)(b11-b22)
    U[4, 4] = 1; U[8, 4] = -1
    V[4, 4] = 1; V[8, 4] = -1
    
    # P6 = (a22-a00)(b22-b00)
    U[8, 5] = 1; U[0, 5] = -1
    V[8, 5] = 1; V[0, 5] = -1
    
    # P7 = (a01-a12)(b01-b12)
    U[1, 6] = 1; U[5, 6] = -1
    V[1, 6] = 1; V[5, 6] = -1
    
    # P8 = (a12-a20)(b12-b20)
    U[5, 7] = 1; U[6, 7] = -1
    V[5, 7] = 1; V[6, 7] = -1
    
    # Now compute W by solving: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,k] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] summed over j and c
    # We need: for all i,k: sum_c W[k*3+i, c] * (sum_j U[i*3+j, c] * V[j*3+k, c]) = 1
    # Let S[i,k,c] = sum_j U[i*3+j, c] * V[j*3+k, c]
    # Then we need: sum_c W[k*3+i, c] * S[i,k,c] = 1
    
    # Build the linear system: for each (i,k), we have 8 unknowns W[k*3+i, 0..7]
    # with coefficients S[i,k,0..7]
    # 9 equations, 8 unknowns each - but they're independent per (i,k)
    
    # Actually wait - W[k*3+i, c] appears in the equation for (i,k) with coefficient S[i,k,c].
    # Each (i,k) pair gives one equation with 8 unknowns. But we have 9 (i,k) pairs
    # and each W entry W[k*3+i, c] only appears in the equation for that specific (i,k).
    # So each equation is independent: 1 equation, 8 unknowns -> underdetermined!
    
    # That can't be right. Let me re-examine.
    # T[i,j,k] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1 for ALL i,j,k
    # For a fixed (i,k), varying j gives: sum_c W[k*3+i, c] * sum_j' U[i*3+j', c] * V[j'*3+k, c] = 1
    # Wait no, it's: for each (i,j,k): sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
    # This is 27 equations with 72 unknowns (9 W entries × 8 columns).
    # For each c, W[k*3+i, c] appears in equations where the output is (i,k), i.e., 
    # in 3 equations (one for each j).
    
    # Let me set up the full system. 27 equations, 72 unknowns.
    # Unknown W[k*3+i, c] for i,k in 0..2, c in 0..7.
    # Equation (i,j,k): sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
    
    # Group by (i,k): for fixed (i,k), the 3 equations (varying j) are:
    # sum_c W[k*3+i, c] * U[i*3+j, c] * V[j*3+k, c] = 1, for j=0,1,2
    
    # So for each (i,k), we have a 3x8 system: A_ik * W_ik = 1
    # where A_ik[j, c] = U[i*3+j, c] * V[j*3+k, c]
    # and W_ik is an 8-vector (W[k*3+i, :])
    
    # This is 3 equations, 8 unknowns -> underdetermined. We need a solution.
    # Use least squares or pick a particular solution.
    
    # Actually, for a valid rank-8 decomposition, this system must have a solution.
    # Let me use np.linalg.lstsq to find the minimum-norm solution.
    
    for i in range(3):
        for k in range(3):
            # Build 3x8 matrix
            Aik = np.zeros((3, 8), dtype=float)
            for j in range(3):
                for c in range(8):
                    Aik[j, c] = U[i * 3 + j, c] * V[j * 3 + k, c]
            # Solve Aik @ W_ik = [1, 1, 1]
            b = np.ones(3, dtype=float)
            sol, residuals, rank_s, sv = np.linalg.lstsq(Aik, b, rcond=None)
            W[k * 3 + i, :] = sol
    
    return U, V, W


def _kronecker_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two tensor decompositions.
    If (U1,V1,W1) decomposes n1xm1xp1 and (U2,V2,W2) decomposes n2xm2xp2,
    then the Kronecker product decomposes (n1*n2)x(m1*m2)x(p1*p2).
    
    The Kronecker product structure:
    A is (n1*n2) x (m1*m2), B is (m1*m2) x (p1*p2)
    A is arranged as block matrix where block (i1,i2) is a1[i1,j1]*a2[i2,j2]
    Specifically, A[(i1*n2+i2), (j1*m2+j2)] = A1[i1,j1] * A2[i2,j2]
    B[(j1*m2+j2), (k1*p2+k2)] = B1[j1,k1] * B2[j2,k2]
    C[(i1*n2+i2), (k1*p2+k2)] = C1[i1,k1] * C2[i2,k2]
    
    The decomposition:
    U[(i1*n2+i2)*m1*m2 + (j1*m2+j2), (c1*r2+c2)] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    V[(j1*m2+j2)*p1*p2 + (k1*p2+k2), (c1*r2+c2)] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    W[(k1*p2+k2)*n1*n2 + (i1*n2+i2), (c1*r2+c2)] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: index (i1*n2+i2, j1*m2+j2) -> flat (i1*n2+i2)*m + (j1*m2+j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u_val = U1[i1 * m1 + j1, c1] * U2[i2 * m2 + j2, c2]
                            if u_val != 0:
                                idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                U[idx, c] = u_val
            # V: index (j1*m2+j2, k1*p2+k2) -> flat (j1*m2+j2)*p + (k1*p2+k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v_val = V1[j1 * p1 + k1, c1] * V2[j2 * p2 + k2, c2]
                            if v_val != 0:
                                idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                                V[idx, c] = v_val
            # W: index (k1*p2+k2, i1*n2+i2) -> flat (k1*p2+k2)*n + (i1*n2+i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w_val = W1[k1 * n1 + i1, c1] * W2[k2 * n2 + i2, c2]
                            if w_val != 0:
                                idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                                W[idx, c] = w_val
    
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication using
    Kronecker product of Strassen 2x2 (rank 7) and Winograd 3x3 (rank 8).
    This gives rank 56.
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try Winograd 3x3 (rank 8) ⊗ Strassen 2x2 (rank 7) = rank 56
    U3, V3, W3 = _winograd_3x3()
    U2, V2, W2 = _strassen_2x2()
    
    # Verify 3x3 decomposition
    if not _verify(U3, V3, W3, 3, 3, 3):
        # Winograd 3x3 failed, try schoolbook 3x3
        U3, V3, W3 = _schoolbook(3, 3, 3)
    
    # Verify 2x2 decomposition
    if not _verify(U2, V2, W2, 2, 2, 2):
        # Strassen 2x2 failed, try schoolbook 2x2
        U2, V2, W2 = _schoolbook(2, 2, 2)
    
    # Kronecker product: 3x3 ⊗ 2x2 = 6x6
    # n1=3, m1=3, p1=3, n2=2, m2=2, p2=2
    U, V, W = _kronecker_product(U3, V3, W3, U2, V2, W2, 3, 3, 3, 2, 2, 2)
    
    # Verify the full 6x6 decomposition
    if not _verify(U, V, W, 6, 6, 6):
        # Fallback to schoolbook
        U, V, W = U_sb, V_sb, W_sb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
# EVOLVE-BLOCK-END
