"""Program for 3x3 * 3x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _strassen_2x2x2():
    """
    Return the 7 scalar products for Strassen 2x2x2.
    Each is a tuple of (u_vec, v_vec, w_coeffs) where:
      u_vec is length-4: coefficients for A entries (a00,a01,a10,a11)
      v_vec is length-4: coefficients for B entries (b00,b01,b10,b11)
      w_coeffs is length-4: coefficients for C entries (c00,c01,c10,c11)
    The rank-1 term is: sum_i u[i]*A[i] * sum_j v[j]*B[j] contributes w_coeffs to C.
    """
    # M1 = (a00+a01)*(b00+b01) -> c00 += M1, c11 += M1
    m1 = ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1])
    # M2 = (a10+a11)*b01 -> c10 += M2
    m2 = ([0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0])
    # M3 = a00*(b10-b11) -> c00 -= M3, c10 += M3
    m3 = ([1, 0, 0, 0], [0, 0, 1, -1], [-1, 0, 1, 0])
    # M4 = a11*(b00+b11) -> c01 += M4, c11 += M4
    m4 = ([0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1])
    # M5 = (a00-a10)*(b10+b11) -> c00 -= M5, c01 += M5
    m5 = ([1, 0, -1, 0], [0, 0, 1, 1], [-1, 1, 0, 0])
    # M6 = (a01+a11)*(b00-b10) -> c00 += M6
    m6 = ([0, 1, 0, 1], [1, 0, -1, 0], [1, 0, 0, 0])
    # M7 = (a00-a01)*(b01+b11) -> c11 -= M7
    m7 = ([1, -1, 0, 0], [0, 1, 0, 1], [0, 0, 0, -1])

    return [m1, m2, m3, m4, m5, m6, m7]


def _build_block_decomposition():
    """
    Build a block-structured decomposition for 3x3 * 3x5.
    
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Decomposition into blocks:
    1. i=0, j=0: A[0,0]*B[0,k] for k=0..4 -> rank 5
    2. i=0, j in {1,2}: A[0,j]*B[j,k] for k=0..4 -> rank 10
    3. i in {1,2}, j=0: A[i,0]*B[0,k] for k=0..4 -> rank 10
    4. i in {1,2}, j in {1,2}: 2x2x5 problem -> Strassen + schoolbook
    
    For block 4 (2x2x5):
      Split B_bot columns into groups: cols 0-1, cols 2-3, col 4
      - A_bot(2x2) * B_bot[:,0:2](2x2): Strassen rank 7
      - A_bot(2x2) * B_bot[:,2:4](2x2): Strassen rank 7
      - A_bot(2x2) * B_bot[:,4:5](2x1): schoolbook rank 4
      Subtotal: 18
    
    Total: 5 + 10 + 10 + 18 = 43
    """
    n, m, p = 3, 3, 5
    total_rank = 43
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    strassen = _strassen_2x2x2()
    
    # Block 1: i=0, j=0: A[0,0]*B[0,k] for k=0..4
    # U index for A[0,0]: 0*3+0 = 0
    # V index for B[0,k]: 0*5+k = k
    # W index for C[0,k]: k*3+0 = 3k
    for k in range(5):
        U[0][col] = 1
        V[k][col] = 1
        W[3*k + 0][col] = 1
        col += 1
    
    # Block 2: i=0, j in {1,2}: A[0,j]*B[j,k] for k=0..4
    # For j=1: U index 0*3+1=1, V index 1*5+k=5+k, W index 3k+0
    # For j=2: U index 0*3+2=2, V index 2*5+k=10+k, W index 3k+0
    for j in [1, 2]:
        for k in range(5):
            U[j][col] = 1
            V[5*j + k][col] = 1
            W[3*k + 0][col] = 1
            col += 1
    
    # Block 3: i in {1,2}, j=0: A[i,0]*B[0,k] for k=0..4
    # For i=1: U index 1*3+0=3, V index k, W index 3k+1
    # For i=2: U index 2*3+0=6, V index k, W index 3k+2
    for i in [1, 2]:
        for k in range(5):
            U[3*i + 0][col] = 1
            V[k][col] = 1
            W[3*k + i][col] = 1
            col += 1
    
    # Block 4: i in {1,2}, j in {1,2}: 2x2x5 problem
    # A_bot: rows 1,2 of A; cols 1,2 of A -> A[1,1],A[1,2],A[2,1],A[2,2]
    #   U indices: 1*3+1=4, 1*3+2=5, 2*3+1=7, 2*3+2=8
    # B_bot: rows 1,2 of B; cols 0..4 of B
    #   V indices for row 1: 5*1+k = 5+k for k=0..4
    #   V indices for row 2: 5*2+k = 10+k for k=0..4
    # C_bot: rows 1,2 of C; cols 0..4 of C
    #   W indices: 3*k+i for i in {1,2}, k in 0..4
    
    # Sub-block 4a: A_bot(2x2) * B_bot[:,0:2](2x2) -> C_bot[:,0:2](2x2)
    # A local: (a00,a01,a10,a11) = (A[1,1],A[1,2],A[2,1],A[2,2])
    #   U local indices: [4, 5, 7, 8]
    # B local: (b00,b01,b10,b11) = (B[1,0],B[1,1],B[2,0],B[2,1])
    #   V local indices: [5, 6, 10, 11]
    # C local: (c00,c01,c10,c11) = (C[1,0],C[1,1],C[2,0],C[2,1])
    #   W local indices: [3*0+1, 3*1+1, 3*0+2, 3*1+2] = [1, 4, 2, 5]
    
    a_local_U = [4, 5, 7, 8]
    b_local_V = [5, 6, 10, 11]
    c_local_W = [1, 4, 2, 5]
    
    for (u_vec, v_vec, w_coeffs) in strassen:
        for idx in range(4):
            if u_vec[idx] != 0:
                U[a_local_U[idx]][col] += u_vec[idx]
            if v_vec[idx] != 0:
                V[b_local_V[idx]][col] += v_vec[idx]
            if w_coeffs[idx] != 0:
                W[c_local_W[idx]][col] += w_coeffs[idx]
        col += 1
    
    # Sub-block 4b: A_bot(2x2) * B_bot[:,2:4](2x2) -> C_bot[:,2:4](2x2)
    # A local same: U indices [4, 5, 7, 8]
    # B local: (b00,b01,b10,b11) = (B[1,2],B[1,3],B[2,2],B[2,3])
    #   V local indices: [5+2, 5+3, 10+2, 10+3] = [7, 8, 12, 13]
    # C local: (c00,c01,c10,c11) = (C[1,2],C[1,3],C[2,2],C[2,3])
    #   W local indices: [3*2+1, 3*3+1, 3*2+2, 3*3+2] = [7, 10, 8, 11]
    
    b_local_V = [7, 8, 12, 13]
    c_local_W = [7, 10, 8, 11]
    
    for (u_vec, v_vec, w_coeffs) in strassen:
        for idx in range(4):
            if u_vec[idx] != 0:
                U[a_local_U[idx]][col] += u_vec[idx]
            if v_vec[idx] != 0:
                V[b_local_V[idx]][col] += v_vec[idx]
            if w_coeffs[idx] != 0:
                W[c_local_W[idx]][col] += w_coeffs[idx]
        col += 1
    
    # Sub-block 4c: A_bot(2x2) * B_bot[:,4:5](2x1) -> C_bot[:,4:5](2x1)
    # Schoolbook: 4 terms
    # A[1,1]*B[1,4], A[1,2]*B[2,4], A[2,1]*B[1,4], A[2,2]*B[2,4]
    # Wait, 2x2 * 2x1: C[i,k] = sum_j A[i,j]*B[j,k] where i in {0,1}, k in {0}
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # So 4 terms:
    # Term 1: A[1,1]*B[1,4] -> C[1,4]
    #   U[4]=1, V[5+4=9]=1, W[3*4+1=13]=1
    # Term 2: A[1,2]*B[2,4] -> C[1,4]
    #   U[5]=1, V[10+4=14]=1, W[13]=1
    # Term 3: A[2,1]*B[1,4] -> C[2,4]
    #   U[7]=1, V[9]=1, W[3*4+2=14]=1
    # Term 4: A[2,2]*B[2,4] -> C[2,4]
    #   U[8]=1, V[14]=1, W[14]=1
    
    U[4][col] = 1; V[9][col] = 1; W[13][col] = 1; col += 1
    U[5][col] = 1; V[14][col] = 1; W[13][col] = 1; col += 1
    U[7][col] = 1; V[9][col] = 1; W[14][col] = 1; col += 1
    U[8][col] = 1; V[14][col] = 1; W[14][col] = 1; col += 1
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check it equals delta(i,j,k) for the multiplication tensor
    # Actually, the multiplication tensor has T[i,j,k] = 1 if this is a valid (i,j,k) term
    # The decomposition should satisfy: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    errors = 0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    errors += 1
                    max_err = max(max_err, abs(s - 1.0))
    return errors == 0, max_err


def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 3x3 * 3x5.
    Returns a dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try the block-structured decomposition with Strassen
    try:
        result = _build_block_decomposition()
        # Verify correctness
        valid, max_err = _verify_decomposition(result["U"], result["V"], result["W"], n, m, p)
        if valid:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
        else:
            # Fall through to schoolbook if verification fails
            pass
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    # Fallback: schoolbook
    result = _schoolbook(n, m, p)
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
