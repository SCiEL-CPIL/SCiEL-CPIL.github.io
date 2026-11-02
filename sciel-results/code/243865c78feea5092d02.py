"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with Strassen rank 7.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2*2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    W row index = k*n+i where k is col of C, i is row of C.
    So: W[0]=c[0,0], W[1]=c[1,0], W[2]=c[0,1], W[3]=c[1,1]
    U row index = i*m+j: U[0]=a[0,0], U[1]=a[0,1], U[2]=a[1,0], U[3]=a[1,1]
    V row index = j*p+k: V[0]=b[0,0], V[1]=b[0,1], V[2]=b[1,0], V[3]=b[1,1]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen: C = A*B, A,B 2x2
    # C[0,0] = P1 - P4 + P5 - P7
    # C[0,1] = P3 + P7
    # C[1,0] = P2 + P6
    # C[1,1] = P1 - P3 + P4 + P6
    #
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    #
    # W row mapping: k*n+i
    # C[0,0]: k=0,i=0 -> W[0]
    # C[1,0]: k=0,i=1 -> W[1]
    # C[0,1]: k=1,i=0 -> W[2]
    # C[1,1]: k=1,i=1 -> W[3]
    
    # P1: contributes +1 to C[0,0] and +1 to C[1,1]
    # W: [1, 0, 0, 1]
    U[:, 0] = [1, 0, 0, 1]  # a00+a11
    V[:, 0] = [1, 0, 0, 1]  # b00+b11
    W[:, 0] = [1, 0, 0, 1]  # C[0,0]+C[1,1]
    
    # P2: contributes +1 to C[1,0] and +1 to C[1,1]
    # W: [0, 1, 0, 1]
    U[:, 1] = [0, 0, 1, 1]  # a10+a11
    V[:, 1] = [1, 0, 0, 0]  # b00
    W[:, 1] = [0, 1, 0, 1]  # C[1,0]+C[1,1]
    
    # P3: contributes +1 to C[0,1] and -1 to C[1,1]
    # W: [0, 0, 1, -1]
    U[:, 2] = [1, 0, 0, 0]  # a00
    V[:, 2] = [0, 1, 0, -1]  # b01-b11
    W[:, 2] = [0, 0, 1, -1]  # C[0,1]-C[1,1]
    
    # P4: contributes -1 to C[0,0] and +1 to C[1,1]
    # W: [-1, 0, 0, 1]
    U[:, 3] = [0, 0, 0, 1]  # a11
    V[:, 3] = [-1, 0, 1, 0]  # b10-b00
    W[:, 3] = [-1, 0, 0, 1]  # -C[0,0]+C[1,1]
    
    # P5: contributes +1 to C[0,0] and +1 to C[1,1]
    # W: [1, 0, 0, 1]
    U[:, 4] = [1, 1, 0, 0]  # a00+a01
    V[:, 4] = [0, 0, 0, 1]  # b11
    W[:, 4] = [1, 0, 0, 1]  # C[0,0]+C[1,1]
    
    # P6: contributes +1 to C[1,0] and +1 to C[1,1]
    # W: [0, 1, 0, 1]
    U[:, 5] = [-1, 0, 1, 0]  # a10-a00
    V[:, 5] = [1, 1, 0, 0]  # b00+b01
    W[:, 5] = [0, 1, 0, 1]  # C[1,0]+C[1,1]
    
    # P7: contributes +1 to C[0,1] and -1 to C[1,1]
    # W: [0, 0, 1, -1]
    U[:, 6] = [0, 1, 0, -1]  # a01-a11
    V[:, 6] = [0, 0, 1, 1]  # b10+b11
    W[:, 6] = [0, 0, 1, -1]  # C[0,1]-C[1,1]
    
    return U, V, W

def _identity_block(n, m, p):
    """Return trivial rank n*m*p decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _decompose_nm_p(n, m, p):
    """Decompose n x m * m x p matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2 blocks.
    Returns U, V, W as numpy arrays.
    """
    # Base case: if any dimension is 0, return empty
    if n == 0 or m == 0 or p == 0:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W
    
    # Base case: 2x2*2x2 -> Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Base case: if m == 1, it's just outer product scaled
    if m == 1:
        rank = n * p
        U = np.zeros((n*1, rank))
        V = np.zeros((1*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[0, col] = 1
                W[k*n + i, col] = 1
                col += 1
        return U, V, W
    
    # If n == 1
    if n == 1:
        rank = m * p
        U = np.zeros((1*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*1, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j*p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
    
    # If p == 1
    if p == 1:
        rank = n * m
        U = np.zeros((n*m, rank))
        V = np.zeros((m*1, rank))
        W = np.zeros((1*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
    
    # Try to split m into m1 + m2 where m1, m2 > 0
    # Choose split that minimizes total rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        # Get decompositions for (n, m1, p) and (n, m2, p)
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            # Combine: U1 and U2 have different row indices because m1 != m2
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Map U1 rows: U1[i*m1 + j, c] -> U[i*m + j, c] for j in [0, m1)
            for i in range(n):
                for j in range(m1):
                    u1_row = i * m1 + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            # Map U2 rows: U2[i*m2 + j, c] -> U[i*m + (m1+j), c] for j in [0, m2)
            for i in range(n):
                for j in range(m2):
                    u2_row = i * m2 + j
                    full_row = i * m + (m1 + j)
                    U[full_row, r1:] = U2[u2_row, :]
            
            # V1: rows (j,k) for j in [0,m1), k in [0,p) -> full row j*p+k
            V[:m1*p, :r1] = V1
            # V2: rows (j,k) for j in [0,m2), k in [0,p) -> full row (m1+j)*p+k
            V[m1*p:, r1:] = V2
            
            # W is the same for both since output dimensions are the same
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U = U
            best_V = V
            best_W = W
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback: schoolbook
    return _identity_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose_nm_p(n, m, p)
    rank = U.shape[1]
    
    # Verify correctness: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # This is expensive for large ranks, so only do a spot check
    if rank <= 100:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        # Fall back to schoolbook if verification fails
                        U, V, W = _identity_block(n, m, p)
                        rank = n * m * p
                        break
                else:
                    continue
                break
            else:
                continue
            break
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in U.tolist()]
    V_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in V.tolist()]
    W_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
