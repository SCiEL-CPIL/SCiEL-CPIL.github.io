import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _strassen_2x2x2():
    """Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: 0=a(i=0,j=0), 1=b(i=0,j=1), 2=c(i=1,j=0), 3=d(i=1,j=1)
    # V rows: 0=e(j=0,k=0), 1=f(j=0,k=1), 2=g(j=1,k=0), 3=h(j=1,k=1)
    # W rows: 0=c11(k=0,i=0), 1=c12(k=1,i=0), 2=c21(k=0,i=1), 3=c22(k=1,i=1)
    
    # M1 = (c+d)*e
    U[2,0] = 1; U[3,0] = 1
    V[0,0] = 1
    W[1,0] = 1  # c12
    
    # M2 = (c+d)*(f-h)
    U[2,1] = 1; U[3,1] = 1
    V[1,1] = 1; V[3,1] = -1
    W[0,1] = -1  # c11
    W[1,1] = 1   # c12
    
    # M3 = a*(e-f)
    U[0,2] = 1
    V[0,2] = 1; V[1,2] = -1
    W[2,2] = 1   # c21
    W[3,2] = -1  # c22
    
    # M4 = d*(g-h)
    U[3,3] = 1
    V[2,3] = 1; V[3,3] = -1
    W[0,3] = 1   # c11
    W[2,3] = 1   # c21
    
    # M5 = (a+b)*h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1   # c11
    W[3,4] = 1   # c22
    
    # M6 = (c-a)*(e+f)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # c11
    
    # M7 = (b-d)*(g+h)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = -1  # c22
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for l in range(rank):
                    u_val = U[i*m + j, l]
                    v_val = V[j*p + k, l]
                    w_val = W[k*n + i, l]
                    rhs += u_val * v_val * w_val
                if abs(lhs - rhs) > tol:
                    return False
    return True

def _build_block_decomp(n, m, p):
    """Build a block-structured decomposition using Strassen for 2x2x2 sub-blocks.
    Partition p=7 into [2,2,2,1].
    For each 2-column group, use two 2x2x2 Strassen blocks (rank 14 each).
    For the 1-column group, use schoolbook (rank 8).
    Total: 3*14 + 8 = 50.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Partition p=7 into [2,2,2,1]
    p_groups = [2, 2, 2, 1]
    
    for pg in p_groups:
        if pg == 2:
            # 2x4x2: split m=4 into [2,2], use two Strassen 2x2x2 blocks
            for j_offset in [0, 2]:
                sU, sV, sW = _strassen_2x2x2()
                U_b = np.zeros((n*m, 7))
                V_b = np.zeros((m*p, 7))
                W_b = np.zeros((p*n, 7))
                
                # Map local Strassen indices to global
                # U local: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
                # Global U row: i*m + j
                u_map = [0*m + (j_offset+0), 0*m + (j_offset+1), 1*m + (j_offset+0), 1*m + (j_offset+1)]
                
                # V local: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
                # Global V row: j*p + k, where k is within the 2-column group
                # The k values for this group are at positions k_offset=0,1 within the group
                # But globally, we need to figure out which k values
                # Actually, for a 2x4x2 block at column positions k0,k1:
                # V local row 0 = (j=0, k=k0), row 1 = (j=0, k=k1)
                # V local row 2 = (j=1, k=k0), row 3 = (j=1, k=k1)
                # Global: V[j*p + k]
                
                # We need to track which global k positions this group uses
                # Let's restructure: track k_offset
                pass
                
            # Let me redo this more carefully
            # For the 2x4x2 block at j_offset, k_offset:
            # U rows: i*m + j for i in {0,1}, j in {j_offset, j_offset+1}
            # V rows: j*p + k for j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
            # W rows: k*n + i for k in {k_offset, k_offset+1}, i in {0,1}
            
            # But we're partitioning p, so k_offset varies
            # Let me track k_offset explicitly
            pass
            
        else:
            # pg == 1: schoolbook for 2x4x1
            # rank = 2*4*1 = 8
            U_b = np.zeros((n*m, 8))
            V_b = np.zeros((m*p, 8))
            W_b = np.zeros((p*n, 8))
            col = 0
            for i in range(n):
                for j in range(m):
                    U_b[i*m+j, col] = 1
                    V_b[j*p + k_offset, col] = 1  # k=k_offset
                    W_b[k_offset*n+i, col] = 1
                    col += 1
            
            U_parts.append(U_b)
            V_parts.append(V_b)
            W_parts.append(W_b)
    
    return U_parts, V_parts, W_parts

def _build_decomp_strassen_blocks(n, m, p):
    """Build decomposition using Strassen 2x2x2 blocks.
    Partition: m=4 into [2,2], p=7 into [2,2,2,1]
    For each (j_block, k_block) pair where both have size 2: use Strassen (rank 7)
    For (j_block, k_block) where k_block has size 1: use schoolbook (rank 4)
    
    j_blocks: [0,2] (each size 2)
    k_blocks: [0,2,4,6] (sizes [2,2,2,1])
    
    For j_block=0 (j=0,1) and k_block=0 (k=0,1): Strassen 2x2x2, rank 7
    For j_block=0 (j=0,1) and k_block=1 (k=2,3): Strassen 2x2x2, rank 7
    For j_block=0 (j=0,1) and k_block=2 (k=4,5): Strassen 2x2x2, rank 7
    For j_block=0 (j=0,1) and k_block=3 (k=6): schoolbook 2x2x1, rank 4
    For j_block=1 (j=2,3) and k_block=0 (k=0,1): Strassen 2x2x2, rank 7
    For j_block=1 (j=2,3) and k_block=1 (k=2,3): Strassen 2x2x2, rank 7
    For j_block=1 (j=2,3) and k_block=2 (k=4,5): Strassen 2x2x2, rank 7
    For j_block=1 (j=2,3) and k_block=3 (k=6): schoolbook 2x2x1, rank 4
    
    Total rank: 6*7 + 2*4 = 42 + 8 = 50
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    j_blocks = [(0, 2), (2, 4)]  # (start, end) for j
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]  # (start, end) for k
    
    for j_start, j_end in j_blocks:
        for k_start, k_end in k_blocks:
            j_size = j_end - j_start
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Strassen 2x2x2
                sU, sV, sW = _strassen_2x2x2()
                U_b = np.zeros((n*m, 7))
                V_b = np.zeros((m*p, 7))
                W_b = np.zeros((p*n, 7))
                
                for l in range(7):
                    # U: local row r -> global row
                    # local 0: i=0, j=j_start -> global 0*m + j_start
                    # local 1: i=0, j=j_start+1 -> global 0*m + j_start+1
                    # local 2: i=1, j=j_start -> global 1*m + j_start
                    # local 3: i=1, j=j_start+1 -> global 1*m + j_start+1
                    U_b[0*m + j_start, l] = sU[0, l]
                    U_b[0*m + j_start+1, l] = sU[1, l]
                    U_b[1*m + j_start, l] = sU[2, l]
                    U_b[1*m + j_start+1, l] = sU[3, l]
                    
                    # V: local row r -> global row
                    # local 0: j=j_start, k=k_start -> global j_start*p + k_start
                    # local 1: j=j_start, k=k_start+1 -> global j_start*p + k_start+1
                    # local 2: j=j_start+1, k=k_start -> global (j_start+1)*p + k_start
                    # local 3: j=j_start+1, k=k_start+1 -> global (j_start+1)*p + k_start+1
                    V_b[j_start*p + k_start, l] = sV[0, l]
                    V_b[j_start*p + k_start+1, l] = sV[1, l]
                    V_b[(j_start+1)*p + k_start, l] = sV[2, l]
                    V_b[(j_start+1)*p + k_start+1, l] = sV[3, l]
                    
                    # W: local row r -> global row
                    # local 0: k=k_start, i=0 -> global k_start*n + 0
                    # local 1: k=k_start+1, i=0 -> global (k_start+1)*n + 0
                    # local 2: k=k_start, i=1 -> global k_start*n + 1
                    # local 3: k=k_start+1, i=1 -> global (k_start+1)*n + 1
                    W_b[k_start*n + 0, l] = sW[0, l]
                    W_b[(k_start+1)*n + 0, l] = sW[1, l]
                    W_b[k_start*n + 1, l] = sW[2, l]
                    W_b[(k_start+1)*n + 1, l] = sW[3, l]
                
                U_parts.append(U_b)
                V_parts.append(V_b)
                W_parts.append(W_b)
                
            elif j_size == 2 and k_size == 1:
                # Schoolbook 2x2x1, rank 4
                U_b = np.zeros((n*m, 4))
                V_b = np.zeros((m*p, 4))
                W_b = np.zeros((p*n, 4))
                col = 0
                for i in range(2):
                    for j in range(j_start, j_end):
                        U_b[i*m + j, col] = 1
                        V_b[j*p + k_start, col] = 1
                        W_b[k_start*n + i, col] = 1
                        col += 1
                
                U_parts.append(U_b)
                V_parts.append(V_b)
                W_parts.append(W_b)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _schoolbook_decomp():
    """Standard (schoolbook) decomposition of 2x4x7 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 56)
    U_school, V_school, W_school = _schoolbook_decomp()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try Strassen block decomposition (rank 50)
    try:
        U_str, V_str, W_str = _build_decomp_strassen_blocks(n, m, p)
        if _verify_decomp(U_str, V_str, W_str, n, m, p, tol=1e-9):
            best_U, best_V, best_W = U_str, V_str, W_str
            best_rank = U_str.shape[1]
    except Exception:
        pass
    
    # Return best result
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
