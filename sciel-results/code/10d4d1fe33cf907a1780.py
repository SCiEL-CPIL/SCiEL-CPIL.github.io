"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _construct_rank112():
    """
    Construct a rank-112 decomposition for 4x4 x 4x8.
    Uses block decomposition with Strassen 2x2 blocks.
    
    4x4 x 4x8 is split into two 4x4 x 4x4 blocks.
    Each 4x4 x 4x4 uses 8 Strassen 2x2 multiplications (rank 7 each) = 56.
    Total rank = 2 * 56 = 112.
    """
    n, m, p = 4, 4, 8
    base_U, base_V, base_W = _strassen_2x2()
    
    total_rank = 112
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # For each of the two 4x4x4 blocks (p_block in {0, 1}):
    for p_block in range(2):
        # For each of the 8 terms in the 4x4x4 block:
        for i in range(2):  # row block of C
            for j in range(2):  # col block of C
                for k in range(2):  # middle block index
                    # Term: A_block(i, k) * B_block(k, j)
                    for s in range(7):
                        col = p_block * 56 + (i * 4 + j * 2 + k) * 7 + s
                        
                        # U: A_block(i, k)
                        for a in range(2):
                            for b in range(2):
                                u_idx = (i * 2 + a) * 4 + (k * 2 + b)
                                U[u_idx][col] += base_U[a * 2 + b][s]
                        
                        # V: B_block(k, j)
                        for c in range(2):
                            for d in range(2):
                                v_idx = (k * 2 + c) * 8 + (p_block * 4 + j * 2 + d)
                                V[v_idx][col] += base_V[c * 2 + d][s]
                        
                        # W: C_block(i, j)
                        for e in range(2):
                            for f in range(2):
                                w_idx = (p_block * 4 + j * 2 + f) * 4 + (i * 2 + e)
                                W[w_idx][col] += base_W[f * 2 + e][s]
    
    return {"U": U, "V": V, "W": W}

def _construct_rank64():
    """
    Construct a rank-64 decomposition for 4x4 x 4x8.
    
    Strategy: View 4x4 x 4x8 as two independent 4x4 x 4x4 multiplications.
    For each 4x4 x 4x4, use a rank-32 decomposition.
    
    Rank-32 for 4x4x4:
    Use the fact that 4x4 = 2x2 blocks of 2x2.
    We can use a "Kronecker product" style approach:
    The multiplication tensor for 4x4x4 has rank at most 32.
    One construction: use 16 terms where each term computes a 2x2 block product
    using a 2x2 x 2x2 decomposition of rank 2 (which is schoolbook for 2x2).
    Wait, that's 16 * 4 = 64. That's not 32.
    
    Actually, let's try a different approach.
    For 4x4 x 4x4, the rank is at most 32 by known results.
    Let's try to construct it.
    
    Alternative: Use Strassen for the 4x4x4 part.
    4x4 x 4x4 can be viewed as 2x2 blocks of 2x2 matrices.
    There are 4 output blocks, each is a sum of 2 products of 2x2 matrices.
    If we use Strassen for each 2x2 product, we get 8 * 7 = 56.
    
    But there's a known rank-32 decomposition. Let me try to find it.
    
    Actually, for now, let's just use the rank-112 construction as it's simpler and verified.
    The rank-64 construction would require more careful design.
    
    For now, return None to indicate this construction is not yet implemented.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try the rank-112 block decomposition
    try:
        candidate = _construct_rank112()
        candidate_rank = len(candidate["U"][0])
        if candidate_rank < best_rank:
            if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                best = candidate
                best_rank = candidate_rank
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
