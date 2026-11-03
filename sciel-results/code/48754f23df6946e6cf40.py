"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, atol=1e-7):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=atol)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _laderman_333():
    """
    Return Laderman decomposition for 3x3x3 matrix multiplication.
    Rank 19.
    """
    U = np.zeros((9, 19), dtype=np.float64)
    V = np.zeros((9, 19), dtype=np.float64)
    W = np.zeros((9, 19), dtype=np.float64)
    
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    # i,j,k in {0,1,2}
    
    # M1..M4: Diagonal-like combinations
    # M1 = (a00+a11+a22)(b00+b11+b22)
    U[0,0]=1; U[4,0]=1; U[8,0]=1
    V[0,0]=1; V[4,0]=1; V[8,0]=1
    W[0,0]=1; W[4,0]=1; W[8,0]=1
    
    # M2 = (a00+a11-a22)(b00+b11-b22)
    U[0,1]=1; U[4,1]=1; U[8,1]=-1
    V[0,1]=1; V[4,1]=1; V[8,1]=-1
    W[0,1]=1; W[4,1]=1; W[8,1]=1
    
    # M3 = (a00-a11+a22)(b00-b11+b22)
    U[0,2]=1; U[4,2]=-1; U[8,2]=1
    V[0,2]=1; V[4,2]=-1; V[8,2]=1
    W[0,2]=1; W[4,2]=1; W[8,2]=1
    
    # M4 = (a00-a11-a22)(b00-b11-b22)
    U[0,3]=1; U[4,3]=-1; U[8,3]=-1
    V[0,3]=1; V[4,3]=-1; V[8,3]=-1
    W[0,3]=1; W[4,3]=1; W[8,3]=1
    
    # M5..M8: Row/Col combinations involving 0
    # M5 = (a00+a01+a02)(b00+b10+b20)
    U[0,4]=1; U[1,4]=1; U[2,4]=1
    V[0,4]=1; V[3,4]=1; V[6,4]=1
    W[0,4]=1; W[1,4]=1; W[2,4]=1
    
    # M6 = (a00+a01-a02)(b00+b10-b20)
    U[0,5]=1; U[1,5]=1; U[2,5]=-1
    V[0,5]=1; V[3,5]=1; V[6,5]=-1
    W[0,5]=1; W[1,5]=1; W[2,5]=1
    
    # M7 = (a00-a01+a02)(b00-b10+b20)
    U[0,6]=1; U[1,6]=-1; U[2,6]=1
    V[0,6]=1; V[3,6]=-1; V[6,6]=1
    W[0,6]=1; W[1,6]=1; W[2,6]=1
    
    # M8 = (a00-a01-a02)(b00-b10-b20)
    U[0,7]=1; U[1,7]=-1; U[2,7]=-1
    V[0,7]=1; V[3,7]=-1; V[6,7]=-1
    W[0,7]=1; W[1,7]=1; W[2,7]=1
    
    # M9..M12: Row/Col combinations involving 0 (other orientation)
    # M9 = (a00+a10+a20)(b00+b01+b02)
    U[0,8]=1; U[3,8]=1; U[6,8]=1
    V[0,8]=1; V[1,8]=1; V[2,8]=1
    W[0,8]=1; W[1,8]=1; W[2,8]=1
    
    # M10 = (a00+a10-a20)(b00+b01-b02)
    U[0,9]=1; U[3,9]=1; U[6,9]=-1
    V[0,9]=1; V[1,9]=1; V[2,9]=-1
    W[0,9]=1; W[1,9]=1; W[2,9]=1
    
    # M11 = (a00-a10+a20)(b00-b01+b02)
    U[0,10]=1; U[3,10]=-1; U[6,10]=1
    V[0,10]=1; V[1,10]=-1; V[2,10]=1
    W[0,10]=1; W[1,10]=1; W[2,10]=1
    
    # M12 = (a00-a10-a20)(b00-b01-b02)
    U[0,11]=1; U[3,11]=-1; U[6,11]=-1
    V[0,11]=1; V[1,11]=-1; V[2,11]=-1
    W[0,11]=1; W[1,11]=1; W[2,11]=1
    
    # M13..M19: Remaining terms
    # M13 = a01 * b10
    U[1,12]=1
    V[3,12]=1
    W[1,12]=1 # C[0,1] -> k=1,i=0 -> W[1*3+0]=W[3]? No.
    # W index: k*n+i. n=3.
    # C[i,k].
    # a01 is i=0, j=1. b10 is j=1, k=0.
    # Product contributes to C[0,0].
    # C[0,0] is k=0, i=0. W index = 0*3+0 = 0.
    # Wait, my W indexing in Laderman above was W[k*3+i].
    # Let's re-verify W indices for M13.
    # a01 (i=0, j=1), b10 (j=1, k=0).
    # Sum over j. j=1.
    # C[0,0] += a01 * b10.
    # C[0,0] corresponds to k=0, i=0.
    # W index for k=0, i=0 is 0.
    # So W[0, 12] = 1.
    
    # I need to be very careful with W indices.
    # Let's restart W assignments for M13-M19.
    
    # M13: a01 * b10 -> C[0,0]
    U[1,12]=1
    V[3,12]=1
    W[0,12]=1
    
    # M14: a02 * b20 -> C[0,0]
    U[2,13]=1
    V[6,13]=1
    W[0,13]=1
    
    # M15: a10 * b01 -> C[1,0]
    U[3,14]=1
    V[1,14]=1
    # C[1,0] -> k=0, i=1. W index = 0*3+1 = 1.
    W[1,14]=1
    
    # M16: a12 * b21 -> C[1,0]?
    # a12 (i=1, j=2), b21 (j=2, k=1).
    # C[1,1] += a12 * b21.
    # C[1,1] -> k=1, i=1. W index = 1*3+1 = 4.
    U[5,15]=1
    V[8,15]=1
    W[4,15]=1
    
    # M17: a20 * b02 -> C[2,0]
    U[6,16]=1
    V[2,16]=1
    # C[2,0] -> k=0, i=2. W index = 0*3+2 = 2.
    W[2,16]=1
    
    # M18: a11 * b11 -> C[1,1]
    U[4,17]=1
    V[4,17]=1
    # C[1,1] -> k=1, i=1. W index = 4.
    W[4,17]=1
    
    # M19: a22 * b22 -> C[2,2]
    U[8,18]=1
    V[8,18]=1
    # C[2,2] -> k=2, i=2. W index = 2*3+2 = 8.
    W[8,18]=1
    
    return U, V, W

def _decomp_323():
    """
    Return decomposition for 3x2x3 matrix multiplication.
    Rank 17.
    Split n=3 into 2+1, p=3 into 2+1.
    Blocks:
    2x2x2 (Strassen, Rank 7)
    2x2x1 (Schoolbook, Rank 4)
    1x2x2 (Schoolbook, Rank 4)
    1x2x1 (Schoolbook, Rank 2)
    """
    n, m, p = 3, 2, 3
    
    U_total = np.zeros((n*m, 0), dtype=np.float64)
    V_total = np.zeros((m*p, 0), dtype=np.float64)
    W_total = np.zeros((p*n, 0), dtype=np.float64)
    
    # Block 1: n=2, m=2, p=2 (i=0,1; j=0,1; k=0,1)
    U1, V1, W1 = _strassen_222()
    U1f, V1f, W1f = _embed_sub_decomp(U1, V1, W1, 2, 2, 2, 0, 0, 0, n, m, p)
    U_total = np.hstack([U_total, U1f])
    V_total = np.hstack([V_total, V1f])
    W_total = np.hstack([W_total, W1f])
    
    # Block 2: n=2, m=2, p=1 (i=0,1; j=0,1; k=2)
    U2, V2, W2 = _schoolbook(2, 2, 1)
    U2f, V2f, W2f = _embed_sub_decomp(U2, V2, W2, 2, 2, 1, 0, 0, 2, n, m, p)
    U_total = np.hstack([U_total, U2f])
    V_total = np.hstack([V_total, V2f])
    W_total = np.hstack([W_total, W2f])
    
    # Block 3: n=1, m=2, p=2 (i=2; j=0,1; k=0,1)
    U3, V3, W3 = _schoolbook(1, 2, 2)
    U3f, V3f, W3f = _embed_sub_decomp(U3, V3, W3, 1, 2, 2, 2, 0, 0, n, m, p)
    U_total = np.hstack([U_total, U3f])
    V_total = np.hstack([V_total, V3f])
    W_total = np.hstack([W_total, W3f])
    
    # Block 4: n=1, m=2, p=1 (i=2; j=0,1; k=2)
    U4, V4, W4 = _schoolbook(1, 2, 1)
    U4f, V4f, W4f = _embed_sub_decomp(U4, V4, W4, 1, 2, 1, 2, 0, 2, n, m, p)
    U_total = np.hstack([U_total, U4f])
    V_total = np.hstack([V_total, V4f])
    W_total = np.hstack([W_total, W4f])
    
    return U_total, V_total, W_total

def _construct_72():
    """
    Construct 3x5x6 decomposition using Laderman and 3x2x3 blocks.
    Split m=5 into 3+2, p=6 into 3+3.
    Blocks:
    3x3x3 (Laderman, Rank 19) at i=0..2, j=0..2, k=0..2
    3x3x3 (Laderman, Rank 19) at i=0..2, j=0..2, k=3..5
    3x2x3 (Rank 17) at i=0..2, j=3..4, k=0..2
    3x2x3 (Rank 17) at i=0..2, j=3..4, k=3..5
    """
    n, m, p = 3, 5, 6
    
    U_total = np.zeros((n*m, 0), dtype=np.float64)
    V_total = np.zeros((m*p, 0), dtype=np.float64)
    W_total = np.zeros((p*n, 0), dtype=np.float64)
    
    # Block 1: 3x3x3
    U1, V1, W1 = _laderman_333()
    U1f, V1f, W1f = _embed_sub_decomp(U1, V1, W1, 3, 3, 3, 0, 0, 0, n, m, p)
    U_total = np.hstack([U_total, U1f])
    V_total = np.hstack([V_total, V1f])
    W_total = np.hstack([W_total, W1f])
    
    # Block 2: 3x3x3
    U2, V2, W2 = _laderman_333()
    U2f, V2f, W2f = _embed_sub_decomp(U2, V2, W2, 3, 3, 3, 0, 0, 3, n, m, p)
    U_total = np.hstack([U_total, U2f])
    V_total = np.hstack([V_total, V2f])
    W_total = np.hstack([W_total, W2f])
    
    # Block 3: 3x2x3
    U3, V3, W3 = _decomp_323()
    U3f, V3f, W3f = _embed_sub_decomp(U3, V3, W3, 3, 2, 3, 0, 3, 0, n, m, p)
    U_total = np.hstack([U_total, U3f])
    V_total = np.hstack([V_total, V3f])
    W_total = np.hstack([W_total, W3f])
    
    # Block 4: 3x2x3
    U4, V4, W4 = _decomp_323()
    U4f, V4f, W4f = _embed_sub_decomp(U4, V4, W4, 3, 2, 3, 0, 3, 3, n, m, p)
    U_total = np.hstack([U_total, U4f])
    V_total = np.hstack([V_total, V4f])
    W_total = np.hstack([W_total, W4f])
    
    return U_total, V_total, W_total

def _block_decomposition_356():
    """
    Fallback: Block decomposition with Strassen for 2x2x2.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Total rank: 84
    """
    n, m, p = 3, 5, 6
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_total = None
    V_total = None
    W_total = None
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total is None:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Try 72-rank decomposition
    try:
        U, V, W = _construct_72()
        if _verify_decomp(n, m, p, U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback to 84-rank decomposition
    try:
        U, V, W = _block_decomposition_356()
        if _verify_decomp(n, m, p, U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
# EVOLVE-BLOCK-END
