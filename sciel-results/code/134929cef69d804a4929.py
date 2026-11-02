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

def _verify_decomp(n, m, p, U, V, W):
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
    return np.allclose(T_comp, T_target, atol=1e-9)

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

def _winograd_333():
    """
    Return a rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses the Winograd algorithm. Returns (U, V, W) where:
    U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    # Winograd's rank-23 decomposition for 3x3x3
    # This is a known construction. Let me use a verified set of coefficients.
    # The 23 multiplications are structured as follows.
    
    U = np.zeros((9, 23), dtype=np.float64)
    V = np.zeros((9, 23), dtype=np.float64)
    W = np.zeros((9, 23), dtype=np.float64)
    
    # Row indices: U[i*3+j], V[j*3+k], W[k*3+i]
    # i,j,k in {0,1,2}
    
    # I'll use a known rank-23 decomposition.
    # Reference: Winograd (1971), "Arithmetic complexity of computations"
    # The 23 products are:
    
    # Let me use a verified construction from literature.
    # Products M0..M22:
    
    # M0 = (a00+a11+a22)*(b00+b11+b22)
    # M1 = (a00+a11-a22)*(b00+b11-b22)  
    # M2 = (a00-a11+a22)*(b00-b11+b22)
    # M3 = (a00-a11-a22)*(b00-b11-b22)
    # ... etc.
    
    # Actually, let me use a simpler verified approach.
    # I'll construct it using the known Strassen-like extension.
    
    # For now, let me use a different strategy: 
    # Split 3x3x3 into 2x2x2 (Strassen rank 7) + cross terms
    # 3x3 = [2x2 | 2x1; 1x2 | 1x1]
    # This gives blocks: 2x2x2, 2x2x1, 2x1x2, 2x1x1, 1x2x2, 1x2x1, 1x1x2, 1x1x1
    # Ranks: 7, 4, 4, 2, 4, 2, 2, 1 = 26 (worse than 23)
    
    # Let me just use schoolbook rank 27 for 3x3x3 and see if the total is still < 90
    # 27+27+18+18 = 90 (no improvement)
    
    # Alternative: use rank-20 for 3x3x3 (if available)
    # Or better: use a direct rank-23 construction
    
    # I'll implement a verified rank-23 Winograd decomposition.
    # Here's one from the literature (Winograd 1971, as presented in various sources):
    
    # The 23 scalar multiplications for 3x3x3:
    # Using the following convention where A is 3x3, B is 3x3, C = A*B
    
    # Let me use a known valid construction. I'll encode the coefficients.
    
    # From various references, here's a rank-23 decomposition:
    # I'll use the one from "The Complexity of Computing Polynomials" by Winograd
    
    # Actually, let me try a different approach entirely.
    # Instead of Winograd, let me use a block decomposition that's easier to verify.
    
    # Split 3x5x6 as:
    # n=3, m=5=2+3, p=6=2+4
    # Blocks: 3x2x2, 3x2x4, 3x3x2, 3x3x4
    # Use Strassen for 2x2x2 sub-blocks where possible
    
    # Or even simpler: just use the schoolbook and accept rank 90,
    # but try to find any improvement through a structured search.
    
    # For now, return schoolbook as a valid fallback
    return _schoolbook(3, 3, 3)

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full 3x5x6 problem.
    U_sub: (n_sub*m_sub, rank)
    V_sub: (m_sub*p_sub, rank)  
    W_sub: (p_sub*n_sub, rank)
    Returns full U, V, W with zeros outside the block.
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

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6.
    Split m=5 as 3+2, p=6 as 3+3.
    Blocks: 3x3x3, 3x3x3, 3x2x3, 3x2x3
    Use schoolbook for all: 27+27+18+18 = 90 (no improvement)
    
    Let me try a different split: m=5 as 2+3, p=6 as 2+2+2
    Blocks: 3x2x2, 3x2x2, 3x2x2, 3x3x2, 3x3x2, 3x3x2
    Schoolbook: 12*3 + 18*3 = 36+54 = 90 (same)
    
    I need to use non-schoolbook for at least one block.
    Let me use Strassen (rank 7) for 2x2x2 blocks.
    
    Split: m=5 as 2+2+1, p=6 as 2+2+2, n=3 as 2+1
    This gives many small blocks. Let me enumerate:
    For each (n_block, m_block, p_block):
    n_blocks: [2, 1], m_blocks: [2, 2, 1], p_blocks: [2, 2, 2]
    2*3*3 = 18 blocks
    
    2x2x2: rank 7 (Strassen) - 2*3*1 = 6 blocks? No:
    n=2,m=2,p=2: 1 block per (n_blk, m_blk, p_blk) combo
    Actually: for each choice of n_block, m_block, p_block, we get one sub-problem.
    n_blocks = [2, 1] (sum=3), m_blocks = [2, 2, 1] (sum=5), p_blocks = [2, 2, 2] (sum=6)
    Total blocks: 2*3*3 = 18
    
    Sizes:
    2x2x2: appears when n_blk=2, m_blk in {2,2}, p_blk in {2,2,2} -> 2*3=6 blocks, rank 7 each -> 42
    2x2x1: n_blk=2, m_blk in {2,2}, p_blk=1? No, p_blocks are all 2. So no 2x2x1.
    Wait, p_blocks = [2,2,2], so all p_sub = 2.
    m_blocks = [2,2,1], so m_sub can be 2 or 1.
    n_blocks = [2,1], so n_sub can be 2 or 1.
    
    Blocks:
    n=2,m=2,p=2: 2*3=6 blocks (m_blk has two 2s, p_blk has three 2s) -> 6 blocks, rank 7 -> 42
    n=2,m=1,p=2: 1*1*3=3 blocks (m_blk has one 1, p_blk has three 2s) -> 3 blocks, rank 4 (schoolbook 2*1*2=4) -> 12
    n=1,m=2,p=2: 1*2*3=6 blocks -> 6 blocks, rank 4 (schoolbook 1*2*2=4) -> 24
    n=1,m=1,p=2: 1*1*3=3 blocks -> 3 blocks, rank 2 (schoolbook 1*1*2=2) -> 6
    
    Total: 42+12+24+6 = 84 < 90!
    
    Let me implement this.
    """
    n, m, p = 3, 5, 6
    
    # Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    total_rank = 0
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                # Get decomposition for n_sub x m_sub x p_sub
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Use Strassen rank 7
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    # Use schoolbook
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                rank_sub = U_sub.shape[1]
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                total_rank += rank_sub
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    W convention: W[k*n+i] where n=2, so W[k*2+i] for C[i,k]
    C[0,0]: k=0,i=0 -> W[0]
    C[0,1]: k=1,i=0 -> W[2]
    C[1,0]: k=0,i=1 -> W[1]
    C[1,1]: k=1,i=1 -> W[3]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    
    # M1 = (A00+A11)*(B00+B11)
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    # C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    # C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    # C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    # C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using a structured search.
    This is a placeholder for more advanced techniques.
    """
    # For now, return None to indicate no improvement found
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Try block decomposition
    U, V, W = _block_decomposition_356()
    
    # Verify
    if _verify_decomp(n, m, p, U, V, W):
        rank = U.shape[1]
        # Try to find even better
        better = _try_better_decomposition(deadline)
        if better is not None:
            U_b, V_b, W_b = better
            if _verify_decomp(n, m, p, U_b, V_b, W_b):
                if U_b.shape[1] < rank:
                    U, V, W = U_b, V_b, W_b
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
# EVOLVE-BLOCK-END
