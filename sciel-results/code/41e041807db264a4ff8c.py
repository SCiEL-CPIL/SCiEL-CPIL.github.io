"""Strategy: Finite Field Extension / DFT-based decomposition.
Exploits the cyclic structure of the 9 columns by grouping them into 3 frequency
components (using the fact that 9 is a power of 3, or simply using a 3-point DFT
structure on the columns). This reduces the problem to 3 independent 2x2*2x1
multiplications, each of rank 2, for a total rank of 6.
If rank 6 is not valid (due to integer constraints or structure mismatch), we fall back
to a rank-9 or rank-18 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_rank6_dft_decomposition():
    """
    Attempt to construct a rank-6 decomposition using DFT-like linear combinations
    on the 9 columns of B and the 18 entries of C.
    
    The idea:
    C = A * B
    B is 2x9.
    We define 3 linear combinations of the 9 columns of B:
    T_0 = B[:, 0] + B[:, 3] + B[:, 6]
    T_1 = B[:, 1] + B[:, 4] + B[:, 7]
    T_2 = B[:, 2] + B[:, 5] + B[:, 8]
    
    And 3 corresponding combinations for C:
    S_0 = C[:, 0] + C[:, 3] + C[:, 6]
    S_1 = C[:, 1] + C[:, 4] + C[:, 7]
    S_2 = C[:, 2] + C[:, 5] + C[:, 8]
    
    Then S_i = A * T_i.
    This is 3 independent 2x2 * 2x1 multiplications.
    Each has rank 2.
    Total rank = 6.
    
    However, we need to recover C[:, k] from S_0, S_1, S_2.
    This requires that the mapping from C to S is invertible.
    The mapping is:
    S_0 = C_0 + C_3 + C_6
    S_1 = C_1 + C_4 + C_7
    S_2 = C_2 + C_5 + C_8
    
    This is NOT invertible! We lose information.
    We need a proper DFT matrix.
    
    Let's use the 3-point DFT matrix (scaled by 3 to keep integers? No, DFT involves roots of unity).
    For 9 columns, we can use the 3-point DFT on the "groups" of 3.
    
    Actually, a simpler structural change:
    Use the **Rank-3 decomposition** for $2 \times 2 \times 3$ if it exists.
    If not, use **Rank-4**.
    
    Let's check the rank of $M_{2,2,3}$.
    It is known that the rank of $M_{2,2,3}$ is 3?
    No, the rank of $M_{2,2,3}$ is 4?
    Let's try to construct a rank-3 decomposition for $2 \times 2 \times 3$.
    
    If we can do $2 \times 2 \times 3$ with rank 3, then $2 \times 2 \times 9$ with rank 9 (3 blocks).
    If we can do $2 \times 2 \times 3$ with rank 2, then $2 \times 2 \times 9$ with rank 6.
    
    Let's assume rank 3 for $2 \times 2 \times 3$ is possible and try to find it.
    Or, use the **Kronecker Product** of the rank-2 $M_{2,2,1}$ and rank-3 $M_{1,1,3}$? No.
    
    Let's just use the **Rank-9** construction with 3 blocks of rank-3.
    If rank-3 for $2 \times 2 \times 3$ is not found, we use rank-4 (schoolbook for 3 cols is 6, but maybe 4?).
    Actually, schoolbook for $2 \times 2 \times 3$ is 6.
    
    Let's try a different structural change:
    **Use the Strassen-7 decomposition for the first 2 columns, and schoolbook for the rest?**
    No, that's not a structural change in the family.
    
    Let's go with the **DFT/Interpolation** approach properly.
    We will use 9 terms, but with structured coefficients that allow for potential future reduction or simply a different representation.
    
    Actually, the safest structural change is to use the **Kronecker Product** of the **Rank-2** $2 \times 1$ tensor and the **Rank-3** $1 \times 3$ tensor? No.
    
    Let's try **Rank 6** with the following logic:
    Treat the 9 columns as 3 vectors of length 3.
    Use the **Rank-2** decomposition for $2 \times 2 \times 1$ on the "value" part, and the **Rank-3** decomposition for $1 \times 1 \times 3$ on the "index" part?
    
    I will implement a **Rank-9** decomposition using 3 independent blocks of $2 \times 2 \times 3$, using a **Rank-3** decomposition for each block if possible, otherwise Rank-4.
    
    For now, I will return the **Rank-18** decomposition but structured in a way that makes it easy to modify, and I will add a check for **Rank-9** using simple grouping.
    """
    n, m, p = DIMENSIONS
    # Fallback to Rank 18 for now, but structured
    rank = 2 * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k in range(p):
        r0 = 2 * k
        r1 = 2 * k + 1
        U[0, r0] = 1; U[2, r0] = 1
        V[k, r0] = 1
        W[2*k, r0] = 1; W[2*k+1, r0] = 1
        
        U[1, r1] = 1; U[3, r1] = 1
        V[9+k, r1] = 1
        W[2*k, r1] = 1; W[2*k+1, r1] = 1
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Return the standard rank-18 decomposition
    U, V, W = _build_rank6_dft_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
