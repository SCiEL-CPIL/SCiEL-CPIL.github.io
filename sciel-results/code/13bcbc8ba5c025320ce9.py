"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition.

Strategy: exploit the block structure along the p-axis (15 columns).
We split the 15 columns into 7 groups of 2 columns + 1 residual column.
- For each pair of columns, use the optimal rank-7 Strassen decomposition for 2x2x2.
- For the residual single column, we can potentially do better than rank-4 by
  sharing structure across the A dimension.

Key insight: For a 2x2x1 block (single column), the tensor is:
  T = sum_{i,j,k} T[i,j,k] e_i^A e_j^B e_k^C
where k is fixed. This is a 2x2 matrix multiplication tensor, which has rank 4
(classical) but can it be done in rank < 4? No, the tensor rank of 2x2x1 is 4.

However, we can try to combine the residual column with one of the Strassen blocks
to share some terms. Specifically, we can merge column 14 into one of the pairs.

Alternative structural approach: Instead of 7 pairs + 1 single, try:
- 6 pairs (columns 0-11) with Strassen rank 7 each = 42
- 3 columns (12,13,14) handled as a 2x2x3 block

The tensor rank of 2x2x3 matrix multiplication: We need to multiply a 2x2 matrix
by a 2x3 matrix. The naive rank is 12, Strassen on two 2x2 blocks gives 14, but
we can do better. Let's think about it.

Actually, for 2x2x3: C = A * B where A is 2x2, B is 2x3, C is 2x3.
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2}.

This has 6 output entries, each a sum of 2 products. The tensor is:
T = sum_{i=0}^{1} sum_{k=0}^{2} sum_{j=0}^{1} e_{i}^A e_{j}^B e_{k}^C e_{j}^A... 

Wait, let me be more careful. The multiplication tensor for (n,m,p) = (2,2,3):
T = sum_{i=0}^{1} sum_{k=0}^{2} sum_{j=0}^{1} a_{i,j} b_{j,k} c_{i,k}

The tensor rank r(2,2,3) is known. Let me compute it.

The naive algorithm uses 2*2*3 = 12 multiplications.
Strassen-based: split B into [B_01, B_2] where B_01 is 2x2 and B_2 is 2x1.
Then rank = r(2,2,2) + r(2,2,1) = 7 + 4 = 11.

Can we do better? Let's try rank 10 or less.

For rank 10: we need 10 bilinear forms. Let's see if we can find them.

Actually, let me try a different split. What about:
- Use 3 columns together with a clever decomposition.

The tensor T_{2,2,3} has 12 nonzero entries (a_{ij} b_{jk} c_{ik} for i,j in {0,1}, k in {0,1,2}).

Let me try to find a rank-10 decomposition by hand or by search.

Given the time constraint, let me try a different approach: use a known result or
do a small search for rank-10 decomposition of 2x2x3.

Actually, I recall that r(2,2,3) = 10 is achievable. Let me construct it.

The idea: we have 6 outputs C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
Naive: 12 products.

Strassen on first 2 columns: 7 products for C[:,0:2], plus 4 for C[:,2] = 11.

To get 10, we need to share 1 product. Let's try:
P1 = (A00+A11)(B00+B11) -- but this is for 2x2x2, not directly applicable.

Let me try a different approach. Write:
C00 = A00*B00 + A01*B10
C10 = A10*B00 + A11*B10
C01 = A00*B01 + A01*B11
C11 = A10*B01 + A11*B11
C02 = A00*B02 + A01*B12
C12 = A10*B02 + A11*B12

Group as:
C00 = A00*B00 + A01*B10
C10 = A10*B00 + A11*B10
C01 = A00*B01 + A01*B11
C11 = A10*B01 + A11*B11
C02 = A00*B02 + A01*B12
C12 = A10*B02 + A11*B12

Let me define:
P1 = (A00 + A10) * B00
P2 = (A01 + A11) * B00  -- no, this doesn't work cleanly.

Let me try:
P1 = (A00 + A11) * (B00 + B10)  -- this gives A00B00 + A00B10 + A11B00 + A11B10
P2 = (A00 - A11) * (B00 - B10)  -- this gives A00B00 - A00B10 - A11B00 + A11B10
P3 = A00 * (B10 - B00)
P4 = A11 * (B00 - B10)
P5 = (A00 + A01) * B11
P6 = (A10 - A00) * (B00 + B01)
P7 = (A01 - A11) * (B10 + B11)

This is getting complicated. Let me just use the rank-11 decomposition (6 Strassen blocks + 1 three-column block with rank 11) and see if the total is 6*7 + 11 = 53, same as before.

Wait, that's the same rank. Let me reconsider.

Current: 7 Strassen (2x2x2) blocks = 49, plus 1 schoolbook (2x2x1) = 4. Total = 53.

Alternative: 6 Strassen blocks = 42, plus 1 three-column block (2x2x3). If r(2,2,3) = 10, total = 52.

Let me try to find a rank-10 decomposition of 2x2x3.

I'll do a small search using the fact that we need integer/half-integer coefficients.

Let me try to construct it. The 12 tensor entries are:
T[0,0,0] = 1  (A00 * B00 -> C00)
T[0,0,1] = 1  (A00 * B01 -> C01)
T[0,0,2] = 1  (A00 * B02 -> C02)
T[0,1,0] = 1  (A01 * B10 -> C00)
T[0,1,1] = 1  (A01 * B11 -> C01)
T[0,1,2] = 1  (A01 * B12 -> C02)
T[1,0,0] = 1  (A10 * B00 -> C10)
T[1,0,1] = 1  (A10 * B01 -> C11)
T[1,0,2] = 1  (A10 * B02 -> C12)
T[1,1,0] = 1  (A11 * B10 -> C10)
T[1,1,1] = 1  (A11 * B11 -> C11)
T[1,1,2] = 1  (A11 * B12 -> C12)

I'll search for a rank-10 decomposition with small integer coefficients.
"""
import time
import itertools
import numpy as np

_t0 = time.time()
_DEADLINE = _t0 + 135.0  # safety margin before 150s kill


def _find_2x2x3_rank10():
    """Search for a rank-10 decomposition of the 2x2x3 matrix multiplication tensor.
    
    Returns (U, V, W) as 2D arrays or None if not found.
    U: 4x10, V: 6x10, W: 6x10
    U rows: A00, A01, A10, A11
    V rows: B00, B01, B02, B10, B11, B12
    W rows: C00, C01, C02, C10, C11, C12
    """
    # Try to construct by extending Strassen
    # Use 7 Strassen-like terms for the first 2 columns, plus 3 more for column 2
    
    # Actually, let me try a direct approach: use 10 rank-1 tensors.
    # Each rank-1 tensor is (u, v, w) where u in R^4, v in R^6, w in R^6.
    # The sum of outer products must equal the tensor.
    
    # The tensor T has entries T[i,j,k] = 1 if (i,k) is a valid output and j is the 
    # column of A / row of B, else 0.
    
    # T[i,j,k] = 1 iff i is row of A/C, j is col of A / row of B, k is col of B/C
    # and the product A[i,j]*B[j,k] contributes to C[i,k].
    
    # So T[i,j,k] = 1 for all valid (i,j,k) triples where j is the shared index.
    # i in {0,1}, j in {0,1}, k in {0,1,2}
    
    # Let's try a structured approach:
    # Terms 1-7: Strassen for columns 0,1
    # Terms 8-10: handle column 2 with 3 terms instead of 4
    
    # For column 2, we need:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Can we do this in 3 terms? The rank of 2x2x1 is 4, so no.
    # But we can share with existing terms.
    
    # Let me try: add B02 and B12 to existing Strassen terms.
    
    # Strassen for cols 0,1:
    # P1 = (A00+A11)(B00+B10) -> contributes to C00, C11
    # P2 = (A10+A11)B00 -> contributes to C10, C11
    # P3 = A00(B01-B10) -> contributes to C01, C11
    # P4 = A11(B10-B00) -> contributes to C00, C10
    # P5 = (A00+A01)B10 -> contributes to C00, C01
    # P6 = (A10-A00)(B00+B01) -> contributes to C11
    # P7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # Wait, I need to be more careful with the Strassen formulas.
    
    # Standard Strassen for 2x2:
    # C00 = P1 + P4 - P5 + P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P3 + P6
    
    # Where:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For 2 columns (k=0,1), B0k means B[0,k], B1k means B[1,k].
    # So for the 2x2x2 block with columns k0, k1:
    # B00 -> B[0,k0], B01 -> B[0,k1], B10 -> B[1,k0], B11 -> B[1,k1]
    
    # Now for the third column k2, I want to reuse some of these.
    
    # Let me try a different decomposition. I'll use a search.
    
    # Given the complexity, let me just return the known rank-11 and accept rank 53.
    # But first, let me try a quick search for rank 10.
    
    pass  # Placeholder - will implement below if time permits


def _build_strassen_block(k0, k1, base_rank, U, V, W, n=2, m=2, p=15):
    """Build a Strassen 2x2x2 block for columns k0, k1 starting at base_rank."""
    # Strassen terms for 2x2x2
    # P1 = (A00+A11)(B0k0+B1k1)
    # P2 = (A10+A11)B0k0
    # P3 = A00(B0k1-B1k1)
    # P4 = A11(B1k0-B0k0)
    # P5 = (A00+A01)B1k1
    # P6 = (A10-A00)(B0k0+B0k1)
    # P7 = (A01-A11)(B1k0+B1k1)
    
    # U contributions (rows: 0=A00, 1=A01, 2=A10, 3=A11)
    U_terms = [
        [(0, 1), (3, 1)],   # P1
        [(2, 1), (3, 1)],   # P2
        [(0, 1)],           # P3
        [(3, 1)],           # P4
        [(0, 1), (1, 1)],   # P5
        [(2, 1), (0, -1)],  # P6
        [(1, 1), (3, -1)],  # P7
    ]
    
    # V contributions (rows: j*p + k, so B0k0 = 0*p+k0, B0k1 = 0*p+k1, B1k0 = 1*p+k0, B1k1 = 1*p+k1)
    V_rows = [k0, k1, p + k0, p + k1]  # B0k0, B0k1, B1k0, B1k1
    V_terms = [
        [(V_rows[0], 1), (V_rows[3], 1)],   # P1: B0k0 + B1k1
        [(V_rows[0], 1)],                    # P2: B0k0
        [(V_rows[1], 1), (V_rows[3], -1)],  # P3: B0k1 - B1k1
        [(V_rows[2], 1), (V_rows[0], -1)],  # P4: B1k0 - B0k0
        [(V_rows[3], 1)],                    # P5: B1k1
        [(V_rows[0], 1), (V_rows[1], 1)],   # P6: B0k0 + B0k1
        [(V_rows[2], 1), (V_rows[3], 1)],   # P7: B1k0 + B1k1
    ]
    
    # W contributions (rows: k*n + i, so C0k0 = k0*2+0, C0k1 = k1*2+0, C1k0 = k0*2+1, C1k1 = k1*2+1)
    W_rows = [k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1]  # C0k0, C0k1, C1k0, C1k1
    W_terms = [
        [(W_rows[0], 1), (W_rows[3], 1)],    # P1: C0k0 + C1k1
        [(W_rows[2], 1), (W_rows[3], -1)],   # P2: C1k0 - C1k1
        [(W_rows[1], 1), (W_rows[3], 1)],    # P3: C0k1 + C1k1
        [(W_rows[0], 1), (W_rows[2], 1)],    # P4: C0k0 + C1k0
        [(W_rows[0], -1), (W_rows[1], 1)],   # P5: -C0k0 + C0k1
        [(W_rows[3], 1)],                    # P6: C1k1
        [(W_rows[0], 1)],                    # P7: C0k0
    ]
    
    for t in range(7):
        r = base_rank + t
        for (row, c) in U_terms[t]:
            U[row][r] += c
        for (row, c) in V_terms[t]:
            V[row][r] += c
        for (row, c) in W_terms[t]:
            W[row][r] += c
    
    return base_rank + 7


def _build_2x2x1_block(k, base_rank, U, V, W, n=2, m=2, p=15):
    """Build a schoolbook 2x2x1 block for column k starting at base_rank. Rank 4."""
    # C0k = A00*B0k + A01*B1k
    # C1k = A10*B0k + A11*B1k
    
    # Term 0: A00 * B0k -> C0k
    U[0][base_rank + 0] = 1
    V[0 * p + k][base_rank + 0] = 1
    W[k * n + 0][base_rank + 0] = 1
    
    # Term 1: A01 * B1k -> C0k
    U[1][base_rank + 1] = 1
    V[1 * p + k][base_rank + 1] = 1
    W[k * n + 0][base_rank + 1] = 1
    
    # Term 2: A10 * B0k -> C1k
    U[2][base_rank + 2] = 1
    V[0 * p + k][base_rank + 2] = 1
    W[k * n + 1][base_rank + 2] = 1
    
    # Term 3: A11 * B1k -> C1k
    U[3][base_rank + 3] = 1
    V[1 * p + k][base_rank + 3] = 1
    W[k * n + 1][base_rank + 3] = 1
    
    return base_rank + 4


def construct():
    n, m, p = 2, 2, 15
    rank = 53  # 7*7 + 4
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 53
    V = [[0] * rank for _ in range(m * p)]   # 30 x 53
    W = [[0] * rank for _ in range(p * n)]   # 30 x 53
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for blk in range(7):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        base = 7 * blk
        _build_strassen_block(k0, k1, base, U, V, W, n, m, p)
    
    # 1 schoolbook block for column 14
    b14 = 49
    _build_2x2x1_block(14, b14, U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
