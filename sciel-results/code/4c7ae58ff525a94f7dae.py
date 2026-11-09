"""Matrix multiplication 5x5x5 using a block decomposition with Strassen for 2x2x2 blocks."""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _trivial_ump(n, m, p):
    """Return (U, V, W) for trivial multiplication of n x m by m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_222():
    """Return (U, V, W) for Strassen 2x2x2 decomposition.
    
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Row indices for U: (i, j) with i in 0..1, j in 0..1 -> row = i*2 + j
    Row indices for V: (j, k) with j in 0..1, k in 0..1 -> row = j*2 + k
    Row indices for W: (k, i) with k in 0..1, i in 0..1 -> row = k*2 + i
    """
    # 7 bilinear forms for Strassen
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U matrix: rows indexed by (i, j), columns by form
    # Form 1: (i=0,j=0) and (i=1,j=1) get +1
    # Form 2: (i=1,j=0) and (i=1,j=1) get +1
    # Form 3: (i=0,j=0) gets +1
    # Form 4: (i=1,j=1) gets +1
    # Form 5: (i=0,j=0) and (i=0,j=1) get +1
    # Form 6: (i=1,j=0) gets +1, (i=0,j=0) gets -1
    # Form 7: (i=0,j=1) gets +1, (i=1,j=1) gets -1
    
    U = np.zeros((4, 7), dtype=float)
    # (i,j) -> row: i*2+j
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Form 1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    # Form 2
    U[2, 1] = 1.0; U[3, 1] = 1.0
    # Form 3
    U[0, 2] = 1.0
    # Form 4
    U[3, 3] = 1.0
    # Form 5
    U[0, 4] = 1.0; U[1, 4] = 1.0
    # Form 6
    U[2, 5] = 1.0; U[0, 5] = -1.0
    # Form 7
    U[1, 6] = 1.0; U[3, 6] = -1.0
    
    # V matrix: rows indexed by (j, k), columns by form
    # Form 1: (j=0,k=0) and (j=1,k=1) get +1
    # Form 2: (j=0,k=0) gets +1
    # Form 3: (j=0,k=1) gets +1, (j=1,k=1) gets -1
    # Form 4: (j=1,k=0) gets +1, (j=0,k=0) gets -1
    # Form 5: (j=1,k=1) gets +1
    # Form 6: (j=0,k=0) gets +1, (j=0,k=1) gets +1
    # Form 7: (j=1,k=0) gets +1, (j=1,k=1) gets +1
    
    V = np.zeros((4, 7), dtype=float)
    # (j,k) -> row: j*2+k
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Form 1
    V[0, 0] = 1.0; V[3, 0] = 1.0
    # Form 2
    V[0, 1] = 1.0
    # Form 3
    V[1, 2] = 1.0; V[3, 2] = -1.0
    # Form 4
    V[2, 3] = 1.0; V[0, 3] = -1.0
    # Form 5
    V[3, 4] = 1.0
    # Form 6
    V[0, 5] = 1.0; V[1, 5] = 1.0
    # Form 7
    V[2, 6] = 1.0; V[3, 6] = 1.0
    
    # W matrix: rows indexed by (k, i), columns by form
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M7
    
    # W rows: (k,i) -> row: k*2+i
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # k=0,i=0 -> C11 -> row 0
    # k=0,i=1 -> C12 -> row 1
    # k=1,i=0 -> C21 -> row 2
    # k=1,i=1 -> C22 -> row 3
    
    W = np.zeros((4, 7), dtype=float)
    # C11 = M1 - M4 - M6 + M7
    W[0, 0] = 1.0; W[0, 3] = -1.0; W[0, 5] = -1.0; W[0, 6] = 1.0
    # C12 = M3 + M5
    W[1, 2] = 1.0; W[1, 4] = 1.0
    # C21 = M2 + M4
    W[2, 1] = 1.0; W[2, 3] = 1.0
    # C22 = M1 - M3 + M5 - M7
    W[3, 0] = 1.0; W[3, 2] = -1.0; W[3, 4] = 1.0; W[3, 6] = -1.0
    
    return U, V, W

def _get_block_ump(n, m, p):
    """Return (U, V, W) for multiplication of n x m by m x p using best known method."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    else:
        return _trivial_ump(n, m, p)

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partition 5 = 2 + 2 + 1
    # Block sizes for A rows: [2, 2, 1]
    # Block sizes for A cols / B rows: [2, 2, 1]
    # Block sizes for B cols / C cols: [2, 2, 1]
    
    block_sizes = [2, 2, 1]
    num_blocks = len(block_sizes)
    
    # Precompute cumulative offsets
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)
    
    # Get base decompositions for each block combination
    # We need U, V, W for each (bi, bj, bk) combination
    block_umps = {}
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                bn = block_sizes[bi]
                bm = block_sizes[bj]
                bp = block_sizes[bk]
                block_umps[(bi, bj, bk)] = _get_block_ump(bn, bm, bp)
    
    # Calculate total rank
    total_rank = 0
    for (bi, bj, bk), (U, V, W) in block_umps.items():
        total_rank += U.shape[1]
    
    # Build global U, V, W
    # U has shape (n*m, total_rank) = (25, total_rank)
    # V has shape (m*p, total_rank) = (25, total_rank)
    # W has shape (p*n, total_rank) = (25, total_rank)
    
    U_global = np.zeros((n * m, total_rank), dtype=float)
    V_global = np.zeros((m * p, total_rank), dtype=float)
    W_global = np.zeros((p * n, total_rank), dtype=float)
    
    rank_offset = 0
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                U_b, V_b, W_b = block_umps[(bi, bj, bk)]
                rank_b = U_b.shape[1]
                
                if rank_b == 0:
                    continue
                
                # Map local indices to global indices
                # For U: local row (i, j) where i in block bi, j in block bj
                #   global row = (offsets[bi] + i) * m + (offsets[bj] + j)
                # For V: local row (j, k) where j in block bj, k in block bk
                #   global row = (offsets[bj] + j) * p + (offsets[bk] + k)
                # For W: local row (k, i) where k in block bk, i in block bi
                #   global row = (offsets[bk] + k) * n + (offsets[bi] + i)
                
                bn = block_sizes[bi]
                bm = block_sizes[bj]
                bp = block_sizes[bk]
                
                for r in range(rank_b):
                    col = rank_offset + r
                    
                    # Fill U
                    for lu in range(U_b.shape[0]):
                        if U_b[lu, r] != 0:
                            # Local (i, j): i = lu // bm, j = lu % bm
                            li = lu // bm
                            lj = lu % bm
                            gi = offsets[bi] + li
                            gj = offsets[bj] + lj
                            U_global[gi * m + gj, col] = U_b[lu, r]
                    
                    # Fill V
                    for lv in range(V_b.shape[0]):
                        if V_b[lv, r] != 0:
                            # Local (j, k): j = lv // bp, k = lv % bp
                            lj = lv // bp
                            lk = lv % bp
                            gj = offsets[bj] + lj
                            gk = offsets[bk] + lk
                            V_global[gj * p + gk, col] = V_b[lv, r]
                    
                    # Fill W
                    for lw in range(W_b.shape[0]):
                        if W_b[lw, r] != 0:
                            # Local (k, i): k = lw // bn, i = lw % bn
                            lk = lw // bn
                            li = lw % bn
                            gk = offsets[bk] + lk
                            gi = offsets[bi] + li
                            W_global[gk * n + gi, col] = W_b[lw, r]
                
                rank_offset += rank_b
    
    # Convert to lists
    U_list = U_global.tolist()
    V_list = V_global.tolist()
    W_list = W_global.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
