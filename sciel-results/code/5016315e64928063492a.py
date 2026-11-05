import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Build a library of block decompositions
    # Each block covers a contiguous set of j-indices and k-indices
    # For each block, we have multiple decomposition options with different ranks
    
    # Decomposition options for a 2x2x2 block (i=0,1; j=j1,j2; k=k1,k2):
    # Option A: Strassen (rank 7)
    # Option B: Schoolbook (rank 8)
    
    # For 2x2x1 block (i=0,1; j=j1,j2; k=k):
    # Option: Schoolbook (rank 4)
    
    # For 2x1x2 block (i=0,1; j=j; k=k1,k2):
    # Option: Schoolbook (rank 4)
    
    # For 1x2x2 block (i=i; j=j1,j2; k=k1,k2):
    # Option: Schoolbook (rank 4)
    
    # For 1x1x2, 1x2x1, 2x1x1, 1x1x1 blocks: schoolbook
    
    # We'll enumerate all ways to partition the j-dimension (size 4) and k-dimension (size 5)
    # into contiguous blocks, then for each block choose the best decomposition.
    
    # A partition of range [0, size) into contiguous blocks is a composition of size
    # into parts. For size=4: compositions are (4), (1,3),(2,2),(3,1),(1,1,2),(1,2,1),(2,1,1),
    # (1,1,1,1) = 2^(4-1) = 8 compositions
    # For size=5: 2^(5-1) = 16 compositions
    
    def compositions(size):
        """Generate all compositions of size into positive integers (contiguous block sizes)."""
        if size == 1:
            yield [1]
            return
        for first in range(1, size + 1):
            for rest in compositions(size - first):
                yield [first] + rest
    
    # For a block of dimensions (ni, nj, nk) where ni is always 2 (since n=2),
    # determine the rank of the best known decomposition:
    # If nj=2 and nk=2: Strassen rank 7, or schoolbook rank 8
    # If nj=2 and nk=1: schoolbook rank 4
    # If nj=1 and nk=2: schoolbook rank 4
    # If nj=1 and nk=1: schoolbook rank 2
    # In general: ni*nj*nk for schoolbook, or 7 if ni=nj=nk=2
    
    def block_rank(nj, nk):
        """Rank of best decomposition for a 2 x nj x nk block."""
        if nj == 2 and nk == 2:
            return 7  # Strassen
        else:
            return 2 * nj * nk  # schoolbook
    
    # Generate all j-partitions and k-partitions
    j_parts_list = list(compositions(m))  # m=4
    k_parts_list = list(compositions(p))  # p=5
    
    best_rank = n * m * p  # fallback: full schoolbook = 40
    best_decomp = None
    
    # For each combination of j-partition and k-partition, compute total rank
    # and check if it's better than current best
    for j_parts in j_parts_list:
        for k_parts in k_parts_list:
            total_rank = 0
            for nj in j_parts:
                for nk in k_parts:
                    total_rank += block_rank(nj, nk)
            if total_rank < best_rank:
                best_rank = total_rank
                best_decomp = (j_parts, k_parts)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_decomp is None:
        # Fallback to full schoolbook
        best_rank = n * m * p
        best_decomp = ([m], [p])
    
    j_parts, k_parts = best_decomp
    
    # Now build the decomposition
    rank = best_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1
        c = col_offset + 0
        U[a11, c] += 1; U[a22, c] += 1
        V[b11, c] += 1; V[b22, c] += 1
        W[c11, c] += 1; W[c22, c] += 1
        
        # M2
        c = col_offset + 1
        U[a21, c] += 1; U[a22, c] += 1
        V[b11, c] += 1
        W[c21, c] += 1; W[c22, c] += -1
        
        # M3
        c = col_offset + 2
        U[a11, c] += 1
        V[b12, c] += 1; V[b22, c] += -1
        W[c12, c] += 1; W[c22, c] += 1
        
        # M4
        c = col_offset + 3
        U[a22, c] += 1
        V[b21, c] += 1; V[b11, c] += -1
        W[c11, c] += 1; W[c21, c] += 1
        
        # M5
        c = col_offset + 4
        U[a11, c] += 1; U[a12, c] += 1
        V[b22, c] += 1
        W[c11, c] += -1; W[c12, c] += 1
        
        # M6
        c = col_offset + 5
        U[a21, c] += 1; U[a11, c] += -1
        V[b11, c] += 1; V[b12, c] += 1
        W[c22, c] += 1
        
        # M7
        c = col_offset + 6
        U[a12, c] += 1; U[a22, c] += -1
        V[b21, c] += 1; V[b22, c] += 1
        W[c11, c] += 1
        
        return col_offset + 7
    
    def add_schoolbook_block(i1, i2, j1, j2, k1, k2, col_offset):
        """Schoolbook decomposition for a 2 x nj x nk block."""
        for i in range(i1, i2 + 1):
            for j in range(j1, j2 + 1):
                for k in range(k1, k2 + 1):
                    U[i * m + j, col_offset] += 1
                    V[j * p + k, col_offset] += 1
                    W[k * n + i, col_offset] += 1
                    col_offset += 1
        return col_offset
    
    j_start = 0
    for nj in j_parts:
        j_end = j_start + nj - 1
        k_start = 0
        for nk in k_parts:
            k_end = k_start + nk - 1
            if nj == 2 and nk == 2:
                col = add_strassen(0, 1, j_start, j_end, k_start, k_end, col)
            else:
                col = add_schoolbook_block(0, 1, j_start, j_end, k_start, k_end, col)
            k_start = k_end + 1
        j_start = j_end + 1
    
    # Verify
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=1e-9):
        # Fallback to schoolbook rank 40
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s))
        V_s = np.zeros((m * p, rank_s))
        W_s = np.zeros((p * n, rank_s))
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c2] = 1
                    V_s[j * p + k, c2] = 1
                    W_s[k * n + i, c2] = 1
                    c2 += 1
        return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
