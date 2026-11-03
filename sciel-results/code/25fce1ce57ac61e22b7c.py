import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    def verify(U, V, W):
        # Check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
            
        return np.allclose(T, 1.0, atol=1e-9)

    # Identify 2x2x2 blocks for Strassen
    # We want to find disjoint sets of indices (I, J, K) where |I|=2, |J|=2, |K|=2
    # For n=3, m=5, p=7:
    # We can fit blocks using i in {0,1} or {1,2} or {0,2}
    # Let's try to maximize the number of disjoint blocks.
    
    # Greedy approach:
    # List all possible 2x2x2 blocks
    # A block is defined by I (set of 2 i's), J (set of 2 j's), K (set of 2 k's)
    # We need to select a set of blocks such that no index (i,j,k) is in more than one block.
    
    # Since n=3 is small, we can just enumerate.
    # Possible I sets: {0,1}, {1,2}, {0,2}
    # Possible J sets: {0,1}, {1,2}, {2,3}, {3,4}, {0,2}, {0,3}, {0,4}, {1,3}, {1,4}, {2,4}
    # Possible K sets: {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}, ...
    
    # Actually, simpler:
    # We know 6 blocks fit with I={0,1}.
    # Let's define the 6 blocks found in inspiration 2/3.
    
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({0, 1}, {0, 1}, {2, 3}),
        ({0, 1}, {2, 3}, {2, 3}),
        ({0, 1}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {4, 5}),
    ]
    
    # Verify these blocks are disjoint
    covered = set()
    valid_blocks = []
    for I, J, K in blocks:
        block_terms = set()
        for i in I:
            for j in J:
                for k in K:
                    block_terms.add((i, j, k))
        if block_terms & covered:
            continue # Skip if overlaps
        covered |= block_terms
        valid_blocks.append((I, J, K))
        
    # It's possible we can add more blocks using i=2?
    # If we used I={0,1} for all, i=2 is unused in Strassen.
    # Can we add a block with I={1,2}?
    # It would overlap with I={0,1} on i=1. So no.
    # Can we add a block with I={2,?}? Only 2 is left, need 2 indices. No.
    # So 6 blocks is the max if we stick to one I-set.
    
    # What if we mix I-sets?
    # E.g., one block with I={0,1} and one with I={1,2}? They overlap on i=1.
    # So we must partition the i-index? No, the blocks are on the tensor indices.
    # If two blocks share an i-index, they might still be disjoint if they differ in j or k?
    # No, the term (i,j,k) is unique.
    # If Block A has i=1 and Block B has i=1, they are disjoint only if (j,k) pairs are disjoint.
    # Let's try to find a better packing.
    
    # Let's just use the 6 blocks we found. Rank 99.
    
    U = np.zeros((n * m, len(valid_blocks) * 7 + (n*m*p - len(valid_blocks)*8)), dtype=float)
    V = np.zeros((m * p, len(valid_blocks) * 7 + (n*m*p - len(valid_blocks)*8)), dtype=float)
    W = np.zeros((p * n, len(valid_blocks) * 7 + (n*m*p - len(valid_blocks)*8)), dtype=float)
    
    col = 0
    
    # Add Strassen blocks
    for I, J, K in valid_blocks:
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U, V, W vectors for each term
        # Term 1
        U[i0*m + j0, col] += 1
        U[i1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        W[k0*n + i0, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # Term 2
        U[i1*m + j0, col] += 1
        U[i1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + i1, col] += 1
        W[k1*n + i1, col] -= 1
        col += 1
        
        # Term 3
        U[i0*m + j0, col] += 1
        V[j0*p + k1, col] += 1
        V[j1*p + k1, col] -= 1
        W[k1*n + i0, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # Term 4
        U[i1*m + j1, col] += 1
        V[j1*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + i1, col] += 1
        col += 1
        
        # Term 5
        U[i0*m + j0, col] += 1
        U[i0*m + j1, col] += 1
        V[j1*p + k1, col] += 1
        W[k1*n + i0, col] += 1
        W[k0*n + i0, col] -= 1
        col += 1
        
        # Term 6
        U[i1*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + k1, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # Term 7
        U[i0*m + j1, col] += 1
        U[i1*m + j1, col] -= 1
        V[j1*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        W[k0*n + i0, col] += 1
        col += 1
        
        if time.time() > deadline:
            break
            
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if time.time() > deadline:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Trim unused columns if any (should be exact)
    if col < U.shape[1]:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
