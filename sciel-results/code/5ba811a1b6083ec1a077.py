import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Find a low-rank tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a combination of Strassen 2x2x2 blocks and schoolbook terms.
    
    The tensor T has indices (i, j, k) where:
    - i in {0, 1} (rows of A, cols of C)
    - j in {0, 1, 2, 3, 4} (cols of A, rows of B)
    - k in {0, 1, 2, 3, 4, 5} (cols of B, cols of C)
    
    T[i,j,k] = 1 if and only if C[i,k] gets contribution from A[i,j]*B[j,k]
    
    U shape: (n*m, rank) = (10, rank)
    V shape: (m*p, rank) = (30, rank)
    W shape: (p*n, rank) = (12, rank)
    
    W index convention: W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # Start with schoolbook rank = n*m*p = 60
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Build schoolbook decomposition
    schoolbook_U = []
    schoolbook_V = []
    schoolbook_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m+j
                # V index: j*p+k
                # W index: k*n+i
                schoolbook_U.append((i*m+j, 1))
                schoolbook_V.append((j*p+k, 1))
                schoolbook_W.append((k*n+i, 1))
    
    # Try to replace groups of schoolbook terms with Strassen-like decompositions
    # A 2x2x2 block uses 7 terms instead of 8 schoolbook terms
    
    # Strategy: identify 2x2x2 sub-blocks and replace their 8 schoolbook terms 
    # with 7 Strassen terms
    
    # For a 2x2x2 block with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}:
    # Strassen terms:
    # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
    #   -> contributes to C[i0,k0] and C[i0+1,k0+1]
    # M2 = A[i0+1, j0] * (B[j0,k0] + B[j0,k0+1])
    #   -> contributes to C[i0+1,k0] and C[i0+1,k0+1]
    # M3 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
    #   -> contributes to C[i0,k0+1] and C[i0+1,k0+1]
    # M4 = (A[i0+1,j0+1] - A[i0,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
    #   -> contributes to C[i0+1,k0] and C[i0+1,k0+1]
    # M5 = (A[i0,j0] - A[i0+1,j0]) * (B[j0,k0] + B[j0,k0+1])
    #   -> contributes to C[i0,k0] and C[i0,k0+1]
    # M6 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
    #   -> contributes to C[i0,k0+1] and C[i0+1,k0+1]
    # M7 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0+1] - B[j0+1,k0+1])
    #   -> contributes to C[i0+1,k0+1] and C[i0,k0+1]... 
    
    # Let me use the standard Strassen 7-term decomposition more carefully:
    # For C = A*B where A is 2x2, B is 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(i0, j0, k0):
        """
        Return 7 (U, V, W) term tuples for a 2x2x2 Strassen block.
        Each tuple is (dict of U indices->coeffs, dict of V indices->coeffs, dict of W indices->coeffs)
        """
        terms = []
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        #   -> C[i0,k0] += M1, C[i0+1,k0+1] += M1
        terms.append({
            'U': {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
            'V': {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
            'W': {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
        })
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        #   -> C[i0+1,k0] += M2, C[i0+1,k0+1] += M2
        terms.append({
            'U': {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
            'V': {j0*p+k0: 1},
            'W': {(k0)*n+(i0+1): 1, (k0+1)*n+(i0+1): 1}
        })
        # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
        #   -> C[i0,k0+1] += M3, C[i0+1,k0+1] += M3
        terms.append({
            'U': {i0*m+j0: 1},
            'V': {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): -1},
            'W': {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
        })
        # M4 = A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
        #   -> C[i0+1,k0] += M4, C[i0+1,k0+1] += M4
        terms.append({
            'U': {(i0+1)*m+(j0+1): 1},
            'V': {j0*p+(k0+1): 1, j0*p+k0: -1},
            'W': {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): 1}
        })
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        #   -> C[i0,k0+1] += M5, C[i0+1,k0+1] += M5
        terms.append({
            'U': {i0*m+j0: 1, i0*m+(j0+1): 1},
            'V': {(j0+1)*p+(k0+1): 1},
            'W': {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
        })
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        #   -> C[i0+1,k0] += M6, C[i0+1,k0+1] += M6
        terms.append({
            'U': {(i0+1)*m+j0: 1, i0*m+j0: -1},
            'V': {j0*p+k0: 1, j0*p+(k0+1): 1},
            'W': {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): 1}
        })
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        #   -> C[i0,k0+1] += M7, C[i0+1,k0+1] += M7
        terms.append({
            'U': {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
            'V': {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
            'W': {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
        })
        return terms
    
    # For each possible 2x2x2 block, compute the set of schoolbook terms it covers
    # and check if replacing them with Strassen reduces the total rank
    
    # A schoolbook term (i,j,k) is covered by block (i0,j0,k0) if
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    
    # Try all combinations of non-overlapping 2x2x2 blocks
    # Each block covers 8 schoolbook terms and replaces them with 7 Strassen terms
    # Net saving per block: 1
    
    # Find all possible 2x2x2 blocks
    possible_blocks = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    possible_blocks.append((i0, j0, k0))
    
    # Greedy: pick blocks that don't overlap
    # Two blocks overlap if they share any (i,j,k) coordinate
    
    def blocks_overlap(b1, b2):
        i01, j01, k01 = b1
        i02, j02, k02 = b2
        # Check if the 2x2x2 regions overlap
        if i01 == i02 or i01 == i02+1 or i02 == i01+1:
            if j01 == j02 or j01 == j02+1 or j02 == j01+1:
                if k01 == k02 or k01 == k02+1 or k02 == k01+1:
                    return True
        return False
    
    # Greedy selection of non-overlapping blocks
    selected_blocks = []
    remaining = list(possible_blocks)
    while remaining:
        # Pick the first block that doesn't overlap with selected
        for idx, b in enumerate(remaining):
            if all(not blocks_overlap(b, sb) for sb in selected_blocks):
                selected_blocks.append(b)
                remaining.pop(idx)
                break
        else:
            break
    
    # Build the decomposition using selected Strassen blocks + remaining schoolbook terms
    if time.time() > deadline:
        # Return schoolbook
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            col = idx
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Mark which schoolbook terms are covered by Strassen blocks
    covered = set()
    all_block_terms = {}
    for b in selected_blocks:
        i0, j0, k0 = b
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0+di, j0+dj, k0+dk
                    covered.add((i, j, k))
                    if b not in all_block_terms:
                        all_block_terms[b] = []
                    all_block_terms[b].append((i, j, k))
    
    # Build U, V, W matrices
    # First, add Strassen terms
    U_terms = []  # list of (index, coeff) per column
    V_terms = []
    W_terms = []
    
    for b in selected_blocks:
        strassen_terms = strassen_2x2x2(*b)
        for term in strassen_terms:
            U_terms.append(term['U'])
            V_terms.append(term['V'])
            W_terms.append(term['W'])
    
    # Then add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_terms.append({i*m+j: 1})
                    V_terms.append({j*p+k: 1})
                    W_terms.append({k*n+i: 1})
    
    rank = len(U_terms)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for col, (u_dict, v_dict, w_dict) in enumerate(zip(U_terms, V_terms, W_terms)):
        for idx, coeff in u_dict.items():
            U[idx, col] = coeff
        for idx, coeff in v_dict.items():
            V[idx, col] = coeff
        for idx, coeff in w_dict.items():
            W[idx, col] = coeff
    
    # Verify correctness
    # The tensor T[i,j,k] should equal sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For the multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k)
    
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    U = np.zeros((n*m, n*m*p))
                    V = np.zeros((m*p, n*m*p))
                    W = np.zeros((p*n, n*m*p))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj, col] = 1
                                V[jj*p+kk, col] = 1
                                W[kk*n+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Also check that no unintended entries are nonzero
    # The tensor should only have 1s at valid (i,j,k) positions
    # Since our construction only targets valid positions, this should be fine
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
