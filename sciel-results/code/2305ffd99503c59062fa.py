import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Find a low-rank tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen 2x2x2 blocks where possible, plus schoolbook for the rest.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin for 150s limit
    
    # Build schoolbook decomposition
    schoolbook_U = []
    schoolbook_V = []
    schoolbook_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U.append((i*m+j, 1))
                schoolbook_V.append((j*p+k, 1))
                schoolbook_W.append((k*n+i, 1))
    
    # Strassen 2x2x2 block
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
            'W': {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): 1}
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
    
    # Find all possible 2x2x2 blocks
    possible_blocks = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    possible_blocks.append((i0, j0, k0))
    
    # Greedy selection of non-overlapping blocks
    selected_blocks = []
    remaining = list(possible_blocks)
    while remaining:
        for idx, b in enumerate(remaining):
            if all(not blocks_overlap(b, sb) for sb in selected_blocks):
                selected_blocks.append(b)
                remaining.pop(idx)
                break
        else:
            break
    
    # Mark which schoolbook terms are covered by Strassen blocks
    covered = set()
    for b in selected_blocks:
        i0, j0, k0 = b
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0+di, j0+dj, k0+dk
                    covered.add((i, j, k))
    
    # Build U, V, W matrices
    U_terms = []
    V_terms = []
    W_terms = []
    
    for b in selected_blocks:
        strassen_terms = strassen_2x2x2(*b)
        for term in strassen_terms:
            U_terms.append(term['U'])
            V_terms.append(term['V'])
            W_terms.append(term['W'])
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def blocks_overlap(b1, b2):
    i01, j01, k01 = b1
    i02, j02, k02 = b2
    if i01 == i02 or i01 == i02+1 or i02 == i01+1:
        if j01 == j02 or j01 == j02+1 or j02 == j01+1:
            if k01 == k02 or k01 == k02+1 or k02 == k01+1:
                return True
    return False
