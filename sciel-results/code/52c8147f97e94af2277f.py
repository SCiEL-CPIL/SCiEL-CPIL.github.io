"""Optimized rank-minimal tensor decomposition for 4x4 x 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Create the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W


def _pairwise_merge(U, V, W, n, m, p, deadline):
    """
    Try to find and remove terms that are scalar multiples of other terms.
    """
    start_time = time.time()
    rank = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    keep = [True] * rank
    
    for c1 in range(rank):
        if time.time() > deadline - 20:
            break
        if not keep[c1]:
            continue
        for c2 in range(c1+1, rank):
            if not keep[c2]:
                continue
            
            U1, U2 = current_U[:, c1], current_U[:, c2]
            V1, V2 = current_V[:, c1], current_V[:, c2]
            W1, W2 = current_W[:, c1], current_W[:, c2]
            
            mask = U1 != 0
            if not np.any(mask):
                continue
            alphas = U2[mask] / U1[mask]
            if not np.allclose(alphas, alphas[0], atol=1e-9):
                continue
            alpha = alphas[0]
            
            maskV = V1 != 0
            if np.any(maskV):
                alphasV = V2[maskV] / V1[maskV]
                if not np.allclose(alphasV, alpha, atol=1e-9):
                    continue
            
            maskW = W1 != 0
            if np.any(maskW):
                alphasW = W2[maskW] / W1[maskW]
                if not np.allclose(alphasW, alpha, atol=1e-9):
                    continue
            
            keep[c2] = False
            current_rank -= 1
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            if keep[c]:
                                s += current_U[i*m+j, c] * current_V[j*p+k, c] * current_W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                keep[c2] = True
                current_rank += 1
    
    if current_rank < rank:
        keep_indices = [c for c in range(rank) if keep[c]]
        return current_U[:, keep_indices], current_V[:, keep_indices], current_W[:, keep_indices]
    return U, V, W


def _greedy_reduce(U, V, W, n, m, p, deadline):
    """
    Greedy rank reduction with redundancy-based ordering.
    """
    start_time = time.time()
    rank = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    keep = [True] * rank
    
    # Compute redundancy scores
    term_redundancy = []
    for c in range(rank):
        support = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if current_U[i*m+j, c] != 0 and current_V[j*p+k, c] != 0 and current_W[k*n+i, c] != 0:
                        support.add((i,j,k))
        
        overlap = 0
        for c2 in range(rank):
            if c2 == c or not keep[c2]:
                continue
            for pos in support:
                i, j, k = pos
                if current_U[i*m+j, c2] != 0 and current_V[j*p+k, c2] != 0 and current_W[k*n+i, c2] != 0:
                    overlap += 1
        
        term_redundancy.append((overlap, c))
    
    term_redundancy.sort(reverse=True)
    
    for overlap, idx in term_redundancy:
        if time.time() > deadline - 30:
            break
        if not keep[idx]:
            continue
        
        keep[idx] = False
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        if keep[c]:
                            s += current_U[i*m+j, c] * current_V[j*p+k, c] * current_W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            current_rank -= 1
        else:
            keep[idx] = True
    
    if current_rank < rank:
        keep_indices = [c for c in range(rank) if keep[c]]
        return current_U[:, keep_indices], current_V[:, keep_indices], current_W[:, keep_indices]
    return U, V, W


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms


def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 via Kronecker product of 2x2 Strassen."""
    base_terms = _strassen_2x2_terms()
    terms_4x4 = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base_terms[c1][0][I1*2+J1] * base_terms[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base_terms[c1][1][J1*2+K1] * base_terms[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base_terms[c1][2][K1*2+I1] * base_terms[c2][2][K0*2+I0]
            terms_4x4.append((U4, V4, W4))
    return terms_4x4


def _make_4x4x7_decomposition():
    """Create a rank-97 decomposition: Strassen 4x4x4 (49) + schoolbook 4x4x3 (48)."""
    terms_4x4 = _strassen_4x4_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    # First 4 columns: Strassen 4x4x4 (49 terms)
    for U4, V4, W4 in terms_4x4:
        V1 = [0] * 28
        W1 = [0] * 28
        for J in range(4):
            for K in range(4):
                V1[J*7+K] = V4[J*4+K]
        for K in range(4):
            for I in range(4):
                W1[K*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V1)
        W_list.append(W1)
    
    # Last 3 columns: schoolbook (4*4*3 = 48 terms)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_col = [0] * 16
                V_col = [0] * 28
                W_col = [0] * 28
                U_col[i*4+j] = 1
                V_col[j*7+(k+4)] = 1
                W_col[(k+4)*4+i] = 1
                U_list.append(U_col)
                V_list.append(V_col)
                W_list.append(W_col)
    
    rank = len(U_list)
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    for c in range(rank):
        U[:, c] = U_list[c]
        V[:, c] = V_list[c]
        W[:, c] = W_list[c]
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U, V, W = _make_4x4x7_decomposition()
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() < deadline - 40:
        U, V, W = _pairwise_merge(U, V, W, n, m, p, deadline)
    
    if time.time() < deadline - 20:
        U, V, W = _greedy_reduce(U, V, W, n, m, p, deadline)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _make_4x4x7_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
