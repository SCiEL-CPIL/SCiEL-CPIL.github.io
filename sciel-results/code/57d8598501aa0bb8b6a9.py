"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (c+d)(e)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = a(f-h)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4 = d(g-e)
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    # M5 = (a+b)(h)
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6 = (c-a)(e+f)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h)
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms

def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 matrix multiplication using 2x2 blocks."""
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

def _make_4x4x7_decomposition_from_4x4x4(terms_4x4x4):
    """Construct 4x4x7 decomposition from a list of 4x4x4 terms."""
    U_list = []
    V_list = []
    W_list = []
    
    for U4, V4, W4 in terms_4x4x4:
        # First group: columns 0-3
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
        
        # Second group: columns 4-6 (padded to 4)
        V2 = [0] * 28
        W2 = [0] * 28
        for J in range(4):
            for K in range(3):
                V2[J*7+(K+4)] = V4[J*4+K]
        for K in range(3):
            for I in range(4):
                W2[(K+4)*4+I] = W4[K*4+I]
        U_list.append(list(U4))
        V_list.append(V2)
        W_list.append(W2)
    
    rank = len(U_list)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    for c in range(rank):
        for i in range(16):
            U[i][c] = U_list[c][i]
        for i in range(28):
            V[i][c] = V_list[c][i]
        for i in range(28):
            W[i][c] = W_list[c][i]
            
    return U, V, W

def _search_4x4x4_decomposition(deadline):
    """Search for a rank < 49 decomposition of 4x4x4."""
    # We know rank 49 is possible. Let's try to find rank 48, 47...
    # Random search with small integers is very slow.
    # Instead, we can try to merge terms in the Strassen decomposition.
    # Or just return the Strassen terms if no time left.
    
    terms = _strassen_4x4_terms()
    
    # Try to reduce rank by checking for linear dependencies among the 49 terms.
    # This is complex. Let's just use the Strassen terms for now.
    # A more advanced search would involve random perturbations and re-fitting.
    
    # For now, return the standard 49-term decomposition.
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Get the 4x4x4 decomposition
    terms_4x4 = _search_4x4x4_decomposition(deadline)
    
    # Construct the 4x4x7 decomposition
    U, V, W = _make_4x4x7_decomposition_from_4x4x4(terms_4x4)
    
    # Verify
    if not _check_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
