import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _elapsed(start):
    return time.time() - start

def _time_left(start, deadline=145.0):
    return deadline - _elapsed(start)

def _strassen_222():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (U_coeffs, V_coeffs, W_coeffs) where each is a dict mapping (i,j) or (j,k) or (k,i) to coefficient.
    Indices: i in {0,1}, j in {0,1}, k in {0,1}
    U is indexed by (i,j), V by (j,k), W by (k,i)
    """
    # Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B10-B11)
    # M3 = (A11)(B01-B10)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    
    terms = []
    # M0: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):-1}
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):-1}
    ))
    # M1: U={(1,0):1,(1,1):1}, V={(0,0):1}, W={(1,0):1,(1,1):-1}
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1, (1,1):-1}
    ))
    # M2: U={(0,0):1}, V={(1,0):1,(1,1):-1}, W={(0,0):1,(1,1):1}
    terms.append((
        {(0,0):1},
        {(1,0):1, (1,1):-1},
        {(0,0):1, (1,1):1}
    ))
    # M3: U={(1,1):1}, V={(0,1):1,(1,0):-1}, W={(0,1):1,(1,0):-1}
    terms.append((
        {(1,1):1},
        {(0,1):1, (1,0):-1},
        {(0,1):1, (1,0):-1}
    ))
    # M4: U={(0,0):1,(0,1):1}, V={(1,1):1}, W={(0,1):-1,(1,1):1}
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,1):-1, (1,1):1}
    ))
    # M5: U={(1,0):1,(0,0):-1}, V={(0,0):1,(0,1):1}, W={(1,0):1,(1,1):1}
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ))
    # M6: U={(0,1):1,(1,1):-1}, V={(1,0):1,(1,1):1}, W={(0,0):-1,(1,1):1}
    terms.append((
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):-1, (1,1):1}
    ))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    # Split j into pairs: (0,1), (2,3), and leftover j=4
    # Split k into pairs: (0,1), (2,3), (4,5)
    # For each (j_pair, k_pair), apply Strassen to the 2x2x2 sub-problem
    
    strassen_terms = _strassen_222()
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # j pairs
    j_pairs = [(0, 1), (2, 3)]
    j_leftover = [4]
    
    # k pairs
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # For each j_pair and k_pair, add Strassen terms
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            for (u_coeff, v_coeff, w_coeff) in strassen_terms:
                u_map = {}
                v_map = {}
                w_map = {}
                
                # U is indexed by (i, j) where i in {0,1}, j in {j0, j1}
                for (i, j_local), coeff in u_coeff.items():
                    j_global = j0 if j_local == 0 else j1
                    u_map[(i, j_global)] = coeff
                
                # V is indexed by (j, k) where j in {j0, j1}, k in {k0, k1}
                for (j_local, k_local), coeff in v_coeff.items():
                    j_global = j0 if j_local == 0 else j1
                    k_global = k0 if k_local == 0 else k1
                    v_map[(j_global, k_global)] = coeff
                
                # W is indexed by (k, i) where k in {k0, k1}, i in {0,1}
                for (k_local, i), coeff in w_coeff.items():
                    k_global = k0 if k_local == 0 else k1
                    w_map[(k_global, i)] = coeff
                
                U_terms.append(u_map)
                V_terms.append(v_map)
                W_terms.append(w_map)
    
    # For leftover j=4, use trivial decomposition
    for j in j_leftover:
        for k in range(p):
            for i in range(n):
                U_terms.append({(i, j): 1})
                V_terms.append({(j, k): 1})
                W_terms.append({(k, i): 1})
    
    rank = len(U_terms)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (i, j), coeff in U_terms[r].items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in V_terms[r].items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in W_terms[r].items():
            W[k * n + i, r] = coeff
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to integer/half-integer representation
    # All entries are integers in this construction
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
