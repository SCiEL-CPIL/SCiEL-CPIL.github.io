"""Rewritten program for matrix multiplication 3x4x7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns list of (U_flat, V_flat, W_flat) where:
      U_flat has length n*m = 4 (indexed by i*m+a)
      V_flat has length m*p = 4 (indexed by a*p+b)
      W_flat has length p*n = 4 (indexed by k*n+i)
    """
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    terms = []
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (C00+=1, C11+=1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,1] (C10+=1, C11+=1)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,1]))
    # M3: U=[1,0,0,0], V=[0,0,1,-1], W=[0,1,0,1] (C01+=1, C11+=1)
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    # M4: U=[0,0,0,1], V=[-1,1,0,0], W=[1,0,1,0] (C00+=1, C10+=1)
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,1] (C01+=1, C11+=1)
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,1]))
    # M6: U=[-1,0,1,0], V=[1,1,0,0], W=[1,0,1,0] (C00+=1, C10+=1)
    terms.append(([-1,0,1,0], [1,1,0,0], [1,0,1,0]))
    # M7: U=[0,1,0,-1], V=[0,0,1,1], W=[0,1,0,1] (C01+=1, C11+=1)
    terms.append(([0,1,0,-1], [0,0,1,1], [0,1,0,1]))
    return terms

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    terms = []
    for i in range(n):
        for a in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + a] = 1
                v[a * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))
    return terms

def _embed_split_m(terms1, terms2, n, m1, m2, p):
    """Embed two decompositions (for m1 and m2 columns) into full m=m1+m2."""
    all_terms = []
    for u, v, w in terms1:
        u_full = list(u) + [0] * (n * m2)
        v_full = list(v) + [0] * (m2 * p)
        w_full = list(w)
        all_terms.append((u_full, v_full, w_full))
    for u, v, w in terms2:
        u_full = [0] * (n * m1) + list(u)
        v_full = [0] * (m1 * p) + list(v)
        w_full = list(w)
        all_terms.append((u_full, v_full, w_full))
    return all_terms

def _embed_split_n(terms1, terms2, n1, n2, m, p):
    """Embed two decompositions (for n1 and n2 rows) into full n=n1+n2."""
    all_terms = []
    for u, v, w in terms1:
        u_full = list(u) + [0] * (n2 * m)
        v_full = list(v)
        # w has length p*n, first p*n1 entries for first block
        w_full = list(w[:p*n1]) + [0] * (p*n2)
        all_terms.append((u_full, v_full, w_full))
    for u, v, w in terms2:
        u_full = [0] * (n1 * m) + list(u)
        v_full = list(v)
        w_full = [0] * (p*n1) + list(w[p*n1:])
        all_terms.append((u_full, v_full, w_full))
    return all_terms

def _embed_split_p(terms1, terms2, n, m, p1, p2):
    """Embed two decompositions (for p1 and p2 columns) into full p=p1+p2."""
    all_terms = []
    for u, v, w in terms1:
        u_full = list(u)
        v_full = list(v) + [0] * (m * p2)
        w_full = list(w[:p1*n]) + [0] * (p2*n)
        all_terms.append((u_full, v_full, w_full))
    for u, v, w in terms2:
        u_full = list(u)
        v_full = [0] * (m * p1) + list(v)
        w_full = [0] * (p1*n) + list(w[p1*n:])
        all_terms.append((u_full, v_full, w_full))
    return all_terms

def _build_decomp(n, m, p):
    """Recursively build a low-rank decomposition for n x m x p matrix mult."""
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return [([1], [1], [1])]
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Try splitting m (most beneficial for our case since m=4)
    if m >= 2 and m % 2 == 0:
        m1 = m // 2
        m2 = m - m1
        terms1 = _build_decomp(n, m1, p)
        terms2 = _build_decomp(n, m2, p)
        return _embed_split_m(terms1, terms2, n, m1, m2, p)
    
    # Try splitting p
    if p >= 2 and p % 2 == 0:
        p1 = p // 2
        p2 = p - p1
        terms1 = _build_decomp(n, m, p1)
        terms2 = _build_decomp(n, m, p2)
        return _embed_split_p(terms1, terms2, n, m, p1, p2)
    
    # Try splitting n
    if n >= 2 and n % 2 == 0:
        n1 = n // 2
        n2 = n - n1
        terms1 = _build_decomp(n1, m, p)
        terms2 = _build_decomp(n2, m, p)
        return _embed_split_n(terms1, terms2, n1, n2, m, p)
    
    # Try odd splits if even doesn't work
    if m >= 3:
        m1 = m // 2
        m2 = m - m1
        terms1 = _build_decomp(n, m1, p)
        terms2 = _build_decomp(n, m2, p)
        return _embed_split_m(terms1, terms2, n, m1, m2, p)
    
    if p >= 3:
        p1 = p // 2
        p2 = p - p1
        terms1 = _build_decomp(n, m, p1)
        terms2 = _build_decomp(n, m, p2)
        return _embed_split_p(terms1, terms2, n, m, p1, p2)
    
    if n >= 3:
        n1 = n // 2
        n2 = n - n1
        terms1 = _build_decomp(n1, m, p)
        terms2 = _build_decomp(n2, m, p)
        return _embed_split_n(terms1, terms2, n1, n2, m, p)
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for a in range(m):
            for k in range(p):
                for b in range(m):
                    expected = 1.0 if a == b else 0.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i * m + a, c] * V[a * p + b, c] * W[k * n + i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build the decomposition using recursive splitting
    terms = _build_decomp(n, m, p)
    
    # Build U, V, W matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = np.array(u, dtype=float)
        V[:, c] = np.array(v, dtype=float)
        W[:, c] = np.array(w, dtype=float)
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        terms = _schoolbook(n, m, p)
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (u, v, w) in enumerate(terms):
            U[:, c] = np.array(u, dtype=float)
            V[:, c] = np.array(v, dtype=float)
            W[:, c] = np.array(w, dtype=float)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - start_time
    # Should be well within deadline since this is a deterministic construction
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
