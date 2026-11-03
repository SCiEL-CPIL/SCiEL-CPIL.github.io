"""Rewritten program: recursive block-splitting with Strassen for 2x2x2 sub-blocks.
Produces a rank-71 exact tensor decomposition for (3,5,5) matrix multiplication."""

from functools import lru_cache

def construct():
    """Return {'U','V','W'} with shapes (15,rank),(25,rank),(15,rank) for (3,5,5)."""
    
    # Schoolbook terms for (n,m,p)
    def schoolbook_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_dict = {(i,j): 1}
                    V_dict = {(j,k): 1}
                    W_dict = {(k,i): 1}
                    terms.append((U_dict, V_dict, W_dict))
        return terms
    
    # Strassen terms for (2,2,2)
    def strassen_terms():
        terms = []
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
        return terms
    
    @lru_cache(maxsize=None)
    def min_rank(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return 7
        if n <= 1 or m <= 1 or p <= 1:
            return n * m * p
        
        best = float('inf')
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            r = min_rank(n, m1, p) + min_rank(n, m2, p)
            if r < best:
                best = r
        
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            r = min_rank(n, m, p1) + min_rank(n, m, p2)
            if r < best:
                best = r
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            r = min_rank(n1, m, p) + min_rank(n2, m, p)
            if r < best:
                best = r
        
        return best
    
    def decompose_general(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_terms()
        if n <= 1 or m <= 1 or p <= 1:
            return schoolbook_terms(n, m, p)
        
        best_terms = None
        best_len = min_rank(n, m, p)
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            r1 = min_rank(n, m1, p)
            r2 = min_rank(n, m2, p)
            if r1 + r2 == best_len:
                terms1 = decompose_general(n, m1, p)
                terms2 = decompose_general(n, m2, p)
                shifted_terms2 = []
                for U_d, V_d, W_d in terms2:
                    new_U_d = {(i, j + m1): val for (i, j), val in U_d.items()}
                    new_V_d = {(j + m1, k): val for (j, k), val in V_d.items()}
                    shifted_terms2.append((new_U_d, new_V_d, W_d))
                best_terms = terms1 + shifted_terms2
                break
        
        if best_terms is not None:
            return best_terms
        
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            r1 = min_rank(n, m, p1)
            r2 = min_rank(n, m, p2)
            if r1 + r2 == best_len:
                terms1 = decompose_general(n, m, p1)
                terms2 = decompose_general(n, m, p2)
                shifted_terms2 = []
                for U_d, V_d, W_d in terms2:
                    new_V_d = {(j, k + p1): val for (j, k), val in V_d.items()}
                    new_W_d = {(k + p1, i): val for (k, i), val in W_d.items()}
                    shifted_terms2.append((U_d, new_V_d, new_W_d))
                best_terms = terms1 + shifted_terms2
                break
        
        if best_terms is not None:
            return best_terms
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            r1 = min_rank(n1, m, p)
            r2 = min_rank(n2, m, p)
            if r1 + r2 == best_len:
                terms1 = decompose_general(n1, m, p)
                terms2 = decompose_general(n2, m, p)
                shifted_terms2 = []
                for U_d, V_d, W_d in terms2:
                    new_U_d = {(i + n1, j): val for (i, j), val in U_d.items()}
                    new_W_d = {(k, i + n1): val for (k, i), val in W_d.items()}
                    shifted_terms2.append((new_U_d, V_d, new_W_d))
                best_terms = terms1 + shifted_terms2
                break
        
        if best_terms is None:
            return schoolbook_terms(n, m, p)
        return best_terms
    
    all_terms = decompose_general(3, 5, 5)
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(15)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(15)]
    
    n, m, p = 3, 5, 5
    for c, (U_d, V_d, W_d) in enumerate(all_terms):
        for (i, j), val in U_d.items():
            U[i * m + j][c] = val
        for (j, k), val in V_d.items():
            V[j * p + k][c] = val
        for (k, i), val in W_d.items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
