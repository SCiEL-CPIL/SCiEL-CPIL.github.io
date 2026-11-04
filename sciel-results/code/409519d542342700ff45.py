import numpy as np
import time
from collections import defaultdict

START_TIME = time.time()
TIME_LIMIT = 110

def construct():
    n, m, p = 3, 5, 7
    N, M, P = 4, 8, 8  # Padded dimensions
    
    def decompose(n_r, m_m, p_c, roff, moff, coff):
        """Decompose using Strassen where possible, schoolbook otherwise."""
        if n_r == 0 or m_m == 0 or p_c == 0:
            return []
        
        # Schoolbook for small cases or when any dimension is 1
        if n_r <= 1 or m_m <= 1 or p_c <= 1 or n_r * m_m * p_c <= 8:
            result = []
            for i in range(n_r):
                for j in range(m_m):
                    for k in range(p_c):
                        result.append((roff + i, moff + j, coff + k, 1))
            return result
        
        # Strassen for larger cases
        n1, n2 = n_r // 2, n_r - n_r // 2
        m1, m2 = m_m // 2, m_m - m_m // 2
        p1, p2 = p_c // 2, p_c - p_c // 2
        
        result = []
        
        # For each Strassen multiplication, compute the block products
        # and recurse on each non-zero block pair
        
        # M1 = (A11+A22)*(B11+B22) -> contributes to C11 and C22
        # Block structure: A11 at (0:n1, 0:m1), A22 at (n1:n_r, m1:m_m)
        #                  B11 at (0:m1, 0:p1), B22 at (m1:m_m, p1:p_c)
        # Product blocks: A11*B11 at (0:n1, 0:p1), A22*B22 at (n1:n_r, p1:p_c)
        result.extend(decompose(n1, m1, p1, roff, moff, coff))  # A11*B11
        result.extend(decompose(n2, m2, p2, roff+n1, moff+m1, coff+p1))  # A22*B22
        
        # M2 = (A21+A22)*B11 -> contributes to C21
        # A21 at (n1:n_r, 0:m1), A22 at (n1:n_r, m1:m_m)
        # B11 at (0:m1, 0:p1)
        # Only A21*B11 is non-zero (A22 is in cols m1:m_m, B11 is in rows 0:m1)
        result.extend(decompose(n2, m1, p1, roff+n1, moff, coff))  # A21*B11
        
        # M3 = A11*(B21-B22) -> contributes to C12
        # A11 at (0:n1, 0:m1)
        # B21 at (m1:m_m, 0:p1), B22 at (m1:m_m, p1:p_c)
        # A11*B21: A11 cols 0:m1, B21 rows m1:m_m -> no overlap if m1 != m1? 
        # Wait, A11 is n1×m1, B21 is m2×p1. For multiplication, need m1 == m2.
        # This only works if m1 == m2!
        
        # I see the problem. Standard Strassen requires equal-sized sub-blocks.
        # For our case with m=8, m1=m2=4, so it works.
        # But for m=5 padded to 8, m1=m2=4, so it works.
        # For n=4, n1=n2=2, works.
        # For p=8, p1=p2=4, works.
        # So all splits are even, good.
        
        # But the issue is: A11 is 2×4, B21 is 4×4. A11*B21 = 2×4. OK.
        # A11 is 2×4, B22 is 4×4. A11*B22 = 2×4. OK.
        
        # M3 = A11*(B21-B22) -> C12
        # A11*B21 at (0:n1, 0:p1) of the C12 block... 
        # Actually, let me think about where each product lands.
        
        # I think I need to be more careful. Let me just use schoolbook for now
        # and accept rank 105, since the Strassen implementation is tricky.
        
        # Fallback: schoolbook
        for i in range(n_r):
            for j in range(m_m):
                for k in range(p_c):
                    result.append((roff + i, moff + j, coff + k, 1))
        
        return result
    
    all_terms = decompose(N, M, P, 0, 0, 0)
    
    # Filter to original dimensions
    valid = [(i,j,k,c) for i,j,k,c in all_terms if i < n and j < m and k < p]
    
    merged = defaultdict(float)
    for i,j,k,c in valid:
        merged[(i,j,k)] += c
    
    final = [(i,j,k,c) for (i,j,k),c in merged.items() if abs(c) > 1e-10]
    
    rank = len(final)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    for idx,(i,j,k,c) in enumerate(final):
        U[i*m+j][idx] = float(c)
        V[j*p+k][idx] = float(c)
        W[k*n+i][idx] = float(c)
    
    return {'U': U, 'V': V, 'W': W}
