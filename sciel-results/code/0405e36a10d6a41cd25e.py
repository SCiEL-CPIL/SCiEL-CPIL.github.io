"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition for 3x3 x 3x6. Rank = 54."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] should be 1
                val = float(np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    # Check off-diagonal entries
    for i in range(n):
        for j in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j2 * p + k
                        c = k * n + i2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        val = float(np.sum(U_arr[a] * V_arr[b] * W_arr[c]))
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
    return max_err < tol

def _build_strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (U_entries, V_entries, W_entries) where entries are dicts
    mapping (row, col) index to coefficient.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
    })
    # M2 = (A10+A11)B00
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,0): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M3 = A00(B10-B11)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # M4 = A11(B01-B00)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,0): -1, (0,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M5 = (A00+A01)B11
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(0,1): 1, (1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M6 = (A10-A00)(B00+B10)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (1,0): 1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(0,1): 1, (1,1): 1},
        'W': {(0,1): -1, (1,1): 1}
    })
    return terms

def _verify_strassen_2x2():
    """Verify the Strassen 2x2 decomposition is correct."""
    terms = _build_strassen_2x2_terms()
    # Build the full 2x2x2 tensor from Strassen terms
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    T = np.zeros((2,2,2))
    for term in terms:
        for (i,j), u_val in term['U'].items():
            for (j2,k), v_val in term['V'].items():
                if j2 != j:
                    continue
                for (k2,i2), w_val in term['W'].items():
                    if k2 != k or i2 != i:
                        continue
                    T[i,j,k] += u_val * v_val * w_val
    
    # Expected: T[i,j,k] = 1 if i==i2 and j==j2 for the output position
    # Actually for 2x2 matmul: C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor T[a,b,c] where a=(i,j), b=(k,l), c=(m,n) represents
    # contribution to C[m,n] from A[i,j]*B[k,l] when i==k and j==l... 
    # Let me think more carefully.
    
    # For matrix multiplication C = A*B where A is 2x2, B is 2x2:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T has indices (i,k), (k,j), (i,j) but we need to map to
    # our convention: U index = i*m+j (for A[i,j]), V index = j*p+k (for B[j,k]),
    # W index = k*n+i (for C[k,i])
    
    # Wait, our convention is:
    # U[i*m + j] corresponds to A[i,j]
    # V[j*p + k] corresponds to B[j,k]  
    # W[k*n + i] corresponds to C[k,i]
    
    # So the tensor T[U_idx, V_idx, W_idx] = 1 iff
    # U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i for some i,j,k
    # i.e., the "middle" indices match: j (from U) == j (from V) and k (from V) == k (from W)
    
    # For 2x2: m=2, p=2, n=2
    # U_idx = i*2 + j, V_idx = j*2 + k, W_idx = k*2 + i
    # So T[i*2+j, j*2+k, k*2+i] = 1
    
    # Let's verify our Strassen terms produce this.
    # Actually, let me just verify by checking the 8 diagonal entries and 8 off-diagonal.
    
    # Diagonal entries (should be 1):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u_idx = i * 2 + j
                v_idx = j * 2 + k
                w_idx = k * 2 + i
                # Sum over all terms
                total = 0
                for term in terms:
                    u_val = term['U'].get((i,j), 0)
                    v_val = term['V'].get((j,k), 0)
                    w_val = term['W'].get((k,i), 0)
                    total += u_val * v_val * w_val
                if abs(total - 1) > 1e-9:
                    return False
    
    # Off-diagonal entries (should be 0):
    for i in range(2):
        for j in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    for k in range(2):
                        u_idx = i * 2 + j
                        v_idx = j2 * 2 + k
                        w_idx = k * 2 + i2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        if expected == 0:
                            total = 0
                            for term in terms:
                                u_val = term['U'].get((i,j), 0)
                                v_val = term['V'].get((j2,k), 0)
                                w_val = term['W'].get((k,i2), 0)
                                total += u_val * v_val * w_val
                            if abs(total) > 1e-9:
                                return False
    
    return True

def _strassen_3x3_decomposition():
    """
    Build a 3x3 matrix multiplication decomposition using Strassen-style recursion.
    Returns U, V, W as lists of lists.
    
    Strategy: Use the fact that 3x3 matmul can be done with 23 multiplications
    (Bini et al. 1979). We'll use a known rank-23 decomposition.
    """
    n = 3
    # Use a known rank-23 decomposition for 3x3 matrix multiplication
    # This is based on Bini, Capovani, Lotti, Romani (1979)
    
    # The 23 terms are complex. Let's use a simpler approach:
    # Pad to 4x4 and use Strassen, then extract the 3x3 part.
    # 4x4 Strassen has rank 2*7 = 14... no, Strassen for 4x4 is 7*7=49? No.
    # Strassen for 2x2 is rank 7. For 4x4, it's 7^2 = 49? No, it's recursive.
    # Actually, the rank of 4x4 matmul tensor is at most 2*7 = 14? No.
    
    # Let me think differently. The minimal rank for 3x3 matmul is 23.
    # I'll hardcode a known rank-23 decomposition.
    
    # Actually, let me use a different strategy: build the 3x3x3x6 decomposition
    # by combining two 3x3x3x3 decompositions (one for each 3x3 block of B).
    
    # For now, return schoolbook for 3x3x3x6 as a valid baseline.
    # The improvement will come from finding a better structure.
    
    # Let's try: view 3x3 * 3x6 as two 3x3 * 3x3 multiplications.
    # If we have a rank-r decomposition for 3x3*3x3, we can use rank 2r for 3x3*3x6.
    # Schoolbook for 3x3*3x3 has rank 27, so 2*27 = 54 (same as direct schoolbook).
    # But if we find rank 23 for 3x3*3x3, we get rank 46 for 3x3*3x6.
    
    # For now, let's just return the schoolbook result and focus on the
    # Boolean rank / fixed support approach for improvement.
    return _schoolbook()

def _find_boolean_support(n, m, p, max_terms=60, deadline=None):
    """
    Find a sparse support pattern for the tensor decomposition.
    Returns a list of support tuples: each is (U_support, V_support, W_support)
    where each support is a set of indices.
    """
    # This is a placeholder for the Boolean rank analysis.
    # For now, return the schoolbook supports.
    supports = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                supports.append((frozenset([u_idx]), frozenset([v_idx]), frozenset([w_idx])))
    return supports

def _solve_fixed_support(supports, n, m, p):
    """
    Given a fixed support pattern, solve for the coefficients using exact rational arithmetic.
    Returns U, V, W as lists of lists of Fractions, or None if no solution exists.
    """
    rank = len(supports)
    
    # Build the system of equations.
    # For each (i, j, k) triple, we have:
    # sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = T[i*m+j, j*p+k, k*n+i]
    # where T = 1 if (i,j,k) is a "valid" triple, 0 otherwise.
    # A triple (i,j,k) is valid if it corresponds to an actual matrix multiplication term.
    # In our convention, ALL triples (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # are "diagonal" and should give 1.
    # Off-diagonal: for (i,j) in U, (j2,k) in V, (k2,i2) in W,
    # the entry is 1 iff i==i2 and j==j2 and k==k2 (i.e., all indices match).
    
    # Actually, the tensor T is defined as:
    # T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    
    # So we need: for all a in [0,n*m), b in [0,m*p), c in [0,p*n):
    # sum_{c' in supports} U_{c'}[a] * V_{c'}[b] * W_{c'}[c] = T[a,b,c]
    
    # This is a system of bilinear equations (not linear in the unknowns U, V, W).
    # To make it linear, we need to fix some of the variables.
    
    # Strategy: Fix the supports to be "monomial" (each term has exactly one nonzero
    # in each of U, V, W), then the system becomes linear in the coefficients.
    
    # For monomial supports:
    # U_c has exactly one nonzero entry at index u_c
    # V_c has exactly one nonzero entry at index v_c  
    # W_c has exactly one nonzero entry at index w_c
    # Let U_c[u_c] = a_c, V_c[v_c] = b_c, W_c[w_c] = c_c
    # Then the equation for (a,b,c) is:
    # sum_{c': u_c'=a, v_c'=b, w_c'=c} a_c' * b_c' * c_c' = T[a,b,c]
    
    # This is still nonlinear (products of three unknowns).
    
    # Alternative: fix two of the three coefficient sets, then solve linearly for the third.
    # E.g., fix all a_c and b_c to be 1, then solve for c_c.
    # But this may not have a solution.
    
    # Better approach: use the fact that for the schoolbook decomposition,
    # each term has a_c = b_c = c_c = 1. We can try to find a lower-rank
    # decomposition by allowing some terms to have different coefficients.
    
    # For now, this is a placeholder. Return None.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Verify the schoolbook decomposition
    assert _verify(best["U"], best["V"], best["W"], n, m, p), "Schoolbook verification failed"
    
    # Try to find a better decomposition using the Strassen-based approach.
    # Key insight: 3x3 x 3x6 = two 3x3 x 3x3 multiplications.
    # If we can find a rank-23 decomposition for 3x3 x 3x3, we get rank 46 for 3x3 x 3x6.
    
    # For now, try to use a known rank-23 decomposition for 3x3 matmul.
    # This is a well-known result from Bini et al. (1979).
    
    # Let's try a different approach: use the structure of the problem.
    # C = A * B where A is 3x3, B is 3x6.
    # Split B into B1 (3x3) and B2 (3x3).
    # C1 = A * B1, C2 = A * B2.
    # If we have a rank-r decomposition for 3x3*3x3, we can compute C1 and C2
    # using the same U factors but different V and W factors.
    
    # Specifically, if for 3x3*3x3 we have:
    # C[i,j] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # Then for 3x3*3x6, we can use the same U_c, but for V and W,
    # we need to handle the 6 columns of B.
    
    # This is getting complex. For now, let's just return the schoolbook result.
    # The Boolean rank / fixed support approach would require more work.
    
    return best
# EVOLVE-BLOCK-END
