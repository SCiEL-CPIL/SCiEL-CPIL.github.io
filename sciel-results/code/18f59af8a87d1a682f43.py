import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Check the "on" positions: for each (i,j,k), 
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    # Check a random sample of "off" positions
    rng = np.random.RandomState(42)
    for _ in range(min(2000, nm * mp * pn // 5)):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        # Check if this is an "on" position
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False, (a, b, c, val)
    
    return True, None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_2x2_block(n, m, p, i0, j0, k0):
    """
    Embed a 2x2*2x2 Strassen decomposition into a larger matrix multiplication.
    This handles the sub-problem: A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    into C[i0:i0+2, k0:k0+2]
    """
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Strassen's 7 products for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For our flattened indices:
    # A[i,j] is at index i*m + j
    # B[j,k] is at index j*p + k
    # C[i,k] is at index k*n + i
    
    # Define the 7 Strassen terms
    # Each term is (U_coeff, V_coeff, W_coeff) where these are dictionaries
    # mapping (index_in_U, value), etc.
    
    # M1: (A00+A11)*(B00+B11)
    # U: {i0*m+j0: 1, (i0+1)*m+(j0+1): 1}
    # V: {j0*p+k0: 1, (j0+1)*p+(k0+1): 1}
    # W: {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
    U_terms.append({i0*m+j0: 1, (i0+1)*m+(j0+1): 1})
    V_terms.append({j0*p+k0: 1, (j0+1)*p+(k0+1): 1})
    W_terms.append({k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
    
    # M2: (A10+A11)*B00
    # U: {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1}
    # V: {j0*p+k0: 1}
    # W: {k0*n+(i0+1): 1}
    U_terms.append({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1})
    V_terms.append({j0*p+k0: 1})
    W_terms.append({k0*n+(i0+1): 1})
    
    # M3: A00*(B01-B11)
    # U: {i0*m+j0: 1}
    # V: {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1}
    # W: {k0*n+i0: 1}
    U_terms.append({i0*m+j0: 1})
    V_terms.append({j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1})
    W_terms.append({k0*n+i0: 1})
    
    # M4: A11*(B10-B00)
    # U: {(i0+1)*m+(j0+1): 1}
    # V: {(j0+1)*p+k0: 1, j0*p+k0: -1}
    # W: {k0*n+(i0+1): 1}
    U_terms.append({(i0+1)*m+(j0+1): 1})
    V_terms.append({(j0+1)*p+k0: 1, j0*p+k0: -1})
    W_terms.append({k0*n+(i0+1): 1})
    
    # M5: (A00+A10)*B11
    # U: {i0*m+j0: 1, (i0+1)*m+j0: 1}
    # V: {(j0+1)*p+(k0+1): 1}
    # W: {(k0+1)*n+i0: 1}
    U_terms.append({i0*m+j0: 1, (i0+1)*m+j0: 1})
    V_terms.append({(j0+1)*p+(k0+1): 1})
    W_terms.append({(k0+1)*n+i0: 1})
    
    # M6: (A11-A00)*(B00+B01)
    # U: {(i0+1)*m+(j0+1): 1, i0*m+j0: -1}
    # V: {j0*p+k0: 1, j0*p+(k0+1): 1}
    # W: {(k0+1)*n+(i0+1): 1, k0*n+i0: 1}
    U_terms.append({(i0+1)*m+(j0+1): 1, i0*m+j0: -1})
    V_terms.append({j0*p+k0: 1, j0*p+(k0+1): 1})
    W_terms.append({(k0+1)*n+(i0+1): 1, k0*n+i0: 1})
    
    # M7: (A01-A11)*(B10+B11)
    # U: {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1}
    # V: {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1}
    # W: {k0*n+i0: 1, (k0+1)*n+i0: 1}
    U_terms.append({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1})
    V_terms.append({(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1})
    W_terms.append({k0*n+i0: 1, (k0+1)*n+i0: 1})
    
    return U_terms, V_terms, W_terms

def _combine_terms(U_terms, V_terms, W_terms, n, m, p):
    """Combine a list of term dictionaries into full matrices."""
    # Determine the maximum row indices
    max_u = max(max(U_terms[r].keys()) for r in range(len(U_terms))) if U_terms else 0
    max_v = max(max(V_terms[r].keys()) for r in range(len(V_terms))) if V_terms else 0
    max_w = max(max(W_terms[r].keys()) for r in range(len(W_terms))) if W_terms else 0
    
    rank = len(U_terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for r in range(rank):
        for idx, val in U_terms[r].items():
            U[idx, r] = val
        for idx, val in V_terms[r].items():
            V[idx, r] = val
        for idx, val in W_terms[r].items():
            W[idx, r] = val
    
    return U, V, W

def _construct_decomposition(n, m, p):
    """
    Construct a decomposition for n x m * m x p.
    
    Strategy: Use Strassen for 2x2 blocks where possible, and schoolbook for the rest.
    """
    U_terms = []
    V_terms = []
    W_terms = []
    
    # For 3x3 * 3x5:
    # We can decompose this as:
    # - A 2x2 * 2x2 block using Strassen (rank 7)
    # - The remaining terms using schoolbook
    #
    # Let's break it down more carefully.
    #
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # We can group terms by (i,j) pairs and use Strassen for 2x2 blocks.
    #
    # Actually, let's use a simpler approach:
    # Use Strassen for the (i,j) in {0,1} x {0,1} block, covering k in {0,1}.
    # Then handle the remaining terms with schoolbook.
    
    # Use Strassen for the top-left 2x2 * 2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    sU, sV, sW = _strassen_2x2_block(n, m, p, 0, 0, 0)
    U_terms.extend(sU)
    V_terms.extend(sV)
    W_terms.extend(sW)
    
    # Now handle the remaining terms with schoolbook.
    # The Strassen block covered: i in {0,1}, j in {0,1}, k in {0,1}
    # We need to cover all (i,j,k) triples.
    
    # Let's be more systematic. We'll use Strassen for all possible 2x2*2x2 sub-blocks
    # and then fill in the gaps with schoolbook terms, avoiding double-counting.
    
    # Actually, this is getting complicated. Let's use a cleaner approach:
    # Decompose the problem into 2x2 blocks where possible, and use schoolbook for the rest.
    
    # For 3x3 * 3x5:
    # We can write this as a sum of 2x2*2x2 blocks plus remainder.
    #
    # Let's just use schoolbook for everything except we try to optimize using Strassen
    # for specific sub-blocks.
    
    # Simpler approach: Use Strassen for the first 2x2*2x2 sub-block, and schoolbook
    # for all other (i,j,k) triples.
    
    # The Strassen block covered (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}.
    # We need to add schoolbook terms for all other (i,j,k).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this was already covered by Strassen
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                # Add schoolbook term
                U_terms.append({i * m + j: 1})
                V_terms.append({j * p + k: 1})
                W_terms.append({k * n + i: 1})
    
    U, V, W = _combine_terms(U_terms, V_terms, W_terms, n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try the Strassen-based decomposition
    U_try, V_try, W_try = _construct_decomposition(n, m, p)
    valid, err = _validate(U_try, V_try, W_try, n, m, p, tol=1e-6)
    if valid and U_try.shape[1] < best_rank:
        best_rank = U_try.shape[1]
        best_result = (U_try, V_try, W_try)
    
    # Return the best result found
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
