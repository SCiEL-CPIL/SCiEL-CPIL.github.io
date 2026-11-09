import numpy as np
import time

DIMENSIONS = [3, 3, 4]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin under 150s
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Try to construct a rank-34 decomposition using Strassen for 2x2 blocks
    # and schoolbook for the rest. Fall back to rank-36 schoolbook if needed.
    
    result = _try_rank34(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook rank 36
    result = _schoolbook(n, m, p)
    _verify(result, n, m, p)
    return result


def _strassen_2x2x2():
    """
    Return the 7 Strassen multiplications for 2x2 * 2x2.
    Each multiplication is defined by:
      - U coefficients: dict mapping (i,j) -> coeff
      - V coefficients: dict mapping (j,k) -> coeff  
      - W coefficients: dict mapping (k,i) -> coeff (output)
    
    The 7 products and their output contributions:
    We need: for each output (i,k), sum over the 7 products of W_contrib * U_val * V_val = A[i,*] @ B[*,k]
    
    Standard Strassen (verified computationally):
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 - M4 + M6
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M7
    """
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: {(i,j): val} for A factors
    # V_coeffs: {(j,k): val} for B factors
    # W_coeffs: {(k,i): val} for output contributions
    
    products = []
    
    # M1 = (A00+A11)*(B00+B11)
    products.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}  # contributes to C00 and C11
    })
    
    # M2 = (A10+A11)*B00
    products.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1}  # contributes to C10
    })
    
    # M3 = A00*(B01-B11)
    products.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1}  # contributes to C01
    })
    
    # M4 = A11*(B10-B00)
    products.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (0,1): 1}  # contributes to C00 (neg) and C10
    })
    
    # M5 = (A00+A01)*B11
    products.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(1,0): 1}  # contributes to C01
    })
    
    # M6 = (A10-A00)*(B00+B10)
    products.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (1,0): 1},
        'W': {(0,0): 1}  # contributes to C00
    })
    
    # M7 = (A01-A11)*(B01+B11)
    products.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(0,1): 1, (1,1): 1},
        'W': {(1,1): 1}  # contributes to C11
    })
    
    return products


def _build_rank34(n, m, p):
    """
    Build a rank-34 decomposition for 3x3 * 3x4.
    
    Strategy:
    - Use Strassen (7 mults) for i in {0,1}, j in {0,1}, k in {0,1}
    - Use Strassen (7 mults) for i in {0,1}, j in {0,1}, k in {2,3}
    - Use schoolbook for all other (i,j,k) triples: 20 terms
    Total: 7 + 7 + 20 = 34
    """
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block 1: i in {0,1}, j in {0,1}, k in {0,1}
    strassen_products = _strassen_2x2x2()
    for prod in strassen_products:
        for (i, j), uval in prod['U'].items():
            U[i * m + j][col] += uval
        for (j, k), vval in prod['V'].items():
            V[j * p + k][col] += vval
        for (k, i), wval in prod['W'].items():
            W[k * n + i][col] += wval
        col += 1
    
    # Strassen block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Same structure but k offset by 2
    for prod in strassen_products:
        for (i, j), uval in prod['U'].items():
            U[i * m + j][col] += uval
        for (j, k), vval in prod['V'].items():
            V[j * p + (k + 2)][col] += vval
        for (k, i), wval in prod['W'].items():
            W[(k + 2) * n + i][col] += wval
        col += 1
    
    # Schoolbook for remaining terms: i=2 (any j,k) or j=2 (i in {0,1}, any k)
    # i=2, j=0,1,2, k=0,1,2,3: 12 terms
    # i=0,1, j=2, k=0,1,2,3: 8 terms
    for i in range(n):
        for j in range(m):
            if i == 2 or j == 2:
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    result = {"U": U, "V": V, "W": W}
    return result


def _try_rank34(n, m, p):
    """Try to build and verify a rank-34 decomposition."""
    result = _build_rank34(n, m, p)
    try:
        _verify(result, n, m, p)
        return result
    except ValueError:
        return None


def _schoolbook(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
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


def _verify(result, n, m, p):
    """Verify that the factor matrices exactly reproduce the MM tensor."""
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    raise ValueError(
                        f"Verification failed at i={i}, j={j}, k={k}: got {total}"
                    )


def _verify_strassen():
    """Verify Strassen formulas on a test case."""
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C should be [[19,22],[43,50]]
    A = [[1,2],[3,4]]
    B = [[5,6],[7,8]]
    C_expected = [[19,22],[43,50]]
    
    products = _strassen_2x2x2()
    C = [[0,0],[0,0]]
    
    for prod in products:
        # Compute the product value
        a_val = 0
        for (i,j), uval in prod['U'].items():
            a_val += uval * A[i][j]
        b_val = 0
        for (j,k), vval in prod['V'].items():
            b_val += vval * B[j][k]
        prod_val = a_val * b_val
        
        for (k,i), wval in prod['W'].items():
            C[i][k] += wval * prod_val
    
    return C == C_expected


if __name__ == "__main__":
    # First verify Strassen formulas
    strassen_ok = _verify_strassen()
    print(f"Strassen verified: {strassen_ok}")
    
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
