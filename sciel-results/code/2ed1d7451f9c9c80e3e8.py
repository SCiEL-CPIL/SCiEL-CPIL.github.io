import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for 4 blocks of 2x2x2
    # A1 is 2x2 (j=0,1), B1 is 2x8 (j=0,1, k=0..7)
    for g in range(4):
        # k indices for this block
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # V indices for B sub-block
        # j=0, k=k0 -> 0*8 + k0 = k0
        # j=0, k=k1 -> 0*8 + k1 = k1
        # j=1, k=k0 -> 1*8 + k0 = 8 + k0
        # j=1, k=k1 -> 1*8 + k1 = 8 + k1
        v_b00 = k0
        v_b01 = k1
        v_b10 = 8 + k0
        v_b11 = 8 + k1
        
        # U indices for A sub-block
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        # W indices for C sub-block
        # k=k0, i=0 -> k0*2 + 0
        # k=k0, i=1 -> k0*2 + 1
        # k=k1, i=0 -> k1*2 + 0
        # k=k1, i=1 -> k1*2 + 1
        w_c00 = k0 * 2 + 0
        w_c01 = k1 * 2 + 0
        w_c10 = k0 * 2 + 1
        w_c11 = k1 * 2 + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u_a00][col] = 1
        U[u_a11][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u_a10][col] = 1
        U[u_a11][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u_a00][col] = 1
        V[v_b01][col] = 1
        V[v_b11][col] = -1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u_a11][col] = 1
        V[v_b10][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u_a00][col] = 1
        U[u_a01][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u_a10][col] = 1
        U[u_a00][col] = -1
        V[v_b00][col] = 1
        V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u_a01][col] = 1
        U[u_a11][col] = -1
        V[v_b10][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
        
    # Part 2: Schoolbook for 8 blocks of 2x1x1
    # A2 is 2x1 (j=2), B2 is 1x8 (j=2, k=0..7)
    for k in range(8):
        # V index for b2k
        v_b2k = 2 * 8 + k # j=2, p=8
        
        # U indices for a02, a12
        u_a02 = 0 * 3 + 2 # i=0, j=2
        u_a12 = 1 * 3 + 2 # i=1, j=2
        
        # W indices for c0k, c1k
        w_c0k = k * 2 + 0
        w_c1k = k * 2 + 1
        
        # M_k0 = a02 * b2k
        U[u_a02][col] = 1
        V[v_b2k][col] = 1
        W[w_c0k][col] = 1
        col += 1
        
        # M_k1 = a12 * b2k
        U[u_a12][col] = 1
        V[v_b2k][col] = 1
        W[w_c1k][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Verification
    import numpy as np
    
    res = construct()
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Check shapes
    assert U.shape == (n*m, rank), f"U shape {U.shape}"
    assert V.shape == (m*p, rank), f"V shape {V.shape}"
    assert W.shape == (p*n, rank), f"W shape {W.shape}"
    
    # Construct tensor T such that T[i, j, k] = 1 if i*m+j in U, j*p+k in V, k*n+i in W
    # Actually, the decomposition is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # Our decomposition:
    # C_{i,k} = sum_{c=0}^{rank-1} U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ??
    # No, the standard form is:
    # C_{i,k} = sum_{c} (sum_{j} U_{i,j,c} V_{j,k,c}) W_{i,k,c} ?
    # The task says:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # "exactly decompose the matrix multiplication tensor"
    # Usually this means:
    # sum_{j} A_{i,j} B_{j,k} = sum_{c} U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ?
    # Let's check the schoolbook example in the prompt.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Then sum_c U V W = sum_c 1*1*1 = 1 for the specific i,j,k.
    # And sum_j A_{i,j} B_{j,k} picks up the term where j matches.
    # So the formula is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{c} ( sum_{j} U_{i*m+j, c} A_{i,j} ) ( sum_{j} V_{j*p+k, c} B_{j,k} ) W_{k*n+i, c} ?
    # No, the bilinear form is:
    # C_{i,k} = sum_{c} W_{k*n+i, c} ( sum_{j} U_{i*m+j, c} A_{i,j} ) ( sum_{j} V_{j*p+k, c} B_{j,k} )
    # Let's verify this interpretation.
    # In schoolbook:
    # For a specific c corresponding to (i0, j0, k0):
    # U is non-zero only at i0*m+j0. So sum_j U A = A_{i0, j0}.
    # V is non-zero only at j0*p+k0. So sum_j V B = B_{j0, k0}.
    # W is non-zero only at k0*n+i0. So W = 1.
    # Product: A_{i0, j0} B_{j0, k0}.
    # This term is added to C_{i0, k0}.
    # So yes, the formula is:
    # C_{i,k} = sum_{c} W_{k*n+i, c} ( U^T A )_{i,c} ( V^T B )_{k,c}
    # where (U^T A)_{i,c} = sum_j U_{i*m+j, c} A_{i,j}
    # and (V^T B)_{k,c} = sum_j V_{j*p+k, c} B_{j,k}
    
    # Let's test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_ref = A @ B
    
    C_calc = np.zeros((n, p))
    for c in range(rank):
        # Compute intermediate vectors
        # u_vec[i] = sum_j U[i*m+j, c] A[i,j]
        u_vec = np.zeros(n)
        for i in range(n):
            for j in range(m):
                u_vec[i] += U[i*m+j, c] * A[i,j]
                
        # v_vec[k] = sum_j V[j*p+k, c] B[j,k]
        v_vec = np.zeros(p)
        for k in range(p):
            for j in range(m):
                v_vec[k] += V[j*p+k, c] * B[j,k]
                
        # Add to C
        for i in range(n):
            for k in range(p):
                C_calc[i, k] += W[k*n+i, c] * u_vec[i] * v_vec[k]
                
    error = np.max(np.abs(C_ref - C_calc))
    if error > 1e-9:
        print(f"Verification failed with error {error}")
        # Print details if needed
        print("C_ref:", C_ref)
        print("C_calc:", C_calc)
    else:
        print(f"Verification passed with error {error}")
