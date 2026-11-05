import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Verify a decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    def verify_decomposition(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    # Strassen 2x2 rank-7 algorithm
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # m0 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # m1 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        # m2 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        # m3 = A11(B10-B00)
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[1, 3] = 1

        # m4 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # m5 = (A10-A00)(B00+B01)
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # m6 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    # Bini 3x3 rank-21 algorithm
    # This implementation is a candidate. If it fails verification, we fall back.
    def get_bini_3x3():
        U = np.zeros((9, 21), dtype=int)
        V = np.zeros((9, 21), dtype=int)
        W = np.zeros((9, 21), dtype=int)
        
        # A verified set of 21 terms for 3x3 matrix multiplication
        # Based on Bini's rank-21 algorithm
        # The terms are (coeff, i, j, k) where U[i,j], V[j,k], W[k,i] are involved
        # Note: The standard Bini algorithm is not trivially symmetric in this simple index mapping
        # without careful construction.
        
        # To ensure correctness, we use a known valid decomposition if available.
        # If not, this function might return invalid data, caught by verify.
        
        # Here is a standard Bini 3x3 decomposition (Rank 21)
        # Source: Various computational algebra references
        bini_terms = [
            # 9 terms: A_ii B_ii
            (1, 0, 0, 0), (1, 1, 1, 1), (1, 2, 2, 2),
            # 6 terms: A_ij B_ji (i != j)
            (1, 0, 1, 1), (1, 1, 0, 0),
            (1, 0, 2, 2), (1, 2, 0, 0),
            (1, 1, 2, 2), (1, 2, 1, 1),
            # 6 terms: (A_ii + A_jj + A_kk) (B_ii + B_jj + B_kk) variants
            (1, 0, 0, 1), (1, 1, 1, 0), (1, 2, 2, 0),
            (1, 0, 0, 2), (1, 1, 1, 2), (1, 2, 2, 1),
            # 0 terms left? 9+6+6 = 21.
        ]
        
        # This specific set of terms is NOT guaranteed to be correct for the tensor product.
        # The Bini algorithm requires specific linear combinations.
        # For safety, we will NOT use this unverified hardcoded set.
        # Instead, we return None to signal failure.
        return None

    # Schoolbook 3x3 rank-27
    def get_schoolbook_3x3():
        U = np.zeros((9, 27), dtype=int)
        V = np.zeros((9, 27), dtype=int)
        W = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U[i * 3 + j, r] = 1
                    V[j * 3 + k, r] = 1
                    W[k * 3 + i, r] = 1
        return U, V, W

    # Kronecker product of two decompositions
    def kron_decomp(U1, V1, W1, U2, V2, W2, n1, n2):
        rank1, rank2 = U1.shape[1], U2.shape[1]
        n = n1 * n2
        rank = rank1 * rank2
        
        U = np.zeros((n*n, rank), dtype=int)
        V = np.zeros((n*n, rank), dtype=int)
        W = np.zeros((n*n, rank), dtype=int)
        
        for r1 in range(rank1):
            for r2 in range(rank2):
                r = r1 * rank2 + r2
                for i1 in range(n1):
                    for j1 in range(n1):
                        if U1[i1*n1+j1, r1] != 0:
                            for i2 in range(n2):
                                for j2 in range(n2):
                                    if U2[i2*n2+j2, r2] != 0:
                                        i = i1*n2 + i2
                                        j = j1*n2 + j2
                                        U[i*n+j, r] += U1[i1*n1+j1, r1] * U2[i2*n2+j2, r2]
                
                for j1 in range(n1):
                    for k1 in range(n1):
                        if V1[j1*n1+k1, r1] != 0:
                            for j2 in range(n2):
                                for k2 in range(n2):
                                    if V2[j2*n2+k2, r2] != 0:
                                        j = j1*n2 + j2
                                        k = k1*n2 + k2
                                        V[j*n+k, r] += V1[j1*n1+k1, r1] * V2[j2*n2+k2, r2]
                
                for k1 in range(n1):
                    for i1 in range(n1):
                        if W1[k1*n1+i1, r1] != 0:
                            for k2 in range(n2):
                                for i2 in range(n2):
                                    if W2[k2*n2+i2, r2] != 0:
                                        k = k1*n2 + k2
                                        i = i1*n2 + i2
                                        W[k*n+i, r] += W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
        
        return U, V, W

    # Try Strassen 2x2 + Schoolbook 3x3 = rank 189 (Guaranteed)
    U2, V2, W2 = get_strassen_2x2()
    strassen_valid = verify_decomposition(U2, V2, W2, 2, 2, 2)
    
    U3s, V3s, W3s = get_schoolbook_3x3()
    
    if strassen_valid:
        U6, V6, W6 = kron_decomp(U2, V2, W2, U3s, V3s, W3s, 2, 3)
        if verify_decomposition(U6, V6, W6, 6, 6, 6):
            # Rank 189
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Ultimate fallback: pure schoolbook 6x6 (rank 216)
    U6s = np.zeros((36, 216), dtype=int)
    V6s = np.zeros((36, 216), dtype=int)
    W6s = np.zeros((36, 216), dtype=int)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                r = i * 36 + j * 6 + k
                U6s[i * 6 + j, r] = 1
                V6s[j * 6 + k, r] = 1
                W6s[k * 6 + i, r] = 1
    
    return {"U": U6s.tolist(), "V": V6s.tolist(), "W": W6s.tolist()}
# EVOLVE-BLOCK-END
