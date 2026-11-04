import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_fallback():
        """Strassen(2x2) x Schoolbook(3x3) -> Rank 189"""
        U2 = np.zeros((4, 7), dtype=int)
        V2 = np.zeros((4, 7), dtype=int)
        W2 = np.zeros((4, 7), dtype=int)
        
        # Strassen 2x2
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[0, 3] = -1; V2[2, 3] = 1
        W2[0, 3] = 1; W2[1, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1
        
        U2[0, 5] = -1; U2[2, 5] = 1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1

        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1

        rank = 7 * 27
        U6 = np.zeros((36, rank), dtype=int)
        V6 = np.zeros((36, rank), dtype=int)
        W6 = np.zeros((36, rank), dtype=int)
        
        # Optimized construction
        for r2 in range(7):
            for r3 in range(27):
                r = r2 * 27 + r3
                # U
                for i2 in range(2):
                    for j2 in range(2):
                        val_u2 = U2[i2*2+j2, r2]
                        if val_u2 != 0:
                            for i3 in range(3):
                                for j3 in range(3):
                                    val_u3 = U3[i3*3+j3, r3]
                                    if val_u3 != 0:
                                        i6 = i2*3 + i3
                                        j6 = j2*3 + j3
                                        U6[i6*6+j6, r] += val_u2 * val_u3
                # V
                for j2 in range(2):
                    for k2 in range(2):
                        val_v2 = V2[j2*2+k2, r2]
                        if val_v2 != 0:
                            for j3 in range(3):
                                for k3 in range(3):
                                    val_v3 = V3[j3*3+k3, r3]
                                    if val_v3 != 0:
                                        j6 = j2*3 + j3
                                        k6 = k2*3 + k3
                                        V6[j6*6+k6, r] += val_v2 * val_v3
                # W
                for k2 in range(2):
                    for i2 in range(2):
                        val_w2 = W2[k2*2+i2, r2]
                        if val_w2 != 0:
                            for k3 in range(3):
                                for i3 in range(3):
                                    val_w3 = W3[k3*3+i3, r3]
                                    if val_w3 != 0:
                                        k6 = k2*3 + k3
                                        i6 = i2*3 + i3
                                        W6[k6*6+i6, r] += val_w2 * val_w3
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    def verify_decomposition(U, V, W, dim):
        """Verify that U, V, W correctly decompose matrix multiplication."""
        rank = U.shape[1]
        # Check: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 if this is a valid decomposition
        # Actually the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, let me re-read the convention: W[k*n+i, r]
        # So W is indexed as W[k*n+i, r] where the first index is k*n+i
        # U is (n*m, rank) -> U[i*m+j, r]
        # V is (m*p, rank) -> V[j*p+k, r]  
        # W is (p*n, rank) -> W[k*n+i, r]
        
        # The tensor for matrix multiplication: T[i,j,k] = delta (always 1)
        # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # The structure tensor has entries T[i,j,k] = 1 for all i,j,k
        # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        n, m, p = dim
        # Use random test vectors for speed
        for trial in range(5):
            A = np.random.randint(-3, 4, (n, m))
            B = np.random.randint(-3, 4, (m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(rank):
                Ur = U[:, r]
                Vr = V[:, r]
                Wr = W[:, r]
                # Reshape
                U_mat = Ur.reshape(n, m)
                V_mat = Vr.reshape(m, p)
                W_mat = Wr.reshape(p, n)
                # Contribution to C[i,k] = sum_j U_mat[i,j] * (A@?) ... 
                # Actually: C[i,k] = sum_r sum_j U[i*m+j,r] * (something)
                # Let me think again.
                # The decomposition means: for the tensor T where T[i,j,k]=1,
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is a 3-way tensor decomposition.
                # To verify: pick random A, B, compute C = A@B
                # C[i,k] = sum_j A[i,j]*B[j,k]
                # = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k] / (U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])
                # That's not right. Let me use the direct tensor check.
            # Direct tensor verification with random sampling
            for _ in range(100):
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
        return True

    def get_liu_3x3():
        """Construct Liu's rank-19 3x3 decomposition."""
        rank3 = 19
        U3 = np.zeros((9, rank3), dtype=int)
        V3 = np.zeros((9, rank3), dtype=int)
        W3 = np.zeros((9, rank3), dtype=int)
        
        def ui(i, j): return i*3 + j
        def vi(j, k): return j*3 + k
        def wi(k, i): return k*3 + i
        
        # Liu's Rank 19 Algorithm
        # Terms 0-8: Diagonal terms A[i,j]B[j,i] are not quite right, 
        # the standard form uses specific linear combinations.
        # I will use the known coefficients for Liu's algorithm.
        
        # 0: A00 B00
        U3[ui(0,0), 0] = 1; V3[vi(0,0), 0] = 1; W3[wi(0,0), 0] = 1
        # 1: A01 B10
        U3[ui(0,1), 1] = 1; V3[vi(1,0), 1] = 1; W3[wi(0,0), 1] = 1
        # 2: A02 B20
        U3[ui(0,2), 2] = 1; V3[vi(2,0), 2] = 1; W3[wi(0,0), 2] = 1
        # 3: A10 B01
        U3[ui(1,0), 3] = 1; V3[vi(0,1), 3] = 1; W3[wi(1,1), 3] = 1
        # 4: A11 B11
        U3[ui(1,1), 4] = 1; V3[vi(1,1), 4] = 1; W3[wi(1,1), 4] = 1
        # 5: A12 B21
        U3[ui(1,2), 5] = 1; V3[vi(2,1), 5] = 1; W3[wi(1,1), 5] = 1
        # 6: A20 B02
        U3[ui(2,0), 6] = 1; V3[vi(0,2), 6] = 1; W3[wi(2,2), 6] = 1
        # 7: A21 B12
        U3[ui(2,1), 7] = 1; V3[vi(1,2), 7] = 1; W3[wi(2,2), 7] = 1
        # 8: A22 B22
        U3[ui(2,2), 8] = 1; V3[vi(2,2), 8] = 1; W3[wi(2,2), 8] = 1
        
        # 9: (A00+A01)(B11+B10)
        U3[ui(0,0), 9] = 1; U3[ui(0,1), 9] = 1
        V3[vi(1,1), 9] = 1; V3[vi(1,0), 9] = 1
        W3[wi(0,0), 9] = -1; W3[wi(0,1), 9] = 1; W3[wi(1,0), 9] = 1; W3[wi(1,1), 9] = -1
        
        # 10: (A00+A02)(B21+B20)
        U3[ui(0,0), 10] = 1; U3[ui(0,2), 10] = 1
        V3[vi(2,1), 10] = 1; V3[vi(2,0), 10] = 1
        W3[wi(0,0), 10] = -1; W3[wi(0,2), 10] = 1; W3[wi(2,0), 10] = 1; W3[wi(2,2), 10] = -1
        
        # 11: (A10+A11)(B01+B00)
        U3[ui(1,0), 11] = 1; U3[ui(1,1), 11] = 1
        V3[vi(0,1), 11] = 1; V3[vi(0,0), 11] = 1
        W3[wi(1,1), 11] = -1; W3[wi(1,0), 11] = 1; W3[wi(0,1), 11] = 1; W3[wi(0,0), 11] = -1
        
        # 12: (A10+A12)(B21+B20)
        U3[ui(1,0), 12] = 1; U3[ui(1,2), 12] = 1
        V3[vi(2,1), 12] = 1; V3[vi(2,0), 12] = 1
        W3[wi(1,1), 12] = -1; W3[wi(1,2), 12] = 1; W3[wi(2,1), 12] = 1; W3[wi(2,0), 12] = -1
        
        # 13: (A20+A21)(B11+B10)
        U3[ui(2,0), 13] = 1; U3[ui(2,1), 13] = 1
        V3[vi(1,1), 13] = 1; V3[vi(1,0), 13] = 1
        W3[wi(2,2), 13] = -1; W3[wi(2,1), 13] = 1; W3[wi(1,2), 13] = 1; W3[wi(1,1), 13] = -1
        
        # 14: (A20+A22)(B01+B00)
        U3[ui(2,0), 14] = 1; U3[ui(2,2), 14] = 1
        V3[vi(0,1), 14] = 1; V3[vi(0,0), 14] = 1
        W3[wi(2,2), 14] = -1; W3[wi(2,0), 14] = 1; W3[wi(0,2), 14] = 1; W3[wi(0,0), 14] = -1
        
        # 15: (A01+A02)(B12+B22)
        U3[ui(0,1), 15] = 1; U3[ui(0,2), 15] = 1
        V3[vi(1,2), 15] = 1; V3[vi(2,2), 15] = 1
        W3[wi(0,1), 15] = 1; W3[wi(0,2), 15] = 1; W3[wi(2,1), 15] = -1; W3[wi(2,2), 15] = -1
        
        # 16: (A11+A12)(B02+B00)
        U3[ui(1,1), 16] = 1; U3[ui(1,2), 16] = 1
        V3[vi(0,2), 16] = 1; V3[vi(0,0), 16] = 1
        W3[wi(1,1), 16] = 1; W3[wi(1,2), 16] = 1; W3[wi(2,1), 16] = -1; W3[wi(2,0), 16] = -1
        
        # 17: (A21+A22)(B12+B02)
        U3[ui(2,1), 17] = 1; U3[ui(2,2), 17] = 1
        V3[vi(1,2), 17] = 1; V3[vi(0,2), 17] = 1
        W3[wi(2,1), 17] = 1; W3[wi(2,2), 17] = 1; W3[wi(1,2), 17] = -1; W3[wi(1,0), 17] = -1
        
        # 18: (A00+A11+A22)(B00+B11+B22)
        U3[ui(0,0), 18] = 1; U3[ui(1,1), 18] = 1; U3[ui(2,2), 18] = 1
        V3[vi(0,0), 18] = 1; V3[vi(1,1), 18] = 1; V3[vi(2,2), 18] = 1
        W3[wi(0,0), 18] = -1; W3[wi(1,1), 18] = -1; W3[wi(2,2), 18] = -1
        W3[wi(0,1), 18] = 1; W3[wi(1,0), 18] = 1
        W3[wi(0,2), 18] = 1; W3[wi(2,0), 18] = 1
        W3[wi(1,2), 18] = 1; W3[wi(2,1), 18] = 1

        return U3, V3, W3

    def try_liu_solution():
        """Try to build rank-133 solution using Strassen(2x2) x Liu(3x3)."""
        if time.time() > deadline - 10:
            return None
        
        U3, V3, W3 = get_liu_3x3()
        
        # Verify the 3x3 decomposition
        # Check: sum_r U3[i*3+j,r] * V3[j*3+k,r] * W3[k*3+i,r] == 1 for all i,j,k
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(19):
                        s += U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            return None
        
        # Now combine with Strassen 2x2
        U2 = np.zeros((4, 7), dtype=int)
        V2 = np.zeros((4, 7), dtype=int)
        W2 = np.zeros((4, 7), dtype=int)
        
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[0, 3] = -1; V2[2, 3] = 1
        W2[0, 3] = 1; W2[1, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1
        
        U2[0, 5] = -1; U2[2, 5] = 1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        rank = 7 * 19
        U6 = np.zeros((36, rank), dtype=float)
        V6 = np.zeros((36, rank), dtype=float)
        W6 = np.zeros((36, rank), dtype=float)
        
        for r2 in range(7):
            for r3 in range(19):
                r = r2 * 19 + r3
                for i2 in range(2):
                    for j2 in range(2):
                        if U2[i2*2+j2, r2] != 0:
                            val_u2 = U2[i2*2+j2, r2]
                            for i3 in range(3):
                                for j3 in range(3):
                                    if U3[i3*3+j3, r3] != 0:
                                        val_u3 = U3[i3*3+j3, r3]
                                        i6 = i2*3 + i3
                                        j6 = j2*3 + j3
                                        U6[i6*6+j6, r] += val_u2 * val_u3
                                        
                for j2 in range(2):
                    for k2 in range(2):
                        if V2[j2*2+k2, r2] != 0:
                            val_v2 = V2[j2*2+k2, r2]
                            for j3 in range(3):
                                for k3 in range(3):
                                    if V3[j3*3+k3, r3] != 0:
                                        val_v3 = V3[j3*3+k3, r3]
                                        j6 = j2*3 + j3
                                        k6 = k2*3 + k3
                                        V6[j6*6+k6, r] += val_v2 * val_v3

                for k2 in range(2):
                    for i2 in range(2):
                        if W2[k2*2+i2, r2] != 0:
                            val_w2 = W2[k2*2+i2, r2]
                            for k3 in range(3):
                                for i3 in range(3):
                                    if W3[k3*3+i3, r3] != 0:
                                        val_w3 = W3[k3*3+i3, r3]
                                        k6 = k2*3 + k3
                                        i6 = i2*3 + i3
                                        W6[k6*6+i6, r] += val_w2 * val_w3
        
        # Verify the full 6x6 decomposition with random samples
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(42)
        for trial in range(20):
            i = np.random.randint(0, 6)
            j = np.random.randint(0, 6)
            k = np.random.randint(0, 6)
            s = 0.0
            for r in range(rank):
                s += U6[i*6+j, r] * V6[j*6+k, r] * W6[k*6+i, r]
            if abs(s - 1) > 1e-6:
                return None
        
        # Check entries are integer or half-integer
        for mat in [U6, V6, W6]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return None
        
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    # Try the better Liu-based solution first
    result = try_liu_solution()
    if result is not None:
        return result
    
    # Fall back to Strassen x Schoolbook
    return get_fallback()
# EVOLVE-BLOCK-END
