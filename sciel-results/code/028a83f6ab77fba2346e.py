"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact using numpy for speed."""
        rank = len(U[0]) if U else 0
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i * m + j] * V_np[j * p + k], W_np[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def build_rank32():
        """
        Build a rank-32 decomposition using 4 Strassen blocks (rank 7 each)
        for column pairs (0,1), (2,3), (4,5), (6,7) and 1 schoolbook block
        (rank 4) for column 8.
        
        A is 2x2: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx2, A[1,1]=idx3
        B is 2x9: B[0,k]=idx k, B[1,k]=idx 9+k
        C is 2x9: C[0,k]=idx 2k, C[1,k]=idx 2k+1
        """
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # U indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
            # V indices: k0=B[0,k0], k1=B[0,k1], 9+k0=B[1,k0], 9+k1=B[1,k1]
            # W indices: 2*k0=C[0,k0], 2*k0+1=C[1,k0], 2*k1=C[0,k1], 2*k1+1=C[1,k1]

            # M1 = (a+d)(e+h) -> C[0,k0] += 1, C[1,k1] += 1
            U[0][col] = 1; U[3][col] = 1
            V[k0][col] = 1; V[9 + k1][col] = 1
            W[2 * k0][col] = 1; W[2 * k1 + 1][col] = 1
            col += 1

            # M2 = (c+d)e -> C[1,k0] += 1, C[1,k1] -= 1
            U[2][col] = 1; U[3][col] = 1
            V[k0][col] = 1
            W[2 * k0 + 1][col] = 1; W[2 * k1 + 1][col] = -1
            col += 1

            # M3 = a(f-h) -> C[0,k1] += 1, C[1,k1] += 1
            U[0][col] = 1
            V[k1][col] = 1; V[9 + k1][col] = -1
            W[2 * k1][col] = 1; W[2 * k1 + 1][col] = 1
            col += 1

            # M4 = d(g-e) -> C[0,k0] += 1, C[1,k0] += 1
            U[3][col] = 1
            V[9 + k0][col] = 1; V[k0][col] = -1
            W[2 * k0][col] = 1; W[2 * k0 + 1][col] = 1
            col += 1

            # M5 = (a+b)h -> C[0,k0] -= 1, C[0,k1] += 1
            U[0][col] = 1; U[1][col] = 1
            V[9 + k1][col] = 1
            W[2 * k0][col] = -1; W[2 * k1][col] = 1
            col += 1

            # M6 = (c-a)(e+f) -> C[1,k1] += 1
            U[2][col] = 1; U[0][col] = -1
            V[k0][col] = 1; V[k1][col] = 1
            W[2 * k1 + 1][col] = 1
            col += 1

            # M7 = (b-d)(g+h) -> C[0,k0] += 1
            U[1][col] = 1; U[3][col] = -1
            V[9 + k0][col] = 1; V[9 + k1][col] = 1
            W[2 * k0][col] = 1
            col += 1

        def add_schoolbook(k):
            nonlocal col
            # S1 = A[0,0]*B[0,k] -> C[0,k]
            U[0][col] = 1; V[k][col] = 1; W[2 * k][col] = 1
            col += 1
            # S2 = A[0,1]*B[1,k] -> C[0,k]
            U[1][col] = 1; V[9 + k][col] = 1; W[2 * k][col] = 1
            col += 1
            # S3 = A[1,0]*B[0,k] -> C[1,k]
            U[2][col] = 1; V[k][col] = 1; W[2 * k + 1][col] = 1
            col += 1
            # S4 = A[1,1]*B[1,k] -> C[1,k]
            U[3][col] = 1; V[9 + k][col] = 1; W[2 * k + 1][col] = 1
            col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_schoolbook(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    def try_rank31():
        """
        Attempt to find a rank-31 decomposition by checking if any of the 32
        Strassen/schoolbook terms is linearly dependent on the others in the
        36-dimensional tensor space.
        
        The tensor T has 36 non-zero entries: T[(i,j,k)] = 1 for all valid (i,j,k).
        Each rank-1 term c defines a vector v_c in R^36 where
        v_c[(i,j,k)] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        
        We need sum_c v_c = target (all ones in 36 positions).
        If v_c = sum_{c' != c} alpha_{c'} v_{c'} for some c, then we can
        absorb it and get a rank-31 decomposition.
        """
        if time.time() > deadline - 30:
            return None

        result32 = build_rank32()
        U32 = np.array(result32['U'], dtype=float)
        V32 = np.array(result32['V'], dtype=float)
        W32 = np.array(result32['W'], dtype=float)
        rank = 32

        # Build the 36-dimensional target and term vectors
        def tensor_index(i, j, k):
            return i * m * p + j * p + k

        target = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[tensor_index(i, j, k)] = 1.0

        terms = np.zeros((rank, n * m * p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms[c, tensor_index(i, j, k)] = (
                            U32[i * m + j, c] * V32[j * p + k, c] * W32[k * n + i, c]
                        )

        # Verify the rank-32 construction sums to target
        if not np.allclose(terms.sum(axis=0), target, atol=1e-9):
            return None

        # For each term c_drop, check if it's in the span of the other 31
        for c_drop in range(rank):
            if time.time() > deadline - 10:
                return None

            mask = [i for i in range(rank) if i != c_drop]
            A = terms[mask, :]  # shape (31, 36)
            b = terms[c_drop, :]  # shape (36,)

            # Solve A^T x = b (36 equations, 31 unknowns)
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A.T, b, rcond=None)
                residual = np.linalg.norm(A.T @ x - b)

                if residual < 1e-8:
                    # Found dependency! Build rank-31 decomposition
                    # target = sum_all terms = sum_{c != c_drop} terms[c] + terms[c_drop]
                    # terms[c_drop] = sum_{c != c_drop} alpha_c * terms[c]
                    # target = sum_{c != c_drop} (1 + alpha_c) * terms[c]
                    # Scale each remaining term by (1 + alpha_c)
                    new_coeffs = 1.0 + x

                    U31 = U32[:, mask].copy()
                    V31 = V32[:, mask].copy()
                    W31 = W32[:, mask].copy()

                    for idx, coeff in enumerate(new_coeffs):
                        U31[:, idx] *= coeff
                        V31[:, idx] *= coeff
                        W31[:, idx] *= coeff

                    U_list = U31.tolist()
                    V_list = V31.tolist()
                    W_list = W31.tolist()

                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}

            except np.linalg.LinAlgError:
                continue

        return None

    # Try rank 31 first
    result31 = try_rank31()
    if result31 is not None:
        return result31

    # Fallback to rank 32
    result = build_rank32()
    if verify(result["U"], result["V"], result["W"]):
        return result

    # Final fallback: schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
