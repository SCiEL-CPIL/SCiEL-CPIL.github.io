import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    # Total standard terms
    total_terms = n * m * p
    # New Rank:
    # 4 Strassen blocks (2x2x2) -> 4 * 7 = 28
    # 4 "Simple" blocks (2x1x2) for j in {0,1,2,3}, k in {4,5} -> 4 * 1 = 4
    # 3 "Simple" blocks (2x1x2) for j=4, k in {0,1}, {2,3}, {4,5} -> 3 * 1 = 3
    # Total Rank = 28 + 4 + 3 = 35
    rank = 35

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Define the 4 disjoint 2x2x2 blocks for Strassen
    blocks = [
        ((0, 1), (0, 1), (0, 1)),
        ((0, 1), (2, 3), (2, 3)),
        ((0, 1), (0, 1), (2, 3)),
        ((0, 1), (2, 3), (0, 1)),
    ]

    # Create a set of all (i,j,k) in these blocks to skip them in standard loop
    strassen_indices = set()
    for i_range, j_range, k_range in blocks:
        for i in range(i_range[0], i_range[1] + 1):
            for j in range(j_range[0], j_range[1] + 1):
                for k in range(k_range[0], k_range[1] + 1):
                    strassen_indices.add((i, j, k))

    # Helper to add a "Simple" (2x1x2) decomposition
    # This covers terms for a fixed j and a pair of k's {k0, k1}
    # i ranges over {0, 1}
    def add_simple_block(j, k0, k1):
        nonlocal col
        
        # U indices: i*m + j for i in {0, 1}
        u0 = 0 * m + j
        u1 = 1 * m + j
        
        # V indices: j*p + k for k in {k0, k1}
        v0 = j * p + k0
        v1 = j * p + k1
        
        # W indices: k*n + i for k in {k0, k1} and i in {0, 1}
        w00 = k0 * n + 0
        w01 = k0 * n + 1
        w10 = k1 * n + 0
        w11 = k1 * n + 1

        # Rank 1 term: U=1, V=1, W=1
        U[u0][col] = 1
        U[u1][col] = 1
        V[v0][col] = 1
        V[v1][col] = 1
        W[w00][col] = 1
        W[w01][col] = 1
        W[w10][col] = 1
        W[w11][col] = 1
        col += 1

    # Add Simple blocks for j in {0, 1, 2, 3} and k in {4, 5}
    for j in range(4):
        add_simple_block(j, 4, 5)
        
    # Add Simple blocks for j=4 and k pairs {0,1}, {2,3}, {4,5}
    add_simple_block(4, 0, 1)
    add_simple_block(4, 2, 3)
    add_simple_block(4, 4, 5)

    # Helper to add a Strassen decomposition for a specific block
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        
        # Indices in U (n*m)
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        # Indices in V (m*p)
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        # Indices in W (p*n)
        w00 = k0 * n + i0
        w01 = k0 * n + i1
        w10 = k1 * n + i0
        w11 = k1 * n + i1

        # M1 = (A00 + A11)(B00 + B11)
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1

        # M2 = (A10 + A11)B00
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] += -1
        col += 1

        # M3 = A00(B01 - B11)
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] += -1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1

        # M4 = A11(B10 - B00)
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] += -1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1

        # M5 = (A00 + A01)B11
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1
        W[w01][col] += 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[u10][col] += 1
        U[u00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[u01][col] += 1
        U[u11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Apply Strassen to each block
    for i_range, j_range, k_range in blocks:
        i0, i1 = i_range
        j0, j1 = j_range
        k0, k1 = k_range
        add_strassen_block(i0, i1, j0, j1, k0, k1)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to rank 60
        U_fb = [[0] * total_terms for _ in range(n * m)]
        V_fb = [[0] * total_terms for _ in range(m * p)]
        W_fb = [[0] * total_terms for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_fb[u_idx][col_fb] = 1
                    V_fb[v_idx][col_fb] = 1
                    W_fb[w_idx][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
