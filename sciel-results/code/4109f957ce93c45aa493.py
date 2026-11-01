import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    n, m, p = DIMENSIONS
    # Start with a rank upper bound; we'll build up and verify
    max_rank = 60

    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= max_rank:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)

        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1

        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1

        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1

        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> C11 += 1
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> C00 += 1
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1

        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= max_rank:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        g = (j0 + 1) * p + k0

        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        # P1 = a*e
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1

        # P2 = b*g
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1

        # P3 = c*e
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1

        # P4 = d*g
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1

        return True

    # Process blocks: 2 blocks in j-dim (each 2 wide), 4 blocks in k-dim (each 2 wide except last)
    for j_start in [0, 2]:
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        for k_start in [0, 2, 4]:
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            if not add_strassen_222(0, j_start, k_start):
                break
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        if not add_schoolbook_221(0, j_start, 6):
            break

    # Now try to reduce rank by finding linear dependencies
    # Convert to numpy arrays for efficient linear algebra
    U_np = np.array(U[:n * m][:, :col], dtype=float)
    V_np = np.array(V[:m * p][:, :col], dtype=float)
    W_np = np.array(W[:p * n][:, :col], dtype=float)

    # The decomposition is valid if for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{correct}
    # We need to find the minimum subset of columns that still works.
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # The target is: T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication entry, else 0
    # For 2x4x7: A is 2x4, B is 4x7, C is 2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k) triples, and 0 otherwise.
    # Actually, the tensor T[i,j,k] = A[i,j]*B[j,k]*C[k,i]... no.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Let's verify our construction first
    def verify(U_a, V_a, W_a, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_a[i * m + j][r] * V_a[j * p + k][r] * W_a[k * n + i][r]
                    expected = 1.0
                    if abs(val - expected) > 1e-9:
                        return False
        return True

    # Verify current decomposition
    if not verify(U, V, W, col):
        # Something went wrong, return what we have
        return {"U": U, "V": V, "W": W}

    # Now try to find rank reduction using linear algebra
    # We have col columns. We want to find the minimum number of columns that suffice.
    # This is a hard problem, but we can try greedy approaches.
    
    # Alternative: use the fact that the tensor is fixed, and find a rank-minimal decomposition
    # via iterative methods. But given time constraints, let's try a simpler approach.
    
    # Check if we can eliminate any column by seeing if it's a linear combination of others
    # in a tensor sense. This is complex, so let's try a different strategy.
    
    # For now, trim to actual used rank
    used_rank = col
    U_trimmed = [row[:used_rank] for row in U[:n * m]]
    V_trimmed = [row[:used_rank] for row in V[:m * p]]
    W_trimmed = [row[:used_rank] for row in W[:p * n]]
    
    # Quick verification
    if not verify(U_trimmed, V_trimmed, W_trimmed, used_rank):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
