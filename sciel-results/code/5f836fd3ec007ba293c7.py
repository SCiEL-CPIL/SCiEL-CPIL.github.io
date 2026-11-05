"""Matrix multiplication 3,4,7 - rank-minimal decomposition via Strassen substitutions."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    def schoolbook():
        """Return the schoolbook (rank n*m*p) decomposition.
        U[i*m+k, r] = A[i,k] coefficient
        V[k*p+j, r] = B[k,j] coefficient
        W[j*n+i, r] = C[i,j] coefficient (indexed as W[k*n+i] where k=j_col, i=row)
        """
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # A[i,j] * B[j,k] contributes to C[i,k]
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True

    def try_strassen(U, V, W, i0, k0, j0):
        """Replace 8 schoolbook terms for A rows {i0,i0+1}, B cols {j0,j0+1}, inner {k0,k0+1}
        with 7 Strassen terms.
        i0: row index in A (0..n-2)
        j0: col index in B (0..p-2)  
        k0: inner dimension (0..m-2)
        """
        if i0 + 1 >= n or j0 + 1 >= p or k0 + 1 >= m:
            return None

        positions = [(i, j, k) for i in [i0, i0+1] for j in [k0, k0+1] for k in [j0, j0+1]]
        
        rank = len(U[0])
        touching_terms = set()
        for (i, j, k) in positions:
            for r in range(rank):
                if U[i*m+j][r] != 0 and V[j*p+k][r] != 0 and W[k*n+i][r] != 0:
                    touching_terms.add(r)
        
        if len(touching_terms) != 8:
            return None
        
        for r in touching_terms:
            count = 0
            for (i, j, k) in positions:
                if U[i*m+j][r] == 1 and V[j*p+k][r] == 1 and W[k*n+i][r] == 1:
                    count += 1
            if count != 1:
                return None
        
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        for r in touching_terms:
            for (i, j, k) in positions:
                new_U[i*m+j][r] = 0
                new_V[j*p+k][r] = 0
                new_W[k*n+i][r] = 0
        
        new_rank = rank + 7
        for row in new_U:
            row.extend([0] * 7)
        for row in new_V:
            row.extend([0] * 7)
        for row in new_W:
            row.extend([0] * 7)
        
        # Strassen for C[i,j] = sum_k A[i,k]*B[k,j]
        # Block: A rows i0,i0+1; B cols j0,j0+1; inner k0,k0+1
        # C[i0,j0]   = A[i0,k0]*B[k0,j0] + A[i0,k0+1]*B[k0+1,j0]
        # C[i0,j0+1] = A[i0,k0]*B[k0,j0+1] + A[i0,k0+1]*B[k0+1,j0+1]
        # C[i0+1,j0]   = A[i0+1,k0]*B[k0,j0] + A[i0+1,k0+1]*B[k0+1,j0]
        # C[i0+1,j0+1] = A[i0+1,k0]*B[k0,j0+1] + A[i0+1,k0+1]*B[k0+1,j0+1]
        
        # m1 = (A[i0,k0]+A[i0+1,k0+1])*(B[k0,j0]+B[k0+1,j0+1])
        r = rank
        new_U[i0*m+k0][r] += 1
        new_U[(i0+1)*m+(k0+1)][r] += 1
        new_V[k0*p+j0][r] += 1
        new_V[(k0+1)*p+(j0+1)][r] += 1
        new_W[j0*n+i0][r] += 1
        new_W[(j0+1)*n+i0][r] -= 1
        new_W[j0*n+(i0+1)][r] -= 1
        new_W[(j0+1)*n+(i0+1)][r] += 1
        r += 1

        # m2 = (A[i0+1,k0]+A[i0+1,k0+1])*B[k0,j0]
        new_U[(i0+1)*m+k0][r] += 1
        new_U[(i0+1)*m+(k0+1)][r] += 1
        new_V[k0*p+j0][r] += 1
        new_W[j0*n+(i0+1)][r] += 1
        r += 1

        # m3 = A[i0,k0]*(B[k0,j0+1]-B[k0+1,j0+1])
        new_U[i0*m+k0][r] += 1
        new_V[k0*p+(j0+1)][r] += 1
        new_V[(k0+1)*p+(j0+1)][r] -= 1
        new_W[(j0+1)*n+i0][r] += 1
        r += 1

        # m4 = A[i0+1,k0+1]*(B[k0+1,j0]-B[k0,j0])
        new_U[(i0+1)*m+(k0+1)][r] += 1
        new_V[(k0+1)*p+j0][r] += 1
        new_V[k0*p+j0][r] -= 1
        new_W[j0*n+i0][r] += 1
        new_W[(j0+1)*n+(i0+1)][r] += 1
        r += 1

        # m5 = (A[i0,k0]+A[i0,k0+1])*B[k0+1,j0+1]
        new_U[i0*m+k0][r] += 1
        new_U[i0*m+(k0+1)][r] += 1
        new_V[(k0+1)*p+(j0+1)][r] += 1
        new_W[(j0+1)*n+i0][r] -= 1
        new_W[(j0+1)*n+(i0+1)][r] += 1
        r += 1

        # m6 = (A[i0+1,k0]-A[i0,k0])*(B[k0,j0]+B[k0,j0+1])
        new_U[(i0+1)*m+k0][r] += 1
        new_U[i0*m+k0][r] -= 1
        new_V[k0*p+j0][r] += 1
        new_V[k0*p+(j0+1)][r] += 1
        new_W[(j0+1)*n+(i0+1)][r] += 1
        r += 1

        # m7 = (A[i0,k0+1]-A[i0+1,k0+1])*(B[k0+1,j0]+B[k0+1,j0+1])
        new_U[i0*m+(k0+1)][r] += 1
        new_U[(i0+1)*m+(k0+1)][r] -= 1
        new_V[(k0+1)*p+j0][r] += 1
        new_V[(k0+1)*p+(j0+1)][r] += 1
        new_W[j0*n+i0][r] += 1
        r += 1

        if verify(new_U, new_V, new_W):
            return new_U, new_V, new_W
        return None

    best = schoolbook()
    U, V, W = best["U"], best["V"], best["W"]
    best_rank = len(U[0])
    applied = set()

    # Multiple passes to allow cascading substitutions
    for _ in range(3):
        for i0 in range(n - 1):
            for j0 in range(p - 1):
                for k0 in range(m - 1):
                    if time.time() > deadline - 5:
                        return best
                    if (i0, j0, k0) in applied:
                        continue
                    result = try_strassen(U, V, W, i0, k0, j0)
                    if result is not None:
                        U, V, W = result
                        applied.add((i0, j0, k0))
                        best_rank = len(U[0])
                        best = {"U": U, "V": V, "W": W}

    return best


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank} (schoolbook: {3*4*7})")
    print(f"Score: {3*4*7/rank:.4f}")
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    valid = all(
        sum(U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] for r in range(len(U[0]))) == 1
        for i in range(n) for j in range(m) for k in range(p)
    )
    print(f"Valid: {valid}")
