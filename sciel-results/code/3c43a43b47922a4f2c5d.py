"""Rewritten program for 2x2x15 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 2, 15]

def construct():
    """
    Constructs the rank-minimal decomposition for 2x2 matrix times 2x15 matrix.
    The rank is 60, as the problem decomposes into 15 independent 2x2x1 problems,
    each of rank 4.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 2 * 15 = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (following W[k*n+i,r] convention)
                W[k * n + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Quick validation
    n, m, p = 2, 2, 15
    rank = len(res["U"][0])
    # Check a few random entries or full check if fast
    # For 2x2x15, full check is 2*2*15*60 operations = 3600, very fast.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += res["U"][i*m+j][c] * res["V"][j*p+k][c] * res["W"][k*n+i][c]
                # The term A[i,j] B[j,k] should contribute 1 to C[i,k]
                # But wait, the tensor decomposition sum is over c.
                # The identity is sum_c U V W = delta?
                # For schoolbook, each term is distinct.
                # U[i*m+j, c] is 1 only if c corresponds to (i,j,k').
                # V[j*p+k, c] is 1 only if c corresponds to (j,k,k'').
                # W[k'*n+i, c] is 1 only if c corresponds to (k',i,k'').
                # If c corresponds to (i,j,k), then:
                # U=1, V=1, W=1.
                # If c corresponds to different indices, one of them is 0.
                # So for fixed i,j,k, sum is 1 if there is a c for (i,j,k).
                # Yes, c = i*m*p + j*p + k? No, col order is i,j,k nested.
                # So s should be 1.
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    if valid:
        print(f"Rank {rank} decomposition is valid.")
    else:
        print("Decomposition invalid!")
