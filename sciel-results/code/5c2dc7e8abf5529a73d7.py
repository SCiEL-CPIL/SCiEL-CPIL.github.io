import random
import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions 3x4x8 using the schoolbook decomposition.
    
    The decomposition is:
    C = A * B
    where A is 3x4, B is 4x8, C is 3x8.
    
    The tensor is T = sum_{i,j,k} e_{i,j}^A \otimes f_{j,k}^B \otimes g_{i,k}^C.
    
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    Note: The problem statement says W[k*n+i, r]. This implies the index for W is k*n + i.
    Standard index for C is i*n + k? Or k*n + i?
    The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    This usually means the linear index in W is determined by the indices of C.
    If C is n x p, and we flatten it, do we flatten row-major or column-major?
    "W[k*n+i, r]" suggests that for C[i, k], the index in W is k*n + i.
    This corresponds to a column-major flattening of C (where i is the row, k is the column, and we index by column k then row i).
    Or it simply defines the mapping from (i,k) to the row index of W.
    Let's follow the prompt exactly: W index = k*n + i.
    
    U index: i*m + j (row i, col j of A)
    V index: j*p + k (row j, col k of B)
    W index: k*n + i (row k, col i? No, C is n x p. i is row 0..n-1, k is col 0..p-1.
    If W index is k*n + i, then for fixed k, i varies from 0 to n-1.
    This maps C[i, k] to W[k*n + i, :].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index for A[i, j]
                u_idx = i * m + j
                # V index for B[j, k]
                v_idx = j * p + k
                # W index for C[i, k] following W[k*n + i, r]
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-check to ensure correctness and timing
    start_time = time.time()
    res = construct()
    end_time = time.time()
    
    # Verify dimensions
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    assert len(res["U"]) == n * m
    assert len(res["V"]) == m * p
    assert len(res["W"]) == p * n
    assert rank == n * m * p
    
    # Verify a random case
    import random
    random.seed(42)
    A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[i][k] = s
            
    # Compute C' using U, V, W
    # C'[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
    # Wait, the tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # The bilinear form is sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r].
    # Here idx_u depends on i, j. idx_v depends on j, k. idx_w depends on i, k.
    # But in the sum over r, the indices i, j, k are fixed for the output C[i, k].
    # The inputs are A and B.
    # The formula for C[i, k] is sum_{j} A[i, j] B[j, k].
    # The tensor decomposition approximates this.
    # Specifically, C[i, k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]?
    # No, the standard definition is:
    # C[i, k] = sum_r U_{i, j, r} ... this is not right.
    # The tensor T has components T_{i, j, k}.
    # T_{i, j, k} = 1 if the indices match the multiplication rule?
    # The multiplication tensor is T = sum_{i, j, k} e_{i, j} \otimes f_{j, k} \otimes g_{i, k}.
    # So T_{u, v, w} = 1 if u corresponds to (i, j), v to (j, k), w to (i, k).
    # Then C[i, k] = sum_{u, v, w} T_{u, v, w} A_u B_v.
    # Using the rank decomposition T = sum_r u_r \otimes v_r \otimes w_r:
    # C[i, k] = sum_r w_r[i, k] * (sum_u u_r[u] A_u) * (sum_v v_r[v] B_v).
    # In our case:
    # u_r[u] is 1 if u corresponds to (i_r, j_r) for this r?
    # No, in schoolbook, each r corresponds to a specific (i, j, k).
    # So for a given r, there is a specific (i, j, k).
    # u_r is 1 at index i*m+j, 0 elsewhere.
    # v_r is 1 at index j*p+k, 0 elsewhere.
    # w_r is 1 at index k*n+i, 0 elsewhere.
    # Then sum_u u_r[u] A_u = A[i, j].
    # sum_v v_r[v] B_v = B[j, k].
    # w_r[i, k] (mapped to linear index) is 1 if the linear index matches k*n+i.
    # So the term is 1 * A[i, j] * B[j, k] if the linear index of w_r matches the current (i, k).
    # Wait, the sum is over r.
    # C[i, k] = sum_r (sum_u U[u, r] A[u]) (sum_v V[v, r] B[v]) W[w, r] where w = k*n+i.
    # For a specific r, let it correspond to (i_r, j_r, k_r).
    # U[u, r] is 1 if u = i_r*m + j_r.
    # V[v, r] is 1 if v = j_r*p + k_r.
    # W[w, r] is 1 if w = k_r*n + i_r.
    # So the term is non-zero only if:
    # 1. u = i_r*m + j_r -> A[i_r, j_r]
    # 2. v = j_r*p + k_r -> B[j_r, k_r]
    # 3. w = k_r*n + i_r -> This must match the target w = k*n+i.
    # So we sum over all r such that k_r = k and i_r = i.
    # For such r, j_r can be anything?
    # In schoolbook, r is unique for each (i, j, k).
    # So for fixed i, k, we sum over j.
    # The term is A[i, j] * B[j, k] * 1.
    # Summing over j gives C[i, k].
    # Correct.
    
    C_calc = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            w_idx = k * n + i
            s = 0
            for r in range(rank):
                if W[w_idx][r] != 0:
                    # Find u and v for this r
                    # This is inefficient for verification, but rank is small.
                    # We can precompute u_idx, v_idx for each r.
                    pass
            
    # More efficient verification:
    # Since U, V, W are sparse (one 1 per column), we can just iterate.
    # But for n=3, m=4, p=8, rank=96, it's fast enough.
    
    for i in range(n):
        for k in range(p):
            w_idx = k * n + i
            s = 0
            for r in range(rank):
                w_val = W[w_idx][r]
                if w_val == 0:
                    continue
                # Find the unique u and v where U and V are non-zero for this r
                # In schoolbook, there is exactly one.
                u_val_sum = 0
                v_val_sum = 0
                for u in range(n * m):
                    if U[u][r] != 0:
                        u_val_sum += U[u][r] * A[u // m][u % m]
                for v in range(m * p):
                    if V[v][r] != 0:
                        v_val_sum += V[v][r] * B[v // p][v % p]
                s += w_val * u_val_sum * v_val_sum
            C_calc[i][k] = s
            
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_calc[i][k]:
                print(f"Mismatch at {i}, {k}: {C[i][k]} vs {C_calc[i][k]}")
                sys.exit(1)
                
    print(f"Verification passed in {end_time - start_time:.4f}s")
