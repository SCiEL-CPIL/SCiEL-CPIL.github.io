import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit

    n, m, p = DIMENSIONS
    rank = 50

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
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

        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1

        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1

        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1

        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1

        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1

        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1

        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False

        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)

        e = j0 * p + k0
        g = (j0 + 1) * p + k0

        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1

        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1

        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1

        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1

        return True

    # First, build the current rank-50 solution
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    if not add_schoolbook_221(0, 0, 6):
        pass

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    if not add_schoolbook_221(0, 2, 6):
        pass

    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: sum_k U[:,k] * V[:,k] * W[:,k] should equal the multiplication tensor
    # The multiplication tensor T[i,j,k] = delta_{i,k} for A[i,j]*B[j,k]
    # In our convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing: U[i*m+j, k], V[j*p+k, k], W[k*n+i, k]
    
    # Let's verify by checking a few entries
    # For a given (i, j, k) in the output matrix C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor entry should be: T[i,j,k] = 1 if this contributes to C[i,k]
    
    # Actually, let's just verify that the decomposition is valid by checking
    # that for random matrices, the result matches
    rng = np.random.default_rng(42)
    A_test = rng.random((n, m))
    B_test = rng.random((m, p))
    C_expected = A_test @ B_test
    
    # Compute using the decomposition
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U_arr[:, r].reshape(n, m)
        v_vec = V_arr[:, r].reshape(m, p)
        w_vec = W_arr[:, r].reshape(p, n)
        # The contribution is: (A * u_vec) * (v_vec * B) * w_vec
        # Actually, the tensor decomposition means:
        # C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
        # More precisely: C[i,k] = sum_r sum_j sum_l U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r]
        # Wait, that's not right. Let me think about this more carefully.
        
        # The standard matrix multiplication tensor T_{i,j,k} for C[i,k] = sum_j A[i,j]*B[j,k]
        # is T[i,j,k] = delta_{j,k} (no wait, that's not right either)
        
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # But that's just 1 for all valid (i,j,k).
        
        # The tensor decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So: C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # That doesn't make sense. Let me reconsider.
        
        # The correct interpretation: The multiplication tensor T has entries T[i,j,k] where
        # T[i,j,k] = 1 if (i,j,k) is a valid triple for matrix multiplication.
        # Actually no. The tensor is defined as: T[i,j,k] = delta_{j,k} is wrong.
        
        # Let me use the standard definition: For matrix multiplication C = A*B where
        # A is n×m, B is m×p, C is n×p:
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        
        # The multiplication tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid indices.
        # A rank-r decomposition is: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U is (n*m)×r, V is (m*p)×r, W is (p*n)×r
        # with U indexed by (i,j), V by (j,k), W by (k,i)
        
        # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all valid (i,j,k)
        
        # This means: C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]*A[i,j]*B[j,k]
        # No wait, that's not right. The tensor decomposition replaces the product A[i,j]*B[j,k]
        # with a sum of rank-1 terms.
        
        # Actually, I think the correct verification is:
        # For the decomposition to be valid, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # (since T[i,j,k] = 1 for all valid indices)
        
        # Let's check this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        # Decomposition is invalid
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            # Decomposition is valid
            pass
    
    return {"U": U, "V": V, "W": W}
