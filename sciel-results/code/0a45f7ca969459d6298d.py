import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U is of shape (n*m, rank)
    # V is of shape (m*p, rank)
    # W is of shape (p*n, rank)
    # 
    # The matrix multiplication tensor T is defined such that:
    # (A * B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This can be viewed as a tensor T_{i,j,k} = delta_{i,j} delta_{j,k} is wrong.
    # The bilinear map is (A, B) -> C.
    # The tensor is T_{i, j, k} which is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the decomposition U_{(i,j), c} V_{(j,k), c} W_{(k,i), c}
    # Wait, the interface specifies:
    # U shape: (n*m, rank) -> index (i*m+j)
    # V shape: (m*p, rank) -> index (j*p+k)
    # W shape: (p*n, rank) -> index (k*n+i)
    # 
    # The contraction is sum_c U[(i,j), c] * V[(j,k), c] * W[(k,i), c]
    # This must equal delta_{i,i'} delta_{k,k'} (where the output index is (k,i) and input is fixed? No.)
    # 
    # Let's trace the indices.
    # Output C has indices (i, k).
    # The tensor T_{i, j, k} is non-zero when we multiply A_{i, j} and B_{j, k} to get C_{i, k}.
    # Actually, the standard decomposition for matmul is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a sum of rank-1 terms? No, it's a bilinear form.
    # The tensor T has entries T_{i, j, k} = 1.
    # We want to write T = sum_c u_c \otimes v_c \otimes w_c
    # where (u_c \otimes v_c \otimes w_c)_{i, j, k} = u_{c,i} v_{c,j} w_{c,k}
    # But the interface groups indices differently.
    # U index: a = i*m + j
    # V index: b = j*p + k
    # W index: c_idx = k*n + i
    #
    # The term in the sum is U[a, col] * V[b, col] * W[c_idx, col]
    # We need this to reconstruct the multiplication.
    # The multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be seen as a tensor T_{a, b, c_idx} where:
    # a = (i,j), b = (j,k), c_idx = (k,i).
    # T_{a, b, c_idx} = 1 if the indices match consistently for the product term A_{i,j}B_{j,k} -> C_{i,k}.
    # Specifically, for a fixed j, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So for each j, and for all i, k, we have a term?
    # No, the sum is over j.
    # The bilinear form is sum_j A_{i,j} B_{j,k} \delta_{i,i} \delta_{k,k} ? No.
    # 
    # Let's look at the schoolbook decomposition provided in the seed.
    # For each i, j, k:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    #
    # This creates a term for every triplet (i,j,k).
    # Sum over col: U[a, col] V[b, col] W[c, col]
    # Non-zero only if a, b, c correspond to the same (i,j,k).
    # If a = i*m+j, b = j*p+k, c = k*n+i, then the product is 1.
    # This effectively selects the term A_{i,j} B_{j,k} and assigns it to C_{i,k} (via W index k*n+i).
    # Since there is exactly one column for each (i,j,k), the sum over columns picks out the specific term.
    # The resulting value at output index (k,n+i) is sum_{j} (term for i,j,k) = sum_j A_{i,j} B_{j,k}.
    # This is correct.
    #
    # Rank is n*m*p.
    # 
    # I will generate these matrices using NumPy for speed and code distinctness.
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    # Convert to lists of lists to match the expected return type if strict, 
    # though numpy arrays are often accepted if convertible. 
    # The prompt says "factor matrices... with integer/half-integer entries".
    # Returning numpy arrays is usually fine, but lists are safer for strict JSON-like interfaces.
    # However, for rank 36, lists are small.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
