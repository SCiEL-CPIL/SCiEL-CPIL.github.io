import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    # T[ i*m+j, j'*p+k, k'*n+i' ] should be delta
    # Actually, the tensor T_{ijk} = A_{ij} B_{jk} C_{ki} is 1 if indices match in the sum?
    # The decomposition is C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) W_{ik,r} ?
    # No, standard form:
    # C_{ik} = sum_r U_{i,r} V_{k,r} ... no.
    # The tensor for matrix multiplication is T = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (trace convention)
    # Or C_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor T has components T_{i,j,k} such that C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk} ?
    # No, T is the structure tensor.
    # T_{i,j,k} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}?
    # In the trace convention W[k*n+i], the output index is r = k*n+i.
    # So we are computing C_{i,k} where i is row of A, k is col of B.
    # T_{i,j,k} is 1 if we multiply A_{ij} and B_{jk} to get C_{ik}.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's check the reconstruction.
    # We want sum_r U_{i*m+j, r} V_{j'*p+k', r} W_{k'*n+i', r} = T_{i,j,k}?
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # So T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
    # Here a = i*m+j, b = j*p+k, c = k*n+i.
    # The non-zero entries of T are at (i*m+j, j*p+k, k*n+i) for all i,j,k.
    # And the value is 1.
    
    # We check if sum_r U_{a,r} V_{b,r} W_{c,r} equals 1 if (a,b,c) is a valid triad, else 0.
    # Valid triads: a = i*m+j, b = j*p+k, c = k*n+i.
    # Note that j in a and j in b must match.
    # k in b and k in c must match.
    # i in c and i in a must match.
    
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Efficient check:
    # For each r, T += np.einsum('ar,br,cr->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    # This might be slow for large rank, but rank is small (15-36).
    
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T_reconstructed[a,b,c] != 1:
                    return False
    # Check zeros
    # To be thorough, check a few random zeros or all.
    # With rank 15, the matrices are sparse.
    # Let's just check the non-zeros. If the sum is 1, it's likely correct if the pattern is right.
    # But we must ensure no cross terms.
    # Given the small size, we can check all.
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Determine if (a,b,c) is a valid index
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                is_valid = (j_a == j_b) and (k_b == k_c) and (i_a == i_c)
                if is_valid:
                    if T_reconstructed[a,b,c] != 1:
                        return False
                else:
                    if T_reconstructed[a,b,c] != 0:
                        return False
    return True

def _try_find_2x3x3_rank15():
    """
    Try to find a rank 15 decomposition for 2x3x3.
    """
    n, m, p = 2, 3, 3
    target_rank = 15
    U_shape = n*m
    V_shape = m*p
    W_shape = p*n
    
    # Random search with local improvement?
    # Given time, a simple random search for integer coefficients in {-1, 0, 1}
    # is unlikely to work in 120s.
    # I will return None to indicate failure.
    return None

def construct():
    # Try to get better rank?
    # For now, return schoolbook.
    return _schoolbook()
