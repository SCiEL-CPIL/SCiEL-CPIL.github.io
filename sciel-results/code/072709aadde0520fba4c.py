import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    - Split the 2x14 matrix B into 7 blocks of 2x2 matrices.
    - Use Strassen's algorithm (rank 7) for each 2x2 * 2x2 block multiplication.
    - The total rank is 7 * 7 = 49.
    
    Interface:
    - U: shape (n*m, rank) = (4, 49)
    - V: shape (m*p, rank) = (28, 49)
    - W: shape (p*n, rank) = (28, 49)
    
    Entries are integers.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49
    
    # Initialize matrices with zeros
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    # Strassen's algorithm coefficients for 2x2 * 2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Terms: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {idx: coeff}
    # B_coeffs: dict {idx: coeff}
    # C_coeffs: dict {idx: coeff}
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # Fill U (A part)
            # U[i*m + j, col]
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            # Fill V (B part)
            # V[j*p + k, col]
            # Block t corresponds to columns 2*t and 2*t+1 of B
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            # Fill W (C part)
            # W[k*n + i, col]
            # Block t corresponds to columns 2*t and 2*t+1 of C
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff
                
    # Verification
    # Check T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 if (i,j,k) is a valid mult term, else 0
    # Actually, the tensor T is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition must satisfy: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # for all i, j, k? No.
    # The product is sum_j A[i,j] B[j,k].
    # In the tensor decomposition, the "multiplication" is sum_c (U . A) (V . B) (W . C)? No.
    # The standard check is:
    # For all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if the indices form a valid term in the sum?
    # No, the tensor T_{i,j,k} is 1 if it corresponds to a multiplication term.
    # But here we have multiple j's.
    # The correct check is that the reconstructed matrix multiplication is correct.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition provides: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j])? No.
    # U is a linear form in A. Let a_c = U[:, c] . A_vec.
    # V is a linear form in B. Let b_c = V[:, c] . B_vec.
    # W is a linear form in C.
    # The condition is: C = sum_c a_c b_c w_c^T? No.
    # The tensor is T_{i,j,k}.
    # T_{i,j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # This sum must equal 1 if the indices (i,j,k) are such that they contribute to the product?
    # Actually, the multiplication tensor is defined by T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} exists.
    # Wait, the indices are (i, j, k).
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So for a fixed (i,k), the sum over j of A_{i,j} B_{j,k} is the result.
    # The tensor decomposition expresses the *operation*.
    # The condition is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k?
    # No, that would imply A_{i,j} B_{j,k} = 1?
    # The correct identity is:
    # For any A, B, C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The decomposition says: C_{i,k} = sum_c (U_{i,*} A)_c (V_{*,k} B)_c W_{k,i,c}? No.
    # Standard tensor decomposition for matrix mult:
    # C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) W_{k,i,c}? No, indices don't match.
    
    # Correct check:
    # The tensor T has entries T_{i,j,k}.
    # T_{i,j,k} = 1 if the pair (i,j,k) is a valid term in the multiplication?
    # No, the multiplication is a map from (A,B) to C.
    # The tensor T is in A* \otimes B* \otimes C.
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}?
    # No, the term is A_{i,j} B_{j,k} -> C_{i,k}.
    # So T_{i,j,k} is non-zero only if the indices are consistent.
    # Specifically, T_{i,j,k} = 1 for all i,j,k? No.
    # The term is A_{i,j} B_{j,k}. This contributes to C_{i,k}.
    # So the tensor entry T_{i, j, k} where the third index is the output index?
    # The problem statement says:
    # U: (n*m, rank) -> U[i*m+j, c]
    # V: (m*p, rank) -> V[j*p+k, c]
    # W: (p*n, rank) -> W[k*n+i, c]
    # Reconstruction: C[i,k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] * A[i,j] * B[j,k]?
    # No, U, V, W are the factors.
    # The identity is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k?
    # Let's test with schoolbook rank 1 term for i,j,k.
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Sum = 1.
    # If we sum this over j for fixed i,k:
    # sum_j 1 = m.
    # But C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition must reproduce the sum.
    # So the condition is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    
    # Let's verify this condition with NumPy.
    # Construct the tensor T from U, V, W.
    # T[i*m+j, j*p+k, k*n+i] should be 1?
    # No, the indices are coupled.
    # We need to check:
    # For all i in n, j in m, k in p:
    # val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # val should be 1.
    
    # This check is O(n*m*p*rank) = 2*2*14*49 = 2744 operations.
    # Very fast.
    
    # However, doing it in Python loops is slow.
    # We can use NumPy einsum or broadcasting.
    
    # U: (nm, rank)
    # V: (mp, rank)
    # W: (pn, rank)
    
    # We want to check M[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Note the indices of U, V, W depend on i, j, k in a specific way.
    
    # Let's just iterate. It's only 56 triples.
    # 56 * 49 = 2744 multiplications. Trivial.
    
    valid = True
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                # sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    # Print debug info if needed
                    # print(f"Fail at i={i}, j={j}, k={k}, val={val}")
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if Strassen construction is incorrect
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    rank = col # Update rank for return? No, rank is fixed.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
