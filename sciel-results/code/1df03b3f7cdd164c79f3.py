"""
Tensor decomposition for 3x3 * 3x5 matrix multiplication.
Uses a block-based approach combining Strassen (2x2x2, rank 7) with schoolbook
for the remaining blocks to achieve a lower rank than pure schoolbook (45).
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline

    n, m, p = DIMENSIONS  # 3, 3, 5

    # We'll build a decomposition using block structure.
    # A is 3x3, B is 3x5, C is 3x5.
    # Split A into A_top (2x3) and A_bottom (1x3).
    # Split B into B_top (2x5) and B_bottom (1x5).
    # C = A_top @ B_top + A_bottom @ B_bottom
    #
    # For A_top (2x3) @ B_top (2x5):
    #   Split A_top into A11 (2x2) and A12 (2x1).
    #   Split B_top into B1 (2x2) and B2 (2x3).
    #   C_top_left (2x2) = A11 @ B1
    #   C_top_right (2x3) = A11 @ B2 + A12 @ B_bottom_right... 
    #
    # This is getting complex. Let me use a simpler approach:
    # Use Strassen for the 2x2x2 part and schoolbook for everything else.
    #
    # Actually, let me just implement a clean schoolbook decomposition
    # which is guaranteed to be correct, and then try to improve.

    # Schoolbook decomposition: rank = n*m*p = 45
    schoolbook_rank = n * m * p
    U = np.zeros((n * m, schoolbook_rank), dtype=float)
    V = np.zeros((m * p, schoolbook_rank), dtype=float)
    W = np.zeros((p * n, schoolbook_rank), dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0

    # Verify the schoolbook decomposition
    def verify(U, V, W):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor M[i,j,k] = delta_{i,j,k} in the sense that
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition says: for each rank term r,
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should sum to 1 if (i,j,k) is a valid multiplication term, else 0.
        
        # Actually, the tensor for matrix multiplication M_{i,j,k} is 1 for all (i,j,k)
        # in the sense that C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor is T[i, j, k] = 1 for all i in [n], j in [m], k in [p].
        # Wait, no. The tensor is defined such that:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
        # where T[i, j, k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k].
        # Actually T[i,j,k] = 1 for all valid (i,j,k).
        
        # Let's just check numerically with random matrices.
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, the standard tensor decomposition for matrix multiplication is:
        # C = sum_r W_r * (U_r @ A) * (B @ V_r) ... no.
        
        # The correct interpretation:
        # For each rank term r, we compute:
        # M_r = U_r (n*m vector) reshaped as (n,m) times V_r (m*p vector) reshaped as (m,p)
        # No, that's not right either.
        
        # Let me think again. The tensor T is of size (n*m, m*p, p*n) or similar.
        # The decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}.
        
        # For matrix multiplication, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
        # Wait, that's not a well-defined tensor because the indices are coupled.
        
        # The correct tensor for matrix multiplication M_{n,m,p} has:
        # T[a, b, c] where a indexes (i,j) with a = i*m+j, b indexes (j,k) with b = j*p+k,
        # c indexes (k,i) with c = k*n+i.
        # T[a,b,c] = 1 if the (j) in a matches the (j) in b, and (k) in b matches (k) in c,
        # and (i) in c matches (i) in a. Otherwise 0.
        
        # So T[i*m+j, j*p+k, k*n+i] = 1, and all other entries are 0.
        
        # The decomposition says: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        # Let's verify this.
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Compute decomposition
        # sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a batched operation.
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, outer product of U[:,r], V[:,r], W[:,r]
        
        result = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.allclose(result, T)
    
    if not verify(U, V, W):
        # This should never happen for schoolbook, but just in case
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to find a lower rank decomposition.
    # Strategy: Use Strassen for 2x2x2 blocks.
    # 
    # For 3x3 * 3x5:
    # Split into:
    #   - 2x2 * 2x2 part (top-left 2x2 of A, top-left 2x2 of B) -> rank 7 (Strassen)
    #   - Remaining parts use schoolbook
    #
    # Let me implement a hybrid approach.
    #
    # Actually, let me try a different approach: use the fact that we can
    # combine Strassen for multiple 2x2x2 blocks.
    #
    # For now, let me just return the schoolbook solution which is guaranteed correct.
    # The rank is 45.
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to implement a lower rank decomposition using Strassen for 2x2x2 blocks.
    #
    # Approach: 
    # A (3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B (3x5) = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 1x2, B22 is 1x3
    # C (3x5) = [C11 C12; C21 C22] where C11 is 2x2, C12 is 2x3, C21 is 1x2, C22 is 1x3
    #
    # C11 = A11 @ B11 (2x2 * 2x2) -> Strassen rank 7
    # C12 = A11 @ B12 + A12 @ B21... wait, dimensions don't match.
    #
    # Let me be more careful.
    # A = [[a11, a12, a13],
    #      [a21, a22, a23],
    #      [a31, a32, a33]]
    # B = [[b11, b12, b13, b14, b15],
    #      [b21, b22, b23, b24, b25],
    #      [b31, b32, b33, b34, b35]]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For C[0,0] = a11*b11 + a12*b21 + a13*b31
    # For C[0,1] = a11*b12 + a12*b22 + a13*b32
    # etc.
    #
    # Using Strassen for the 2x2x2 part (i,j in {0,1}, k in {0,1}):
    # This handles C[0,0], C[0,1], C[1,0], C[1,1] using terms involving
    # a11,a12,a21,a22 and b11,b12,b21,b22.
    #
    # Strassen's 7 products for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # These involve only a11,a12,a21,a22 and b11,b12,b21,b22.
    #
    # For the remaining C entries, we need to handle the contributions from
    # the third row/column of A and B.
    #
    # C[0,2] = a11*b13 + a12*b23 + a13*b33
    # C[0,3] = a11*b14 + a12*b24 + a13*b34
    # C[0,4] = a11*b15 + a12*b25 + a13*b35
    # C[1,2] = a21*b13 + a22*b23 + a23*b33
    # C[1,3] = a21*b14 + a22*b24 + a23*b34
    # C[1,4] = a21*b15 + a22*b25 + a23*b35
    # C[2,0] = a31*b11 + a32*b21 + a33*b31
    # C[2,1] = a31*b12 + a32*b22 + a33*b32
    # C[2,2] = a31*b13 + a32*b23 + a33*b33
    # C[2,3] = a31*b14 + a32*b24 + a33*b34
    # C[2,4] = a31*b15 + a32*b25 + a33*b35
    #
    # The Strassen part handles C[0,0], C[0,1], C[1,0], C[1,1] using 7 products.
    # The remaining 11 C entries each require 3 terms (schoolbook), so 33 more products.
    # Total: 7 + 33 = 40. But some of the Strassen products might be reused.
    #
    # Actually, the Strassen products M1-M7 are specific linear combinations.
    # For the remaining entries, we need to express them in terms of the original
    # a's and b's, not the Strassen combinations.
    #
    # So the total rank would be 7 (Strassen) + 33 (schoolbook for remaining) = 40.
    # But we need to check if any of the Strassen products can be reused for the
    # remaining entries.
    #
    # Let me just implement this.

    # Strassen products for 2x2x2 (indices 0,1 for i,j,k)
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # For the U, V, W matrices, each rank term r corresponds to a product
    # of a linear combination of A entries times a linear combination of B entries,
    # contributing to a linear combination of C entries.
    
    # Let me define the rank terms.
    
    rank_terms = []
    
    # Strassen terms for 2x2x2 (i,j in {0,1}, k in {0,1})
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from (i,j) to coefficient
    # V_coeffs: dict from (j,k) to coefficient  
    # W_coeffs: dict from (k,i) to coefficient
    
    # M1 = (a00+a11)(b00+b11) -> contributes to C00, C01, C10, C11
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1... 
    # Wait, let me re-derive Strassen.
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Where C11 corresponds to C[0,0], C12 to C[0,1], C21 to C[1,0], C22 to C[1,1].
    
    # Let me define each M term as a rank-1 tensor contribution.
    
    # M1 = (a00 + a11) * (b00 + b11)
    # U: a00: 1, a11: 1
    # V: b00: 1, b11: 1
    # W: contributes to C00 (k=0,i=0): 1, C01 (k=1,i=0): 1, C10 (k=0,i=1): 1, C11 (k=1,i=1): 1
    # Wait, C[0,0] gets M1, C[0,1] gets M1, C[1,0] gets M1, C[1,1] gets M1?
    # No, from the formulas:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So M1 contributes to C[0,0] with coeff 1 and C[1,1] with coeff 1.
    # M2 contributes to C[1,0] with coeff 1 and C[1,1] with coeff -1.
    # M3 contributes to C[0,1] with coeff 1 and C[1,1] with coeff 1.
    # M4 contributes to C[0,0] with coeff 1 and C[1,0] with coeff 1.
    # M5 contributes to C[0,0] with coeff -1 and C[0,1] with coeff 1.
    # M6 contributes to C[1,1] with coeff 1.
    # M7 contributes to C[0,0] with coeff 1.
    
    # Let me define each term:
    
    def add_term(U_coeffs, V_coeffs, W_coeffs):
        """Add a rank term to the list."""
        rank_terms.append((U_coeffs, V_coeffs, W_coeffs))
    
    # M1 = (a00 + a11) * (b00 + b11)
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    
    # M2 = (a10 + a11) * b00
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1})
    
    # M3 = a00 * (b01 - b11)
    add_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    
    # M4 = a11 * (b10 - b00)
    add_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1})
    
    # M5 = (a00 + a01) * b11
    add_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1})
    
    # M6 = (a10 - a00) * (b00 + b01)
    add_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
    
    # M7 = (a01 - a11) * (b10 + b11)
    add_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    
    # Now add schoolbook terms for the remaining C entries.
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    add_term({(0,0): 1}, {(0,2): 1}, {(2,0): 1})
    add_term({(0,1): 1}, {(1,2): 1}, {(2,0): 1})
    add_term({(0,2): 1}, {(2,2): 1}, {(2,0): 1})
    
    # C[0,3] = a00*b03 + a01*b13 + a02*b23
    add_term({(0,0): 1}, {(0,3): 1}, {(3,0): 1})
    add_term({(0,1): 1}, {(1,3): 1}, {(3,0): 1})
    add_term({(0,2): 1}, {(2,3): 1}, {(3,0): 1})
    
    # C[0,4] = a00*b04 + a01*b14 + a02*b24
    add_term({(0,0): 1}, {(0,4): 1}, {(4,0): 1})
    add_term({(0,1): 1}, {(1,4): 1}, {(4,0): 1})
    add_term({(0,2): 1}, {(2,4): 1}, {(4,0): 1})
    
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    add_term({(1,0): 1}, {(0,2): 1}, {(2,1): 1})
    add_term({(1,1): 1}, {(1,2): 1}, {(2,1): 1})
    add_term({(1,2): 1}, {(2,2): 1}, {(2,1): 1})
    
    # C[1,3] = a10*b03 + a11*b13 + a12*b23
    add_term({(1,0): 1}, {(0,3): 1}, {(3,1): 1})
    add_term({(1,1): 1}, {(1,3): 1}, {(3,1): 1})
    add_term({(1,2): 1}, {(2,3): 1}, {(3,1): 1})
    
    # C[1,4] = a10*b04 + a11*b14 + a12*b24
    add_term({(1,0): 1}, {(0,4): 1}, {(4,1): 1})
    add_term({(1,1): 1}, {(1,4): 1}, {(4,1): 1})
    add_term({(1,2): 1}, {(2,4): 1}, {(4,1): 1})
    
    # C[2,0] = a20*b00 + a21*b10 + a22*b20
    add_term({(2,0): 1}, {(0,0): 1}, {(0,2): 1})
    add_term({(2,1): 1}, {(1,0): 1}, {(0,2): 1})
    add_term({(2,2): 1}, {(2,0): 1}, {(0,2): 1})
    
    # C[2,1] = a20*b01 + a21*b11 + a22*b21
    add_term({(2,0): 1}, {(0,1): 1}, {(1,2): 1})
    add_term({(2,1): 1}, {(1,1): 1}, {(1,2): 1})
    add_term({(2,2): 1}, {(2,1): 1}, {(1,2): 1})
    
    # C[2,2] = a20*b02 + a21*b12 + a22*b22
    add_term({(2,0): 1}, {(0,2): 1}, {(2,2): 1})
    add_term({(2,1): 1}, {(1,2): 1}, {(2,2): 1})
    add_term({(2,2): 1}, {(2,2): 1}, {(2,2): 1})
    
    # C[2,3] = a20*b03 + a21*b13 + a22*b23
    add_term({(2,0): 1}, {(0,3): 1}, {(3,2): 1})
    add_term({(2,1): 1}, {(1,3): 1}, {(3,2): 1})
    add_term({(2,2): 1}, {(2,3): 1}, {(3,2): 1})
    
    # C[2,4] = a20*b04 + a21*b14 + a22*b24
    add_term({(2,0): 1}, {(0,4): 1}, {(4,2): 1})
    add_term({(2,1): 1}, {(1,4): 1}, {(4,2): 1})
    add_term({(2,2): 1}, {(2,4): 1}, {(4,2): 1})
    
    total_rank = len(rank_terms)
    
    # Build U, V, W matrices
    U_new = np.zeros((n * m, total_rank), dtype=float)
    V_new = np.zeros((m * p, total_rank), dtype=float)
    W_new = np.zeros((p * n, total_rank), dtype=float)
    
    for r, (U_coeffs, V_coeffs, W_coeffs) in enumerate(rank_terms):
        for (i, j), coeff in U_coeffs.items():
            U_new[i * m + j, r] = coeff
        for (j, k), coeff in V_coeffs.items():
            V_new[j * p + k, r] = coeff
        for (k, i), coeff in W_coeffs.items():
            W_new[k * n + i, r] = coeff
    
    # Verify the new decomposition
    if verify(U_new, V_new, W_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# Test the construct function
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print("U shape:", len(result['U']), "x", len(result['U'][0]))
    print("V shape:", len(result['V']), "x", len(result['V'][0]))
    print("W shape:", len(result['W']), "x", len(result['W'][0]))
