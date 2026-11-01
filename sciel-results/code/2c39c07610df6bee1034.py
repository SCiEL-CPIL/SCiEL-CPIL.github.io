import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Strassen's 7-term algorithm for 2x2 matrix multiplication.
    # For matrices A (2x2) and B (2x2), C = A*B with 7 products:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In the tensor decomposition framework:
    #   T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m+j (row i, col j of A), b = j*p+k (row j, col k of B),
    #       c = k*n+i (row k, col i of C)  [official trace convention: W[k*n+i, r]]
    #
    # For the full 2x2x8 problem, we have 8 independent 2x2 multiplications
    # (one per column k of B). Each uses the same A but a different 2x2 slice of B.
    # We replicate the Strassen factors for each k, giving 7*8 = 56 factors?
    # No! The A-side factors are shared across all k. But in the tensor framework,
    # U is indexed by (i,j) only, V by (j,k), W by (k,i).
    # 
    # Actually, let's think more carefully. The tensor has 32 non-zero entries:
    # (i,j,k) for i in {0,1}, j in {0,1}, k in {0,...,7}.
    # 
    # The schoolbook uses 32 rank-1 terms. Strassen reduces the 2x2 part to 7,
    # so for each of the 8 k-values, we need 7 terms. But the U factors for
    # different k's can be the same (they depend only on i,j), while V and W
    # differ. However, in the tensor decomposition, each rank-1 term has its own
    # U, V, W vectors. So we need 7*8 = 56 terms? That's worse than 32.
    #
    # Wait - let me reconsider. The key is that the A-side is shared.
    # In Strassen, the 7 "products" involve linear combinations of A entries
    # multiplied by linear combinations of B entries. For the full 2x2x8:
    #   A is 2x2 (4 entries), B is 2x8 (16 entries), C is 2x8 (16 entries).
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Strassen for 2x2x2 reduces 4 products to 7 bilinear forms.
    # For 2x2x8, we can view it as 8 independent 2x2x2 problems (one per k),
    # but they share the same A. The standard approach: apply Strassen to each
    # 2x2 block of B independently. This gives 7*8 = 56 terms, which is worse.
    #
    # Alternative: Can we do better by exploiting the shared A structure?
    # The tensor M_{2,2,8} has a specific structure. Let's think about it as:
    #   T[i,j,k] = 1 for all i,j,k (in the coupled index sense).
    #
    # Actually, the correct way to think about this: the matrix multiplication
    # tensor M_{n,m,p} has rank at most n*m*p (schoolbook). For M_{2,2,8},
    # schoolbook rank is 32. The question is whether we can do better.
    #
    # One approach: use Strassen's 7-term formula for the 2x2 part, but since
    # the B matrix has 8 columns, we need to handle all 8 columns. The 7 Strassen
    # products each involve a 2x2 sub-problem. If we apply them per column, we get
    # 56 terms. But we can potentially share the A-side linear combinations.
    #
    # Let me try a different approach: directly construct a rank-28 decomposition.
    # 
    # Actually, the simplest correct approach: use the schoolbook (rank 32) as a
    # fallback, but first try to construct a valid lower-rank decomposition.
    #
    # Let me try the most straightforward correct construction:
    # For each (i, j, k), we have one term. That's 32 terms (schoolbook).
    # To reduce, we need to combine terms.
    #
    # Let's try a direct construction with rank 28:
    # Use Strassen for the 2x2x2 core, extended to 2x2x8.
    # 
    # Key insight: The 2x2x8 multiplication can be seen as computing 8 output
    # entries C[i,k] for i in {0,1}, k in {0,...,7}, each requiring a 2x2 dot
    # product. Strassen reduces each 2x2 dot product from 4 to 7 bilinear forms,
    # but that's an increase! Strassen is for 2x2 MATRIX multiplication (4 outputs
    # from 4*2=8 input pairs reduced to 7 products). For a single dot product
    # (1 output from 2 input pairs), you can't do better than 2.
    #
    # So the right decomposition: for each of the 8 columns k, compute
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  (2 products)
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  (2 products)
    # Total: 8 * 2 * 2 = 32 products. That's schoolbook.
    #
    # To reduce below 32, we need to share computation across columns.
    # One idea: group columns. E.g., process 2 columns at a time using
    # Strassen's 2x2 matrix multiplication (4 outputs from 7 products instead
    # of 8). For 8 columns, that's 4 groups of 2 columns, each using 7 products
    # = 28 total. This gives rank 28!
    #
    # Let's verify: For columns (2q, 2q+1), we compute:
    #   C[:, 2q:2q+2] = A * B[:, 2q:2q+2]
    # This is a 2x2 * 2x2 = 2x2 matrix multiplication, which Strassen does in 7 products.
    # 4 such groups * 7 = 28 products.
    #
    # Now let's construct the tensor decomposition for this rank-28 scheme.
    #
    # For group q (q = 0, 1, 2, 3), columns k1=2q, k2=2q+1:
    #   B_sub = [[B[0,k1], B[0,k2]],
    #            [B[1,k1], B[1,k2]]]
    #   C_sub = A * B_sub = [[C[0,k1], C[0,k2]],
    #                         [C[1,k1], C[1,k2]]]
    #
    # Strassen's 7 products for this sub-problem:
    #   M1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2])
    #   M2 = (A[1,0]+A[1,1]) * B[0,k1]
    #   M3 = A[0,0] * (B[0,k2]-B[1,k2])
    #   M4 = A[1,1] * (B[1,k1]-B[0,k1])
    #   M5 = (A[0,0]+A[0,1]) * B[1,k2]
    #   M6 = (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2])
    #   M7 = (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2])
    #
    # Outputs:
    #   C[0,k1] = M1 + M4 - M5 + M7
    #   C[0,k2] = M3 + M5
    #   C[1,k1] = M2 + M4
    #   C[1,k2] = M1 - M2 + M3 + M6
    #
    # In the tensor framework, each product M_r corresponds to a rank-1 term.
    # For product r in group q:
    #   U factor: linear combination of A entries
    #   V factor: linear combination of B entries (for columns k1, k2)
    #   W factor: contributes to C entries (for columns k1, k2)
    #
    # Let's define the factors precisely.
    # U has shape (n*m, rank) = (4, 28). Index: i*m+j = i*2+j.
    # V has shape (m*p, rank) = (16, 28). Index: j*p+k = j*8+k.
    # W has shape (p*n, rank) = (16, 28). Index: k*n+i = k*2+i.
    #
    # For each group q and Strassen product r (0..6), we create one rank-1 term.
    # Term index: t = q*7 + r.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for each product:
    # Product r: (coeff_A, coeff_B, coeff_C)
    # where coeff_A is a 2x2 matrix giving the linear combo of A,
    #       coeff_B is a 2x2 matrix giving the linear combo of B_sub,
    #       coeff_C gives how this product contributes to C_sub.
    
    # For product r, the bilinear form is:
    #   M_r = (sum_{i,j} a_r[i,j] * A[i,j]) * (sum_{j',k'} b_r[j',k'] * B_sub[j',k'])
    # and it contributes to C_sub[i,k'] with coefficient c_r[i,k'].
    
    # Let's define each product:
    # r=0 (M1): A_combo = [[1,0],[0,1]], B_combo = [[1,0],[0,1]]
    #   M1 = (A00+A11)*(B00+B11)  [using sub-matrix indices]
    #   Wait, let me re-derive. In Strassen:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   So A_combo picks A00 and A11 (coeff 1 each), B_combo picks B00 and B11.
    #   Contribution to C: C00 += M1, C11 += M1
    
    # Let me be very explicit with 2x2 sub-matrix indices (0,0),(0,1),(1,0),(1,1)
    # corresponding to (i,j) for A and (j,k) for B_sub.
    
    # Product definitions: (A_coeffs dict, B_coeffs dict, C_contrib dict)
    # A_coeffs: {(i,j): coeff}
    # B_coeffs: {(j,k_sub): coeff} where k_sub is 0 or 1 (local column)
    # C_contrib: {(i, k_sub): coeff}
    
    strassen_products = [
        # M1
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients
            # B_sub[j, k_sub] corresponds to B[j, k1 + k_sub]
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions
            # C_sub[i, k_sub] corresponds to C[i, k1 + k_sub]
            # W index: k*n + i = k_global * n + i
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verify the decomposition
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    if not np.allclose(T_computed, T_target, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Convert to lists for the return interface
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
