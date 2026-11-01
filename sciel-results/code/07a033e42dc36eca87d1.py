import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Deadline guard: evaluator kills at 150s wall-clock.
    # We use a tight safety margin since the construction is deterministic and fast.
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: Block-diagonal Strassen decomposition.
    # We group the 8 columns of B into 4 pairs.
    # Each pair forms a 2x2 sub-matrix B_sub.
    # We apply Strassen's 7-term algorithm to each pair independently.
    # Total rank: 4 * 7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen products definitions for 2x2 matrix multiplication.
    # For A (2x2), B_sub (2x2), C_sub (2x2):
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00      -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)      -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)      -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11      -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01)-> C11+=M6
    # M7 = (A01-A11)(B10+B11)-> C00+=M7
    
    # Coefficients are defined for local indices in the 2x2 sub-block.
    # A: (i, j) where i,j in {0,1}
    # B: (j, k_local) where j in {0,1}, k_local in {0,1}
    # C: (i, k_local) where i in {0,1}, k_local in {0,1}
    
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
    
    # Construct the factor matrices
    for q in range(p // 2):  # 4 groups
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients
            # U index: i*m + j
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients
            # V index: j*p + k_global
            # k_global = k1 + k_sub
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions
            # W index: k_global*n + i  (official trace convention)
            # k_global = k1 + k_sub
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Lightweight verification
    # Check that matrices are not empty
    if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
        # Fallback to schoolbook if something went wrong (should not happen)
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Optional: Quick sanity check with random matrices if time permits
    # This is very fast (microseconds) and ensures correctness.
    if time.time() < deadline - 1.0:
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        # Compute C using the decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can use einsum for efficiency
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W is indexed by (k, i) in the tensor, so W_r[k, i, r]
        # Let's be careful with the W reshaping.
        # W has shape (p*n, rank). Index is k*n + i.
        # So W.reshape(p, n, rank) gives W_r[k, i, r].
        # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
        # C[i, k] = sum_j T[i, j, k]
        
        # Efficient computation:
        # C[i, k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is not a standard matrix product.
        # Let's just do it with loops for the small dimensions to be safe, or use einsum.
        # np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r)
        C_computed = np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback if verification fails
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
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
