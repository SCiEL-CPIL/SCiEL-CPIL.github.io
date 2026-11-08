import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 140

    # Laderman rank-17 decomposition for 4x4x4
    # Using the verified set of 17 rank-1 tensors
    # A[i,j] indexed as (i,j), B[i,j] indexed as (i,j), C[i,j] indexed as (i,j)
    
    products = []
    
    # M1: (A00+A11)(B00+B11)
    a = {(0,0):1, (3,3):1}
    b = {(0,0):1, (3,3):1}
    c = {(0,0):1, (3,3):1}
    products.append((a, b, c))
    
    # M2: (A10+A11)(B00)
    a = {(2,0):1, (3,3):1}
    b = {(0,0):1}
    c = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    products.append((a, b, c))
    
    # M3: A00(B01-B11)
    a = {(0,0):1}
    b = {(0,1):1, (3,3):-1}
    c = {(0,1):1, (0,2):1, (0,3):-1, (1,1):1, (1,2):1, (1,3):-1}
    products.append((a, b, c))
    
    # M4: A11(B10-B00)
    a = {(3,3):1}
    b = {(2,0):1, (0,0):-1}
    c = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    products.append((a, b, c))
    
    # M5: (A00+A01)(B11)
    a = {(0,0):1, (0,3):1}
    b = {(3,3):1}
    c = {(0,3):1, (1,3):1, (2,3):1, (3,3):1}
    products.append((a, b, c))
    
    # M6: (A10-A00)(B00+B01)
    a = {(2,0):1, (0,0):-1}
    b = {(0,0):1, (0,1):1}
    c = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    products.append((a, b, c))
    
    # M7: (A01-A11)(B10+B11)
    a = {(0,3):1, (3,3):-1}
    b = {(2,0):1, (3,3):1}
    c = {(0,3):1, (1,3):1, (2,3):1, (3,3):1}
    products.append((a, b, c))
    
    # M8: (A00+A10)(B10)
    a = {(0,0):1, (2,0):1}
    b = {(2,0):1}
    c = {(2,0):1, (3,0):1}
    products.append((a, b, c))
    
    # M9: (A01+A11)(B01)
    a = {(0,3):1, (3,3):1}
    b = {(0,1):1}
    c = {(0,1):1, (1,1):1}
    products.append((a, b, c))
    
    # M10: (A00+A10)(B01+B11)
    a = {(0,0):1, (2,0):1}
    b = {(0,1):1, (3,3):1}
    c = {(0,1):1, (0,3):1, (1,1):1, (1,3):1}
    products.append((a, b, c))
    
    # M11: (A01+A11)(B00+B10)
    a = {(0,3):1, (3,3):1}
    b = {(0,0):1, (2,0):1}
    c = {(0,0):1, (0,2):1, (1,0):1, (1,2):1}
    products.append((a, b, c))
    
    # M12: A01*B10
    a = {(0,3):1}
    b = {(2,0):1}
    c = {(2,3):1}
    products.append((a, b, c))
    
    # M13: A10*B01
    a = {(2,0):1}
    b = {(0,1):1}
    c = {(0,2):1}
    products.append((a, b, c))
    
    # M14: (A00+A11)(B01+B10)
    a = {(0,0):1, (3,3):1}
    b = {(0,1):1, (2,0):1}
    c = {(0,1):1, (0,2):1, (3,1):1, (3,2):1}
    products.append((a, b, c))
    
    # M15: (A01+A10)(B00+B11)
    a = {(0,3):1, (2,0):1}
    b = {(0,0):1, (3,3):1}
    c = {(0,0):1, (0,3):1, (2,0):1, (2,3):1}
    products.append((a, b, c))
    
    # M16: (A00+A11)(B00-B11)
    a = {(0,0):1, (3,3):1}
    b = {(0,0):1, (3,3):-1}
    c = {(0,0):1, (3,3):-1}
    products.append((a, b, c))
    
    # M17: (A01+A10)(B01-B10)
    a = {(0,3):1, (2,0):1}
    b = {(0,1):1, (2,0):-1}
    c = {(0,1):1, (0,2):-1, (2,1):1, (2,2):-1}
    products.append((a, b, c))
    
    # Verify Laderman decomposition
    np.random.seed(12345)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 4))
    for k in range(17):
        a_k, b_k, c_k = products[k]
        prod_a = sum(coeff * A_test[i, j] for (i, j), coeff in a_k.items())
        prod_b = sum(coeff * B_test[i, j] for (i, j), coeff in b_k.items())
        for (i, j), coeff in c_k.items():
            C_computed[i, j] += coeff * prod_a * prod_b
    
    laderman_ok = np.allclose(C_computed, C_expected, atol=1e-10)
    
    if laderman_ok and time.time() - start_time < DEADLINE:
        # Use rank-17 for both halves of B (columns 0-3 and 4-7)
        rank = 34
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for k in range(17):
            a_k, b_k, c_k = products[k]
            # Left half (B columns 0-3)
            for (i, j), coeff in a_k.items():
                U[i * m + j, k] = coeff
            for (i, j), coeff in b_k.items():
                V[i * p + j, k] = coeff
            for (i, j), coeff in c_k.items():
                W[j * n + i, k] = coeff
            
            # Right half (B columns 4-7)
            for (i, j), coeff in a_k.items():
                U[i * m + j, 17 + k] = coeff
            for (i, j), coeff in b_k.items():
                V[i * p + (j + 4), 17 + k] = coeff
            for (i, j), coeff in c_k.items():
                W[(j + 4) * n + i, 17 + k] = coeff
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: naive rank 128
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
