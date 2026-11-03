import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def _construct_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=1e-9)

def _strassen_decomposition(n, m, p, k_offset=0):
    """
    Returns U, V, W for 3x3x3 Strassen algorithm.
    k_offset shifts the k index in b and c.
    """
    if n != 3 or m != 3 or p != 3:
        return None
    
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A20+A22)B02
    # M3 = A00(B02-B22)
    # M4 = A22(B20-B00)
    # M5 = (A00+A10)B20
    # M6 = (A12+A22)(B00+B10)
    # M7 = (A01+A21)(B22+B12)
    
    # Indices: a = i*3+j, b = j*3+k, c = k*3+i
    # Note: p=3 here for the sub-problem
    
    def add_term(u_indices, v_indices, w_indices, sign=1):
        # This is not how it works, we need to build the factors.
        pass

    # Explicit factors for Strassen
    # We define the 7 rank-1 tensors.
    # Each term r corresponds to U[:,r], V[:,r], W[:,r]
    
    # M1: (A00+A22)(B00+B22)
    # A part: A00, A22
    # B part: B00, B22
    # C part: C00 += M1, C02 += M1, C20 += M1, C22 += M1 (Wait, C indices depend on i,k)
    # C00 (i=0,k=0): A00 B00. M1 contributes A00 B00.
    # C02 (i=0,k=2): A00 B02 + ... M1 contributes A00 B00? No.
    # Let's trace M1 = (A00+A22)(B00+B22) = A00B00 + A00B22 + A22B00 + A22B22.
    # C00 = A00B00.
    # C02 = A00B02 + A01B12 + A02B22.
    # C20 = A20B00 + A21B10 + A22B20.
    # C22 = A20B02 + A21B12 + A22B22.
    
    # This manual construction is error prone. I will use the known integer coefficients.
    
    # U factors (A indices)
    # V factors (B indices)
    # W factors (C indices)
    
    # r=0: M1
    U[:, 0] = [1, 0, 0, 0, 0, 0, 0, 0, 1] # A00, A22
    V[:, 0] = [1, 0, 0, 0, 0, 0, 0, 0, 1] # B00, B22 (j=0,k=0; j=2,k=2)
    # W indices: c = k*3+i
    # C00 (k=0,i=0) -> 0
    # C02 (k=2,i=0) -> 2*3+0=6
    # C20 (k=0,i=2) -> 0*3+2=2
    # C22 (k=2,i=2) -> 2*3+2=8
    W[:, 0] = [0, 0, 1, 0, 0, 0, 1, 0, 1] # Wait, C00 is 1?
    # M1 = A00B00 + A00B22 + A22B00 + A22B22
    # C00 gets A00B00.
    # C02 gets A00B22.
    # C20 gets A22B00.
    # C22 gets A22B22.
    # So W[0]=1, W[6]=1, W[2]=1, W[8]=1.
    W[:, 0] = [1, 0, 1, 0, 0, 0, 1, 0, 1]
    
    # r=1: M2 = (A20+A22)B02
    U[:, 1] = [0, 0, 0, 0, 0, 0, 1, 0, 1] # A20, A22
    V[:, 1] = [0, 0, 0, 0, 0, 1, 0, 0, 0] # B02 (j=0,k=2)
    # C: A20B02 -> C22 (i=2,k=2) -> 8. A22B02 -> C22? No.
    # C22 = A20B02 + A21B12 + A22B22.
    # So M2 contributes to C22.
    W[:, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    # r=2: M3 = A00(B02-B22)
    U[:, 2] = [1, 0, 0, 0, 0, 0, 0, 0, 0] # A00
    V[:, 2] = [0, 0, 1, 0, 0, 0, 0, 0, -1] # B02 (1), B22 (-1)
    # C: A00B02 -> C02 (6). A00B22 -> C02? No, C02 has A02B22.
    # C02 = A00B02 + A01B12 + A02B22.
    # So M3 contributes to C02.
    W[:, 2] = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    
    # r=3: M4 = A22(B20-B00)
    U[:, 3] = [0, 0, 0, 0, 0, 0, 0, 0, 1] # A22
    V[:, 3] = [0, 0, -1, 0, 0, 0, 0, 0, 1] # B00 (-1), B20 (1)
    # C: A22B20 -> C20 (2). A22B00 -> C20? No, C20 has A20B00.
    # C20 = A20B00 + A21B10 + A22B20.
    # So M4 contributes to C20.
    W[:, 3] = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    # r=4: M5 = (A00+A10)B20
    U[:, 4] = [1, 0, 0, 0, 0, 0, 0, 1, 0] # A00, A10
    V[:, 4] = [0, 0, 0, 0, 0, 0, 0, 0, 1] # B20 (j=2,k=0)
    # C: A00B20 -> C00 (0). A10B20 -> C10 (3).
    W[:, 4] = [1, 0, 0, 1, 0, 0, 0, 0, 0]
    
    # r=5: M6 = (A12+A22)(B00+B10)
    U[:, 5] = [0, 0, 0, 0, 0, 1, 0, 0, 1] # A12, A22
    V[:, 5] = [1, 0, 0, 1, 0, 0, 0, 0, 0] # B00, B10
    # C: A12B00 -> C12 (5). A12B10 -> C12? No.
    # C12 = A10B02 + A11B12 + A12B22.
    # Wait, my Strassen formulas might be mixed up with indices.
    # Standard Strassen for 2x2 is easy. For 3x3 it's more complex.
    # I will skip explicit Strassen construction and rely on ALS to find a low rank decomposition, 
    # or use the schoolbook for the whole thing if ALS fails.
    # Given the risk, I will return the schoolbook (63) if I can't find better quickly.
    # But I will try ALS for rank 19-25.
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(T, target_rank, max_iter=2000, tol=1e-9):
    n, m, p = DIMENSIONS
    shape = T.shape
    U = np.random.randn(shape[0], target_rank)
    V = np.random.randn(shape[1], target_rank)
    W = np.random.randn(shape[2], target_rank)
    
    for it in range(max_iter):
        X = np.einsum('br,cr->bcr', V, W).reshape(-1, target_rank)
        T_matted = T.reshape(shape[0], -1)
        Z, _, _, _ = np.linalg.lstsq(X, T_matted.T, rcond=None)
        U = Z.T
        
        X = np.einsum('ar,cr->acr', U, W).reshape(-1, target_rank)
        T_matted = T.transpose(1, 0, 2).reshape(shape[1], -1)
        Z, _, _, _ = np.linalg.lstsq(X, T_matted.T, rcond=None)
        V = Z.T
        
        X = np.einsum('ar,br->abr', U, V).reshape(-1, target_rank)
        T_matted = T.transpose(2, 0, 1).reshape(shape[2], -1)
        Z, _, _, _ = np.linalg.lstsq(X, T_matted.T, rcond=None)
        W = Z.T
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T) / np.linalg.norm(T)
        if err < tol:
            return U, V, W, err
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(recon - T) / np.linalg.norm(T)
    return U, V, W, err

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    T = _construct_tensor()
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try ranks 19 to 30
    for target_rank in range(19, 31):
        if time.time() > deadline - 30:
            break
        found = False
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            np.random.seed(trial * 1000 + target_rank)
            U, V, W, err = _try_rank_decomposition(T, target_rank, max_iter=1000, tol=1e-9)
            if err < 1e-8:
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
                if np.allclose(recon, T, atol=1e-8):
                    U_list = [[float(x) for x in row] for row in U_arr]
                    V_list = [[float(x) for x in row] for row in V_arr]
                    W_list = [[float(x) for x in row] for row in W_arr]
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        if found:
            continue
        else:
            break
            
    return best
