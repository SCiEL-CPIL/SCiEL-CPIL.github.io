import time
import numpy as np
from fractions import Fraction
from itertools import product
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 5
    
    # 1. Build the target tensor T[i, j, k] = 1 if k==j else 0 (for A[i,k]*B[k,j] -> C[i,j])
    # Wait, the tensor for matrix multiplication C = A @ B is T_{i,k,j} = delta_{k,j}?
    # No, C_{i,j} = sum_k A_{i,k} B_{k,j}.
    # The tensor T has indices (i, k, j) where T[i,k,j] = 1 if we are computing C[i,j] from A[i,k] and B[k,j].
    # Actually, the standard matrix multiplication tensor T_{i,j,k} (or similar) is defined such that
    # C_{i,j} = sum_k A_{i,k} B_{k,j}.
    # In tensor decomposition terms: T_{i,j,k} = 1 if the term A_{i,k} B_{k,j} contributes to C_{i,j}.
    # So T[i, j, k] = 1 if k is the summation index connecting A and B?
    # Let's stick to the definition: T is a 3-tensor of size n x p x m (if C is n x m, A is n x p, B is p x m).
    # Here A is 4x4 (n=4, p=4), B is 4x5 (p=4, m=5). C is 4x5 (n=4, m=5).
    # C[i, j] = sum_{k=0}^{3} A[i, k] B[k, j].
    # The tensor T has entries T[i, j, k]. For a fixed (i, j), the coefficient of A[i, k] B[k, j] is 1.
    # So T[i, j, k] = 1 if the indices match the summation structure?
    # Actually, the tensor is simply T[i, j, k] = 1 for all i, j, k? No.
    # The tensor T is defined by T_{i,j,k} = 1 if the k-th term in the sum for C_{i,j} involves A_{i,k} and B_{k,j}.
    # Since C_{i,j} = sum_k A_{i,k} B_{k,j}, the coefficient of the product A_{i,k} B_{k,j} is 1.
    # Thus, the tensor T is a 3-way tensor of size (n, m, p) where T[i, j, k] = 1 if the term exists.
    # Wait, the "term" is defined by the index k.
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)?
    # No, that would mean C[i,j] = sum_k A[i,k] B[k,j] * 1.
    # Yes, T[i, j, k] = 1 for all valid i, j, k?
    # Let's check the shape.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # This suggests the tensor is flattened.
    # Let's define the tensor T of shape (n, p, m) or (n, m, p)?
    # Standard MM tensor: T_{i,k,j} = 1.
    # The decomposition is: T_{i,k,j} = sum_r U_{(i,k), r} V_{(k,j), r} W_{(i,j), r}?
    # No, the factors are U (A-space), V (B-space), W (C-space).
    # A-space dimension: n*p = 4*4 = 16.
    # B-space dimension: p*m = 4*5 = 20.
    # C-space dimension: n*m = 4*5 = 20.
    # The tensor T is of shape (n*p, p*m, n*m)? No, that's not right.
    # The tensor T maps A (size 16) and B (size 20) to C (size 20).
    # T_{a, b, c} = 1 if A_a * B_b contributes to C_c.
    # A_a is A[i, k] where a = i*p + k.
    # B_b is B[k, j] where b = k*m + j.
    # C_c is C[i, j] where c = i*m + j.
    # Contribution happens if a=(i,k) and b=(k,j) and c=(i,j).
    # So T[a, b, c] = 1 if:
    #   a = i*p + k
    #   b = k*m + j
    #   c = i*m + j
    # for some i, j, k.
    
    # Let's construct T as a sparse dictionary or dense array.
    # Dense array size: 16 * 20 * 20 = 6400. Small enough.
    T = np.zeros((16, 20, 20), dtype=int)
    for i in range(4):
        for k in range(4):
            for j in range(5):
                a = i * 4 + k
                b = k * 5 + j
                c = i * 5 + j
                T[a, b, c] = 1

    # We want to find rank-r decomposition:
    # T[a, b, c] = sum_{r=0}^{R-1} U[a, r] * V[b, r] * W[c, r]
    
    # Strategy:
    # 1. Try random rank 20-30? 
    #    Schoolbook is 80.
    #    Strassen for 4x4x4 is 49.
    #    For 4x4x5, maybe ~50-60?
    #    Let's try to find a rank 30 solution first? Unlikely.
    #    Let's try to find a rank 40 solution.
    
    # Method: Randomized SVD / ALS with exact rational recovery.
    # 1. Generate random U (16xR), V (20xR).
    # 2. Solve for W: For each r, W[:, r] is determined by T and U, V?
    #    No, T is the sum of outer products.
    #    This is a non-linear problem.
    
    # Alternative: Use the fact that T has a specific structure.
    # T is the "matrix multiplication tensor".
    # We can try to use Strassen's algorithm for 4x4 blocks and extend to 4x5.
    
    # Let's try a structured search for rank 40.
    # We can use the "Strassen-like" approach:
    # Partition A into 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01, B02], [B10, B11, B12]]? No, B is 4x5.
    # B = [[B00, B01, B02, B03, B04], ...]
    
    # Let's try a generic random search with a time limit.
    # If we find a valid decomposition of rank R, we return it.
    # If not, we return the schoolbook (rank 80).
    
    best_rank = 80
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook init
    U_sb = np.zeros((16, 80))
    V_sb = np.zeros((20, 80))
    W_sb = np.zeros((20, 80))
    col = 0
    for i in range(4):
        for k in range(4):
            for j in range(5):
                a = i * 4 + k
                b = k * 5 + j
                c = i * 5 + j
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    def verify(U, V, W):
        # Check if U V W^T == T
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum
        try:
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_rec, T, atol=1e-9)
        except:
            return False

    # Try to find a lower rank solution.
    # We will try ranks 40, 50, 60, 70.
    # Use random initialization and ALS (Alternating Least Squares) to minimize ||T - U V W^T||_F.
    # Then check for exactness. If close enough, try to round to rationals.
    
    def try_rank(R, time_limit=2.0):
        start = time.time()
        # Initialize U, V, W randomly
        # Scale matters.
        U = np.random.randn(16, R)
        V = np.random.randn(20, R)
        W = np.random.randn(20, R)
        
        # ALS
        for step in range(50):
            if time.time() - start > time_limit:
                break
            
            # Update W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed U, V, W is linear.
            # Reshape T to (16*20, 20) -> (320, 20)
            # Reshape U, V to (320, R)
            T_mat = T.reshape(320, 20)
            UV_mat = np.einsum('ar,br->ab r', U, V).reshape(320, R)
            # Solve UV_mat @ W.T = T_mat
            # W.T (R x 20) = UV_mat.T (R x 320) @ T_mat (320 x 20)
            W_new = np.linalg.lstsq(UV_mat.T, T_mat, rcond=None)[0].T
            W = W_new
            
            # Update U
            # T[a,b,c] = U[a,r] * (sum_b V[b,r] W[c,r])
            # For fixed V, W, U is linear.
            # T_mat (320, 20) = U_mat (16, R) @ (V @ W.T) (R, 20)?
            # Let's check indices.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix a, c. Sum over b? No.
            # T[a, :, c] (20,) = sum_r U[a,r] * (V[:, r] * W[c,r]) (20,)
            # Let M[a, r, b] = U[a,r] V[b,r].
            # T[a, b, c] = sum_r M[a, r, b] W[c, r]
            # For fixed V, W:
            # T[a, b, c] = sum_r U[a,r] * (V[b,r] W[c,r])
            # Let K[b, r, c] = V[b,r] W[c,r].
            # T[a, b, c] = sum_r U[a,r] K[b, r, c]
            # Reshape T to (16, 20*20) = (16, 400)
            # Reshape K to (20, R, 20) -> (20, 20*R)?
            # T_mat_ac (16, 400) where index (b*20+c)
            # K_mat (20, 400) where index (r*20+c)? No.
            # K[b, r, c]. We want to sum over r.
            # T[a, b, c] = sum_r U[a,r] K[b,r,c]
            # Let T_2d[a, b*20+c] = T[a,b,c]
            # Let K_2d[b*20+c, r] = K[b,r,c]
            # Then T_2d = U @ K_2d.T
            # U (16, R) @ K_2d.T (R, 400) = T_2d (16, 400)
            
            T_2d = T.reshape(16, 400)
            K = np.einsum('br,cr->brc', V, W) # (20, R, 20)
            K_2d = K.reshape(20, -1).T # (400, R) ? 
            # K[b, r, c]. We want index (b, c) -> r.
            # K_2d[(b*20+c), r] = K[b,r,c]
            K_2d = K.transpose(0, 2, 1).reshape(400, R)
            
            U_new = np.linalg.lstsq(K_2d.T, T_2d, rcond=None)[0]
            U = U_new
            
            # Update V
            # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            # T[a, b, c] = sum_r V[b,r] (U[a,r] W[c,r])
            # Let L[a, r, c] = U[a,r] W[c,r]
            # T[a, b, c] = sum_r V[b,r] L[a,r,c]
            # Reshape T to (20, 16*20) = (20, 320) where index (a*20+c)
            # Reshape L to (320, R) where index (a*20+c, r)
            T_2d = T.transpose(1, 0, 2).reshape(20, 320) # (b, a*20+c)
            L = np.einsum('ar,cr->arc', U, W) # (16, R, 20)
            L_2d = L.transpose(0, 2, 1).reshape(320, R)
            
            V_new = np.linalg.lstsq(L_2d.T, T_2d, rcond=None)[0]
            V = V_new
            
        # Check exactness
        if verify(U, V, W):
            return U, V, W
        
        # Try to round to rationals
        # If close, try rounding to 1/2, 1, 0, -1, etc.
        if np.allclose(np.einsum('ar,br,cr->abc', U, V, W), T, atol=0.1):
             # Try to recover rational forms
             # This is hard. Just return None for now unless it's very close.
             pass
             
        return None

    # Try ranks
    for R in [40, 50, 60, 70]:
        if time.time() > deadline - 10:
            break
        result = try_rank(R, time_limit=10.0)
        if result:
            U, V, W = result
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
