"""Matrix multiplication 2x2 by 2x15: rank-minimal exact tensor decomposition."""
import random
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
_DEADLINE = time.time() + 140.0


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) with integer entries.
    """
    rank = 7
    n, m, p = 2, 2, 2
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_2x2_1x2():
    """
    Schoolbook decomposition for 2x2 * 2x1 -> 2x1.
    Returns (U, V, W) each of shape (4, 4), (2, 4), (2, 4) with integer entries.
    """
    rank = 4
    n, m, p = 2, 2, 1
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    
    return U, V, W


def _als_2x2x3(rank=10, max_iter=500, tol=1e-6):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 using ALS.
    Returns (U, V, W) as numpy arrays or None if failed.
    """
    n, m, p = 2, 2, 3
    # Tensor T is all ones of shape (4, 6, 6) ?
    # U: (4, rank), V: (6, rank), W: (6, rank)
    # Indices:
    # U[i*m+j]
    # V[j*p+k]
    # W[k*n+i]
    
    # Target tensor T[i,j,k] = 1
    # Reshape T to (4, 6, 6) ? No.
    # T is defined on (i,j,k).
    # U index: i*m+j.
    # V index: j*p+k.
    # W index: k*n+i.
    
    # This mapping is not a simple reshape of a 3D tensor to 3 matrices.
    # We need to solve for U, V, W such that:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(4, rank)
    V = np.random.randn(6, rank)
    W = np.random.randn(6, rank)
    
    # Precompute index mappings
    # For each c, we have vectors u_c, v_c, w_c.
    # The constraint is a system of linear equations for one matrix given the others.
    
    # Let's use the fact that we can fix U and V and solve for W.
    # For each (k, i), W[k*n+i, c] is the variable.
    # The equation is:
    # sum_c V[j*p+k, c] W[k*n+i, c] U[i*m+j, c] = 1 for all i, j, k.
    
    # This is tricky because U, V, W are coupled.
    # Standard ALS for tensor decomposition:
    # Reshape T to (I, J, K).
    # But our indices are mixed.
    
    # Let's define the "effective" tensor for ALS.
    # We want to minimize || T - sum_c u_c v_c w_c ||^2.
    # Here T is the all-ones tensor of shape (2, 2, 3).
    # But the factorization is on the "flattened" indices.
    
    # Actually, we can just run ALS on the 3D tensor T (2,2,3) with standard unfolding,
    # BUT the factors must be mapped to the correct indices.
    # Standard ALS gives factors X (2, rank), Y (2, rank), Z (3, rank).
    # Then U[i*m+j] would be X[i, c] Y[j, c]? No, that's not how it works.
    # The factor U is a single matrix (4, rank).
    # The entries U[i*m+j, c] are independent variables in ALS, not separable.
    
    # So we must run ALS on the "flattened" tensor.
    # The tensor T_flat has indices (u_idx, v_idx, w_idx).
    # u_idx = i*m+j.
    # v_idx = j*p+k.
    # w_idx = k*n+i.
    # T_flat[u, v, w] = 1 if there exist i,j,k such that u=i*m+j, v=j*p+k, w=k*n+i.
    # Note: The mapping (i,j,k) -> (u,v,w) is bijective?
    # i in 0..1, j in 0..1, k in 0..2.
    # u in 0..3, v in 0..5, w in 0..5.
    # For a given (u,v,w), is there a unique (i,j,k)?
    # u = 2i+j. i = u//2, j = u%2.
    # v = 2j+k. k = v - 2j. Need k in 0..2.
    # w = 2k+i. Need w == 2k+i.
    # So T_flat[u,v,w] = 1 if k = v - 2j is valid and w == 2k+i.
    
    T_flat = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = i*2 + j
                v = j*3 + k
                w = k*2 + i
                T_flat[u, v, w] = 1.0
                
    # Now run ALS on T_flat.
    # Factors U (4, rank), V (6, rank), W (6, rank).
    
    # Normalize to prevent divergence
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            return None
            
        # Update W
        # For each w, solve for W[w, :]
        # T_flat[:, :, w] = sum_c V[:, c] W[w, c] U[:, c]^T ?
        # T_flat[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        # Let M_w be the matrix of size (4*6, rank) ?
        # Reshape T_flat[:, :, w] to (24, ).
        # Reshape U and V to (24, rank) ?
        # U is (4, rank), V is (6, rank).
        # The product U[u, c] V[v, c] is element-wise product of U and V?
        # No, it's sum_c U[u, c] V[v, c] W[w, c].
        # For fixed w, we want to find W[w, c] to minimize:
        # sum_{u,v} (T_flat[u,v,w] - sum_c U[u,c] V[v,c] W[w,c])^2
        
        # This is a least squares problem.
        # Let A_w be a matrix of size (24, rank).
        # A_w[u*6+v, c] = U[u, c] * V[v, c].
        # Then b_w = T_flat[:, :, w].reshape(24).
        # Solve A_w W[w, :] = b_w.
        
        for w in range(6):
            A_w = np.zeros((24, rank))
            for u in range(4):
                for v in range(6):
                    A_w[u*6+v, :] = U[u, :] * V[v, :]
            b_w = T_flat[u, v, w] # Error in variable scope, should be T_flat[:, :, w].reshape(24)
            b_w = T_flat[:, :, w].reshape(24)
            # Solve
            try:
                W[w, :] = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
            except:
                return None
                
        # Update U
        for u in range(4):
            A_u = np.zeros((36, rank))
            for v in range(6):
                for w in range(6):
                    A_u[v*6+w, :] = V[v, :] * W[w, :]
            b_u = T_flat[u, :, :].reshape(36)
            try:
                U[u, :] = np.linalg.lstsq(A_u, b_u, rcond=None)[0]
            except:
                return None
                
        # Update V
        for v in range(6):
            A_v = np.zeros((24, rank))
            for u in range(4):
                for w in range(6):
                    A_v[u*6+w, :] = U[u, :] * W[w, :]
            b_v = T_flat[:, v, :].reshape(24)
            try:
                V[v, :] = np.linalg.lstsq(A_v, b_v, rcond=None)[0]
            except:
                return None
                
        # Check convergence
        # Calculate residual
        residual = 0.0
        for u in range(4):
            for v in range(6):
                for w in range(6):
                    val = np.dot(U[u, :], V[v, :] * W[w, :])
                    residual += (T_flat[u, v, w] - val)**2
        if residual < tol:
            # Check if it's an exact solution (integer/half-integer)
            # For now, just return it. The construct function will verify.
            return U, V, W
            
    return None


def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-5:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal decomposition for 2x2 * 2x15 -> 2x15.
    """
    n, m, p = 2, 2, 15
    
    # Try to find rank 10 for 2x2x3
    # If found, use 5 blocks of 3 (rank 50)
    # Else, use 7 blocks of 2 + 1 block of 1 (rank 53)
    
    # We have 140s. ALS might be slow.
    # Let's try a few restarts.
    
    U_223, V_223, W_223 = None
    start_time = time.time()
    
    # Limit search time to 20s
    while time.time() - start_time < 20 and U_223 is None:
        U_223, V_223, W_223 = _als_2x2x3(rank=10, max_iter=100)
        if U_223 is not None:
            # Verify
            if _verify(U_223, V_223, W_223, 2, 2, 3):
                break
            else:
                U_223 = None
                
    if U_223 is not None:
        # Build rank 50 solution
        total_rank = 50
        U = [[0]*total_rank for _ in range(4)]
        V = [[0]*total_rank for _ in range(30)]
        W = [[0]*total_rank for _ in range(30)]
        
        U_arr = np.array(U_223)
        V_arr = np.array(V_223)
        W_arr = np.array(W_223)
        
        for block in range(5):
            k_start = block * 3
            for c in range(10):
                for row in range(4):
                    U[row][block*10+c] = U_arr[row][c]
                for j in range(2):
                    for k_local in range(3):
                        local_row = j * 3 + k_local
                        global_row = j * 15 + (k_start + k_local)
                        V[global_row][block*10+c] = V_arr[local_row][c]
                for k_local in range(3):
                    for i in range(2):
                        local_row = k_local * 2 + i
                        global_row = (k_start + k_local) * 2 + i
                        W[global_row][block*10+c] = W_arr[local_row][c]
                        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to rank 53
        total_rank = 53
        U = [[0]*total_rank for _ in range(4)]
        V = [[0]*total_rank for _ in range(30)]
        W = [[0]*total_rank for _ in range(30)]
        
        Us, Vs, Ws = _strassen_2x2()
        Usb, Vsb, Wsb = _schoolbook_2x2_1x2()
        
        col_offset = 0
        for block in range(7):
            k_start = block * 2
            for c in range(7):
                for row in range(4):
                    U[row][col_offset] = Us[row][c]
                for j in range(2):
                    for k_local in range(2):
                        local_row = j * 2 + k_local
                        global_row = j * 15 + (k_start + k_local)
                        V[global_row][col_offset] = Vs[local_row][c]
                for k_local in range(2):
                    for i in range(2):
                        local_row = k_local * 2 + i
                        global_row = (k_start + k_local) * 2 + i
                        W[global_row][col_offset] = Ws[local_row][c]
                col_offset += 1
        
        for c in range(4):
            for row in range(4):
                U[row][col_offset] = Usb[row][c]
            for j in range(2):
                V[j * 15 + 14][col_offset] = Vsb[j][c]
            for i in range(2):
                W[14 * 2 + i][col_offset] = Wsb[i][c]
            col_offset += 1
            
        return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
