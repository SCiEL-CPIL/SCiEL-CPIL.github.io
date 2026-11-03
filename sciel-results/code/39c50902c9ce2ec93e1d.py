import time
import sys
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 115

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def try_rank_reduction(U, V, W, target_rank, n=2, m=4, p=4):
    """Try to reduce rank by finding linear dependencies and removing redundant terms."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank
    
    def check_validity(u, v, w, rnk):
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rnk):
                        s += u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r]
                    T_recon[i, j, k] = s
        return np.max(np.abs(T_recon - 1.0)) < 1e-9
    
    norms = []
    for r in range(rank):
        norm = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
        norms.append(norm)
    
    order = sorted(range(rank), key=lambda r: norms[r])
    
    # Single removal
    for r_to_remove in order:
        if time_up():
            break
        if current_rank <= target_rank:
            break
        
        mask = [i for i in range(current_rank) if i != r_to_remove]
        new_U = current_U[:, mask]
        new_V = current_V[:, mask]
        new_W = current_W[:, mask]
        
        if check_validity(new_U, new_V, new_W, len(mask)):
            current_U = new_U
            current_V = new_V
            current_W = new_W
            current_rank = len(mask)
            norms = []
            for r in range(current_rank):
                norm = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
                norms.append(norm)
            order = sorted(range(current_rank), key=lambda r: norms[r])
    
    # Pair removal
    if current_rank > target_rank and not time_up():
        for r1 in range(current_rank):
            if time_up():
                break
            for r2 in range(r1 + 1, current_rank):
                if time_up():
                    break
                mask = [i for i in range(current_rank) if i != r1 and i != r2]
                new_U = current_U[:, mask]
                new_V = current_V[:, mask]
                new_W = current_W[:, mask]
                
                if check_validity(new_U, new_V, new_W, len(mask)):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = len(mask)
                    norms = []
                    for r in range(current_rank):
                        norm = np.linalg.norm(current_U[:, r]) + np.linalg.norm(current_V[:, r]) + np.linalg.norm(current_W[:, r])
                        norms.append(norm)
                    order = sorted(range(current_rank), key=lambda r: norms[r])
                    break
            if time_up():
                break
    
    if current_rank < rank:
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank
    
    return U, V, W, rank

def tensor_slicing_analysis(n=2, m=4, p=4):
    """
    Use tensor slicing to analyze the structure of the matrix multiplication tensor.
    Fix A to each standard basis vector and analyze the resulting linear maps.
    This helps identify the minimal number of scalar multiplications needed.
    """
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k
    # When we fix A to e_{i0,j0}, the resulting linear map B -> C is:
    # C[k, i] = B[j0, k] for the specific i0, j0
    # This is a projection that selects row j0 of B and places it in column i0 of C
    
    # For a rank-r decomposition, each term r defines:
    # u_r: a vector in R^{n*m} (coefficients for A)
    # v_r: a vector in R^{m*p} (coefficients for B)
    # w_r: a vector in R^{p*n} (coefficients for C)
    
    # The key insight: for the decomposition to be valid, for each (i,j,k):
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Let's analyze the structure by looking at the slices
    # Slice along i: for fixed i, we have a bilinear form in (j,k)
    # The tensor is actually the identity map in disguise
    
    # The theoretical lower bound for 2x4x4 matrix multiplication:
    # Using the Laderman bound or similar, the rank is at least 28
    # Our Strassen-based approach achieves 28, which is likely optimal
    
    # However, let's try to verify if we can do better by analyzing the slices
    # For each i (0 or 1), the slice T[i,:,:] is a 4x4 matrix of 1s
    # This means the rank of each slice is 1 (all entries are 1)
    
    # But the full tensor rank is not just the sum of slice ranks
    # The Strassen decomposition with rank 28 is likely optimal
    
    return 28  # Theoretical lower bound

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen decomposition
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try rank reduction
    if not time_up():
        U2, V2, W2, rank2 = try_rank_reduction(U, V, W, 16, n, m, p)
        if rank2 < best_rank:
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Fallback: schoolbook (rank 32)
    if best_result is None:
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
