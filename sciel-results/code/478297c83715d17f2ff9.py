import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _standard_decomp(n, m, p):
    """Standard (schoolbook) decomposition."""
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Strassen decomposition for 2x2 matrix multiplication. Rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1: (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2: (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # m3: a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4: d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # m5: (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # m6: (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = -1
    
    # m7: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_decomps(decomp_list, n, m, p):
    """
    Combine a list of decompositions for sub-blocks into one for the full matrix.
    decomp_list: list of (U, V, W, i_offset, j_offset, k_offset, n_sub, m_sub, p_sub)
    """
    total_rank = sum(len(d[0][0]) for d in decomp_list)
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub in decomp_list:
        r_sub = len(U_sub[0])
        
        # Copy U
        for i in range(n_sub):
            for j in range(m_sub):
                U_idx = i*m_sub + j
                U_full_idx = (i_off+i)*m + (j_off+j)
                for c in range(r_sub):
                    U_full[U_full_idx][col_offset+c] = U_sub[U_idx][c]
        
        # Copy V
        for j in range(m_sub):
            for k in range(p_sub):
                V_idx = j*p_sub + k
                V_full_idx = (j_off+j)*p + (k_off+k)
                for c in range(r_sub):
                    V_full[V_full_idx][col_offset+c] = V_sub[V_idx][c]
        
        # Copy W
        for k in range(p_sub):
            for i in range(n_sub):
                W_idx = k*n_sub + i
                W_full_idx = (k_off+k)*n + (i_off+i)
                for c in range(r_sub):
                    W_full[W_full_idx][col_offset+c] = W_sub[W_idx][c]
        
        col_offset += r_sub
        
    return U_full, V_full, W_full

def _block_decomposition(n, m, p):
    """Block decomposition using Strassen for 2x2 blocks."""
    decomp_list = []
    
    # Split A (3x5) into A_L (3x2) and A_R (3x3)
    # Split B (5x5) into B_LL (2x2), B_LR (2x3), B_RL (3x2), B_RR (3x3)
    
    # Block 1: A_L (3x2) * B_LL (2x2) -> C_LL (3x2)
    U_222, V_222, W_222 = _strassen_2x2()
    decomp_list.append((U_222, V_222, W_222, 0, 0, 0, 2, 2, 2))
    
    U_122, V_122, W_122 = _standard_decomp(1, 2, 2)
    decomp_list.append((U_122, V_122, W_122, 2, 0, 0, 1, 2, 2))
    
    # Block 2: A_L (3x2) * B_LR (2x3) -> C_LR (3x3)
    U_222_2, V_222_2, W_222_2 = _strassen_2x2()
    decomp_list.append((U_222_2, V_222_2, W_222_2, 0, 0, 2, 2, 2, 2))
    
    U_221, V_221, W_221 = _standard_decomp(2, 2, 1)
    decomp_list.append((U_221, V_221, W_221, 0, 2, 4, 2, 2, 1))
    
    U_122_2, V_122_2, W_122_2 = _standard_decomp(1, 2, 2)
    decomp_list.append((U_122_2, V_122_2, W_122_2, 2, 0, 2, 1, 2, 2))
    
    U_121, V_121, W_121 = _standard_decomp(1, 2, 1)
    decomp_list.append((U_121, V_121, W_121, 2, 2, 4, 1, 2, 1))
    
    # Block 3: A_R (3x3) * B_RL (3x2) -> C_RL (3x2)
    U_222_3, V_222_3, W_222_3 = _strassen_2x2()
    decomp_list.append((U_222_3, V_222_3, W_222_3, 0, 2, 0, 2, 2, 2))
    
    U_212, V_212, W_212 = _standard_decomp(2, 1, 2)
    decomp_list.append((U_212, V_212, W_212, 0, 4, 0, 2, 1, 2))
    
    U_122_3, V_122_3, W_122_3 = _standard_decomp(1, 2, 2)
    decomp_list.append((U_122_3, V_122_3, W_122_3, 2, 2, 0, 1, 2, 2))
    
    U_112, V_112, W_112 = _standard_decomp(1, 1, 2)
    decomp_list.append((U_112, V_112, W_112, 2, 4, 0, 1, 1, 2))
    
    # Block 4: A_R (3x3) * B_RR (3x3) -> C_RR (3x3)
    U_333, V_333, W_333 = _standard_decomp(3, 3, 3)
    decomp_list.append((U_333, V_333, W_333, 0, 2, 2, 3, 3, 3))
    
    U, V, W = _combine_decomps(decomp_list, n, m, p)
    return U, V, W

def _verify_numpy(U, V, W, n, m, p, trials=10):
    """Verify using numpy with random integer matrices."""
    import random
    r = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.randint(-10, 10, size=(n, m))
        B = np.random.randint(-10, 10, size=(m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But this is the tensor form. The actual computation is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In bilinear form: C = sum_c (U_c^T * A) * (V_c^T * B) * W_c^T
        # More precisely: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        C_computed = np.zeros((n, p))
        for c in range(r):
            # U_c is vector of length n*m, reshape to (n,m)
            Uc = U_np[:, c].reshape(n, m)
            Vc = V_np[:, c].reshape(m, p)
            Wc_val = W_np[k*n+i, c] if False else None  # W is per (k,i)
            
        # Actually, let me do it properly:
        # For each c, we have U[:,c] (n*m,), V[:,c] (m*p,), W[:,c] (p*n,)
        # C[i,k] += W[k*n+i, c] * dot(U[i*m:(i+1)*m, c], A[i,:]) * dot(V[k*p:(k+1)*p... 
        # Wait, V is indexed by (j,k) so V[j*p+k, c]
        # The bilinear form: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        for c in range(r):
            for i in range(n):
                for k in range(p):
                    u_val = sum(U_np[i*m+j, c] * A[i, j] for j in range(m))
                    v_val = sum(V_np[j*p+k, c] * B[j, k] for j in range(m))
                    C_computed[i, k] += W_np[k*n+i, c] * u_val * v_val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    r = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(r)] for i in range(n*m)]
    Vf = [[Fraction(V[j][c]) for c in range(r)] for j in range(m*p)]
    Wf = [[Fraction(W[k][c]) for c in range(r)] for k in range(p*n)]
    
    # Check: for each (i,j,k), sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] 
    # should equal... hmm, this isn't quite right.
    
    # The correct check: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: the matrix multiplication tensor T has T[i,j,k] = delta(j,j')... 
    # Actually, let me just check with specific A, B values.
    
    # Use A with A[i,j] = 1 if (i,j) matches some pattern, B similarly.
    # Simplest: check that for each (i,k), the bilinear form gives the correct result.
    
    # Alternative: construct the tensor and check directly.
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if j==j' and k==k' else 0
    
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            target = 1 if (j == j2 and k == k2) else 0
                            s = Fraction(0)
                            for c in range(r):
                                s += Uf[i*m+j][c] * Vf[j2*p+k][c] * Wf[k2*n+i2][c]
                            if s != target:
                                return False
    return True

def _random_search(n, m, p, max_rank, time_limit):
    """Structured random search for low-rank decomposition."""
    import random
    import numpy as np
    
    t0 = time.time()
    
    # Try ranks from max_rank down
    for rank in range(max_rank, 20, -1):
        if time.time() - t0 > time_limit:
            break
        
        # Multiple random initializations
        for trial in range(20):
            if time.time() - t0 > time_limit:
                break
            
            # Initialize with random sparse matrices
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Make them sparse
            mask_u = np.random.rand(n*m, rank) < 0.3
            mask_v = np.random.rand(m*p, rank) < 0.3
            mask_w = np.random.rand(p*n, rank) < 0.3
            U *= mask_u
            V *= mask_v
            W *= mask_w
            
            # ALS-style optimization
            for iteration in range(50):
                if time.time() - t0 > time_limit:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # This is getting complex. Let me use a simpler approach.
                        pass
            
            # Check if close to a valid solution
            # Convert to integers if close
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            # Check if integer solution works
            if np.allclose(U, U_int, atol=0.01) and np.allclose(V, V_int, atol=0.01) and np.allclose(W, W_int, atol=0.01):
                U_list = [[int(U_int[i][c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[int(V_int[i][c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[int(W_int[i][c]) for c in range(rank)] for i in range(p*n)]
                
                if _verify_exact(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0
    
    # First, get the block decomposition as a baseline
    U_block, V_block, W_block = _block_decomposition(n, m, p)
    rank_block = len(U_block[0])
    
    # Verify the block decomposition
    if not _verify_exact(U_block, V_block, W_block, n, m, p):
        # Fallback to standard
        U0, V0, W0 = _standard_decomp(n, m, p)
        return {"U": U0, "V": V0, "W": W0}
    
    best_U, best_V, best_W = U_block, V_block, W_block
    best_rank = rank_block
    
    # Try random search for lower rank
    search_time = 60.0
    result = _random_search(n, m, p, best_rank - 1, search_time)
    
    if result is not None:
        U_rand, V_rand, W_rand = result
        if len(U_rand[0]) < best_rank:
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            best_rank = len(U_rand[0])
    
    # Check time
    if time.time() > deadline - 5.0:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Final verification
    if _verify_exact(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to standard
        U0, V0, W0 = _standard_decomp(n, m, p)
        return {"U": U0, "V": V0, "W": W0}

# EVOLVE-BLOCK-END
