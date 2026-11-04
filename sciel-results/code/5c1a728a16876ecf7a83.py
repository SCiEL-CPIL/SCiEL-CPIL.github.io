import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    # Check integer/half-integer
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0, atol=tol):
                    return False
    return True

def _make_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _greedy_constructive(n, m, p, max_rank, seed, time_budget=10.0):
    """
    Greedy algorithm: Start with zero matrices, iteratively add rank-1 tensors
    that best reduce the residual norm of the target tensor.
    Target tensor T[i,j,k] = 1.
    Current approximation S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    Residual R[i,j,k] = T[i,j,k] - S[i,j,k].
    We want to find u, v, w such that <R, u⊗v⊗w> is maximized.
    This is equivalent to finding the best rank-1 approximation to R.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    # Initialize residual as target tensor
    # Residual[i,j,k] = 1 - sum_r U[i,j,r]V[j,k,r]W[k,i,r]
    # Since U,V,W are 0 initially, Residual is 1 everywhere.
    # We store residual in a flattened array for speed?
    # Let's just recompute or update.
    
    # To make it efficient, we can maintain the residual tensor explicitly.
    # Residual shape: (n, m, p)
    # T[i,j,k] = 1
    # S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Residual = np.ones((n, m, p))
    
    for r in range(max_rank):
        if time.time() - start_time > time_budget:
            break
            
        # Find best rank-1 approximation to Residual
        # This is the NP-hard problem of best rank-1 approximation.
        # We use a random restart approach with a small set of candidates.
        
        best_val = -np.inf
        best_u, best_v, best_w = None, None, None
        
        # Candidate values for entries: -1, 0, 1 (and maybe 0.5, -0.5)
        # To keep it fast, we sample random vectors from a small set.
        # Or use SVD on the flattening?
        # SVD of Residual flattened to (n*m, p) or similar doesn't directly give rank-1 in 3D.
        # However, we can try a few random vectors and compute the optimal others.
        
        # Strategy:
        # 1. Pick random u (normalized? no, we want specific values).
        #    Actually, let's pick u, v, w from a discrete set.
        #    Set: {-1, 0, 1}
        
        # Let's try 100 random candidates
        for _ in range(100):
            u = np.random.choice([-1, 0, 1], size=(n * m,))
            v = np.random.choice([-1, 0, 1], size=(m * p,))
            w = np.random.choice([-1, 0, 1], size=(p * n,))
            
            # Compute inner product <Residual, u⊗v⊗w>
            # sum_{i,j,k} Residual[i,j,k] * u[i,j] * v[j,k] * w[k,i]
            # Reshape u to (n,m), v to (m,p), w to (p,n)
            u_mat = u.reshape((n, m))
            v_mat = v.reshape((m, p))
            w_mat = w.reshape((p, n))
            
            # Compute contraction
            # temp[j,k] = sum_i Residual[i,j,k] * u_mat[i,j]
            # val = sum_{j,k} temp[j,k] * v_mat[j,k] * w_mat[k,j] ? No, w index is k,n.
            # Let's do it step by step to avoid confusion.
            # sum_{i,j,k} Res[i,j,k] u[i,j] v[j,k] w[k,i]
            
            # Step 1: Contract i
            # A[j,k] = sum_i Res[i,j,k] u[i,j]
            A = np.einsum('ijk,ij->jk', Residual, u_mat)
            # Step 2: Contract j
            # B[k] = sum_j A[j,k] v[j,k]
            B = np.einsum('jk,jk->k', A, v_mat)
            # Step 3: Contract k
            # val = sum_k B[k] w[k, ?] -- Wait, w is w[k,i].
            # The sum is sum_{i,j,k} ... w[k,i].
            # My B[k] is sum_{i,j} Res[i,j,k] u[i,j] v[j,k].
            # Then val = sum_k B[k] * ... wait, w depends on i and k.
            # The contraction is sum_{i,j,k} Res[i,j,k] u[i,j] v[j,k] w[k,i].
            # Let's regroup:
            # val = sum_i ( sum_{j,k} Res[i,j,k] u[i,j] v[j,k] w[k,i] )
            # This is not separable simply because w[k,i] links k and i.
            
            # Let's just compute the full sum for the candidate.
            # Since n,m,p are small (4,5,6), we can afford a loop or einsum.
            val = np.einsum('ijk,ij,jk,ki->', Residual, u_mat, v_mat, w_mat)
            
            if val > best_val:
                best_val = val
                best_u, best_v, best_w = u, v, w
        
        if best_val < 1e-9:
            # No improvement possible with these candidates?
            # Maybe try a different strategy or break.
            # For now, just add a zero term or break.
            break
            
        # Update Residual
        u_mat = best_u.reshape((n, m))
        v_mat = best_v.reshape((m, p))
        w_mat = best_w.reshape((p, n))
        
        # Residual -= u⊗v⊗w
        Residual -= np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        
        # Store in U, V, W
        U[:, r] = best_u
        V[:, r] = best_v
        W[:, r] = best_w
        
        # Check if residual is close to zero
        if np.linalg.norm(Residual) < 1e-9:
            # Found exact decomposition with rank r+1
            U = U[:, :r+1]
            V = V[:, :r+1]
            W = W[:, :r+1]
            return U, V, W
            
    return None, None, None

def _block_decomposition():
    """
    Try to construct a decomposition using block structure.
    4x5x6 can be split.
    This is a placeholder for a known lower rank decomposition if available.
    For now, we rely on the greedy search.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    # 1. Schoolbook baseline
    U_sb, V_sb, W_sb = _make_schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # 2. Try greedy constructive search
    # We want to find a rank lower than 120.
    # Let's try to find a decomposition of rank R.
    # The greedy algorithm builds up to max_rank.
    # We can run it with max_rank = 119, 118, etc.
    # But the greedy algorithm stops early if it finds an exact solution.
    # So we can just call it with a high max_rank and see what rank it achieves.
    
    seeds = [42, 123, 999, 777, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
            
        # Give each seed a time budget
        time_left = deadline - time.time()
        budget = min(10.0, time_left / 2.0)
        
        U_out, V_out, W_out = _greedy_constructive(n, m, p, 119, seed, time_budget=budget)
        
        if U_out is not None:
            rank = U_out.shape[1]
            if rank < best_rank:
                # Verify
                if _verify(U_out, V_out, W_out):
                    best_result = {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                    best_rank = rank
                    # If we get close to theoretical lower bound, we might stop
                    # Lower bound for 4x5x6 is not trivial, but < 120 is good.
                    if best_rank <= 110:
                        break

    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = DIMENSIONS
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0):
                    ok = False
                    print(f"Fail at {i},{j},{k}: {val}")
                    break
            if not ok: break
        if not ok: break
    if ok:
        print("Verification Passed")
    else:
        print("Verification Failed")
