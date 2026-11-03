"""
Search for a low-rank tensor decomposition of 3x3 by 3x7 matrix multiplication.
Uses schoolbook (rank 63) as baseline; attempts structured factorization
exploiting Kronecker/block structure to reduce rank.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
n, m, p = DIMENSIONS
SCHOOLBOOK_RANK = n * m * p  # 63
DEADLINE_SECONDS = 120


def _build_tensor(n, m, p):
    """Build the matmul tensor T[i,j,k,r,s,l] = 1 if r==i and s==j and l==k, else 0.
    Flattened to shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-7):
    """Check if sum_c U[:,c] outer V[:,c] outer W[:,c] approximates T."""
    rank = U.shape[1]
    approx = np.einsum('ic,dc,kc->ikd', U, V, W)
    return np.max(np.abs(approx - T)) < tol


def _snap_to_half_integer(U, V, W, tol=1e-6):
    """Snap entries to nearest multiple of 0.5 if close enough."""
    U2 = np.round(2 * U) / 2.0
    V2 = np.round(2 * V) / 2.0
    W2 = np.round(2 * W) / 2.0
    if (np.max(np.abs(U2 - U)) < tol and
            np.max(np.abs(V2 - V)) < tol and
            np.max(np.abs(W2 - W)) < tol):
        return U2, V2, W2, True
    return U, V, W, False


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def _try_kronecker_factor(n, m, p, max_rank=SCHOOLBOOK_RANK - 1):
    """
    Try to decompose the 3x3-by-3x7 matmul tensor using Kronecker product structure.
    
    Idea: The tensor T for (n,m,p) matmul can sometimes be written as a sum of fewer
    rank-1 tensors if the problem has product structure. For instance, if we view the
    3x7 matrix as having 7 columns, each column is a 3-vector. The multiplication
    A * x for each column x_l gives 3 outputs. 
    
    One approach: use the identity that a 3x3 matrix multiply can be written with
    27 scalar multiplications (schoolbook). But there might be savings if we group.
    
    Here we try a structured approach: decompose into block operations.
    We look for U, V, W with rank r < 63 such that they satisfy the tensor equation.
    
    We use a greedy/randomized low-rank approximation approach:
    1. Flatten T to a 2D matrix (n*m) x (m*p * p*n)
    2. Try to find a low-rank factorization
    3. Then reshape into 3-mode decomposition
    """
    T = _build_tensor(n, m, p)
    # For this specific problem, we'll try a different structured approach:
    # Represent the multiplication as C = A * B where A is 3x3, B is 3x7, C is 3x7.
    # C_{i,l} = sum_j A_{i,j} B_{j,l}
    
    # Try rank 27: one multiplication per (i,j) pair, shared across all l.
    # U: shape (n*m, r), for each i,j we have one factor
    # V: shape (m*p, r), for each j,l we need to combine
    # W: shape (p*n, r), for each l,i
    
    # This is just schoolbook. Let's try to find something better.
    
    # Approach: randomized iterative refinement
    # We'll try various ranks and see if we can find a valid decomposition.
    
    np.random.seed(42)
    
    best_rank = max_rank + 1
    best_UVW = None
    
    for rank in range(28, max_rank + 1):
        if rank % 7 == 0:
            # Try structured: rank = (n*m) * (p/r_per_col) ... not obvious
            pass
        
        # Try random initialization with iterative projection
        max_iters = 50
        found = False
        
        for init in range(3):
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize
            for c in range(rank):
                if np.max(np.abs(W[:, c])) > 1e-10:
                    scale = np.max(np.abs(W[:, c]))
                    U[:, c] /= scale
                    V[:, c] /= scale
                    W[:, c] /= scale
            
            for it in range(max_iters):
                # Update W
                for c in range(rank):
                    T_slice = T - np.einsum('ic,dc->id', U[:, c], V[:, c]) * np.eye(p * n) * 0  # wrong
                    # Actually: residual = T - sum_{c'!=c} U_c' ⊗ V_c' ⊗ W_c'
                    # For column c: W_c = argmin ||T - sum U_i⊗V_i⊗W_i||
                    # This is: for each c, solve min_Wc ||T - sum_{all c} U_c⊗V_c⊗W_c||
                    # The optimal W_c given U, V is:
                    # W_c = (sum over c' of (U_c⊗V_c)^T (T_c - T_{c'})) ... complex
                    
                    # Simpler: use ALS (Alternating Least Squares)
                    # For mode-3 (W): fix U, V. The objective is:
                    # min_W sum_c ||T - einsum('ic,dc,kc->ikd', U, V, W)||^2
                    # Expand: for each c, the contribution is U[:,c]⊗V[:,c]⊗W[:,c]
                    # The residual with W=0 is T. 
                    # ALS: W_c = argmin ||T - sum_{c'} U_{c'}⊗V_{c'}⊗W_{c'}||
                    # = argmin ||R_c - U_c⊗V_c⊗W_c|| where R_c = T - sum_{c'≠c} U_{c'}⊗V_{c'}⊗W_{c'}
                    # In vectorized form: vec(R_c) = (V_c⊗U_c) ⊙ W_c (element-wise product after Kronecker)
                    # So W_c = vec(R_c) ./ vec(V_c⊗U_c) (element-wise division)
                    
                    pass  # ALS implementation is complex; skip for now
                
                break  # Skip ALS, just try simple verification
            break
        
        # After the (mostly empty) loop, just verify schoolbook for this rank isn't valid
        # This approach isn't productive. Let's try a different strategy.
        break
    
    # Fallback: return schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    return U, V, W, SCHOOLBOOK_RANK


def _try_direct_rank_search(n, m, p, target_rank, time_budget=10.0):
    """
    Try to find a rank `target_rank` decomposition using iterative methods.
    """
    T = _build_tensor(n, m, p)
    start_time = time.time()
    
    np.random.seed(123)
    
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Normalize columns
    for c in range(target_rank):
        mags = [np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])]
        max_mag = max(mags)
        if max_mag > 1e-10:
            U[:, c] /= max_mag
            V[:, c] /= max_mag
            W[:, c] /= max_mag
    
    best_err = np.inf
    
    for it in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # ALS: update W
        for c in range(target_rank):
            # Compute R = T - sum_{c'≠c} U_c' ⊗ V_c' ⊗ W_c'
            # This is expensive; approximate by only using the current iteration
            # Full residual:
            approx = np.einsum('ic,dc,kc->ikd', U, V, W)
            R = T - approx + np.einsum('ic,dc,kc->ikd', 
                                        U[:, c:c+1], V[:, c:c+1], W[:, c:c+1])
            # Now find W_c to minimize ||R - U_c⊗V_c⊗W_c||
            # vec(R) = vec(U_c⊗V_c⊗W_c) = (V_c⊗U_c)⊙W_c (Kronecker then element-wise)
            # Actually: the 3-way outer product U_c⊗V_c⊗W_c has vec = W_c ⊗ (V_c⊗U_c)?
            # Let me think: if A is n×m, B is p×q, then vec(A⊗B) is not standard.
            # Better: reshape R to 2D: (n*m) x (m*p*p*n) no...
            # For fixed c: min_Wc ||R - outer3(U_c, V_c, W_c)||^2
            # = min_Wc ||vec(R) - kron_kleiber(V_c, U_c, W_c)||^2
            # Using the property: vec3(outer3(a,b,c)) = c ⊗ (b ⊗ a)? No.
            # Let's use: (U_c⊗V_c⊗W_c)[i,d,k] = U_c[i]*V_c[d]*W_c[k]
            # Reshape R to shape (n*m, m*p, p*n), then for each (i,d), R[i,d,:] dot W_c should match
            # sum over all: ||R - U_c[:,None,None]*V_c[None,:,None]*W_c[None,None,:]||^2
            # = sum_{i,d} ||R[i,d,:] - U_c[i]*V_c[d]*W_c||^2
            # Gradient descent or closed form:
            # The optimal W_c minimizes sum_{i,d} ||R[i,d,:] - U_c[i]*V_c[d]*W_c||^2
            # = sum_{i,d} R[i,d,:]^2 - 2*sum_{i,d} R[i,d,:]·(U_c[i]*V_c[d]*W_c) + sum_{i,d} (U_c[i]*V_c[d])^2 * ||W_c||^2
            # d/dW_c: -2*sum_{i,d} R[i,d,:]⊙(U_c[i]*V_c[d]) + 2*sum_{i,d}(U_c[i]*V_c[d])^2 * W_c = 0
            # W_c = [sum_{i,d} (U_c[i]*V_c[d])^2]^{-1} * [sum_{i,d} R[i,d,:]⊙(U_c[i]*V_c[d])]
            
            R2 = R.reshape(n*m, m*p, p*n)
            A = U_c = U[:, c]
            B = V_c = V[:, c]
            AB = np.outer(A, B).reshape(n*m, m*p, 1)
            # sum_{i,d} R[i,d,:] ⊙ (A[i]*B[d])
            numerator = np.tensordot(R2, AB, axes=([0,1],[0,1]))  # shape (p*n,)
            denom = np.tensordot(AB * AB, np.ones((n*m, m*p, 1)), axes=([0,1],[0,1]))
            # Actually denom = sum_{i,d} (A[i]*B[d])^2
            denom = float(np.sum(A**2) * np.sum(B**2))
            if denom > 1e-10:
                W[:, c] = numerator / denom
            else:
                W[:, c] = 0
        
        # Update V similarly (swap roles)
        for c in range(target_rank):
            approx = np.einsum('ic,dc,kc->ikd', U, V, W)
            R = T - approx + np.einsum('ic,dc,kc->ikd',
                                        U[:, c:c+1], V[:, c:c+1], W[:, c:c+1])
            R2 = R.reshape(n*m, m*p, p*n)
            A = U[:, c]
            C = W[:, c]
            AC = np.outer(A, C).reshape(n*m, 1, p*n)
            # For V_c: R[i,d,k] = A[i]*B[d]*C[k], minimize over B
            numerator = np.tensordot(R2, AC, axes=([0,2],[0,2]))  # shape (m*p,)
            denom = float(np.sum(A**2) * np.sum(C**2))
            if denom > 1e-10:
                V[:, c] = numerator / denom
        
        # Update U similarly
        for c in range(target_rank):
            approx = np.einsum('ic,dc,kc->ikd', U, V, W)
            R = T - approx + np.einsum('ic,dc,kc->ikd',
                                        U[:, c:c+1], V[:, c:c+1], W[:, c:c+1])
            R2 = R.reshape(n*m, m*p, p*n)
            B = V[:, c]
            C = W[:, c]
            BC = np.outer(B, C).reshape(1, m*p, p*n)
            numerator = np.tensordot(R2, BC, axes=([1,2],[1,2]))  # shape (n*m,)
            denom = float(np.sum(B**2) * np.sum(C**2))
            if denom > 1e-10:
                U[:, c] = numerator / denom
        
        # Check convergence
        approx = np.einsum('ic,dc,kc->ikd', U, V, W)
        err = np.max(np.abs(approx - T))
        if err < best_err:
            best_err = err
            best_UVW = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-6:
            break
        if time.time() - start_time > time_budget * 1.5:
            break
    
    if best_err < 1e-6 and best_UVW is not None:
        U, V, W = best_UVW
        U, V, W, ok = _snap_to_half_integer(U, V, W)
        if ok:
            # Final verification
            if _verify_decomposition(U, V, W, T, tol=1e-5):
                return U, V, W, target_rank
    
    return None, None, None, None


def construct():
    start = time.time()
    deadline = start + (DEADLINE_SECONDS - 20)  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = SCHOOLBOOK_RANK
    
    # Try to improve via rank search
    T = _build_tensor(n, m, p)
    
    # Try some specific ranks based on theoretical lower bounds
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But actual lower bound is likely higher. Try ranks: 42, 36, 35, 30, 27, 25
    candidates = [42, 36, 35, 32, 30, 27, 25, 24]
    
    for rank in candidates:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 5:
            break
        
        result = _try_direct_rank_search(n, m, p, rank, time_budget=min(15, deadline - time.time() - 10))
        if result[3] is not None:
            U, V, W, found_rank = result
            if found_rank < best_rank:
                best_rank = found_rank
    
    # Return best result found
    if best_rank >= SCHOOLBOOK_RANK:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists
    U_list = [[U[i, j] for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[V[i, j] for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[W[i, j] for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
