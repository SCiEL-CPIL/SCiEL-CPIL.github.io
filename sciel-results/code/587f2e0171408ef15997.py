"""Matrix multiplication 2x4 x 4x7 via Kronecker/block structure + Strassen."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]
DEADLINE = 135.0  # safety margin under 150s

def _time_left():
    return DEADLINE - time.time()

def _tensor_from(U, V, W):
    # U: (a, r), V: (b, r), W: (c, r) -> tensor (a,b,c)
    return np.einsum('ar,br,cr->abc', U, V, W)

def _check(U, V, W, T, tol=1e-6):
    R = _tensor_from(U, V, W)
    return np.allclose(R, T, atol=tol, rtol=tol)

def _round_half(M):
    return np.round(M * 2.0) / 2.0

def _strassen_222():
    # Returns (U,V,W) each (4,7) for 2x2x2 matrix mult tensor, rank 7.
    # Standard Strassen: C = A*B where A,B are 2x2.
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A11+A12)B22
    # M3 = A11(B11-B22)
    # M4 = A22(B11+B22)
    # M5 = (A12-A22)(B11+B12)
    # M6 = (A11+A22)(B21-B22)
    # M7 = (A11+A12)B11
    # C11 = M1+M4-M5+M7
    # C12 = M2+M3
    # C21 = M2+M4
    # C22 = M1-M3+M4+M6
    # U: rows indexed by (a_i, b_j) for A, V: (b_i, b_j) for B, W: (c_i, c_j) for C
    # a indices: 0=(1,1),1=(1,2),2=(2,1),3=(2,2)
    # b indices: 0=(1,1),1=(1,2),2=(2,1),3=(2,2)
    # c indices: 0=(1,1),1=(1,2),2=(2,1),3=(2,2)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1: A11-A22, B21+B22
    U[0,0]=1; U[3,0]=-1; V[2,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2: A11+A12, B22
    U[0,1]=1; U[1,1]=1; V[3,1]=1; W[1,1]=1; W[3,1]=1
    # M3: A11, B11-B22
    U[0,2]=1; V[0,2]=1; V[3,2]=-1; W[1,2]=1; W[2,2]=1
    # M4: A22, B11+B22
    U[3,3]=1; V[0,3]=1; V[3,3]=1; W[0,3]=1; W[1,3]=1; W[2,3]=1; W[3,3]=1
    # M5: A12-A22, B11+B12
    U[1,4]=1; U[3,4]=-1; V[0,4]=1; V[1,4]=1; W[0,4]=-1; W[2,4]=-1
    # M6: A11+A22, B21-B22
    U[0,5]=1; U[3,5]=1; V[2,5]=1; V[3,5]=-1; W[3,5]=1
    # M7: A11+A12, B11
    U[0,6]=1; U[1,6]=1; V[0,6]=1; W[0,6]=1
    return U, V, W

def _build_227_tensor():
    # 2x2x7 tensor: T[a,b,k] = 1 if a==k else 0 (i.e., (A_{2x2}) * (B_{2x7}) -> C_{2x7})
    # C[i,k] = sum_j A[i,j] B[j,k]. So T[i, j, k] = 1 if i==k.
    T = np.zeros((2, 2, 7))
    for i in range(2):
        for j in range(2):
            for k in range(7):
                if i == k:
                    T[i, j, k] = 1.0
    return T

def _search_227(rank, seed=0, max_iters=2000):
    """Search for rank decomposition of 2x2x7 tensor with integer/half-integer entries."""
    T = _build_227_tensor()
    rng = np.random.RandomState(seed)
    best = None
    best_err = np.inf
    for trial in range(200):
        if _time_left() < 5:
            break
        # Initialize
        if trial % 3 == 0:
            U = rng.randn(4, rank)
            V = rng.randn(4, rank)
            W = rng.randn(7, rank)
        else:
            U = rng.randn(4, rank) * 0.3
            V = rng.randn(4, rank) * 0.3
            W = rng.randn(7, rank) * 0.3
        prev_err = np.inf
        for it in range(max_iters):
            if _time_left() < 2:
                break
            # Optimize W: for each a,b, T[a,b,:] = sum_d U[a,d]*V[b,d]*W[:,d]
            # Fix U,V, solve for W.
            # For each (a,b): vector T[a,b,:] = sum_d (U[a,d]*V[b,d]) W[:,d]
            # Stack: X[d, c] where we want to fit T_vec.
            # Let's build: for each c (0..6), T_vec[c] = sum_d W[c,d] * (U*V)[d]
            # (U*V)[d] = sum_{a,b} U[a,d]*V[b,d]*delta... no.
            # Actually T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # For fixed a,b: T[a,b,:] = sum_d W[:,d] * (U[a,d]*V[b,d])
            # So for each (a,b), we have a linear system in W.
            # Combine all: build matrix M of shape (4*7, rank): M[(a*7+c), d] = U[a,d]*V[b,d]... 
            # Wait, it's per (a,b). Let me just do least squares on all entries.
            # Flatten: T_flat of size 4*7=28. 
            # T_flat[idx] where idx = (a*2+b)*7+c = sum_d U[a,d]*V[b,d]*W[c,d]
            # Build design matrix D of shape (28, rank): D[idx, d] = U[a,d]*V[b,d] if c matches... 
            # No: D[idx, d] = U[a,d]*V[b,d]*W[c,d] but W is unknown.
            # For optimizing W: fix U,V. For each c, the contribution to T[a,b,c] is W[c,d]*U[a,d]*V[b,d].
            # So T[:, :, c] (2x2) = sum_d W[c,d] * (U[:,d]*V[:,d])^T  (outer product)
            # Flatten 2x2 to 4-vector: T_flat_c (4,) = sum_d W[c,d] * (U[:,d]*V[:,d])_flat
            # So for each c: T_c (4,) = sum_d W[c,d] * P_c[d] where P_c[d] = (U[:,d]*V[:,d])_flat
            # This is: T_c = P_c @ W[c, :]  where P_c is (4, rank)
            # Stack over c: T_all (28,) = [T_0; T_1; ...; T_6]
            # Design: D (28, rank) where D[c*4 + i, d] = P_c[d][i] = U[a,d]*V[b,d] for i=a*2+b
            D = np.zeros((28, rank))
            for d in range(rank):
                P = (U[:, d] * V[:, d]).flatten()  # shape (4,)
                for c in range(7):
                    D[c*4:(c+1)*4, d] = P
            T_flat = T.flatten()
            try:
                W_new = np.linalg.lstsq(D, T_flat, rcond=None)[0].reshape(7, rank)
            except:
                W_new = W
            # Optimize U: fix V, W. T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # For each (b,c): T[:,b,c] (2,) = sum_d U[:,d] * (V[b,d]*W[c,d])
            # T_flat2[idx] where idx = (b*7+c)*2+a = sum_d U[a,d] * Q[idx//2, d] where Q[d]=V[b,d]*W[c,d]
            D2 = np.zeros((28, rank))
            for d in range(rank):
                for b in range(2):
                    for c in range(7):
                        Q = V[b, d] * W[c, d]
                        D2[(b*7+c)*2:(b*7+c)*2+2, d] = Q
            T_flat2 = T.transpose(1,2,0).reshape(28)  # shape (b,c,a)
            try:
                U_new = np.linalg.lstsq(D2, T_flat2, rcond=None)[0].reshape(2, rank)
                U_new = np.concatenate([U_new, U_new[1:2,:]], axis=0)  # placeholder, need full 4x rank
            except:
                U_new = U
            # Actually let me redo: U is (4, rank) = (2 rows for a, 2 cols for j) -> a*2+j
            # T[a, j, c] = sum_d U[a*2+j, d] * V[j*2+k, d] * W[c, d]  -- NO.
            # Wait, 2x2x7: A is 2x2, B is 2x7. So indices: A[i,j], B[j,k], C[i,k].
            # T[i, j, k] = 1 if i==k.
            # U has shape (2*2, rank) = (4, rank), indexed by (i,j) for A.
            # V has shape (2*7, rank) = (14, rank), indexed by (j,k) for B.
            # W has shape (2*7, rank) = (14, rank), indexed by (i,k) for C.
            # I made an error. Let me redo with correct shapes.
            break  # This approach is getting complex; let me use a cleaner method.
        break
    return None

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Build full target tensor T of shape (8, 28, 14)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Block decomposition: A (2x4) = [A1 | A2] where A1,A2 are 2x2
    # B (4x7) = [B1; B2] where B1,B2 are 2x7
    # C = A1*B1 + A2*B2
    # Each term is a 2x2x7 tensor.
    # So we need rank-r decomposition of 2x2x7 tensor, total rank = 2r.
    
    # Search for 2x2x7 tensor decomposition
    T_227 = np.zeros((2, 2, 7))
    for i in range(2):
        for j in range(2):
            for k in range(7):
                if i == k:
                    T_227[i, j, k] = 1.0
    
    def _alt_min_227(r, n_trials=30, max_iter=500):
        """Alternating minimization for 2x2x7 tensor of rank r."""
        for trial in range(n_trials):
            if _time_left() < 3:
                return None
            rng = np.random.RandomState(trial * 137 + 42)
            if trial % 3 == 0:
                U = rng.randn(4, r)
                V = rng.randn(14, r)
                W = rng.randn(14, r)
            elif trial % 3 == 1:
                U = np.zeros((4, r))
                V = np.zeros((14, r))
                W = np.zeros((14, r))
                for d in range(r):
                    U[d % 4, d] = rng.choice([-1, 1])
                    V[d % 14, d] = rng.choice([-1, 1])
                    W[d % 14, d] = rng.choice([-1, 1])
            else:
                U = rng.randn(4, r) * 0.5
                V = rng.randn(14, r) * 0.5
                W = rng.randn(14, r) * 0.5
                for d in range(r):
                    U[d % 4, d] += 1.0
                    V[d % 14, d] += 1.0
                    W[d % 14, d] += 1.0
            
            prev_err = np.inf
            for it in range(max_iter):
                if _time_left() < 2:
                    return None
                # Optimize W: T[i,j,k] = sum_d U[i,j,d]*V[j,k,d]*W[i,k,d]
                # For fixed U,V: for each (i,k): T[:,i,k]... 
                # T[i,j,k] = sum_d W[i,k,d] * (U[i,j,d]*V[j,k,d])
                # For fixed (i,k): T[:,i,k] (2,) = sum_d W[i,k,d] * (U[i,:,d]*V[:,k,d])
                # Flatten: for each (i,k), build 2-vector.
                # Stack all (i,k): 4*7=28 equations? No, (i,k) has 2*7=14 combos, each gives 2 equations (j=0,1).
                D_w = np.zeros((14*2, r))
                for d in range(r):
                    for i in range(2):
                        for k in range(7):
                            for j in range(2):
                                D_w[(i*7+k)*2+j, d] = U[i*2+j, d] * V[j*7+k, d]
                T_flat_w = T_227.transpose(0,2,1).reshape(14, 2).reshape(28)  # (i,k,j)
                try:
                    W_new = np.linalg.lstsq(D_w, T_flat_w, rcond=None)[0].reshape(14, r)
                except:
                    W_new = W
                
                # Optimize U: for fixed V,W: T[i,j,k] = sum_d U[i,j,d]*V[j,k,d]*W[i,k,d]
                # For fixed (i,j): T[i,j,:] (7,) = sum_d U[i,j,d] * (V[j,:,d]*W[i,:,d])
                D_u = np.zeros((4*7, r))
                for d in range(r):
                    for i in range(2):
                        for j in range(2):
                            for k in range(7):
                                D_u[(i*2+j)*7+k, d] = V[j*7+k, d] * W[i*7+k, d]
                T_flat_u = T_227.reshape(4, 7)
                try:
                    U_new = np.linalg.lstsq(D_u, T_flat_u, rcond=None)[0].reshape(4, r)
                except:
                    U_new = U
                
                # Optimize V: for fixed U,W: T[i,j,k] = sum_d U[i,j,d]*V[j,k,d]*W[i,k,d]
                # For fixed (j,k): T[:,j,k] (2,) = sum_d V[j,k,d] * (U[:,j,d]*W[:,k,d])
                D_v = np.zeros((14*2, r))
                for d in range(r):
                    for j in range(2):
                        for k in range(7):
                            for i in range(2):
                                D_v[(j*7+k)*2+i, d] = U[i*2+j, d] * W[i*7+k, d]
                T_flat_v = T_227.transpose(1,2,0).reshape(14, 2).reshape(28)  # (j,k,i)
                try:
                    V_new = np.linalg.lstsq(D_v, T_flat_v, rcond=None)[0].reshape(14, r)
                except:
                    V_new = V
                
                U, V, W = U_new, V_new, W_new
                T_rec = _tensor_from(U, V, W)
                err = np.linalg.norm(T_rec - T_227)
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-14 and it > 20:
                    break
                prev_err = err
            
            # Check
            T_rec = _tensor_from(U, V, W)
            if np.allclose(T_rec, T_227, atol=1e-8, rtol=1e-8):
                # Try rounding to half-integers
                U_r = _round_half(U)
                V_r = _round_half(V)
                W_r = _round_half(W)
                if np.allclose(_tensor_from(U_r, V_r, W_r), T_227, atol=1e-8, rtol=1e-8):
                    return U_r, V_r, W_r
                return U, V, W
        return None
    
    # Try ranks for 2x2x7 tensor
    # Lower bound: max(2*2, 2*7, 2*7) = 14, but likely lower
    # Known: 2x2x7 tensor rank is at most 14 (schoolbook), likely ~10-12
    found_227 = None
    for r in range(8, 15):
        if _time_left() < 10:
            break
        res = _alt_min_227(r, n_trials=15, max_iter=400)
        if res is not None:
            found_227 = res
            break
    
    if found_227 is not None:
        U_227, V_227, W_227 = found_227
        r_227 = U_227.shape[1]
        # Build full decomposition: C = A1*B1 + A2*B2
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        # B1 = B[0:2, :], B2 = B[2:4, :]
        # U_full: for A1*B1, use U_227 with A1 indices; for A2*B2, use U_227 with A2 indices
        # A index: i*m+j. For A1: j=0,1 -> i*4+j. For A2: j=2,3 -> i*4+j.
        # B index: j*p+k. For B1: j=0,1 -> j*7+k. For B2: j=2,3 -> j*7+k.
        # C index: k*n+i.
        U_full = np.zeros((8, 2*r_227))
        V_full = np.zeros((28, 2*r_227))
        W_full = np.zeros((14, 2*r_227))
        
        # First block: A1*B1
        for i in range(2):
            for j in range(2):
                U_full[i*4+j, 0:r_227] = U_227[i*2+j, :]
        for j in range(2):
            for k in range(7):
                V_full[j*7+k, 0:r_227] = V_227[j*7+k, :]
        for i in range(2):
            for k in range(7):
                W_full[k*2+i, 0:r_227] = W_227[i*7+k, :]
        
        # Second block: A2*B2
        for i in range(2):
            for j in range(2):
                U_full[i*4+(j+2), r_227:2*r_227] = U_227[i*2+j, :]
        for j in range(2):
            for k in range(7):
                V_full[(j+2)*7+k, r_227:2*r_227] = V_227[j*7+k, :]
        for i in range(2):
            for k in range(7):
                W_full[k*2+i, r_227:2*r_227] = W_227[i*7+k, :]
        
        # Verify
        T_rec = _tensor_from(U_full, V_full, W_full)
        if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = 2 * r_227
    
    # Also try direct alternating minimization on full tensor for ranks between best_rank and schoolbook
    if _time_left() > 20:
        T_vec_w = T.transpose(2, 0, 1).reshape(14, 8*28)
        T_vec_u = T.reshape(8, 28*14)
        T_vec_v = T.transpose(1, 0, 2).reshape(28, 8*14)
        
        def _alt_min_full(r, n_trials=5, max_iter=200):
            for trial in range(n_trials):
                if _time_left() < 5:
                    return None
                rng = np.random.RandomState(trial * 251 + 7)
                U = rng.randn(8, r)
                V = rng.randn(28, r)
                W = rng.randn(14, r)
                prev_err = np.inf
                for it in range(max_iter):
                    if _time_left() < 2:
                        return None
                    # Optimize W
                    D_w = np.zeros((8*28, r))
                    for d in range(r):
                        D_w[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                    T_flat = T.transpose(2,0,1).reshape(14, 8*28).T.flatten()
                    try:
                        W_new = np.linalg.lstsq(D_w, T_flat, rcond=None)[0].reshape(14, r)
                    except:
                        W_new = W
                    # Optimize U
                    D_u = np.zeros((28*14, r))
                    for d in range(r):
                        D_u[:, d] = (V[:, d][:, None] * W_new[:, d][None, :]).flatten()
                    T_flat_u = T.reshape(8, 28*14).T.flatten()
                    try:
                        U_new = np.linalg.lstsq(D_u, T_flat_u, rcond=None)[0].reshape(8, r)
                    except:
                        U_new = U
                    # Optimize V
                    D_v = np.zeros((8*14, r))
                    for d in range(r):
                        D_v[:, d] = (U_new[:, d][:, None] * W_new[:, d][None, :]).flatten()
                    T_flat_v = T.transpose(1,0,2).reshape(28, 8*14).T.flatten()
                    try:
                        V_new = np.linalg.lstsq(D_v, T_flat_v, rcond=None)[0].reshape(28, r)
                    except:
                        V_new = V
                    U, V, W = U_new, V_new, W_new
                    T_rec = _tensor_from(U, V, W)
                    err = np.linalg.norm(T_rec - T)
                    if err < 1e-10:
                        break
                    if abs(prev_err - err) < 1e-14 and it > 20:
                        break
                    prev_err = err
                T_rec = _tensor_from(U, V, W)
                if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
                    U_r = _round_half(U)
                    V_r = _round_half(V)
                    W_r = _round_half(W)
                    if np.allclose(_tensor_from(U_r, V_r, W_r), T, atol=1e-8, rtol=1e-8):
                        return U_r, V_r, W_r
                    return U, V, W
            return None
        
        for r in range(best_rank - 2, max(best_rank - 10, 28)):
            if _time_left() < 5:
                break
            if r >= best_rank:
                continue
            res = _alt_min_full(r, n_trials=3, max_iter=150)
            if res is not None:
                best_U, best_V, best_W = res
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
