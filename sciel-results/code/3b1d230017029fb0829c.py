import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Precompute non-zero indices of the tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # We store lists of (v, w) for each u
    T_u = [[] for _ in range(nm)]
    T_v = [[] for _ in range(mp)]
    T_w = [[] for _ in range(pn)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_u[u].append((v, w))
                T_v[v].append((u, w))
                T_w[w].append((u, v))

    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try ranks from 89 down to 50
    # We can jump by 1 to be safe, or try specific known good ranks?
    # Let's try a few random restarts for each rank.
    
    best_result = None
    
    for rank in range(n*m*p - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # ALS iterations
            # We need to solve M_U * U_u = b_u for each u
            # M_U is R x R: M[r, r'] = sum_{v,w} V[v,r] V[v,r'] W[w,r] W[w,r']
            # Note: The sum is over all v, w in the full space?
            # The objective is sum_{u,v,w} (T - sum U V W)^2
            # The term for U[u, r] involves sum_{v,w} T[u,v,w] V[v,r] W[w,r]
            # And sum_{v,w} (sum_{r'} U[u,r'] V[v,r'] W[w,r']) V[v,r] W[w,r]
            # = sum_{r'} U[u,r'] sum_{v,w} V[v,r'] V[v,r] W[w,r'] W[w,r]
            # So yes, M is constant for all u.
            
            # Precompute M for U
            # M_U[r, r'] = (V[:, r] * W[:, r]) dot (V[:, r'] * W[:, r'])
            # Vectorized:
            # VW = V * W? No, V is mp x R, W is pn x R.
            # We need sum_v V[v,r] V[v,r'] sum_w W[w,r] W[w,r']?
            # No, the sum is over v AND w.
            # sum_{v,w} V[v,r] V[v,r'] W[w,r] W[w,r'] = (sum_v V[v,r] V[v,r']) * (sum_w W[w,r] W[w,r'])
            # Let G_V = V.T @ V (R x R)
            # Let G_W = W.T @ W (R x R)
            # Then M_U[r, r'] = G_V[r, r'] * G_W[r, r']?
            # No. sum_{v,w} V[v,r] V[v,r'] W[w,r] W[w,r']
            # = sum_v V[v,r] V[v,r'] sum_w W[w,r] W[w,r']
            # This factorizes only if the sum over v and w are independent.
            # But they are not coupled in the sum?
            # The sum is $\sum_{v=0}^{mp-1} \sum_{w=0}^{pn-1} V[v,r] V[v,r'] W[w,r] W[w,r']$
            # $= (\sum_{v} V[v,r] V[v,r']) (\sum_{w} W[w,r] W[w,r'])$
            # $= (V^T V)_{r,r'} (W^T W)_{r,r'}$
            # So $M_U = (V^T V) \circ (W^T W)$ (Hadamard product).
            
            # Similarly for V:
            # M_V = (U^T U) \circ (W^T W)
            # For W:
            # M_W = (U^T U) \circ (V^T V)
            
            # b_u for U:
            # b_u[r] = sum_{(v,w) in T_u[u]} T[u,v,w] V[v,r] W[w,r]
            # Since T[u,v,w]=1, b_u[r] = sum_{(v,w) in T_u[u]} V[v,r] W[w,r]
            
            # Let's run 50 ALS iterations
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                GV = V.T @ V
                GW = W.T @ W
                M_U = GV * GW
                try:
                    M_U_inv = np.linalg.inv(M_U)
                except np.linalg.LinAlgError:
                    M_U_inv = np.linalg.pinv(M_U)
                
                for u in range(nm):
                    b_u = np.zeros(rank)
                    for v, w in T_u[u]:
                        b_u += V[v, :] * W[w, :]
                    U[u, :] = M_U_inv @ b_u
                    
                # Update V
                GU = U.T @ U
                M_V = GU * GW
                try:
                    M_V_inv = np.linalg.inv(M_V)
                except np.linalg.LinAlgError:
                    M_V_inv = np.linalg.pinv(M_V)
                
                for v in range(mp):
                    b_v = np.zeros(rank)
                    for u, w in T_v[v]:
                        b_v += U[u, :] * W[w, :]
                    V[v, :] = M_V_inv @ b_v
                    
                # Update W
                GV = V.T @ V
                M_W = GU * GV
                try:
                    M_W_inv = np.linalg.inv(M_W)
                except np.linalg.LinAlgError:
                    M_W_inv = np.linalg.pinv(M_W)
                
                for w in range(pn):
                    b_w = np.zeros(rank)
                    for u, v in T_w[w]:
                        b_w += U[u, :] * V[v, :]
                    W[w, :] = M_W_inv @ b_w
            
            # Check error
            # Compute reconstructed tensor
            error = 0.0
            for u in range(nm):
                for v, w in T_u[u]:
                    val = np.dot(U[u, :], V[v, :] * W[w, :])
                    if abs(val - 1.0) > 1e-6:
                        error = 1.0
                        break
                if error > 0:
                    break
            
            if error < 1e-6:
                # Try to round
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                exact = True
                for u in range(nm):
                    for v, w in T_u[u]:
                        val = np.dot(U_r[u, :], V_r[v, :] * W_r[w, :])
                        if abs(val - 1.0) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                
                if exact:
                    # Check if entries are integers or half-integers
                    # np.round(x*2)/2 ensures this, but we should verify
                    if np.allclose(U_r * 2, np.round(U_r * 2)) and \
                       np.allclose(V_r * 2, np.round(V_r * 2)) and \
                       np.allclose(W_r * 2, np.round(W_r * 2)):
                        
                        # Convert to list of lists
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }

    return get_schoolbook()
# EVOLVE-BLOCK-END
