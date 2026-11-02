import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Greedy rank-1 decomposition
    def greedy_decompose(T, max_rank=53, time_limit=120):
        T = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(max_rank):
            if time.time() > time_limit:
                break
            
            # Find best rank-1 approximation of T
            # Try random initializations
            best_err = np.inf
            best_U, best_V, best_W = None, None, None
            
            for trial in range(200):
                np.random.seed(r*1000 + trial)
                # Random initialization
                u = np.random.randn(n*m)
                v = np.random.randn(m*p)
                w = np.random.randn(p*n)
                
                # Iterate to find best rank-1 approx
                for it in range(50):
                    # Update u: u = <T, v⊗w> / ||v⊗w||^2
                    vw = np.outer(v, w)
                    denom = np.sum(vw**2)
                    if denom < 1e-12:
                        break
                    u = np.einsum('abc,bc->a', T, vw) / denom
                    
                    # Update v
                    uw = np.outer(u, w)
                    denom = np.sum(uw**2)
                    if denom < 1e-12:
                        break
                    v = np.einsum('abc,ac->b', T, uw) / denom
                    
                    # Update w
                    uv = np.outer(u, v)
                    denom = np.sum(uv**2)
                    if denom < 1e-12:
                        break
                    w = np.einsum('abc,ab->c', T, uv) / denom
                
                # Compute error
                approx = np.einsum('a,b,c->abc', u, v, w)
                err = np.linalg.norm(T - approx)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = u.copy(), v.copy(), w.copy()
            
            if best_U is None:
                break
            
            U_list.append(best_U)
            V_list.append(best_V)
            W_list.append(best_W)
            
            # Subtract the rank-1 term
            T -= np.einsum('a,b,c->abc', best_U, best_V, best_W)
            
            # Check convergence
            if np.linalg.norm(T) < 1e-8:
                break
        
        if U_list:
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            return len(U_list), U, V, W
        return max_rank, None, None, None
    
    # Try greedy decomposition
    rank, U, V, W = greedy_decompose(T, max_rank=52, time_limit=_DEADLINE - 5)
    
    if U is not None and rank < 53:
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to Strassen + trivial (rank 53)
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
        e_idx, f_idx = k0, k1
        g_idx, h_idx = p+k0, p+k1
        
        U[0][r1]=1; U[3][r1]=1; V[e_idx][r1]=1; V[h_idx][r1]=1
        U[2][r2]=1; U[3][r2]=1; V[e_idx][r2]=1
        U[0][r3]=1; V[f_idx][r3]=1; V[h_idx][r3]=-1
        U[3][r4]=1; V[g_idx][r4]=1; V[e_idx][r4]=-1
        U[0][r5]=1; U[1][r5]=1; V[h_idx][r5]=1
        U[2][r6]=1; U[0][r6]=-1; V[e_idx][r6]=1; V[f_idx][r6]=1
        U[1][r7]=1; U[3][r7]=-1; V[g_idx][r7]=1; V[h_idx][r7]=1
        
        W[2*k0][r1]=1; W[2*k0][r4]=1; W[2*k0][r5]=-1; W[2*k0][r7]=1
        W[2*k0+1][r2]=1; W[2*k0+1][r4]=1
        W[2*k1][r3]=1; W[2*k1][r5]=1
        W[2*k1+1][r1]=1; W[2*k1+1][r2]=-1; W[2*k1+1][r3]=1; W[2*k1+1][r6]=1
    
    for i in range(7):
        add_strassen_block(7*i, 2*i, 2*i+1)
    
    br = 49
    U[0][br]=1; V[14][br]=1; W[28][br]=1
    U[1][br+1]=1; V[29][br+1]=1; W[28][br+1]=1
    U[2][br+2]=1; V[14][br+2]=1; W[29][br+2]=1
    U[3][br+3]=1; V[29][br+3]=1; W[29][br+3]=1
    
    return {"U": U, "V": V, "W": W}
