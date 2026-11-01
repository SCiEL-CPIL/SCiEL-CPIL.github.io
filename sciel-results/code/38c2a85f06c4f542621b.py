import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 12
    nm, mp, pn = n*m, m*p, p*n  # 4, 24, 24
    
    def residual(U, V, W):
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res += (val - 1.0) ** 2
        return res
    
    # Strassen rank-42 solution
    def strassen_solution():
        num_blocks = p // 2
        rank_per_block = 7
        total_rank = num_blocks * rank_per_block
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        for b in range(num_blocks):
            base = b * rank_per_block
            k0, k1 = 2*b, 2*b+1
            
            # M1
            U[0, base+0] += 1; U[3, base+0] += 1
            V[0*2+k0, base+0] += 1; V[0*2+k1, base+0] += 1
            W[k0*2+0, base+0] += 1; W[k1*2+1, base+0] += 1
            # M2
            U[2, base+1] += 1; U[3, base+1] += 1
            V[0*2+k0, base+1] += 1
            W[k0*2+1, base+1] += 1; W[k1*2+1, base+1] += -1
            # M3
            U[0, base+2] += 1
            V[1*2+k0, base+2] += 1; V[1*2+k1, base+2] += -1
            W[k1*2+0, base+2] += 1; W[k1*2+1, base+2] += 1
            # M4
            U[3, base+3] += 1
            V[0*2+k1, base+3] += 1; V[0*2+k0, base+3] += -1
            W[k0*2+0, base+3] += 1; W[k0*2+1, base+3] += 1
            # M5
            U[0, base+4] += 1; U[1, base+4] += 1
            V[1*2+k1, base+4] += 1
            W[k0*2+0, base+4] += -1; W[k1*2+0, base+4] += 1
            # M6
            U[2, base+5] += 1; U[0, base+5] += -1
            V[0*2+k0, base+5] += 1; V[1*2+k0, base+5] += 1
            W[k1*2+1, base+5] += 1
            # M7
            U[1, base+6] += 1; U[3, base+6] += -1
            V[0*2+k1, base+6] += 1; V[1*2+k1, base+6] += 1
            W[k0*2+0, base+6] += 1
        
        return U, V, W, total_rank
    
    U_str, V_str, W_str, rank_str = strassen_solution()
    
    # Verify Strassen
    res_str = residual(U_str, V_str, W_str)
    
    best_U, best_V, best_W = U_str, V_str, W_str
    best_rank = rank_str
    
    # Try to find lower rank via random search
    for target_rank in range(24, rank_str):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_trials = 300
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + trial)
            scale = 1.0
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            prev_res = np.inf
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for idx in range(nm):
                    i, j = divmod(idx, m)
                    D = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    try:
                        U[idx, :] = np.linalg.lstsq(D, np.ones(p), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for idx in range(mp):
                    j, k = divmod(idx, p)
                    D = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    try:
                        V[idx, :] = np.linalg.lstsq(D, np.ones(n), rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for idx in range(pn):
                    k, i = divmod(idx, n)
                    D = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    try:
                        W[idx, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                    except:
                        pass
                
                res = residual(U, V, W)
                if res < 1e-12:
                    break
                if it > 500 and res > prev_res * 1.1:
                    break
                prev_res = res
            
            res = residual(U, V, W)
            if res < 1e-8:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
