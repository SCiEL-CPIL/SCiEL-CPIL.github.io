import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition."""
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                if s == 0:
                    b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1]
                else:
                    b_map = [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify tensor equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None
    
    U56, V56, W56 = build_strassen_56()
    valid, err = verify_decomposition(U56, V56, W56)
    
    if not valid:
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 56
    best_result = (U56, V56, W56)
    
    def try_rank_reduction(target_rank, max_iters=300, seed=42):
        """Try to find a decomposition of given rank using random V,W and solving for U."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Random V and W with small integer entries
            V_rand = rng.randint(-1, 2, size=(32, target_rank))
            W_rand = rng.randint(-1, 2, size=(16, target_rank))
            
            # For each (i,j), solve M @ u = ones(p) where M[k,r] = V[j*p+k,r]*W[k*n+i,r]
            U_sol = np.zeros((8, target_rank))
            feasible = True
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V_rand[j*p+k, :].astype(float) * W_rand[k*n+i, :].astype(float)
                    
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        feasible = False
                        break
                    
                    # Check if rounded solution works
                    u_rounded = np.round(u)
                    err = np.max(np.abs(M @ u_rounded - np.ones(p)))
                    if err > 1e-9:
                        # Try half-integer rounding
                        u_half = np.round(u * 2) / 2.0
                        err_half = np.max(np.abs(M @ u_half - np.ones(p)))
                        if err_half > 1e-9:
                            feasible = False
                            break
                        U_sol[i*m+j, :] = u_half
                    else:
                        U_sol[i*m+j, :] = u_rounded
            
            if not feasible:
                continue
            
            # Verify full decomposition with integer check
            # Convert to common denominator
            U_check = np.round(U_sol * 2) / 2.0
            V_check = V_rand.astype(float)
            W_check = W_rand.astype(float)
            
            all_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U_check[i*m+j, r] * V_check[j*p+k, r] * W_check[k*n+i, r]
                        if abs(s - 1) > 1e-9:
                            all_ok = False
                            break
                    if not all_ok:
                        break
                if not all_ok:
                    break
            
            if all_ok:
                # Convert to integer representation (multiply by 2 if needed)
                U_out = np.round(U_sol * 2) / 2.0
                if np.allclose(U_out, np.round(U_out)):
                    U_out = U_out.astype(int)
                return (U_out, V_rand, W_rand)
        
        return None
    
    # Try ranks from 48 to 55
    for target_rank in range(48, 56):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_reduction(target_rank, max_iters=200, seed=target_rank * 1000 + 7)
        if result is not None:
            U_r, V_r, W_r = result
            # Final verification
            U_f = np.round(U_r * 2) / 2.0 if not np.issubdtype(U_r.dtype, np.integer) else U_r
            valid, _ = verify_decomposition(
                U_f.astype(int) if np.allclose(U_f, np.round(U_f)) else U_f,
                V_r, W_r
            )
            if valid:
                best_rank = target_rank
                best_result = (U_f, V_r, W_r)
                break
    
    if best_rank < 56:
        U_best, V_best, W_best = best_result
        # Ensure proper types
        if not np.issubdtype(U_best.dtype, np.integer):
            U_best = np.round(U_best * 2) / 2.0
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
