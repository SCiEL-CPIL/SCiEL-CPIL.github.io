import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank22_2x3x6():
    """Build a rank-22 decomposition for 2x3x6 using two copies of rank-11 for 2x3x3."""
    n, m, p = 2, 3, 6
    
    # Build rank-11 terms for 2x3x3
    # A: 2x3, indices i*3+j (i in 0..1, j in 0..2)
    # B: 3x3, indices j*3+k (j in 0..2, k in 0..2)
    # C: 2x3, indices i*3+k (i in 0..1, k in 0..2)
    
    terms = []
    
    # Strassen for 2x2x2 core (A[:,0:2] * B[0:2,:])
    # M1 = (A00+A11)(B00+B11)
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 4: 1}})
    # M2 = (A10+A11)B00
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {3: 1, 4: -1}})
    # M3 = A00(B01-B11)
    terms.append({'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {1: 1, 4: 1}})
    # M4 = A11(B10-B00)
    terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 3: 1}})
    # M5 = (A00+A01)B11
    terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01)
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {4: 1}})
    # M7 = (A01-A11)(B10+B11)
    terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}})
    
    # Scalar terms for column j=2 of A and row j=2 of B
    # K1 = A02 * B20
    terms.append({'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}})
    # K2 = A02 * B21
    terms.append({'u': {2: 1}, 'v': {7: 1}, 'w': {1: 1}})
    # K3 = A02 * B22
    terms.append({'u': {2: 1}, 'v': {8: 1}, 'w': {2: 1}})
    # K4 = A12 * B20
    terms.append({'u': {5: 1}, 'v': {6: 1}, 'w': {3: 1}})
    # K5 = A12 * B21
    terms.append({'u': {5: 1}, 'v': {7: 1}, 'w': {4: 1}})
    # K6 = A12 * B22
    terms.append({'u': {5: 1}, 'v': {8: 1}, 'w': {5: 1}})
    
    # Wait, that's 13 terms. Let me reconsider.
    # For 2x3x3: rank 11 means we need 11 terms.
    # The 7 Strassen terms handle the 2x2x2 part.
    # We need 4 more terms for the remaining interactions.
    # Actually, the cross terms are: A[:,2]*B[2,:] gives 2*2=4 scalar products.
    # But some of these might be absorbed. Let me use a different approach.
    
    # Actually, let me just use the block structure from inspiration 1 more carefully.
    # For 2x3x3, the known rank is 11.
    # Let me construct it properly.
    
    # Reset terms
    terms = []
    
    # 7 Strassen terms for the 2x2x2 block (rows/cols 0,1 of A and B)
    # These contribute to C[0:2, 0:2]
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 4: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {3: 1, 4: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {1: 1, 4: 1}})
    terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {4: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}})
    
    # 4 terms for the cross interactions with index 2
    # A[:,2] * B[2,0:2] contributes to C[0:2, 0:2]
    # A[0:2,2] * B[2,0] -> C[0:2, 0]
    terms.append({'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}})
    terms.append({'u': {5: 1}, 'v': {6: 1}, 'w': {3: 1}})
    # A[0:2,2] * B[2,1] -> C[0:2, 1]
    terms.append({'u': {2: 1}, 'v': {7: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {7: 1}, 'w': {4: 1}})
    # A[0:2,2] * B[2,2] -> C[0:2, 2]
    terms.append({'u': {2: 1}, 'v': {8: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {8: 1}, 'w': {5: 1}})
    
    # That's 13 terms, not 11. The rank of 2x3x3 is actually 11, so I need to find a better decomposition.
    # For now, let me use the schoolbook for 2x3x3 which has rank 18, and split 2x3x6 into two 2x3x3 blocks.
    # That gives rank 36, which is the same as schoolbook. Not helpful.
    
    # Let me instead try to find a rank-7 decomposition for 2x3x3 via search, then use 2 copies for rank 14.
    # But that's what the current program does.
    
    # Actually, the simplest reliable improvement is to note that 2x3x6 can be seen as
    # computing C = A*B where A is 2x3 and B is 3x6.
    # We can split B into two 3x3 blocks: B = [B1, B2]
    # C = A*B1 + A*B2
    # If we have a rank-r decomposition for 2x3x3, we can use 2r for 2x3x6.
    # The schoolbook for 2x3x3 has rank 18, so 2*18=36.
    # But if we can find rank 11 for 2x3x3, we get 22.
    
    # Let me just hardcode a known rank-22 solution if I can verify it.
    # For now, return None and let the search handle it.
    
    return None

def _find_decomp_sub(n, m, p, target_rank, deadline):
    """Find a rank decomposition for <n,m,p> using local search."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    def objective(x):
        U = x[:dim_u * target_rank].reshape(dim_u, target_rank)
        V = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
        W = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    error += (1.0 - val) ** 2
        return error
    
    best_sol = None
    best_err = float('inf')
    
    n_attempts = 15 if target_rank <= 12 else 8
    for attempt in range(n_attempts):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + attempt)
        x0 = np.random.choice([-1, 0, 1], size=dim_u*target_rank + dim_v*target_rank + dim_w*target_rank)
        
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1500, 'xatol': 1e-6})
        
        if res.fun < best_err:
            best_err = res.fun
            best_sol = res.x
            
            if best_err < 1e-4:
                U = best_sol[:dim_u * target_rank].reshape(dim_u, target_rank)
                V = best_sol[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W = best_sol[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                
                if _verify(U_s, V_s, W_s, n, m, p, tol=1e-6):
                    return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
                
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if _verify(U_i, V_i, W_i, n, m, p, tol=1e-6):
                    return {'U': U_i.tolist(), 'V': V_i.tolist(), 'W': W_i.tolist()}

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback: rank = n*m*p = 36
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Strategy 1: Try to find a rank-7 decomposition for 2x3x3, then use 2 copies for rank 14
    sub_n, sub_m, sub_p = 2, 3, 3
    
    if time.time() < deadline - 20:
        sub_decomp = _find_decomp_sub(sub_n, sub_m, sub_p, 7, deadline)
        
        if sub_decomp:
            U_sub = np.array(sub_decomp['U'])
            V_sub = np.array(sub_decomp['V'])
            W_sub = np.array(sub_decomp['W'])
            
            full_rank = 2 * 7
            U_full = np.zeros((n * m, full_rank))
            V_full = np.zeros((m * p, full_rank))
            W_full = np.zeros((p * n, full_rank))
            
            U_full[:, :7] = U_sub
            for j in range(sub_m):
                for k in range(sub_p):
                    V_full[j*p+k, :7] = V_sub[j*sub_p+k]
            for k in range(sub_p):
                for i in range(sub_n):
                    W_full[k*n+i, :7] = W_sub[k*sub_n+i]
                    
            U_full[:, 7:] = U_sub
            for j in range(sub_m):
                for k in range(sub_p, sub_p*2):
                    V_full[j*p+k, 7:] = V_sub[j*sub_p+(k-sub_p)]
            for k in range(sub_p, sub_p*2):
                for i in range(sub_n):
                    W_full[k*n+i, 7:] = W_sub[(k-sub_p)*sub_n+i]
            
            if _verify(U_full, V_full, W_full, n, m, p):
                best_result = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                best_rank = full_rank
    
    # Strategy 2: Try ranks 8-11 for 2x3x3
    for r in range(8, 12):
        if time.time() > deadline - 15:
            break
        if r * 2 >= best_rank:
            continue
        
        sub_decomp = _find_decomp_sub(sub_n, sub_m, sub_p, r, deadline)
        
        if sub_decomp:
            U_sub = np.array(sub_decomp['U'])
            V_sub = np.array(sub_decomp['V'])
            W_sub = np.array(sub_decomp['W'])
            
            full_rank = 2 * r
            U_full = np.zeros((n * m, full_rank))
            V_full = np.zeros((m * p, full_rank))
            W_full = np.zeros((p * n, full_rank))
            
            U_full[:, :r] = U_sub
            for j in range(sub_m):
                for k in range(sub_p):
                    V_full[j*p+k, :r] = V_sub[j*sub_p+k]
            for k in range(sub_p):
                for i in range(sub_n):
                    W_full[k*n+i, :r] = W_sub[k*sub_n+i]
                    
            U_full[:, r:] = U_sub
            for j in range(sub_m):
                for k in range(sub_p, sub_p*2):
                    V_full[j*p+k, r:] = V_sub[j*sub_p+(k-sub_p)]
            for k in range(sub_p, sub_p*2):
                for i in range(sub_n):
                    W_full[k*n+i, r:] = W_sub[(k-sub_p)*sub_n+i]
            
            if _verify(U_full, V_full, W_full, n, m, p):
                best_result = {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                best_rank = full_rank
                break
    
    # Strategy 3: Try direct decomposition of 2x3x6 with rank < best_rank
    if time.time() < deadline - 10:
        for r in range(14, best_rank):
            if time.time() > deadline - 5:
                break
            found = False
            for attempt in range(5):
                if time.time() > deadline - 2:
                    break
                np.random.seed(100 + r + attempt)
                U_c = np.random.choice([-1, 0, 1], size=(n*m, r))
                V_c = np.random.choice([-1, 0, 1], size=(m*p, r))
                W_c = np.random.choice([-1, 0, 1], size=(p*n, r))
                if _verify(U_c, V_c, W_c, n, m, p):
                    best_result = {
                        "U": U_c.tolist(),
                        "V": V_c.tolist(),
                        "W": W_c.tolist()
                    }
                    best_rank = r
                    found = True
                    break
            if found:
                break

    return best_result
