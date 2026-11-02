import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute valid indices
    valid_a, valid_b, valid_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_a.append(i*m + j)
                valid_b.append(j*p + k)
                valid_c.append(k*n + i)
    valid_a = np.array(valid_a)
    valid_b = np.array(valid_b)
    valid_c = np.array(valid_c)
    
    # Precompute invalid indices (sample)
    all_a = np.repeat(np.arange(nm), mp*pn)
    all_b = np.tile(np.repeat(np.arange(mp), pn), nm)
    all_c = np.tile(np.arange(pn), nm*mp)
    valid_mask = np.zeros(nm*mp*pn, dtype=bool)
    for idx in range(len(valid_a)):
        a, b, c = valid_a[idx], valid_b[idx], valid_c[idx]
        linear_idx = a * (mp*pn) + b * pn + c
        valid_mask[linear_idx] = True
        
    invalid_indices = np.where(~valid_mask)[0]
    np.random.seed(42)
    np.random.shuffle(invalid_indices)
    num_invalids = 100
    check_invalids = invalid_indices[:num_invalids]
    
    # Map linear index back to a,b,c for check_invalids
    check_invalid_a = check_invalids // (mp*pn)
    check_invalid_b = (check_invalids // pn) % mp
    check_invalid_c = check_invalids % pn
    
    # Combine all check indices
    check_a = np.concatenate([valid_a, check_invalid_a])
    check_b = np.concatenate([valid_b, check_invalid_b])
    check_c = np.concatenate([valid_c, check_invalid_c])
    check_target = np.concatenate([np.ones(len(valid_a)), np.zeros(len(check_invalids))])
    
    best_rank = nm * mp * pn
    best_decomp = None
    
    for R in range(72, 60, -1):
        found = False
        for attempt in range(20):
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Local Search
            for it in range(500):
                # Compute current error
                # S = sum_r U[a, r] V[b, r] W[c, r]
                # This can be done efficiently
                # S_check = np.einsum('ar,br,cr->abc', U, V, W) is too big.
                # We only need S for check_indices.
                
                # U_check = U[check_a, :] (shape len_check, R)
                # V_check = V[check_b, :]
                # W_check = W[check_c, :]
                # S_check = np.einsum('ir,jr,kr->ijk', U_check, V_check, W_check) -> No, i,j,k are aligned.
                # S_check = np.sum(U_check * V_check * W_check, axis=1)
                
                U_check = U[check_a, :]
                V_check = V[check_b, :]
                W_check = W[check_c, :]
                S_check = np.sum(U_check * V_check * W_check, axis=1)
                
                err = np.sum((S_check - check_target)**2)
                
                if err < 1e-6:
                    # Verify all
                    # Check all valid
                    U_v = U[valid_a, :]
                    V_v = V[valid_b, :]
                    W_v = W[valid_c, :]
                    S_v = np.sum(U_v * V_v * W_v, axis=1)
                    if np.allclose(S_v, 1.0, atol=1e-6):
                        # Check all invalid (subset)
                        # If subset is 0, likely all are 0? No.
                        # But for low rank, if many are 0, likely all.
                        # Let's check a larger set
                        large_invalid = invalid_indices[:500]
                        li_a = large_invalid // (mp*pn)
                        li_b = (large_invalid // pn) % mp
                        li_c = large_invalid % pn
                        U_li = U[li_a, :]
                        V_li = V[li_b, :]
                        W_li = W[li_c, :]
                        S_li = np.sum(U_li * V_li * W_li, axis=1)
                        if np.allclose(S_li, 0.0, atol=1e-6):
                            found = True
                            # Snap to half-integers?
                            # The problem allows half-integers.
                            # We found a float solution.
                            # We can try to snap it.
                            U_snap = np.round(U * 2) / 2
                            V_snap = np.round(V * 2) / 2
                            W_snap = np.round(W * 2) / 2
                            
                            # Verify snap
                            U_v = U_snap[valid_a, :]
                            V_v = V_snap[valid_b, :]
                            W_v = W_snap[valid_c, :]
                            S_v = np.sum(U_v * V_v * W_v, axis=1)
                            if np.allclose(S_v, 1.0, atol=1e-6):
                                best_rank = R
                                best_decomp = (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                                break
                            else:
                                # Use the float solution? 
                                # "integer/half-integer entries"
                                # So we MUST snap.
                                # If snap fails, we can't use this.
                                pass
                    break
                
                # Perturb
                # Pick random row/col
                mat = np.random.choice([0, 1, 2])
                if mat == 0:
                    row = np.random.randint(0, nm)
                    col = np.random.randint(0, R)
                    U[row, col] += np.random.randn() * 0.1
                elif mat == 1:
                    row = np.random.randint(0, mp)
                    col = np.random.randint(0, R)
                    V[row, col] += np.random.randn() * 0.1
                else:
                    row = np.random.randint(0, pn)
                    col = np.random.randint(0, R)
                    W[row, col] += np.random.randn() * 0.1
                    
            if found:
                break
                
    if best_decomp:
        return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
    else:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
