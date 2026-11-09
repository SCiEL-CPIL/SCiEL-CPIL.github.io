import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 2, 3, 9
    
    # Target rank 27
    target_rank = 27
    block_rank = 9
    
    # We will search for the sub-decomposition
    # U_sub: 6 x 9
    # V_sub_q: 9 x 9
    # W_sub_q: 6 x 9
    
    # Random search for U_sub, V_sub_0, W_sub_0
    # Then derive V_sub_1, W_sub_1, etc?
    # No, V and W depend on k.
    # But the structure is the same.
    # We can just search for one set and reuse?
    # No, V and W are different.
    # But we can search for all 3 sets simultaneously?
    # Or just search for one set and assume the others are similar?
    # Actually, the problem is identical for each block.
    # So we can find one solution for 2x3x3 and use it for all 3 blocks.
    # But V and W indices are shifted.
    # V[j, k] for block q is V_sub[j, k-3q].
    # W[k, i] for block q is W_sub[k-3q, i].
    # So if we find U_sub, V_sub, W_sub for 2x3x3,
    # we can construct U, V, W for 2x3x9.
    
    # Let's find U_sub, V_sub, W_sub for 2x3x3 with rank 9.
    
    found = False
    for seed in range(1000):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(seed)
        U_sub = rng.choice([-1, 0, 1], size=(6, 9))
        V_sub = rng.choice([-1, 0, 1], size=(9, 9))
        W_sub = rng.choice([-1, 0, 1], size=(6, 9))
        
        # Verify 2x3x3
        ok = True
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(9):
                        total += U_sub[i*3+j, r] * V_sub[j*3+k, r] * W_sub[k*2+i, r]
                    if total != 1:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if ok:
            found = True
            break
            
    if not found:
        # Fallback to 54
        return {"U": [[1 if (i*m+j)%54==r else 0 for r in range(54)] for i in range(2) for j in range(3)],
                "V": [[1 if (j*p+k)%54==r else 0 for r in range(54)] for j in range(3) for k in range(9)],
                "W": [[1 if (k*n+i)%54==r else 0 for r in range(54)] for k in range(9) for i in range(2)]}
        
    # Construct full decomposition
    U = np.zeros((6, 27), dtype=int)
    V = np.zeros((27, 27), dtype=int)
    W = np.zeros((18, 27), dtype=int)
    
    for q in range(3):
        r_offset = q * 9
        U[:, r_offset:r_offset+9] = U_sub
        # V indices: j*p+k. p=9.
        # For block q, k = 3q + k_sub.
        # V[j, 3q+k_sub] = V_sub[j, k_sub]
        for j in range(3):
            for k_sub in range(3):
                V[j*9 + 3*q + k_sub, r_offset:r_offset+9] = V_sub[j*3+k_sub, :]
        # W indices: k*n+i. n=2.
        # For block q, k = 3q + k_sub.
        # W[3q+k_sub, i] = W_sub[k_sub, i]
        for k_sub in range(3):
            for i in range(2):
                W[(3*q+k_sub)*2 + i, r_offset:r_offset+9] = W_sub[k_sub*2+i, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
