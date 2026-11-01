def find_233_decomp_local(target_rank, deadline):
    """Optimized local search for 2x3x3 decomposition."""
    import random
    
    def init_random():
        U = [[random.choice([-1,0,1]) for _ in range(target_rank)] for _ in range(6)]
        V = [[random.choice([-1,0,1]) for _ in range(target_rank)] for _ in range(9)]
        W = [[random.choice([-1,0,1]) for _ in range(target_rank)] for _ in range(6)]
        return U, V, W
    
    def compute_all_sums(U, V, W):
        sums = [[[0]*3 for _ in range(3)] for _ in range(2)]
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(target_rank):
                        s += U[i*3+j][c] * V[j*3+k][c] * W[k*2+i][c]
                    sums[i][j][k] = s
        return sums
    
    def total_error(sums):
        return sum((sums[i][j][k]-1)**2 for i in range(2) for j in range(3) for k in range(3))
    
    for restart in range(30):
        if time.time() > deadline:
            return None
        
        U, V, W = init_random()
        sums = compute_all_sums(U, V, W)
        err = total_error(sums)
        
        if err == 0:
            return {"U": U, "V": V, "W": W}
        
        stuck_count = 0
        for it in range(100000):
            if time.time() > deadline:
                return None
            
            best_delta = 0
            best_flip = None
            
            # Try all U flips
            for ij in range(6):
                i, j = divmod(ij, 3)
                for c in range(target_rank):
                    old = U[ij][c]
                    for new in [-1,0,1]:
                        if new == old:
                            continue
                        # Compute delta in error
                        delta_err = 0
                        for k in range(3):
                            old_s = sums[i][j][k]
                            # New sum = old_s + (new - old) * V[j*3+k][c] * W[k*2+i][c]
                            new_s = old_s + (new - old) * V[j*3+k][c] * W[k*2+i][c]
                            delta_err += (new_s-1)**2 - (old_s-1)**2
                        if delta_err < -best_delta:
                            best_delta = -delta_err
                            best_flip = ('U', ij, c, new)
            
            # Try all V flips
            for jk in range(9):
                j, k = divmod(jk, 3)
                for c in range(target_rank):
                    old = V[jk][c]
                    for new in [-1,0,1]:
                        if new == old:
                            continue
                        delta_err = 0
                        for i in range(2):
                            old_s = sums[i][j][k]
                            new_s = old_s + (new - old) * U[i*3+j][c] * W[k*2+i][c]
                            delta_err += (new_s-1)**2 - (old_s-1)**2
                        if delta_err < -best_delta:
                            best_delta = -delta_err
                            best_flip = ('V', jk, c, new)
            
            # Try all W flips
            for ki in range(6):
                k, i = divmod(ki, 2)
                for c in range(target_rank):
                    old = W[ki][c]
                    for new in [-1,0,1]:
                        if new == old:
                            continue
                        delta_err = 0
                        for j in range(3):
                            old_s = sums[i][j][k]
                            new_s = old_s + (new - old) * U[i*3+j][c] * V[j*3+k][c]
                            delta_err += (new_s-1)**2 - (old_s-1)**2
                        if delta_err < -best_delta:
                            best_delta = -delta_err
                            best_flip = ('W', ki, c, new)
            
            if best_flip is None:
                break
            
            # Apply flip
            arr, idx, c, new_val = best_flip
            if arr == 'U':
                old = U[idx][c]
                U[idx][c] = new_val
                i, j = divmod(idx, 3)
                for k in range(3):
                    sums[i][j][k] += (new_val - old) * V[j*3+k][c] * W[k*2+i][c]
            elif arr == 'V':
                old = V[idx][c]
                V[idx][c] = new_val
                j, k = divmod(idx, 3)
                for i in range(2):
                    sums[i][j][k] += (new_val - old) * U[i*3+j][c] * W[k*2+i][c]
            else:
                old = W[idx][c]
                W[idx][c] = new_val
                k, i = divmod(idx, 2)
                for j in range(3):
                    sums[i][j][k] += (new_val - old) * U[i*3+j][c] * V[j*3+k][c]
            
            err = total_error(sums)
            if err == 0:
                return {"U": U, "V": V, "W": W}
    
    return None
