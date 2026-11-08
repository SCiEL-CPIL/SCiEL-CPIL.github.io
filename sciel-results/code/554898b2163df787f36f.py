import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 110
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Step 1: Find a low-rank decomposition for 3x3x3
    # Tensor T333[i*3+j, j*3+k, k*3+i] = 1
    T333 = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T333[i*3 + j, j*3 + k, k*3 + i] = 1.0
                
    # Flatten T333 for faster access in ALS
    T333_flat_U = T333.reshape(9, 81) # T333[a, :, :]
    T333_flat_V = T333.transpose(1, 0, 2).reshape(9, 81) # T333[:, b, :]
    T333_flat_W = T333.transpose(2, 0, 1).reshape(9, 81) # T333[:, :, c]

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T333."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T333, atol=1e-9)

    def find_333_decomposition(target_rank, max_restarts=200):
        """
        Attempts to find a rank target_rank decomposition for 3x3x3 tensor.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline - 10:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with small random integers/half-integers
            U = np.random.randint(-2, 3, size=(9, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(9, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(9, target_rank)) * 0.5
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                D_U = np.zeros((81, target_rank))
                for r in range(target_rank):
                    D_U[:, r] = (V[:, r].reshape(9, 1) * W[:, r].reshape(1, 9)).reshape(-1)
                
                U_new = np.zeros((9, target_rank))
                for a in range(9):
                    try:
                        U_new[a, :] = np.linalg.lstsq(D_U, T333_flat_U[a, :], rcond=None)[0]
                    except:
                        break
                U = U_new
                
                # Update V
                D_V = np.zeros((81, target_rank))
                for r in range(target_rank):
                    D_V[:, r] = (U[:, r].reshape(9, 1) * W[:, r].reshape(1, 9)).reshape(-1)
                
                V_new = np.zeros((9, target_rank))
                for b in range(9):
                    try:
                        V_new[b, :] = np.linalg.lstsq(D_V, T333_flat_V[b, :], rcond=None)[0]
                    except:
                        break
                V = V_new
                
                # Update W
                D_W = np.zeros((81, target_rank))
                for r in range(target_rank):
                    D_W[:, r] = (U[:, r].reshape(9, 1) * V[:, r].reshape(1, 9)).reshape(-1)
                
                W_new = np.zeros((9, target_rank))
                for c in range(9):
                    try:
                        W_new[c, :] = np.linalg.lstsq(D_W, T333_flat_W[c, :], rcond=None)[0]
                    except:
                        break
                W = W_new
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check exactness
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
                
        return None

    # Step 2: Search for direct 3x3x6 decomposition
    # Tensor T336[a, b, c] where a=i*3+j, b=j*6+k, c=k*3+i
    # T336[i*3+j, j*6+k, k*3+i] = 1
    T336 = np.zeros((9, 18, 18), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T336[i*3 + j, j*6 + k, k*3 + i] = 1.0

    T336_flat_U = T336.reshape(9, 18*18)
    T336_flat_V = T336.transpose(1, 0, 2).reshape(18, 9*18)
    T336_flat_W = T336.transpose(2, 0, 1).reshape(18, 9*18)

    def find_336_decomposition(target_rank, max_restarts=100):
        if time.time() > deadline - 10:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randint(-2, 3, size=(9, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(18, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(18, target_rank)) * 0.5
            
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                D_U = np.zeros((18*18, target_rank))
                for r in range(target_rank):
                    D_U[:, r] = (V[:, r].reshape(18, 1) * W[:, r].reshape(1, 18)).reshape(-1)
                
                U_new = np.zeros((9, target_rank))
                for a in range(9):
                    try:
                        U_new[a, :] = np.linalg.lstsq(D_U, T336_flat_U[a, :], rcond=None)[0]
                    except:
                        break
                U = U_new
                
                # Update V
                D_V = np.zeros((9*18, target_rank))
                for r in range(target_rank):
                    D_V[:, r] = (U[:, r].reshape(9, 1) * W[:, r].reshape(1, 18)).reshape(-1)
                
                V_new = np.zeros((18, target_rank))
                for b in range(18):
                    try:
                        V_new[b, :] = np.linalg.lstsq(D_V, T336_flat_V[b, :], rcond=None)[0]
                    except:
                        break
                V = V_new
                
                # Update W
                D_W = np.zeros((9*18, target_rank))
                for r in range(target_rank):
                    D_W[:, r] = (U[:, r].reshape(9, 1) * V[:, r].reshape(1, 18)).reshape(-1)
                
                W_new = np.zeros((18, target_rank))
                for c in range(18):
                    try:
                        W_new[c, :] = np.linalg.lstsq(D_W, T336_flat_W[c, :], rcond=None)[0]
                    except:
                        break
                W = W_new
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_recon, T336, atol=1e-9):
                return U_r, V_r, W_r
                
        return None

    # Step 3: Construct 3x3x6 decomposition from 3x3x3 (Fallback)
    def construct_from_333(U333, V333, W333):
        rank333 = U333.shape[1]
        total_rank = 2 * rank333
        U = np.zeros((9, total_rank))
        V = np.zeros((18, total_rank))
        W = np.zeros((18, total_rank))
        
        U[:, :rank333] = U333
        for j in range(3):
            for k in range(3):
                V[j*6 + k, :rank333] = V333[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, :rank333] = W333[k*3 + i, :]

        U[:, rank333:] = U333
        for j in range(3):
            for k in range(3, 6):
                k_local = k - 3
                V[j*6 + k, rank333:] = V333[j*3 + k_local, :]
        for k in range(3, 6):
            k_local = k - 3
            for i in range(3):
                W[k*3 + i, rank333:] = W333[k_local*3 + i, :]
                
        return U, V, W

    # Strategy:
    # 1. Try to find a direct 3x3x6 decomposition with rank <= 34 (which is 2*17, better than 2*19=38)
    #    Actually, known bounds for 3x3x6 are not well established, but 3x3x3 is 23.
    #    So 2*23 = 46.
    #    If we can find rank < 46 directly, that's an improvement.
    #    Let's try ranks 30-45 for direct 3x3x6.
    
    best_direct = None
    best_direct_rank = 999
    
    # Try a few ranks for direct 3x3x6
    # Since 3x3x3 is 23, 3x3x6 is at most 46.
    # Let's try 38, 40, 42, 44, 46
    for r in [38, 40, 42, 44, 46]:
        if time.time() > deadline - 30:
            break
        res = find_336_decomposition(r, max_restarts=20)
        if res is not None:
            best_direct = res
            best_direct_rank = r
            break # Found a valid one, stop

    if best_direct is not None:
        return {"U": best_direct[0].tolist(), "V": best_direct[1].tolist(), "W": best_direct[2].tolist()}

    # Fallback: Use 3x3x3 decomposition
    U333 = None
    V333 = None
    W333 = None
    rank333 = 27
    
    search_ranks = [19, 20, 21, 22, 23]
    
    for r in search_ranks:
        if time.time() > deadline - 20:
            break
        restarts = 50
        res = find_333_decomposition(r, max_restarts=restarts)
        if res is not None:
            U333, V333, W333 = res
            rank333 = r
            break

    if U333 is None:
        U333 = np.zeros((9, 27))
        V333 = np.zeros((9, 27))
        W333 = np.zeros((9, 27))
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U333[i*3+j, c] = 1.0
                    V333[j*3+k, c] = 1.0
                    W333[k*3+i, c] = 1.0
                    c += 1
        rank333 = 27

    U, V, W = construct_from_333(U333, V333, W333)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
