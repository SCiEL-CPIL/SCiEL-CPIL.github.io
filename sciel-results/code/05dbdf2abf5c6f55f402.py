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

    def find_333_decomposition(target_rank, max_restarts=500):
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
            # Smaller range might help convergence to sparse/integer solutions
            U = np.random.randint(-1, 2, size=(9, target_rank)) * 0.5
            V = np.random.randint(-1, 2, size=(9, target_rank)) * 0.5
            W = np.random.randint(-1, 2, size=(9, target_rank)) * 0.5
            
            # ALS iterations
            # Reduced iterations, relying on more restarts
            for it in range(20):
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
    # 1. Prioritize finding a good 3x3x3 decomposition (rank 23-26) and lifting it.
    #    This is more likely to succeed than direct 3x3x6 search.
    # 2. If time remains, try direct 3x3x6 search.
    
    U333 = None
    V333 = None
    W333 = None
    rank333 = 27
    
    # Search for 3x3x3 decomposition
    # Start with 23 (optimal), go up to 26
    search_ranks = [23, 24, 25, 26]
    
    for r in search_ranks:
        if time.time() > deadline - 30:
            break
        # Use more restarts for lower ranks as they are harder
        restarts = 300 if r == 23 else 100
        res = find_333_decomposition(r, max_restarts=restarts)
        if res is not None:
            U333, V333, W333 = res
            rank333 = r
            break

    if U333 is None:
        # Fallback to trivial rank 27
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

    # Construct 3x3x6 from 3x3x3
    U, V, W = construct_from_333(U333, V333, W333)
    current_rank = U.shape[1]
    
    # If we have time, try to find a direct 3x3x6 decomposition with rank < current_rank
    if time.time() < deadline - 40:
        # Try ranks slightly below current_rank
        # If current_rank is 46 (2*23), try 44, 42, 40
        # If current_rank is 54 (2*27), try 52, 50, 48, 46
        target_ranks = []
        for r in range(current_rank - 2, 30, -2):
            target_ranks.append(r)
        
        for r in target_ranks:
            if time.time() > deadline - 20:
                break
            res = find_336_decomposition(r, max_restarts=50)
            if res is not None:
                return {"U": res[0].tolist(), "V": res[1].tolist(), "W": res[2].tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
