"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    deadline = time.time() + 140.0
    
    # Precompute tensor mask
    tensor_mask = np.zeros((nm, mp, pn), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_mask[a, b, c] = True
    
    def verify(U, V, W):
        rank = U.shape[1]
        T = np.zeros((nm, mp, pn))
        for r in range(rank):
            T += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        if not np.allclose(T[tensor_mask], 1.0, atol=1e-6):
            return False
        if not np.allclose(T[~tensor_mask], 0.0, atol=1e-6):
            return False
        return True
    
    def coordinate_descent(U, V, W, max_iter=10):
        rank = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            VW = np.einsum('br,cr->bcr', V, W)
            for a in range(nm):
                M = np.einsum('bcr,bsr->rs', VW, VW)
                mask_a = tensor_mask[a, :, :]
                g = np.sum(VW[mask_a, :], axis=0) if np.any(mask_a) else np.zeros(rank)
                U[a, :] = np.linalg.lstsq(M + 1e-9 * np.eye(rank), g, rcond=None)[0]
            
            # Update V
            UW = np.einsum('ar,cr->acr', U, W)
            for b in range(mp):
                M = np.einsum('acr,acs->rs', UW, UW)
                mask_b = tensor_mask[:, b, :]
                g = np.sum(UW[mask_b, :], axis=0) if np.any(mask_b) else np.zeros(rank)
                V[b, :] = np.linalg.lstsq(M + 1e-9 * np.eye(rank), g, rcond=None)[0]
            
            # Update W
            UV = np.einsum('ar,br->abr', U, V)
            for c in range(pn):
                M = np.einsum('abr,abs->rs', UV, UV)
                mask_c = tensor_mask[:, :, c]
                g = np.sum(UV[mask_c, :], axis=0) if np.any(mask_c) else np.zeros(rank)
                W[c, :] = np.linalg.lstsq(M + 1e-9 * np.eye(rank), g, rcond=None)[0]
        
        return U, V, W
    
    # Schoolbook
    def schoolbook():
        rank = nm * mp * pn
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = nm * mp * pn
    
    # Try lower ranks with quick searches
    for rank in range(35, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 7919 + rank * 31)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            U, V, W = coordinate_descent(U, V, W, max_iter=8)
            
            if verify(U, V, W):
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

