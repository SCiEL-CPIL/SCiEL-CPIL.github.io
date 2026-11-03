import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, R):
        U_r = U.reshape((n, m, R))
        V_r = V.reshape((m, p, R))
        W_r = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)

    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen 2x2 core (7 terms)
        r = 0
        U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1

        r = 1
        U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1

        r = 2
        U[idx_u(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
        W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1

        r = 3
        U[idx_u(1, 1), r] = 1
        V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
        W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1

        r = 4
        U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1

        r = 5
        U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
        V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
        W[idx_w(1, 1), r] = 1

        r = 6
        U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
        V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1

        # Standard border terms (19 terms)
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W

    # First, establish the verified rank-26 baseline
    U26, V26, W26 = build_rank26()
    if not verify(U26, V26, W26, 26):
        # Should not happen, but fallback to schoolbook
        return schoolbook()
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26

    # Try to find lower rank decompositions (23-25)
    # Use structured patterns based on known algebraic identities
    
    # Pattern 1: Try to merge some border terms with Strassen terms
    # This is a heuristic search - we'll try a few structured modifications
    
    # Attempt rank 25: Try to merge 1 pair of border terms
    if time.time() < deadline - 10:
        for merge_pair in [(0,2), (1,2), (2,0), (2,1)]:
            i1, j1, k1 = merge_pair
            R = 25
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Copy Strassen core
            for r in range(7):
                for i in range(n):
                    for j in range(m):
                        U[idx_u(i,j), r] = U26[idx_u(i,j), r]
                for j in range(m):
                    for k in range(p):
                        V[idx_v(j,k), r] = V26[idx_v(j,k), r]
                for k in range(p):
                    for i in range(n):
                        W[idx_w(k,i), r] = W26[idx_w(k,i), r]
            
            # Add border terms, skipping the merged pair
            col = 7
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i < 2 and j < 2 and k < 2:
                            continue
                        if (i, j, k) == merge_pair:
                            continue
                        U[idx_u(i, j), col] = 1
                        V[idx_v(j, k), col] = 1
                        W[idx_w(k, i), col] = 1
                        col += 1
            
            # Add the merged term with coefficient 1
            U[idx_u(merge_pair[0], merge_pair[1]), col] = 1
            V[idx_v(merge_pair[1], merge_pair[2]), col] = 1
            W[idx_w(merge_pair[2], merge_pair[0]), col] = 1
            col += 1
            
            if col == R and verify(U, V, W, R):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break

    # If we found rank 25, try to find rank 24 or 23
    if best_rank == 25 and time.time() < deadline - 10:
        # Try to merge 2 more pairs
        for merge_pair1 in [(0,2), (1,2), (2,0), (2,1)]:
            for merge_pair2 in [(0,2), (1,2), (2,0), (2,1)]:
                if merge_pair1 == merge_pair2:
                    continue
                R = 24
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                
                # Copy Strassen core
                for r in range(7):
                    for i in range(n):
                        for j in range(m):
                            U[idx_u(i,j), r] = U26[idx_u(i,j), r]
                    for j in range(m):
                        for k in range(p):
                            V[idx_v(j,k), r] = V26[idx_v(j,k), r]
                    for k in range(p):
                        for i in range(n):
                            W[idx_w(k,i), r] = W26[idx_w(k,i), r]
                
                # Add border terms, skipping the merged pairs
                col = 7
                skip_set = {merge_pair1, merge_pair2}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i < 2 and j < 2 and k < 2:
                                continue
                            if (i, j, k) in skip_set:
                                continue
                            U[idx_u(i, j), col] = 1
                            V[idx_v(j, k), col] = 1
                            W[idx_w(k, i), col] = 1
                            col += 1
                
                # Add the merged terms
                for pair in skip_set:
                    U[idx_u(pair[0], pair[1]), col] = 1
                    V[idx_v(pair[1], pair[2]), col] = 1
                    W[idx_w(pair[2], pair[0]), col] = 1
                    col += 1
                
                if col == R and verify(U, V, W, R):
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    break
            if best_rank == 24:
                break

    # Final verification and return
    if best_rank < 26 and not verify(best_U, best_V, best_W, best_rank):
        # Fallback to rank 26 if verification fails
        best_U, best_V, best_W = U26, V26, W26
        best_rank = 26

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
