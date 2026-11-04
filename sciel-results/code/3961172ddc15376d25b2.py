import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 110

    def verify_full(U_m, V_m, W_m, n, m, p):
        rank_m = U_m.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank_m):
                        s += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank50():
        rank = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1
            W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
            col += 1
            U[i0*m+j0, col] += 1
            V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
            W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            U[(i0+1)*m+(j0+1), col] += 1
            V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
            W[k0*n+i0, col] += 1; W[k0*n+(i0+1), col] += 1
            col += 1
            U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
            col += 1
            U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
            V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
            W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
            V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1
            col += 1
        
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        for i in range(2):
            for j in range(4):
                U[i*m+j, col] = 1
                V[j*p+6, col] = 1
                W[6*n+i, col] = 1
                col += 1
        
        return U, V, W, rank

    U50, V50, W50, rank50 = build_rank50()
    
    if not verify_full(U50, V50, W50, n, m, p):
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    best_rank = rank50

    def try_remove_columns(U, V, W, cols_to_remove, n, m, p):
        remaining = [c for c in range(U.shape[1]) if c not in cols_to_remove]
        if not remaining:
            return None
        U_new = U[:, remaining]
        V_new = V[:, remaining]
        W_new = W[:, remaining]
        if verify_full(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
        return None

    # Iterative removal
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    
    while True:
        if time.time() > deadline - 10:
            break
        found = False
        # Try singles
        for col in range(current_rank):
            if time.time() > deadline - 10:
                break
            result = try_remove_columns(current_U, current_V, current_W, [col], n, m, p)
            if result is not None:
                current_U, current_V, current_W = result
                current_rank -= 1
                found = True
                break
        if found:
            continue
        
        # Try pairs
        for c1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for c2 in range(c1+1, current_rank):
                if time.time() > deadline - 10:
                    break
                result = try_remove_columns(current_U, current_V, current_W, [c1, c2], n, m, p)
                if result is not None:
                    current_U, current_V, current_W = result
                    current_rank -= 2
                    found = True
                    break
            if found:
                break
        if found:
            continue
            
        # Try triples
        for c1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for c2 in range(c1+1, current_rank):
                if time.time() > deadline - 10:
                    break
                for c3 in range(c2+1, current_rank):
                    if time.time() > deadline - 10:
                        break
                    result = try_remove_columns(current_U, current_V, current_W, [c1, c2, c3], n, m, p)
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 3
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue
            
        # If no more removals found, break
        break

    if current_rank < best_rank:
        best_U, best_V, best_W = current_U, current_V, current_W
        best_rank = current_rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
