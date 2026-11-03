import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb
    
    deadline = time.time() + 140  # Safety margin before 150s kill

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        M = np.einsum('ar,cr->acr', U, W)
        T_approx = np.einsum('acr,br->abc', M, V)
        return np.allclose(T_approx, T, atol=tol, rtol=0)
    
    def try_round_and_verify(U, V, W):
        """Try rounding to half-integers and verify"""
        for scale in [2, 4, 8, 16, 32, 64, 128]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None

    def count_nonzero_terms(U, V, W):
        """Count columns that are non-zero in all three matrices"""
        nz = 0
        for r in range(U.shape[1]):
            if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
                nz += 1
        return nz

    def structured_search(target_rank, max_tries=40):
        """Search for a decomposition of given rank using structured patterns"""
        for attempt in range(max_tries):
            if time.time() > deadline - 15:
                return None
                
            if attempt % 6 == 0:
                # Pattern: group by j with Strassen-like additions
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                terms_per_j = target_rank // m
                rem = target_rank % m
                
                col = 0
                for j in range(m):
                    t = terms_per_j + (1 if j < rem else 0)
                    for r in range(t):
                        i1 = (r * 2) % n
                        i2 = (r * 2 + 1) % n
                        k1 = (r * 2) % p
                        k2 = (r * 2 + 1) % p
                        
                        if r % 3 == 0:
                            U[i1*m+j, col] = 1
                            U[i2*m+j, col] = 1
                            V[j*p+k1, col] = 1
                            V[j*p+k2, col] = 1
                            W[k1*n+i1, col] = 1
                            W[k2*n+i2, col] = 1
                        elif r % 3 == 1:
                            U[i1*m+j, col] = 1
                            V[j*p+k1, col] = 1
                            W[k1*n+i1, col] = 1
                            U[i2*m+j, col] = -1
                            V[j*p+k2, col] = -1
                            W[k2*n+i2, col] = -1
                        else:
                            U[i1*m+j, col] = 0.5
                            U[i2*m+j, col] = 0.5
                            V[j*p+k1, col] = 0.5
                            V[j*p+k2, col] = 0.5
                            W[k1*n+i1, col] = 0.5
                            W[k2*n+i2, col] = 0.5
                        col += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 6 == 1:
                # Strassen-like on blocks: split 5 into 2+3
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                col = 0
                # M1 = (A1+A2)(B1+B2)
                for i in range(n):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        for j in range(m):
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
                
                # M2 = A1*(B2-B1)
                if col < target_rank:
                    for i in range(n):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            for j in range(2):
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = -1
                            for j in range(2, 5):
                                V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                
                # M3 = A2*B1
                if col < target_rank:
                    for i in range(n):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            for j in range(2, 5):
                                U[i*m+j, col] = 1
                            for j in range(2):
                                V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 6 == 2:
                # Random with structure
                np.random.seed(42 + attempt)
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                for r in range(target_rank):
                    j = np.random.randint(0, m)
                    for i in range(n):
                        U[i*m+j, r] = float(np.random.choice([-1, 0, 1]))
                    for k in range(p):
                        V[j*p+k, r] = float(np.random.choice([-1, 0, 1]))
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = float(np.random.choice([-1, 0, 1]))
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 6 == 3:
                # Combine schoolbook terms
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                if verify(U, V, W):
                    return U, V, W
                    
            elif attempt % 6 == 4:
                # Gradient-based from structured init
                np.random.seed(100 + attempt)
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
                
                if target_rank <= rank_sb:
                    idx = np.random.choice(rank_sb, size=target_rank, replace=False)
                    U += U_sb[:, idx] * 0.5
                    V += V_sb[:, idx] * 0.5
                    W += W_sb[:, idx] * 0.5
                
                lr = 0.05
                for step in range(300):
                    if time.time() > deadline - 10:
                        break
                    M = np.einsum('ar,cr->acr', U, W)
                    T_approx = np.einsum('acr,br->abc', M, V)
                    R = T - T_approx
                    res = np.sum(R**2)
                    if res < 1e-10:
                        break
                    
                    grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                    grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                    grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                rounded = try_round_and_verify(U, V, W)
                if rounded is not None:
                    return rounded
                
                if verify(U, V, W):
                    return U, V, W
                    
            else:
                # Linear combination of schoolbook terms
                np.random.seed(200 + attempt)
                # Pick target_rank random linear combinations
                coeffs = np.random.randn(rank_sb, target_rank) * 0.5
                U = U_sb @ coeffs
                V = V_sb @ coeffs
                W = W_sb @ coeffs
                
                # Quick gradient descent
                lr = 0.02
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    M = np.einsum('ar,cr->acr', U, W)
                    T_approx = np.einsum('acr,br->abc', M, V)
                    R = T - T_approx
                    res = np.sum(R**2)
                    if res < 1e-10:
                        break
                    
                    grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                    grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                    grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                rounded = try_round_and_verify(U, V, W)
                if rounded is not None:
                    return rounded
                
                if verify(U, V, W):
                    return U, V, W
        
        return None
    
    # Try ranks from 49 down, focusing on structured searches
    for target_rank in range(rank_sb - 1, 24, -1):
        if time.time() > deadline - 25:
            break
            
        result = structured_search(target_rank, max_tries=30)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                actual_rank = count_nonzero_terms(U_r, V_r, W_r)
                if actual_rank < best_rank:
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    best_rank = actual_rank
                continue
    
    # Final: if we found something better, return it; else schoolbook
    return best_result
