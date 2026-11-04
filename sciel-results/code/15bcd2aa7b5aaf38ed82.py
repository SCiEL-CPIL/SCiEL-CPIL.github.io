"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_separable_structure(rank, time_limit):
        """
        Try a separable structure where U[i,j,r] = a[i,r]*b[j,r], V[j,k,r] = b[j,r]*c[k,r],
        W[k,i,r] = c[k,r]*d[i,r]. This gives sum_r a[i,r]*b[j,r]^2*c[k,r]*d[i,r] = 1.
        For this to work for all i,j,k, we need specific relationships.
        
        Alternative: Use the fact that we can have U[i,j,r] = u[i,r], V[j,k,r] = v[k,r] (independent of j),
        then M[k,i,r] = sum_j u[i,r]*v[k,r] = m*u[i,r]*v[k,r].
        Then W[k,i,r] needs sum_r m*u[i,r]*v[k,r]*w[k,i,r] = 1.
        If we set w[k,i,r] = 1/(m*u[i,r]*v[k,r]) when that's a half-integer, it works.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Strategy: Make U depend only on i, V depend only on k, W depends on both
        # U[i,j,r] = u[i,r] for all j
        # V[j,k,r] = v[k,r] for all j  
        # Then M[k,i,r] = sum_j u[i,r]*v[k,r] = m * u[i,r] * v[k,r]
        # Need: sum_r m * u[i,r] * v[k,r] * w[k,i,r] = 1 for all i,k
        
        # Try simple patterns for u and v
        patterns_u = [
            lambda i, r: 1.0 if r == i % rank else 0.0,
            lambda i, r: 1.0 if r < n else (1.0 if i == r else 0.0),
        ]
        patterns_v = [
            lambda k, r: 1.0 if r == k % rank else 0.0,
            lambda k, r: 1.0 if r < p else (1.0 if k == r else 0.0),
        ]
        
        for pu_idx, pu in enumerate(patterns_u):
            for pv_idx, pv in enumerate(patterns_v):
                if time.time() - start_time > time_limit:
                    return None
                
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            U[i*m+j, r] = pu(i, r)
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            V[j*p+k, r] = pv(k, r)
                
                # Now solve for W
                W = np.zeros((p*n, rank))
                valid = True
                for k in range(p):
                    for i in range(n):
                        idx = k*n + i
                        # Need sum_r m * u[i,r] * v[k,r] * w[idx,r] = 1
                        coeffs = []
                        for r in range(rank):
                            c = m * pu(i, r) * pv(k, r)
                            if abs(c) > 1e-9:
                                coeffs.append((r, c))
                        
                        if not coeffs:
                            valid = False
                            break
                        
                        # Try to find w values that are integers or half-integers
                        found = False
                        # Single term
                        for r, c in coeffs:
                            w_val = 1.0 / c
                            if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                                W[idx, r] = w_val
                                found = True
                                break
                        
                        if not found and len(coeffs) >= 2:
                            # Two terms
                            for (r1, c1), (r2, c2) in product(coeffs, repeat=2):
                                if r1 >= r2:
                                    continue
                                for a in range(-4, 5):
                                    for b in range(-4, 5):
                                        if a == 0 and b == 0:
                                            continue
                                        if abs(a*c1 + b*c2 - 1.0) < 1e-9:
                                            W[idx, r1] = a
                                            W[idx, r2] = b
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                        
                        if not found:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and check_decomposition(U, V, W, rank):
                    return (U, V, W)
        
        return None
    
    def try_block_diagonal(rank, time_limit):
        """
        Try block diagonal structure. Divide the 84 schoolbook terms into blocks.
        Each block handles a subset of (i,j,k) triples.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Try to find a rank-r decomposition by randomly assigning the 84 schoolbook terms
        # to r "slots" and seeing if we can combine them
        for trial in range(20):
            if time.time() - start_time > time_limit:
                return None
            
            # Random assignment of (i,j,k) triples to rank slots
            assignment = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        assignment[(i,j,k)] = random.randint(0, rank-1)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # For each rank slot, we have a set of (i,j,k) that use it
            # U[i*m+j, r] = 1 if any (i,j,k) with this i,j is assigned to r
            # V[j*p+k, r] = 1 if any (i,j,k) with this j,k is assigned to r
            # W[k*n+i, r] = 1 if any (i,j,k) with this k,i is assigned to r
            # But then sum_r U*V*W might be > 1
            
            # Instead, use signs to cancel
            signs = {}
            for (i,j,k), r in assignment.items():
                signs[(i,j,k,r)] = random.choice([1.0, -1.0])
            
            for (i,j,k), r in assignment.items():
                U[i*m+j, r] += signs[(i,j,k,r)]
                V[j*p+k, r] += signs[(i,j,k,r)]
                W[k*n+i, r] += signs[(i,j,k,r)]
            
            if check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_linear_algebra_approach(rank, time_limit):
        """
        Use the structure: T[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        Fix U and V randomly, then for each (k,i), we need:
        sum_r (sum_j U[i,j,r]*V[j,k,r]) * W[k,i,r] = 1
        
        This is a linear system for W. If we can find integer/half-integer solutions, great.
        """
        if time.time() - start_time > time_limit:
            return None
        
        for trial in range(30):
            if time.time() - start_time > time_limit:
                return None
            
            # Generate sparse random U, V with entries in {-1, 0, 1}
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            for idx in range(n*m):
                nz = random.randint(1, min(3, rank))
                cols = random.sample(range(rank), nz)
                for c in cols:
                    U[idx, c] = random.choice([-1.0, 1.0])
            for idx in range(m*p):
                nz = random.randint(1, min(3, rank))
                cols = random.sample(range(rank), nz)
                for c in cols:
                    V[idx, c] = random.choice([-1.0, 1.0])
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # For each (k,i), solve M[k,i,:] * W[k,i,:] = 1 with simple W
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                # Find sparse combination of M[idx] that sums to 1
                found = False
                
                # Try single non-zero
                for r in range(rank):
                    if abs(M[idx, r]) > 1e-9:
                        w_val = 1.0 / M[idx, r]
                        if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                            W[idx, r] = w_val
                            found = True
                            break
                
                if not found:
                    # Try two non-zeros
                    for r1 in range(rank):
                        if found:
                            break
                        if abs(M[idx, r1]) < 1e-9:
                            continue
                        for r2 in range(r1+1, rank):
                            if found:
                                break
                            if abs(M[idx, r2]) < 1e-9:
                                continue
                            for a in range(-5, 6):
                                for b in range(-5, 6):
                                    if a == 0 and b == 0:
                                        continue
                                    if abs(a*M[idx, r1] + b*M[idx, r2] - 1.0) < 1e-9:
                                        W[idx, r1] = a
                                        W[idx, r2] = b
                                        found = True
                                        break
                            if found:
                                break
                
                if not found:
                    valid = False
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try various approaches for decreasing ranks
    for rank in range(30, 84):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try separable structure
        result = try_separable_structure(rank, DEADLINE)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break
        
        # Try linear algebra approach
        result = try_linear_algebra_approach(rank, DEADLINE)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break
        
        # Try block diagonal
        result = try_block_diagonal(rank, DEADLINE)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
