import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook initialization (rank = n*m*p)
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # Verification function
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that off-diagonal entries are zero
        # The tensor T[a,b,c] where a in [0,nm), b in [0,mp), c in [0,pn)
        # T[i*m+j, j'*p+k, k'*n+i'] should be 1 if i=i', j=j', k=k' else 0
        # Wait, the tensor definition for matrix multiplication:
        # T_{i,j,k} = 1 for all i,j,k? No.
        # The bilinear map is (A,B) -> AB.
        # (AB)_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor T has indices (i, j, k) corresponding to A_{ij}, B_{jk}, and output C_{ik}.
        # The coefficient of A_{ij} B_{jk} in C_{ik} is 1 if the indices match, else 0?
        # No, the tensor T is defined such that C = T(A,B).
        # T_{i,j,k; l,m,n} is the coefficient of A_{lm} B_{mn} in C_{ik}?
        # The task says: "exactly decompose the matrix multiplication tensor"
        # "output factor follows the official trace convention W[k*n+i,r]"
        # This implies the tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1?
        # Let's check the inspiration 2 verification.
        # It checks: val = sum U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1.0 for all i,j,k.
        # This corresponds to the diagonal of the tensor.
        # But we must also ensure that T[a,b,c] = 0 for other indices.
        # Specifically, T[i*m+j, jp*p+k, kp*n+ip] should be 0 unless i=ip, j=jp, k=kp.
        
        # Let's do a full check for correctness, but it's O(n*m*p * n*m*p * rank) which is expensive.
        # However, n=3, m=4, p=7 -> nm=12, mp=28, pn=21.
        # Total entries = 12 * 28 * 21 = 7056.
        # Rank is at most 84.
        # 7056 * 84 * 2 (mult+add) ~ 1.2M ops. This is fast enough.
        
        # We already checked the 84 diagonal entries.
        # Now check a random sample of off-diagonal entries to save time?
        # Or just check all? 7056 is small.
        
        # To be safe and fast, let's check all.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Determine i, j, k, jp, kp, ip from a, b, c
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ip = c % n
                    
                    if i == ip and j == jp and k == kp:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    
                    if abs(val - target) > tol:
                        return False
        return True

    # Try to reduce rank by removing terms (greedy)
    # This is based on Inspiration 2's _greedy_reduce
    # But the schoolbook decomposition has orthogonal terms, so no term can be removed.
    # So we need to try to find a different decomposition.
    
    # Strategy: 
    # 1. Start with schoolbook.
    # 2. Try to perturb and re-optimize to find a lower rank decomposition?
    #    This is hard.
    # 3. Use the fact that for small matrices, the rank is known or bounded.
    #    Rank(3x4 x 4x7) <= 3*4*7 = 84.
    #    Lower bounds: max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    #    Also, rank >= max(n, m, p) * something?
    #    Actually, the rank of matrix multiplication tensor is known to be close to n*m*p for small sizes,
    #    but can be slightly lower due to "wasteful" terms in schoolbook?
    #    No, schoolbook is optimal for many small cases?
    #    Actually, for 2x2 x 2x2, rank is 7 (Strassen) vs 8 (schoolbook).
    #    For 3x3 x 3x3, rank is 23 (Landsberg) vs 27.
    #    So there is room for improvement.
    
    # However, finding the exact minimal rank is an open problem for many sizes.
    # For 3x4 x 4x7, we don't know the exact minimal rank.
    # The task asks for "rank-minimal exact tensor decomposition".
    # If we can't find it, we return the best we have (schoolbook).
    
    # Let's try a simple heuristic: 
    # Combine terms if possible?
    # Or use a random search for a lower rank decomposition?
    
    # Given the time limit and the difficulty, let's stick to the schoolbook decomposition
    # unless we can quickly verify a lower rank one.
    
    # One idea: check if rank 83 is possible by random search? Unlikely to succeed quickly.
    
    # Let's try to use the "support sparsification" idea from the directive?
    # "formulating the tensor decomposition as a Minimum Weight Perfect Matching problem..."
    # This seems complex to implement correctly in 150s.
    
    # Let's try a simple local search:
    # Start with schoolbook.
    # Try to merge two terms into one?
    # Term r1: u1, v1, w1
    # Term r2: u2, v2, w2
    # Can we replace them with u', v', w' such that u'⊗v'⊗w' = u1⊗v1⊗w1 + u2⊗v2⊗w2?
    # This is only possible if the sum is a rank-1 tensor.
    # u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank 1 only if u1=u2, v1=v2, w1=w2 (up to scale).
    # In schoolbook, u_i,j,k is a unit vector e_{i*m+j}.
    # u1 = e_{a}, u2 = e_{b}. Sum is e_a + e_b, which is not a scalar multiple of a unit vector unless a=b.
    # So schoolbook terms are "far apart" and cannot be merged.
    
    # Therefore, the schoolbook decomposition is likely the best we can easily find.
    # The minimal rank might be lower, but finding it requires sophisticated algorithms (e.g., Border Rank methods)
    # which are not feasible here.
    
    # Let's just return the schoolbook decomposition.
    # But wait, the current program score is 1.0000.
    # If we return the same thing, the score will be 1.0000.
    # Can we do better?
    
    # Let's check if there is a known lower bound or result for 3x4 x 4x7.
    # I don't have access to external knowledge.
    
    # Let's try a random restart with a lower rank constraint?
    # e.g., try to find a decomposition of rank 80.
    # Initialize U, V, W randomly with rank 80.
    # Use ALS (Alternating Least Squares) to minimize the error.
    # If the error reaches 0, we found a valid decomposition.
    
    def try_als(rank, max_iters=100, tol=1e-8):
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Precompute targets
        # T is sparse.
        # We can solve for W given U, V.
        # W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r] / sum_{a,b} U[a,r]^2 V[b,r]^2 ?
        # No, the equation is:
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This is a linear system for W[:, r] for each r?
        # No, it's coupled.
        # Actually, for fixed r, we want to find W_r such that
        # sum_{a,b} T[a,b,c] U[a,r] V[b,r] = W[c,r] ?
        # No, the sum is over r.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a multilinear problem.
        # ALS approach:
        # Fix U, V. Solve for W.
        # W[c, :] is determined by:
        # For each c, W[c, :] is a vector in R^rank.
        # The equations are: for all a, b,
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This doesn't isolate W[c, r] easily because the sum is over r.
        # Wait, T[a,b,c] is 0 or 1.
        # If T[a,b,c] = 1, then sum_r U[a,r] V[b,r] W[c,r] = 1.
        # If T[a,b,c] = 0, then sum_r U[a,r] V[b,r] W[c,r] = 0.
        
        # This is a system of linear equations for the variables W[c, r].
        # There are pn * rank variables.
        # There are nm * mp * pn equations.
        # nm*mp*pn = 12 * 28 * 21 = 7056 equations.
        # Variables: 21 * rank.
        # If rank < 7056/21 = 336, it's underdetermined?
        # No, the equations are not all independent.
        
        # Actually, a better way to view ALS:
        # Reshape the tensor T into a matrix T_(c) of size (nm*mp) x rank?
        # No.
        # Let's use the standard ALS for tensor decomposition.
        # T = U * V * W (mode-1,2,3 unfolding?)
        # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
        # Fix U, V.
        # We want to find W.
        # For each c, W_c is a vector.
        # The contribution of W_c to T_{abc} is W_{cr} * (U_{ar} V_{br}).
        # Let M^{(c)} be the matrix M^{(c)}_{ab, r} = U_{ar} V_{br}.
        # Then T^{(c)}_{ab} = sum_r M^{(c)}_{ab, r} W_{cr}.
        # So T^{(c)} = M^{(c)} W_c.
        # We can solve for W_c using least squares: W_c = (M^{(c)}^T M^{(c)})^{-1} M^{(c)}^T T^{(c)}.
        # But T^{(c)} is a vector of size nm*mp.
        # M^{(c)} is nm*mp x rank.
        # This is a linear least squares problem.
        
        # Let's implement this.
        
        # Precompute T_slices: T_slices[c] is a vector of size nm*mp.
        T_slices = []
        for c in range(pn):
            vec = np.zeros(nm * mp)
            for a in range(nm):
                for b in range(mp):
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ip = c % n
                    if i == ip and j == jp and k == kp:
                        vec[a * mp + b] = 1.0
                    else:
                        vec[a * mp + b] = 0.0
            T_slices.append(vec)
        
        for it in range(max_iters):
            # Update W
            new_W = np.zeros((pn, rank))
            for c in range(pn):
                # M^{(c)}_{ab, r} = U[a, r] V[b, r]
                # M is (nm*mp) x rank
                M = np.zeros((nm * mp, rank))
                for a in range(nm):
                    for b in range(mp):
                        for r in range(rank):
                            M[a * mp + b, r] = U[a, r] * V[b, r]
                
                # Solve M W_c = T_slices[c]
                # Use lstsq
                W_c, _, _, _ = np.linalg.lstsq(M, T_slices[c], rcond=None)
                new_W[c, :] = W_c
            
            # Update U
            new_U = np.zeros((nm, rank))
            for a in range(nm):
                # M^{(a)}_{bc, r} = V[b, r] W[c, r]
                # M is (mp*pn) x rank
                M = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        for r in range(rank):
                            M[b * pn + c, r] = V[b, r] * W[c, r]
                
                # T^{(a)} is a vector of size mp*pn
                T_a = np.zeros(mp * pn)
                for b in range(mp):
                    for c in range(pn):
                        i = a // m
                        j = a % m
                        jp = b // p
                        k = b % p
                        kp = c // n
                        ip = c % n
                        if i == ip and j == jp and k == kp:
                            T_a[b * pn + c] = 1.0
                        else:
                            T_a[b * pn + c] = 0.0
                
                U_a, _, _, _ = np.linalg.lstsq(M, T_a, rcond=None)
                new_U[a, :] = U_a
            
            # Update V
            new_V = np.zeros((mp, rank))
            for b in range(mp):
                # M^{(b)}_{ac, r} = U[a, r] W[c, r]
                # M is (nm*pn) x rank
                M = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        for r in range(rank):
                            M[a * pn + c, r] = U[a, r] * W[c, r]
                
                # T^{(b)} is a vector of size nm*pn
                T_b = np.zeros(nm * pn)
                for a in range(nm):
                    for c in range(pn):
                        i = a // m
                        j = a % m
                        jp = b // p
                        k = b % p
                        kp = c // n
                        ip = c % n
                        if i == ip and j == jp and k == kp:
                            T_b[a * pn + c] = 1.0
                        else:
                            T_b[a * pn + c] = 0.0
                
                V_b, _, _, _ = np.linalg.lstsq(M, T_b, rcond=None)
                new_V[b, :] = V_b
            
            U = new_U
            V = new_V
            W = new_W
            
            # Check convergence
            # Calculate error
            err = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = 0.0
                        for r in range(rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        i = a // m
                        j = a % m
                        jp = b // p
                        k = b % p
                        kp = c // n
                        ip = c % n
                        target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                        err += (val - target) ** 2
            
            if err < tol:
                return U, V, W, True
            
            if time.time() > deadline:
                return None, None, None, False
        
        return None, None, None, False

    # Try ranks from best_rank - 1 downwards
    # But ALS is expensive.
    # Let's try a few ranks.
    
    # Given the complexity, and that schoolbook is 84, let's try 80, 70, 60.
    # If we find a valid one, we return it.
    
    for rank in [80, 70, 60, 50]:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(3):
            U, V, W, success = try_als(rank, max_iters=50, tol=1e-8)
            if success:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        if success:
            break

    # Convert to list
    def to_list(arr):
        return arr.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
