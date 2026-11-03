"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Build target triples
    target_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_triples.append((a, b, c))
    target_set = set(target_triples)
    
    def verify_fast(U, V, W):
        """Vectorized verification."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        # Check target triples
        for (a, b, c) in target_triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check a sample of non-target entries for speed
        # If we have time, check all
        return True
    
    def verify_full(U, V, W):
        """Full verification."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        # Build the full tensor product
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We'll compute this as: T = einsum('ar,br,cr->abc', U, V, W)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check target entries
        for (a, b, c) in target_triples:
            if abs(T[a, b, c] - 1.0) > 1e-9:
                return False
        
        # Check non-target entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in target_set:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline
    
    # Try a structured approach: exploit the tensor structure
    # Idea: Group by i, and try to find a low-rank decomposition for each group
    # then combine
    
    # For now, try a few random searches with alternating minimization
    np.random.seed(42)
    
    # Try to find a decomposition with rank < best_rank
    for target_rank in range(best_rank - 1, max(best_rank - 30, 0), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Alternating minimization
            for it in range(20):
                if time.time() > deadline:
                    break
                
                # Check if current is valid
                if verify_fast(U, V, W):
                    if verify_full(U, V, W):
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        break
                
                # Update W: for each c, solve least squares
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed c: sum_r (U[a,r] V[b,r]) W[c,r] = T[a,b,c]
                # Let M_r = outer(U[:,r], V[:,r]), shape (nm, mp)
                # Then sum_r M_r[a,b] W[c,r] = T[a,b,c]
                # This is a linear system: A_c W[c,:] = t_c
                # where A_c is (nm*mp) x target_rank, A_c[a*mp+b, r] = U[a,r]*V[b,r]
                # t_c[a*mp+b] = 1 if (a,b,c) in target_set else 0
                
                # Build A_c and t_c for all c
                # This is expensive, so we'll do it for a subset of c
                # Actually, let's just do it for all c but vectorize
                
                # Flatten U and V
                # M_r[a,b] = U[a,r] * V[b,r]
                # We want: for each c, sum_r M_r[a,b] * W[c,r] = T[a,b,c]
                
                # Let's build the system for all c at once
                # A is (pn, nm*mp, target_rank)
                # T_flat is (pn, nm*mp)
                
                # This is too memory-intensive, so we'll do it c by c
                for c in range(pn):
                    # Build A_c: (nm*mp) x target_rank
                    A_c = np.zeros((nm * mp, target_rank))
                    t_c = np.zeros(nm * mp)
                    for a in range(nm):
                        for b in range(mp):
                            idx = a * mp + b
                            for r in range(target_rank):
                                A_c[idx, r] = U[a, r] * V[b, r]
                            if (a, b, c) in target_set:
                                t_c[idx] = 1.0
                    
                    # Solve least squares
                    W_c, _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
                    W[c, :] = W_c
                
                # Similarly update U and V
                # This is getting too slow. Let's just check and break
                pass
            
            if best_rank == target_rank:
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
