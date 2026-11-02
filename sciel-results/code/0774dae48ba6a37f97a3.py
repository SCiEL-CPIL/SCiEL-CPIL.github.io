import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_half_int_list(matrix):
        """Convert numpy matrix to list of lists with half-integer entries"""
        return [[float(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]

    def verify_exact(U, V, W, rank, n, m, p):
        """Verify exact decomposition using integer arithmetic (multiply by 2 to clear halves)."""
        # Multiply by 2 to work in integers
        U2 = np.round(U * 2).astype(np.int64)
        V2 = np.round(V * 2).astype(np.int64)
        W2 = np.round(W * 2).astype(np.int64)
        
        # Check if the entries are actually half-integers (i.e., U2 is even? No, U2 is 2*U, so if U is half-integer, U2 is integer).
        # The condition for exactness is that the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] equals 1 for all valid (i,j,k) and 0 otherwise?
        # No, the tensor is defined such that the decomposition reproduces the matrix multiplication.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r (sum_j A[i,j] U[i,j,r]) (sum_k B[j,k] V[j,k,r]) ... wait.
        # Standard form: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] is NOT the standard form.
        # The standard form for matrix multiplication tensor T_{i,j,k} is:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ? No.
        # The tensor T is of shape (n, m, p). T[i,j,k] = 1 if it contributes to C[i,k] via A[i,j]B[j,k].
        # Actually, the tensor T[i,j,k] = 1 for all i,j,k? No.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This can be written as C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] ?
        # Let's check the indices in the current code's verify:
        # total += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This implies the decomposition is:
        # A[i,j] B[j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k] ? No.
        # It implies: 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r] for the "core" tensor?
        # If A and B are arbitrary, then we need:
        # sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i,j,r] ... ) No.
        
        # Let's re-read the verify code:
        # C_comp[i, k] = sum_j sum_r A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This formula is dimensionally consistent if U, V, W are scalars for each (i,j,r), etc.
        # For this to equal A @ B, we need:
        # sum_j sum_r A[i, j] B[j, k] U[i,j,r] V[j,k,r] W[k,i,r] = sum_j A[i,j] B[j,k]
        # This must hold for ALL A, B.
        # This implies that the tensor K[i,j,k] defined by sum_r U[i,j,r] V[j,k,r] W[k,i,r] must be the identity in some sense?
        # No, it implies that the operator defined by the decomposition is the identity on the space of bilinear forms?
        # Actually, if we set A[i,j] = delta[i',j'] and B[j,k] = delta[j',k'], then C[i,k] = delta[i,i'] delta[k,k'].
        # LHS: sum_j sum_r delta[i',j] delta[j',k] U[i,j,r] V[j,k,r] W[k,i,r]
        # = sum_r U[i',j',r] V[j',k',r] W[k',i',r]
        # So we need sum_r U[i',j',r] V[j',k',r] W[k',i',r] = delta[i',i'] delta[k',k'] ?
        # Wait, C[i,k] for A=E_{i'j'}, B=E_{j'k'} is E_{i'k'} (if j=j').
        # So C[i,k] = 1 if i=i' and k=k', else 0.
        # The formula gives sum_r U[i',j',r] V[j',k',r] W[k',i',r].
        # This must equal 1 if i'=i and k'=k, else 0.
        # This is a very strong condition.
        
        # Let's check the schoolbook case:
        # U[i,j,r] = 1 if r corresponds to (i,j,k), else 0.
        # V[j,k,r] = 1 if r corresponds to (i,j,k), else 0.
        # W[k,i,r] = 1 if r corresponds to (i,j,k), else 0.
        # Sum_r ... = 1 if there is an r such that U,V,W are all 1.
        # For schoolbook, r is unique for each (i,j,k).
        # So sum_r is 1 if i=i', j=j', k=k' AND i=i''? No.
        # The term is non-zero only if i=i', j=j', k=k' (from U, V, W definitions).
        # So the sum is 1 if i=i' and k=k' and j=j'?
        # But the condition for C[i,k] to be non-zero is i=i' and k=k'.
        # The sum over r picks out the term where r matches (i',j',k').
        # So the value is 1 if i=i', j=j', k=k'.
        # But we need it to be 1 if i=i' and k=k', regardless of j?
        # No, in the sum over j in C_comp, we sum over j.
        # C_comp[i,k] = sum_j sum_r A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r].
        # If A=E_{i'j'}, B=E_{j'k'}, then A[i,j] is non-zero only if j=j'.
        # So the sum over j collapses to j=j'.
        # C_comp[i,k] = sum_r A[i',j'] B[j',k] U[i',j',r] V[j',k,r] W[k,i',r].
        # B[j',k] is non-zero only if k=k'.
        # So C_comp[i,k] = sum_r A[i',j'] B[j',k'] U[i',j',r] V[j',k',r] W[k',i',r].
        # = A[i',j'] B[j',k'] * (sum_r U[i',j',r] V[j',k',r] W[k',i',r]).
        # We want C_comp[i,k] = C_true[i,k] = delta[i,i'] delta[k,k'].
        # So we need:
        # delta[i,i'] delta[k,k'] = A[i',j'] B[j',k'] * S[i',j',k']
        # where S[i',j',k'] = sum_r U[i',j',r] V[j',k',r] W[k',i',r].
        # Since this must hold for all A, B, we can choose A[i',j']=1, B[j',k']=1.
        # Then delta[i,i'] delta[k,k'] = S[i',j',k'].
        # This implies S[i',j',k'] is 1 if i'=i and k'=k?
        # But S depends on i', j', k'.
        # If i != i', LHS is 0. So S[i',j',k'] must be 0 if i != i'.
        # If i = i' but k != k', LHS is 0. So S[i',j',k'] must be 0 if k != k'.
        # If i = i' and k = k', LHS is 1. So S[i',j',k'] must be 1.
        # So S[i,j,k] = 1 if i=i and k=k? This is tautological.
        # Wait, S[i',j',k'] is a fixed number for given i',j',k'.
        # The condition is: For all i,k, delta[i,i'] delta[k,k'] = S[i',j',k'].
        # This is impossible unless S depends on i,k? But S is fixed by U,V,W.
        # Ah, the sum over r is inside.
        # Let's re-evaluate.
        # C_comp[i,k] = sum_j A[i,j] B[j,k] (sum_r U[i,j,r] V[j,k,r] W[k,i,r]).
        # Let M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # Then C_comp[i,k] = sum_j A[i,j] B[j,k] M[i,j,k].
        # We want C_comp[i,k] = sum_j A[i,j] B[j,k] * 1.
        # So we need M[i,j,k] = 1 for all i,j,k.
        
        # So the condition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i in [n], j in [m], k in [p].
        
        # Let's check schoolbook:
        # M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # For schoolbook, U[i,j,r]=1 iff r=(i,j,k).
        # So M[i,j,k] = U[i,j,(i,j,k)] V[j,k,(i,j,k)] W[k,i,(i,j,k)] = 1*1*1 = 1.
        # Correct.
        
        # So verify_exact should check if M[i,j,k] == 1 for all i,j,k.
        
        # Let's implement this check exactly.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def peel_rank1(T, n, m, p):
        """
        Peel off a rank-1 component from tensor T of shape (nm, mp, pn).
        Uses HOSVD-like approach.
        """
        nm, mp, pn = n * m, m * p, p * n
        T3 = T.reshape(nm, mp, pn)
        
        T1 = T3.reshape(nm, mp * pn)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        
        T2 = T3.transpose(1, 0, 2).reshape(mp, nm * pn)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        
        T3u = T3.transpose(2, 0, 1).reshape(pn, nm * mp)
        U3, S3, Vt3 = np.linalg.svd(T3u, full_matrices=False)
        
        u = U1[:, 0]
        v = U2[:, 0]
        w = U3[:, 0]
        
        alpha = np.einsum('i,j,k,ijk->', u, v, w, T3)
        
        T_residual = T3 - alpha * np.einsum('i,j,k->ijk', u, v, w)
        
        return u, v, w, alpha, T_residual.reshape(nm * mp * pn)

    T_dense = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_dense[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T_dense.reshape(nm * mp * pn)
    
    best_rank = None
    best_factors = None
    
    # Try greedy peeling from rank 42 down to 14 (known lower bound for 2x3x7 is 14)
    # Add random restarts to avoid local minima
    for target_rank in range(42, 13, -1):
        found = False
        for restart in range(5):
            T_remaining = T_flat.copy()
            if restart > 0:
                # Add small noise to break symmetry/local minima
                T_remaining += np.random.randn(T_flat.size) * 1e-5
            
            U_cols = []
            V_cols = []
            W_cols = []
            
            for r in range(target_rank):
                u, v, w, alpha, T_remaining = peel_rank1(T_remaining, n, m, p)
                U_cols.append(alpha * u)
                V_cols.append(v)
                W_cols.append(w)
            
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            
            # Project to nearest half-integer
            U_half = np.round(U * 2) / 2.0
            V_half = np.round(V * 2) / 2.0
            W_half = np.round(W * 2) / 2.0
            
            if verify_exact(U_half, V_half, W_half, target_rank, n, m, p):
                best_rank = target_rank
                best_factors = (U_half, V_half, W_half)
                found = True
                break
        
        if found:
            break
    
    if best_factors is not None:
        U, V, W = best_factors
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    
    # Fallback to schoolbook
    return {"U": to_half_int_list(U_school), "V": to_half_int_list(V_school), "W": to_half_int_list(W_school)}
