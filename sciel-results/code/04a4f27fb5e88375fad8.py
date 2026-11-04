<<<<<<< SEARCH
def _local_search_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    
    # Precompute symmetric a
    a_sym = a + a.T
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            best_j = -1
            best_delta = 0.0
            
            for j in range(i + 1, n):
                # Compute delta for swapping positions i and j
                S = np.dot(a_sym[i] - a_sym[j], D[j] - D[i])
                
                ci = 2*a[i,i] - a[j,i] - a[i,j]
                cj = a[i,j] + a[j,i] - 2*a[j,j]
                
                Ti = ci * (D[j,i] - D[i,i])
                Tj = cj * (D[j,j] - D[i,j])
                
                diag = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
                cross = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                
                delta = S - Ti - Tj + diag + cross
                
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0 and best_delta < -1e-9:
                j = best_j
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += best_delta
                improved = True
    
    return p, current_cost
=======
def _local_search_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps using vectorized delta computation.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    
    # Precompute symmetric a
    a_sym = a + a.T
    a_diag = np.diag(a)
    a_off_diag = a - np.diag(a_diag)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            
            # Vectorized computation of all swap deltas for facility i
            # For each j > i, compute delta for swapping positions i and j
            js = np.arange(i + 1, n)
            
            # S[j] = dot(a_sym[i] - a_sym[j], D[j] - D[i])
            # = a_sym[i] @ D[j] - a_sym[i] @ D[i] - a_sym[j] @ D[j] + a_sym[j] @ D[i]
            a_sym_i_D = a_sym[i] @ D  # shape (n,)
            a_sym_D_i = a_sym @ D[i]  # shape (n,)
            a_sym_D_diag = np.einsum('ij,ij->i', a_sym, D)  # a_sym[j] @ D[j]
            
            S = a_sym_i_D[js] - a_sym_i_D[i] - a_sym_D_diag[js] + a_sym_D_i[js]
            
            # ci = 2*a[i,i] - a[j,i] - a[i,j]
            ci = 2*a[i,i] - a[js, i] - a[i, js]
            
            # cj = a[i,j] + a[j,i] - 2*a[j,j]
            cj = a[i, js] + a[js, i] - 2*a[js, js]
            
            # Ti = ci * (D[j,i] - D[i,i])
            Ti = ci * (D[js, i] - D[i, i])
            
            # Tj = cj * (D[j,j] - D[i,j])
            Tj = cj * (D[js, js] - D[i, js])
            
            # diag = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            diag = (a[i,i] - a[js, js]) * (D[js, js] - D[i, i])
            
            # cross = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - Ti - Tj + diag + cross
            
            best_idx = np.argmin(deltas)
            best_delta = deltas[best_idx]
            
            if best_delta < -1e-9:
                j = js[best_idx]
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += best_delta
                improved = True
    
    return p, current_cost
>>>>>>> REPLACE

