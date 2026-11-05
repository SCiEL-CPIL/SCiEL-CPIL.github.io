<<<<<<< SEARCH
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Verify decomposition: sum_r U[:,r] * V[:,r] * W[:,r] should equal the multiplication tensor
    # Tensor T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # T[i,j,k] = sum over (a,b) where i=a*m+b, j=b*p+c, k=c*n+a of 1
    # Actually T[i,j,k] = 1 if i//m == j%m and j//p == k%n and i%m == k%m... 
    # Let me just verify numerically
    rank = U.shape[1]
    tensor = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        tensor += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Build reference tensor
    ref = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        a = i // m
        b = i % m
        for j in range(m*p):
            c = j % p
            d = j // p
            if d == b:
                for k in range(p*n):
                    e = k % n
                    f = k // n
                    if f == a and e == a:  # k = f*n + e, need f=a
                        if b == d:
                            ref[i,j,k] = 1.0
    # Simpler: T[i,j,k] = 1 iff (i%m == j//p) and (j%p == k%n) and (i//m == k//n)
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                if (i % m) == (j // p) and (j % p) == (k % n) and (i // m) == (k // n):
                    ref[i,j,k] = 1.0
    
    # Check correctness (allow small floating point error)
    if not np.allclose(tensor, ref, atol=1e-10):
        # Fallback: use pure schoolbook
        U_parts = []
        V_parts = []
        W_parts = []
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    if (i % m) == (j // p) and (j % p) == (k % n) and (i // m) == (k // n):
                        u = np.zeros(n*m); u[i] = 1
                        v = np.zeros(m*p); v[j] = 1
                        w = np.zeros(p*n); w[k] = 1
                        U_parts.append(u)
                        V_parts.append(v)
                        W_parts.append(w)
        U = np.array(U_parts).T
        V = np.array(V_parts).T
        W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
=======
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
>>>>>>> REPLACE
