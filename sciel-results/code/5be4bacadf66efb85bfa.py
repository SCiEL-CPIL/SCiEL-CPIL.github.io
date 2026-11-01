def _verify(n, m, p, result):
    """Verify that the decomposition is correct. Returns rank if valid, None otherwise."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    rank = U.shape[1]
    
    if U.shape != (n * m, rank):
        return None
    if V.shape != (m * p, rank):
        return None
    if W.shape != (p * n, rank):
        return None
    
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return None
    
    # Full tensor verification
    for i_prime in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k_prime in range(p):
                    for k in range(p):
                        for i in range(n):
                            val = 0.0
                            for c in range(rank):
                                val += U[i_prime * m + j, c] * V[j_prime * p + k_prime, c] * W[k * n + i, c]
                            
                            target = 1.0 if (i_prime == i and j == j_prime and k_prime == k) else 0.0
                            if abs(val - target) > 1e-9:
                                return None
                                
    return rank
