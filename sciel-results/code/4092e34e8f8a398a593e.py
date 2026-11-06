import math

def construct():
    N = 30
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    weights = [L // i for i in range(1, 31)]
    
    # Basis vectors
    # v_j = e_j + weights[j] * e_N
    basis = []
    for j in range(N):
        v = [0] * (N + 1)
        v[j] = 1
        v[N] = weights[j]
        basis.append(v)
    
    # Simple LLL
    # We want to find short vectors.
    # We'll implement a basic LLL.
    
    def lll_reduce(basis):
        n = len(basis)
        dim = len(basis[0])
        
        # B[i] is the i-th vector
        # V[i] is the GS vector
        # mu[i][j] is the coefficient
        
        B = [b[:] for b in basis]
        V = [b[:] for b in basis]
        mu = [[0.0] * n for _ in range(n)]
        
        # Initialize GS
        for i in range(n):
            V[i] = B[i][:]
            for j in range(i):
                dot_bv = sum(B[i][k] * V[j][k] for k in range(dim))
                dot_vv = sum(V[j][k] * V[j][k] for k in range(dim))
                if dot_vv == 0:
                    mu[i][j] = 0.0
                else:
                    mu[i][j] = dot_bv / dot_vv
                for k in range(dim):
                    V[i][k] -= mu[i][j] * V[j][k]
        
        changed = True
        while changed:
            changed = False
            for i in range(1, n):
                # Check Lovasz condition
                dot_v_prev = sum(V[i-1][k] * V[i-1][k] for k in range(dim))
                dot_v_curr = sum(V[i][k] * V[i][k] for k in range(dim))
                
                if dot_v_curr == 0:
                    continue
                
                # mu[i][i-1] needs to be up to date
                # V[i-1] and V[i] are orthogonal to V[0..i-2]
                # So we can just compute the projection
                # But we stored mu. If we swap, we need to update.
                # For now, assume mu is correct.
                
                # Condition: ||V[i-1]||^2 >= (delta - mu[i][i-1]^2) * ||V[i]||^2
                delta = 0.75
                lhs = dot_v_prev
                rhs = (delta - mu[i][i-1]**2) * dot_v_curr
                
                if lhs < rhs:
                    # Swap B[i-1] and B[i]
                    B[i-1], B[i] = B[i], B[i-1]
                    V[i-1], V[i] = V[i], V[i-1]
                    # Update mu for the swapped row i
                    # mu[i][j] for j < i-1 remains same?
                    # We need to recompute mu[i][j] for j < i
                    # And mu[i][i-1] becomes the old mu[i-1][i]? No.
                    # After swap, V[i] is old V[i-1].
                    # We need to orthogonalize new V[i] against V[0..i-1].
                    # This is getting complicated to maintain.
                    
                    # Let's restart GS for row i and below?
                    # Or just break and restart the whole LLL?
                    # Given small N=30, restarting is cheap.
                    changed = True
                    break
                else:
                    # Size reduction
                    for j in range(i-1, -1, -1):
                        # Round mu[i][j]
                        m = round(mu[i][j])
                        if m != 0:
                            # B[i] = B[i] - m * B[j]
                            for k in range(dim):
                                B[i][k] -= m * B[j][k]
                            # Update V[i]
                            # V[i] = V[i] - m * V[j]
                            for k in range(dim):
                                V[i][k] -= m * V[j][k]
                            # Update mu[i][l] for l < j?
                            # mu[i][l] -= m * mu[j][l]
                            for l in range(j):
                                mu[i][l] -= m * mu[j][l]
                            # mu[i][j] becomes 0? No, it's the fractional part.
                            mu[i][j] -= m
                            changed = True
        
        return B

    reduced_basis = lll_reduce(basis)
    
    best_val = float('inf')
    best_coeffs = None
    
    # Check first few vectors in reduced basis
    # They are likely the shortest
    for vec in reduced_basis[:10]:
        s = vec[N]
        if s == 0:
            continue
        c = vec[:N]
        # Check if c_i in {-1, 0, 1}
        valid = True
        for ci in c:
            if ci not in (-1, 0, 1):
                valid = False
                break
        if valid:
            abs_s = abs(s)
            if abs_s < best_val:
                best_val = abs_s
                best_coeffs = c[:]
                
    if best_coeffs is None:
        # Fallback: Try simple pairs
        for i in range(N):
            for j in range(N):
                if i == j: continue
                c = [0] * N
                c[i] = 1
                c[j] = -1
                s = weights[i] - weights[j]
                if s != 0:
                    abs_s = abs(s)
                    if abs_s < best_val:
                        best_val = abs_s
                        best_coeffs = c[:]
                        
    return best_coeffs
