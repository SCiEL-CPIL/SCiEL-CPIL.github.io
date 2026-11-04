"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import random
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Build the tensor T[a,b,c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    tensor = [[[Fraction(0)] * (p * n) for _ in range(m * p)] for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a][b][c] = Fraction(1)
    
    def try_decompose(target_rank):
        # Greedy extraction with random restarts
        # We try to find target_rank rank-1 terms
        current = [[[tensor[i][j][k] for k in range(p*n)] for j in range(m*p)] for i in range(n*m)]
        
        U = []
        V = []
        W = []
        
        for _ in range(target_rank):
            # Find a non-zero entry
            anchor_a = anchor_b = anchor_c = None
            found = False
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        if current[a][b][c] != 0:
                            anchor_a, anchor_b, anchor_c = a, b, c
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # Tensor is zero, we can pad with zeros
                while len(U) < target_rank:
                    U.append([Fraction(0)] * (n * m))
                    V.append([Fraction(0)] * (m * p))
                    W.append([Fraction(0)] * (p * n))
                return U, V, W
            
            # Construct rank-1 term
            # We use the anchor to define the term
            # To improve chance of success, we can randomize the "direction" slightly
            # But for exact arithmetic, we stick to the anchor-based construction
            # which is the standard "peeling"
            
            denom = current[anchor_a][anchor_b][anchor_c]
            if denom == 0:
                return None
            
            u = [Fraction(0)] * (n * m)
            v = [Fraction(0)] * (m * p)
            w = [Fraction(0)] * (p * n)
            
            u[anchor_a] = Fraction(1)
            v[anchor_b] = Fraction(1)
            w[anchor_c] = Fraction(1)
            
            # Fill u
            for a in range(n * m):
                if a != anchor_a:
                    if current[a][anchor_b][anchor_c] != 0:
                        u[a] = current[a][anchor_b][anchor_c] / (v[anchor_b] * w[anchor_c])
            
            # Fill v
            for b in range(m * p):
                if b != anchor_b:
                    if current[anchor_a][b][anchor_c] != 0:
                        v[b] = current[anchor_a][b][anchor_c] / (u[anchor_a] * w[anchor_c])
            
            # Fill w
            for c in range(p * n):
                if c != anchor_c:
                    if current[anchor_a][anchor_b][c] != 0:
                        w[c] = current[anchor_a][anchor_b][c] / (u[anchor_a] * v[anchor_b])
            
            # Subtract
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        current[a][b][c] -= u[a] * v[b] * w[c]
            
            U.append(u)
            V.append(v)
            W.append(w)
        
        # Verify if the decomposition is exact
        # Reconstruct and compare
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = Fraction(0)
                    for r in range(target_rank):
                        val += U[r][a] * V[r][b] * W[r][c]
                    if val != tensor[a][b][c]:
                        return None
        return U, V, W

    # Try to find a rank lower than schoolbook
    # Known bounds suggest rank might be 39 for 2x4x5
    for target_rank in range(rank_school - 1, max(rank_school - 10, 20), -1):
        # Add some randomization by permuting indices?
        # For now, try the deterministic peeling
        res = try_decompose(target_rank)
        if res is not None:
            U, V, W = res
            def convert_frac_list(lst):
                return [float(x) if x.denominator == 1 else float(x) for x in lst]
            return {
                "U": [convert_frac_list(row) for row in U],
                "V": [convert_frac_list(row) for row in V],
                "W": [convert_frac_list(row) for row in W]
            }
    
    # Fall back to schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
