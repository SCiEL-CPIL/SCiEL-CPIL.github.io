"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def try_rank(R, seed):
        import random
        random.seed(seed)
        U = [[Fraction(0)] * R for _ in range(n * m)]
        V = [[Fraction(0)] * R for _ in range(m * p)]
        for a in range(n * m):
            for c in range(R):
                U[a][c] = Fraction(1) if random.random() < 0.5 else Fraction(0)
        for b in range(m * p):
            for c in range(R):
                V[b][c] = Fraction(1) if random.random() < 0.5 else Fraction(0)
        
        W = [[Fraction(0)] * R for _ in range(p * n)]
        valid = True
        for k in range(p):
            for i in range(n):
                A = [[U[i * m + j][c] * V[j * p + k][c] for c in range(R)] for j in range(m)]
                b_vec = [Fraction(1)] * m
                aug = [A[j][:] + [b_vec[j]] for j in range(m)]
                rows, cols = m, R
                rank = 0
                for col in range(cols):
                    pivot = -1
                    for r in range(rank, rows):
                        if aug[r][col] != 0:
                            pivot = r
                            break
                    if pivot == -1:
                        continue
                    aug[rank], aug[pivot] = aug[pivot], aug[rank]
                    pv = aug[rank][col]
                    for c2 in range(col, cols + 1):
                        aug[rank][c2] = aug[rank][c2] / pv
                    for r in range(rows):
                        if r != rank and aug[r][col] != 0:
                            factor = aug[r][col]
                            for c2 in range(col, cols + 1):
                                aug[r][c2] -= factor * aug[rank][c2]
                    rank += 1
                for r in range(rank, rows):
                    if aug[r][cols] != 0:
                        valid = False
                        break
                if not valid:
                    break
                x = [Fraction(0)] * cols
                for r in range(rank):
                    lead_col = -1
                    for c2 in range(cols):
                        if aug[r][c2] != 0:
                            lead_col = c2
                            break
                    if lead_col != -1:
                        x[lead_col] = aug[r][cols]
                W[k * n + i] = x
            if not valid:
                break
        if not valid:
            return None
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(R):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return None
        return U, V, W
    
    for R in range(48, 128):
        for seed in range(20):
            result = try_rank(R, seed)
            if result is not None:
                U, V, W = result
                U_out = [[float(x) if x.denominator == 1 else x.numerator / x.denominator for x in row] for row in U]
                V_out = [[float(x) if x.denominator == 1 else x.numerator / x.denominator for x in row] for row in V]
                W_out = [[float(x) if x.denominator == 1 else x.numerator / x.denominator for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
