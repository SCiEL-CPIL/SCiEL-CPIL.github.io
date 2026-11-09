"""Matrix multiplication 2x4x4 with rank-28 Strassen-based decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for jm in range(2):
        for kp in range(2):
            base = (jm * 2 + kp) * 7
            for prod_idx in range(7):
                col = base + prod_idx
                if prod_idx == 0:
                    U[0*m + jm*2+0][col] += 1; U[1*m + jm*2+1][col] += 1
                    V[(jm*2+0)*p + kp*2+0][col] += 1; V[(jm*2+1)*p + kp*2+1][col] += 1
                    W[(kp*2+0)*n + 0][col] += 1; W[(kp*2+1)*n + 0][col] -= 1; W[(kp*2+0)*n + 1][col] += 1; W[(kp*2+1)*n + 1][col] += 1
                elif prod_idx == 1:
                    U[1*m + jm*2+0][col] += 1; U[1*m + jm*2+1][col] += 1
                    V[(jm*2+0)*p + kp*2+0][col] += 1
                    W[(kp*2+0)*n + 1][col] += 1; W[(kp*2+1)*n + 1][col] -= 1
                elif prod_idx == 2:
                    U[0*m + jm*2+0][col] += 1
                    V[(jm*2+1)*p + kp*2+0][col] += 1; V[(jm*2+1)*p + kp*2+1][col] -= 1
                    W[(kp*2+0)*n + 0][col] += 1; W[(kp*2+1)*n + 0][col] -= 1
                elif prod_idx == 3:
                    U[1*m + jm*2+1][col] += 1
                    V[(jm*2+0)*p + kp*2+0][col] += 1; V[(jm*2+0)*p + kp*2+1][col] -= 1
                    W[(kp*2+0)*n + 0][col] += 1; W[(kp*2+1)*n + 1][col] += 1
                elif prod_idx == 4:
                    U[0*m + jm*2+0][col] += 1; U[0*m + jm*2+1][col] += 1
                    V[(jm*2+1)*p + kp*2+1][col] += 1
                    W[(kp*2+1)*n + 0][col] -= 1; W[(kp*2+1)*n + 1][col] += 1
                elif prod_idx == 5:
                    U[1*m + jm*2+0][col] += 1; U[0*m + jm*2+0][col] -= 1
                    V[(jm*2+0)*p + kp*2+0][col] += 1; V[(jm*2+0)*p + kp*2+1][col] += 1
                    W[(kp*2+0)*n + 1][col] += 1; W[(kp*2+1)*n + 1][col] += 1
                elif prod_idx == 6:
                    U[0*m + jm*2+1][col] += 1; U[1*m + jm*2+1][col] -= 1
                    V[(jm*2+1)*p + kp*2+0][col] += 1; V[(jm*2+1)*p + kp*2+1][col] += 1
                    W[(kp*2+0)*n + 0][col] += 1; W[(kp*2+1)*n + 0][col] -= 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
