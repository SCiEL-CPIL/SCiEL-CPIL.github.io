"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    strassen_data = [
        # (A_coeffs { (x,y): val }, B_coeffs { (y,z): val }, C_coeffs { (x,z): val })
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
    ]
    
    col = 0
    for jb in range(2):
        if time.time() > deadline: break
        for kb in range(2):
            if time.time() > deadline: break
            for c_local in range(7):
                if time.time() > deadline: break
                a_coeffs, b_coeffs, c_coeffs = strassen_data[c_local]
                
                # Fill U
                for (x, y), val in a_coeffs.items():
                    i = x
                    j = 2 * jb + y
                    U[i * m + j][col] = val
                    
                # Fill V
                for (y, z), val in b_coeffs.items():
                    j = 2 * jb + y
                    k = 2 * kb + z
                    V[j * p + k][col] = val
                    
                # Fill W
                for (x, z), val in c_coeffs.items():
                    i = x
                    k = 2 * kb + z
                    W[k * n + i][col] = val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
