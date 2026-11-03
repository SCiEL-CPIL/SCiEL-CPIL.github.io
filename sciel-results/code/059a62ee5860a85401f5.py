"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    timeout = 150.0
    
    # We partition 5 into 1 + 4.
    # A = [[A11, A14], [A41, A44]]
    # B = [[B11, B14], [B41, B44]]
    # C = AB
    
    # Ranks:
    # A11 B11: 1x1 * 1x1 -> 1x1. Rank 1.
    # A11 B14: 1x1 * 1x4 -> 1x4. Rank 4.
    # A14 B44: 1x4 * 4x4 -> 1x4. Rank 16.
    # A41 B11: 4x1 * 1x1 -> 4x1. Rank 4.
    # A44 B41: 4x4 * 4x1 -> 4x1. Rank 16.
    # A41 B14: 4x1 * 1x4 -> 4x4. Rank 16.
    # A44 B44: 4x4 * 4x4 -> 4x4. Rank 64 (Schoolbook).
    
    total_rank = 1 + 4 + 16 + 4 + 16 + 16 + 64
    # total_rank = 121
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs:
            U[idx][col] += coeff
        for idx, coeff in v_coeffs:
            V[idx][col] += coeff
        for idx, coeff in w_coeffs:
            W[idx][col] += coeff
        col += 1
        
        if time.time() - start_time > timeout - 5:
            return False
        return True

    # 1. A11 B11 (1x1)
    # i=0, j=0, k=0
    # U[0], V[0], W[0]
    if not add_term([(0,1)], [(0,1)], [(0,1)]):
        return {"U": U, "V": V, "W": W}

    # 2. A11 B14 (1x1 * 1x4)
    # i=0, j=0, k=1..4
    # U[0], V[k], W[k*5 + 0]
    for k in range(1, 5):
        if not add_term([(0,1)], [(k,1)], [(k*5,1)]):
            return {"U": U, "V": V, "W": W}

    # 3. A14 B44 (1x4 * 4x4)
    # i=0, j=1..4, k=1..4
    # U[j], V[j*5+k], W[k*5 + 0]
    for j in range(1, 5):
        for k in range(1, 5):
            if not add_term([(j,1)], [(j*5+k,1)], [(k*5,1)]):
                return {"U": U, "V": V, "W": W}

    # 4. A41 B11 (4x1 * 1x1)
    # i=1..4, j=0, k=0
    # U[i*5], V[0], W[0*5 + i]
    for i in range(1, 5):
        if not add_term([(i*5,1)], [(0,1)], [(i,1)]):
            return {"U": U, "V": V, "W": W}

    # 5. A44 B41 (4x4 * 4x1)
    # i=1..4, j=1..4, k=0
    # U[i*5+j], V[j*5], W[0*5 + i]
    for i in range(1, 5):
        for j in range(1, 5):
            if not add_term([(i*5+j,1)], [(j*5,1)], [(i,1)]):
                return {"U": U, "V": V, "W": W}

    # 6. A41 B14 (4x1 * 1x4)
    # i=1..4, j=0, k=1..4
    # U[i*5], V[k], W[k*5 + i]
    for i in range(1, 5):
        for k in range(1, 5):
            if not add_term([(i*5,1)], [(k,1)], [(k*5+i,1)]):
                return {"U": U, "V": V, "W": W}

    # 7. A44 B44 (4x4 * 4x4)
    # i=1..4, j=1..4, k=1..4
    # U[i*5+j], V[j*5+k], W[k*5 + i]
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                if not add_term([(i*5+j,1)], [(j*5+k,1)], [(k*5+i,1)]):
                    return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
