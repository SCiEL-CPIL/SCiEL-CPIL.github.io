"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a decomposition for 5x5 matrix multiplication using block structure.
    Uses Laderman (rank 15) for 3x3*3x3 and schoolbook for rectangular blocks.
    """
    n = m = p = 5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_laderman_3x3(i0, j0, k0):
        """Add Laderman 3x3 decomposition terms."""
        laderman_mults = [
            ({(0,0):1,(0,2):-1,(1,0):-1,(1,2):1,(2,0):1,(2,2):-1},
             {(0,1):1,(1,0):1,(1,2):1,(2,0):-1,(2,2):-1}),
            ({(0,0):1,(0,2):-1},
             {(0,1):1,(1,2):1}),
            ({(0,2):1,(1,2):-1},
             {(0,0):1,(0,1):1}),
            ({(1,0):1,(2,0):-1},
             {(2,0):1,(2,2):1}),
            ({(1,0):1,(1,2):-1},
             {(0,1):1,(2,2):1}),
            ({(1,2):1,(2,2):-1},
             {(0,0):1,(0,1):1}),
            ({(2,0):1,(2,2):-1},
             {(0,1):1,(1,2):1}),
            ({(0,0):1,(1,0):1,(1,2):-1,(2,0):-1,(2,2):1},
             {(0,0):1,(0,1):1,(1,0):-1,(1,2):-1,(2,0):1,(2,2):1}),
            ({(0,0):1,(1,0):1,(1,2):-1,(2,0):-1},
             {(0,1):1,(1,2):1}),
            ({(0,0):1,(0,2):-1,(1,0):1,(1,2):-1,(2,0):1,(2,2):-1},
             {(0,0):1,(0,1):1}),
            ({(0,0):1,(1,0):1,(1,2):-1,(2,0):-1,(2,2):-1},
             {(2,0):1,(2,2):1}),
            ({(0,0):1,(0,2):-1,(1,0):-1,(1,2):1,(2,0):1,(2,2):-1},
             {(0,0):1,(1,0):1,(1,2):1,(2,0):-1,(2,2):-1}),
            ({(0,0):1,(1,0):1,(1,2):-1,(2,0):-1,(2,2):1},
             {(0,1):1,(1,2):1}),
            ({(0,0):1,(0,2):-1,(1,0):1,(1,2):-1,(2,0):-1,(2,2):1},
             {(0,0):1,(0,1):1}),
            ({(0,0):1,(0,2):-1,(1,0):-1,(1,2):1,(2,0):1,(2,2):-1},
             {(2,0):1,(2,2):1}),
        ]
        
        for alpha, beta in laderman_mults:
            for (di, dj), a_coeff in alpha.items():
                for (dj2, dk), b_coeff in beta.items():
                    if dj == dj2:
                        u_idx = (i0 + di) * m + (j0 + dj)
                        v_idx = (j0 + dj) * p + (k0 + dk)
                        w_idx = (k0 + dk) * n + (i0 + di)
                        U_list.append((u_idx, a_coeff))
                        V_list.append((v_idx, b_coeff))
                        W_list.append((w_idx, 1))
    
    def add_schoolbook(i0, i1, j0, j1, k0, k1):
        """Add schoolbook terms for rectangular block."""
        for di in range(i1 - i0):
            for dj in range(j1 - j0):
                for dk in range(k1 - k0):
                    u_idx = (i0 + di) * m + (j0 + dj)
                    v_idx = (j0 + dj) * p + (k0 + dk)
                    w_idx = (k0 + dk) * n + (i0 + di)
                    U_list.append((u_idx, 1))
                    V_list.append((v_idx, 1))
                    W_list.append((w_idx, 1))
    
    # C[0:3,0:3] = A[0:3,0:3]*B[0:3,0:3] + A[0:3,3:5]*B[3:5,0:3]
    add_laderman_3x3(0, 0, 0)
    add_schoolbook(0, 3, 3, 5, 0, 3)
    
    # C[0:3,3:5] = A[0:3,0:3]*B[0:3,3:5] + A[0:3,3:5]*B[3:5,3:5]
    add_schoolbook(0, 3, 0, 3, 3, 5)
    add_schoolbook(0, 3, 3, 5, 3, 5)
    
    # C[3:5,0:3] = A[3:5,0:3]*B[0:3,0:3] + A[3:5,3:5]*B[3:5,0:3]
    add_schoolbook(3, 5, 0, 3, 0, 3)
    add_schoolbook(3, 5, 3, 5, 0, 3)
    
    # C[3:5,3:5] = A[3:5,0:3]*B[0:3,3:5] + A[3:5,3:5]*B[3:5,3:5]
    add_schoolbook(3, 5, 0, 3, 3, 5)
    add_schoolbook(3, 5, 3, 5, 3, 5)
    
    rank = len(U_list)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_idx, u_coeff = U_list[c]
        v_idx, v_coeff = V_list[c]
        w_idx, w_coeff = W_list[c]
        U[u_idx][c] = u_coeff
        V[v_idx][c] = v_coeff
        W[w_idx][c] = w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
