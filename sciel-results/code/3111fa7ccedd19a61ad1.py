"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 140  # Safety margin before 150s limit

def construct():
    """
    Rank-minimal decomposition for 3x3 * 3x7 matrix multiplication.
    
    Uses Strassen's algorithm on the 2x2 x 2x7 subblock to reduce rank.
    
    Block structure:
    A (3x3) = [[A11(2x2), A12(2x1)],
               [A21(1x2), A22(1x1)]]
    B (3x7) = [[B1(2x7)],
               [B2(1x7)]]
    
    C (3x7) = [[C_top(2x7)],
               [C_bot(1x7)]]
    
    C_top = A11*B1 + A12*B2
    C_bot = A21*B1 + A22*B2
    
    A11*B1 uses Strassen (7 multiplications instead of 8)
    A12*B2: 2 multiplications
    A21*B1: 2 multiplications  
    A22*B2: 1 multiplication
    Total: 7 + 2 + 2 + 1 = 12... 
    """
    n, m, p = 3, 3, 7  # A is n x m, B is m x p, C is n x p
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # We'll build the decomposition term by term.
    # Each term r has:
    #   U[i*m + j] = coefficient for A[i,j]
    #   V[j*p + k] = coefficient for B[j,k]
    #   W[k*n + i] = coefficient for C[i,k]
    
    terms = []  # list of (U_vec, V_vec, W_vec)
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a rank-1 term.
        u_coeff: dict mapping (i,j) to coefficient for A[i,j]
        v_coeff: dict mapping (j,k) to coefficient for B[j,k]
        w_coeff: dict mapping (i,k) to coefficient for C[i,k]
        """
        U = [0] * (n * m)
        V = [0] * (m * p)
        W = [0] * (p * n)
        for (i, j), c in u_coeff.items():
            U[i * m + j] = c
        for (j, k), c in v_coeff.items():
            V[j * p + k] = c
        for (i, k), c in w_coeff.items():
            W[k * n + i] = c
        terms.append((U, V, W))
    
    # Strassen for A11 (2x2) * B1 (2x7)
    # A11 = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B1 = [B_top(1x7); B_bot(1x7)] where B_top = B[0,:], B_bot = B[1,:]
    
    # M1 = (a+d) * (B_top + B_bot)
    # Contributes to C_top: M1 appears in both rows of C_top
    # C_top[0] += M1, C_top[1] += M1
    add_term(
        u_coeff={(0,0): 1, (1,1): 1},  # a+d
        v_coeff={(0,k): 1 for k in range(p)} | {(1,k): 1 for k in range(p)},  # B_top + B_bot
        w_coeff={(0,k): 1 for k in range(p)} | {(1,k): 1 for k in range(p)}  # both rows of C_top
    )
    
    # M2 = (c+d) * B_top
    # Contributes to C_top[1]: -M2
    add_term(
        u_coeff={(1,0): 1, (1,1): 1},  # c+d
        v_coeff={(0,k): 1 for k in range(p)},  # B_top
        w_coeff={(1,k): -1 for k in range(p)}  # -M2 in C_top[1]
    )
    
    # M3 = a * (B_top - B_bot)
    # Contributes to C_top[0]: +M3
    add_term(
        u_coeff={(0,0): 1},  # a
        v_coeff={(0,k): 1 for k in range(p)} | {(1,k): -1 for k in range(p)},  # B_top - B_bot
        w_coeff={(0,k): 1 for k in range(p)}  # +M3 in C_top[0]
    )
    
    # M4 = d * (B_bot - B_top)
    # Contributes to C_top[0]: +M4
    add_term(
        u_coeff={(1,1): 1},  # d
        v_coeff={(1,k): 1 for k in range(p)} | {(0,k): -1 for k in range(p)},  # B_bot - B_top
        w_coeff={(0,k): 1 for k in range(p)}  # +M4 in C_top[0]
    )
    
    # M5 = (a+c) * B_bot
    # Contributes to C_top[1]: +M5
    add_term(
        u_coeff={(0,0): 1, (1,0): 1},  # a+c
        v_coeff={(1,k): 1 for k in range(p)},  # B_bot
        w_coeff={(1,k): 1 for k in range(p)}  # +M5 in C_top[1]
    )
    
    # M6 = (b-d) * (B_top + B_bot)
    # Contributes to C_top[0]: -M6
    add_term(
        u_coeff={(0,1): 1, (1,1): -1},  # b-d
        v_coeff={(0,k): 1 for k in range(p)} | {(1,k): 1 for k in range(p)},  # B_top + B_bot
        w_coeff={(0,k): -1 for k in range(p)}  # -M6 in C_top[0]
    )
    
    # M7 = (c-b) * (B_top - B_bot)
    # Contributes to C_top[0]: +M7, C_top[1]: -M7
    add_term(
        u_coeff={(1,0): 1, (0,1): -1},  # c-b
        v_coeff={(0,k): 1 for k in range(p)} | {(1,k): -1 for k in range(p)},  # B_top - B_bot
        w_coeff={(0,k): 1 for k in range(p)} | {(1,k): -1 for k in range(p)}  # +M7 in C_top[0], -M7 in C_top[1]
    )
    
    # A12 * B2: A12 is 2x1, B2 is 1x7
    # A12[0,0] = A[0,2], A12[1,0] = A[1,2]
    # C_top[i] += A12[i,0] * B2[0,k] for i=0,1
    
    # Term for A[0,2] * B[2,k]
    add_term(
        u_coeff={(0,2): 1},
        v_coeff={(2,k): 1 for k in range(p)},
        w_coeff={(0,k): 1 for k in range(p)}
    )
    
    # Term for A[1,2] * B[2,k]
    add_term(
        u_coeff={(1,2): 1},
        v_coeff={(2,k): 1 for k in range(p)},
        w_coeff={(1,k): 1 for k in range(p)}
    )
    
    # A21 * B1: A21 is 1x2, B1 is 2x7
    # A21[0,0] = A[2,0], A21[0,1] = A[2,1]
    # C_bot[k] += A21[0,j] * B1[j,k] for j=0,1
    
    # Term for A[2,0] * B[0,k]
    add_term(
        u_coeff={(2,0): 1},
        v_coeff={(0,k): 1 for k in range(p)},
        w_coeff={(2,k): 1 for k in range(p)}
    )
    
    # Term for A[2,1] * B[1,k]
    add_term(
        u_coeff={(2,1): 1},
        v_coeff={(1,k): 1 for k in range(p)},
        w_coeff={(2,k): 1 for k in range(p)}
    )
    
    # A22 * B2: A22 is 1x1, B2 is 1x7
    # A22[0,0] = A[2,2]
    # C_bot[k] += A22[0,0] * B2[0,k]
    
    add_term(
        u_coeff={(2,2): 1},
        v_coeff={(2,k): 1 for k in range(p)},
        w_coeff={(2,k): 1 for k in range(p)}
    )
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (U_r, V_r, W_r) in enumerate(terms):
        for i in range(n * m):
            U[i][r] = U_r[i]
        for j in range(m * p):
            V[j][r] = V_r[j]
        for k in range(p * n):
            W[k][r] = W_r[k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
