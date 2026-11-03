import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120

    n, m, p = 2, 4, 8
    total_rank = 56

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Strassen 2x2 base block: local indices 0,1,2,3
    # 0=A00, 1=A01, 2=A10, 3=A11
    # W local indices: 0=C00, 1=C01, 2=C10, 3=C11
    #
    # M1: (A00+A11)*(B00+B11) -> C00+=, C11+=
    # M2: (A10+A11)*B00       -> C10+=, C11-=
    # M3: A00*(B01-B11)       -> C01+=, C11+=
    # M4: A11*(B10-B00)       -> C00+=, C10+=
    # M5: (A00+A01)*B11       -> C00-=, C01+=
    # M6: (A10-A00)*(B00+B01) -> C10+=
    # M7: (A01-A11)*(B10+B11) -> C00+=

    # U_local[local_a, s], V_local[local_b, s], W_local[local_c, s]
    U_loc = np.zeros((4, 7), dtype=int)
    V_loc = np.zeros((4, 7), dtype=int)
    W_loc = np.zeros((4, 7), dtype=int)

    # s=0: M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # s=1: M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # s=2: M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # s=3: M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # s=4: M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # s=5: M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # s=6: M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Map local indices to global
    # A: 2x4, index = row*4 + col
    # B: 4x8, index = row*8 + col
    # W (output C): 2x8, index = col*2 + row (trace convention W[k*n+i, r] where k=col, i=row)

    for jg in range(2):  # A row-block group: jg=0 -> A rows 0-1, jg=1 -> A rows 0-1 (same A)
        # Actually, A is 2x4. We split the 4 columns into 2 groups of 2.
        # jg=0: A[:, 0:2], jg=1: A[:, 2:4]
        # B: 4x8. Split rows into 2 groups of 2 (matching A column groups).
        # For jg=0: B rows 0-1 (B[0:2, :])
        # For jg=1: B rows 2-3 (B[2:4, :])
        # Also split B columns into 4 groups of 2.
        # For kg=0: B[:, 0:2], kg=1: B[:, 2:4], kg=2: B[:, 4:6], kg=3: B[:, 6:8]
        
        # A local indices: 0=A[0, 2*jg], 1=A[0, 2*jg+1], 2=A[1, 2*jg], 3=A[1, 2*jg+1]
        a_map = [0 * 4 + 2 * jg + 0,
                 0 * 4 + 2 * jg + 1,
                 1 * 4 + 2 * jg + 0,
                 1 * 4 + 2 * jg + 1]
        
        for kg in range(4):
            # B local indices: 0=B[2*jg, 2*kg], 1=B[2*jg, 2*kg+1], 2=B[2*jg+1, 2*kg], 3=B[2*jg+1, 2*kg+1]
            b_map = [(2 * jg + 0) * 8 + 2 * kg + 0,
                     (2 * jg + 0) * 8 + 2 * kg + 1,
                     (2 * jg + 1) * 8 + 2 * kg + 0,
                     (2 * jg + 1) * 8 + 2 * kg + 1]
            
            # C local indices (trace convention): C[2*kg, 0], C[2*kg+1, 0], C[2*kg, 1], C[2*kg+1, 1]
            # W index = col * 2 + row
            c_map = [(2 * kg + 0) * 2 + 0,   # C[0, 2*kg] -> W[2*kg * 2 + 0]
                     (2 * kg + 1) * 2 + 0,   # C[0, 2*kg+1] -> W[(2*kg+1) * 2 + 0]
                     (2 * kg + 0) * 2 + 1,   # C[1, 2*kg] -> W[2*kg * 2 + 1]
                     (2 * kg + 1) * 2 + 1]   # C[1, 2*kg+1] -> W[(2*kg+1) * 2 + 1]
            
            base = (jg * 4 + kg) * 7
            for s in range(7):
                r = base + s
                for la in range(4):
                    val = U_loc[la, s]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_loc[lb, s]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_loc[lc, s]
                    if val != 0:
                        W[c_map[lc], r] += val

    if time.time() > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
