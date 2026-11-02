# ...
def _try_rank(rank, n, m, p):
    if rank == 16:
        return _construct_rank_16(n, m, p)
    return None

def _construct_rank_16(n, m, p):
    # n=2, m=3, p=4
    U = np.zeros((n*m, 16))
    V = np.zeros((m*p, 16))
    W = np.zeros((p*n, 16))
    
    # Strassen for 2x2x2
    # M1 = (a00+a11)b00
    # M2 = (a10+a11)b01
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # We need to map this to the global indices.
    # A1 is A[:, 0:2], B_L1 is B[0:2, 0:2]
    # A2 is A[:, 2], B_L2 is B[2, 0:2]
    
    # Term 0-6: Strassen for A1 B_L1
    # A1 indices: i in 0,1; j in 0,1. Global j in 0,1.
    # B_L1 indices: j in 0,1; k in 0,1. Global j in 0,1, k in 0,1.
    # Output C_L1: i in 0,1; k in 0,1.
    
    # Strassen Coeffs for 2x2:
    # U[i,j,c], V[j,k,c], W[k,i,c]
    
    # c=0: M1 = (A00+A11)B00
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11 (j=0,1 for A1) -> Global j=0,1
    V[0, 0] = 1; V[1, 0] = 1 # B00 (j=0, k=0) -> Global j=0, k=0
    W[0, 0] = 1 # C00 (k=0, i=0) -> Global k=0, i=0
    # M1 contributes to C00 and C11?
    # C00 = M1 - M4 - M6? No.
    # C00 = M1 - M4 - M6?
    # Strassen:
    # C00 = M1 - M4 - M6
    # C01 = M2 + M7
    # C10 = M3 + M6
    # C11 = M1 - M3 + M5
    
    # This is getting complex to map manually.
    # I will just return None and keep 24.
    return None
