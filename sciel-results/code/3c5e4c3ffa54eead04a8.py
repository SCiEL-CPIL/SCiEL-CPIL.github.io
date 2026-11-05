import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partitions for 5: 2, 2, 1
    # Indices: 0,1 (I1), 2,3 (I2), 4 (I3)
    partitions = [
        [0, 1],
        [2, 3],
        [4]
    ]
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build the decomposition block by block.
    # Block (alpha, beta, gamma)
    # alpha: row block of A/C
    # beta: col block of B/C
    # gamma: col block of A / row block of B
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a rank-1 term
    # u: dict {(i, k): val} for A_{ik}
    # v: dict {(k, j): val} for B_{kj}
    # w: dict {(j, i): val} for C_{ji} (Note: C index is i, j. W index is j*n+i)
    
    def add_term(u_dict, v_dict, w_dict):
        # u_dict keys are local (i, k)
        # v_dict keys are local (k, j)
        # w_dict keys are local (i, j)
        pass # Placeholder

    # We need to map local to global.
    # For block (alpha, beta, gamma):
    # Local i in [0, S_alpha) -> Global I = partitions[alpha][i]
    # Local k in [0, S_gamma) -> Global K = partitions[gamma][k]
    # Local j in [0, S_beta) -> Global J = partitions[beta][j]
    
    # U index: I * m + K
    # V index: K * p + J
    # W index: J * n + I
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                S_a = len(partitions[alpha])
                S_b = len(partitions[beta])
                S_g = len(partitions[gamma])
                
                # Get local decomposition for S_a x S_g x S_b
                # Note: A is S_a x S_g, B is S_g x S_b
                local_decomp = get_local_decomp(S_a, S_g, S_b)
                
                for u_local, v_local, w_local in local_decomp:
                    # u_local is a list of length S_a * S_g? Or dict?
                    # Let's use dicts for clarity in mapping.
                    
                    u_global = [0] * (n * m)
                    v_global = [0] * (m * p)
                    w_global = [0] * (p * n)
                    
                    # Fill u_global
                    for i in range(S_a):
                        for k in range(S_g):
                            val = u_local[i * S_g + k]
                            if val != 0:
                                I = partitions[alpha][i]
                                K = partitions[gamma][k]
                                u_global[I * m + K] = val
                                
                    # Fill v_global
                    for k in range(S_g):
                        for j in range(S_b):
                            val = v_local[k * S_b + j]
                            if val != 0:
                                K = partitions[gamma][k]
                                J = partitions[beta][j]
                                v_global[K * p + J] = val
                                
                    # Fill w_global
                    for i in range(S_a):
                        for j in range(S_b):
                            val = w_local[i * S_b + j]
                            if val != 0:
                                I = partitions[alpha][i]
                                J = partitions[beta][j]
                                w_global[J * n + I] = val
                                
                    U_list.append(u_global)
                    V_list.append(v_global)
                    W_list.append(w_global)
                    total_rank += 1
                    
    # Convert to list of lists
    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}

def get_local_decomp(n, m, p):
    """
    Returns a list of (u, v, w) tuples.
    u: list of length n*m
    v: list of length m*p
    w: list of length p*n
    """
    if n == 2 and m == 2 and p == 2:
        return strassen_decomp()
    else:
        return schoolbook_decomp(n, m, p)

def strassen_decomp():
    """
    Strassen decomposition for 2x2x2.
    Returns 7 terms.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # u: A00=1, A11=1
    # v: B00=1, B11=1
    # w: C00=1, C11=1
    u = [0]*4; u[0]=1; u[3]=1
    v = [0]*4; v[0]=1; v[3]=1
    w = [0]*4; w[0]=1; w[3]=1
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00 -> C10
    # u: A10=1, A11=1
    # v: B00=1
    # w: C10=1
    u = [0]*4; u[2]=1; u[3]=1
    v = [0]*4; v[0]=1
    w = [0]*4; w[2]=1 # C10 index i=1, j=0 -> 0*2+1=1? No.
    # W index is j*n+i. n=2.
    # C10: i=1, j=0. Index 0*2+1 = 1.
    w = [0]*4; w[1]=1
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11) -> C01
    # u: A00=1
    # v: B10=1, B11=-1
    # w: C01=1
    u = [0]*4; u[0]=1
    v = [0]*4; v[2]=1; v[3]=-1
    w = [0]*4; w[0]=1 # C01: i=0, j=1. Index 1*2+0=2.
    w = [0]*4; w[2]=1
    terms.append((u, v, w))
    
    # M4 = A11(B01-B00) -> C00, C10
    # u: A11=1
    # v: B01=1, B00=-1
    # w: C00=1, C10=1
    u = [0]*4; u[3]=1
    v = [0]*4; v[1]=1; v[0]=-1
    w = [0]*4; w[0]=1; w[1]=1
    terms.append((u, v, w))
    
    # M5 = (A00+A10)B11 -> C00, C10
    # u: A00=1, A10=1
    # v: B11=1
    # w: C00=-1, C10=1
    u = [0]*4; u[0]=1; u[2]=1
    v = [0]*4; v[3]=1
    w = [0]*4; w[0]=-1; w[1]=1
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B01) -> C11
    # u: A10=1, A00=-1
    # v: B00=1, B01=1
    # w: C11=1
    u = [0]*4; u[2]=1; u[0]=-1
    v = [0]*4; v[0]=1; v[1]=1
    w = [0]*4; w[3]=1
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    # u: A01=1, A11=-1
    # v: B10=1, B11=1
    # w: C00=1, C11=-1
    u = [0]*4; u[1]=1; u[3]=-1
    v = [0]*4; v[2]=1; v[3]=1
    w = [0]*4; w[0]=1; w[3]=-1
    terms.append((u, v, w))
    
    return terms

def schoolbook_decomp(n, m, p):
    terms = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + k] = 1
                v[k * p + j] = 1
                w[j * n + i] = 1
                
                terms.append((u, v, w))
    return terms

# EVOLVE-BLOCK-START
def construct():
    return construct_impl()
# EVOLVE-BLOCK-END
