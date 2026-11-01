import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U_r[a,b,:] * V_r[c,d,:] * W_r[k,i,:])
                            if abs(s - target) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Rank 24 construction
    # 3 blocks of 2x3x2, each rank 8
    # Total rank 24
    
    R = 24
    U = np.zeros((n*m, R)) # 6 x 24
    V = np.zeros((m*p, R)) # 18 x 24
    W = np.zeros((p*n, R)) # 12 x 24
    
    # Strassen 2x2x2 coefficients
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map local indices (a, b, c, d) to global indices
    # Block s: d_global = 2s + d_loc, k_global = 2s + k_loc
    # b_global = b_loc, c_global = c_loc
    
    for s in range(3):
        base = s * 8
        
        # Part 1: 2x2x2 using b,c in {0,1}
        # Term 0: M1
        # U: a=0,b=0; a=1,b=1
        U[0*3+0, base+0] = 1
        U[1*3+1, base+0] = 1
        # V: c=0,d=0; c=1,d=1
        V[0*6+(2*s+0), base+0] = 1
        V[1*6+(2*s+1), base+0] = 1
        # W: k=0,i=0; k=1,i=1
        W[(2*s+0)*2+0, base+0] = 1
        W[(2*s+1)*2+1, base+0] = 1
        
        # Term 1: M2
        # U: a=1,b=0; a=1,b=1
        U[1*3+0, base+1] = 1
        U[1*3+1, base+1] = 1
        # V: c=0,d=0
        V[0*6+(2*s+0), base+1] = 1
        # W: k=0,i=1
        W[(2*s+0)*2+1, base+1] = 1
        
        # Term 2: M3
        # U: a=0,b=0
        U[0*3+0, base+2] = 1
        # V: c=0,d=1; c=1,d=1
        V[0*6+(2*s+1), base+2] = 1
        V[1*6+(2*s+1), base+2] = -1
        # W: k=1,i=0
        W[(2*s+1)*2+0, base+2] = 1
        
        # Term 3: M4
        # U: a=1,b=1
        U[1*3+1, base+3] = 1
        # V: c=1,d=0; c=0,d=0
        V[1*6+(2*s+0), base+3] = 1
        V[0*6+(2*s+0), base+3] = -1
        # W: k=0,i=1
        W[(2*s+0)*2+1, base+3] = 1
        
        # Term 4: M5
        # U: a=0,b=0; a=0,b=1
        U[0*3+0, base+4] = 1
        U[0*3+1, base+4] = 1
        # V: c=1,d=1
        V[1*6+(2*s+1), base+4] = 1
        # W: k=0,i=0; k=1,i=0
        W[(2*s+0)*2+0, base+4] = 1
        W[(2*s+1)*2+0, base+4] = 1
        
        # Term 5: M6
        # U: a=1,b=0; a=0,b=0
        U[1*3+0, base+5] = 1
        U[0*3+0, base+5] = -1
        # V: c=0,d=0; c=0,d=1
        V[0*6+(2*s+0), base+5] = 1
        V[0*6+(2*s+1), base+5] = 1
        # W: k=1,i=1
        W[(2*s+1)*2+1, base+5] = 1
        
        # Term 6: M7
        # U: a=0,b=1; a=1,b=1
        U[0*3+1, base+6] = 1
        U[1*3+1, base+6] = -1
        # V: c=1,d=0; c=1,d=1
        V[1*6+(2*s+0), base+6] = 1
        V[1*6+(2*s+1), base+6] = 1
        # W: k=0,i=0
        W[(2*s+0)*2+0, base+6] = 1
        
        # Part 2: 2x1x2 using b=2, c=2
        # Term 7: A_a2 * B_2d
        # U: a=0,b=2; a=1,b=2
        U[0*3+2, base+7] = 1
        U[1*3+2, base+7] = 1
        # V: c=2, d=0; c=2, d=1
        V[2*6+(2*s+0), base+7] = 1
        V[2*6+(2*s+1), base+7] = 1
        # W: k=0,i=0; k=1,i=0; k=0,i=1; k=1,i=1
        # Wait, W index is k*n+i.
        # We need W_{k,i} such that sum U V W = delta(a,i) delta(c,k) delta(b,d)
        # For this term, we want it to contribute to C_{a,d} for a=0,1 and d=0,1.
        # The term is A_{a,2} B_{2,d}.
        # So for fixed a,d, we need 1.
        # U_{a,2} = 1. V_{2,d} = 1.
        # We need W_{d, a} = 1?
        # Check: sum_r U_{a,2,r} V_{2,d,r} W_{d,a,r}
        # Only r=base+7.
        # U_{a,2,r} = 1. V_{2,d,r} = 1.
        # So we need W_{d,a,r} = 1.
        W[(2*s+0)*2+0, base+7] = 1 # k=0, i=0
        W[(2*s+0)*2+1, base+7] = 1 # k=0, i=1
        W[(2*s+1)*2+0, base+7] = 1 # k=1, i=0
        W[(2*s+1)*2+1, base+7] = 1 # k=1, i=1

    if time.time() < deadline:
        if check_decomposition_fast(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback
    U_school = np.zeros((6, 36))
    V_school = np.zeros((18, 36))
    W_school = np.zeros((12, 36))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_school[i * 3 + j, col] = 1
                V_school[j * 6 + k, col] = 1
                W_school[k * 2 + i, col] = 1
                col += 1
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

if __name__ == "__main__":
    construct()
