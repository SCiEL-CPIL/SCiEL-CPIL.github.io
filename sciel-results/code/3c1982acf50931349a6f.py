"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def _time_left(limit=145):
    return limit - (time.time() - _start_time)

def construct():
    """Construct a rank-minimal decomposition for 4x5 x 5x5 matrix multiplication."""
    n, m, p = 4, 5, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def make_u():
        return [0] * (n * m)
    
    def make_v():
        return [0] * (m * p)
    
    def make_w():
        return [0] * (p * n)
    
    # Strassen 2x2x2 decomposition
    # For A (2x2) * B (2x2) = C (2x2):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_block(i0, j0, k0):
        """Add Strassen terms for 2x2x2 block starting at (i0,j0,k0).
        i0: row offset in A/C (0 or 2)
        j0: col offset in A / row offset in B (0 or 2)
        k0: col offset in B / col offset in C (0 or 2)
        """
        # M1 = (a+d)(e+h): a=A[i0,j0], d=A[i0+1,j0+1], e=B[j0,k0], h=B[j0+1,k0+1]
        u = make_u(); v = make_v(); w = make_w()
        u[i0*m + j0] = 1
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = 1      # C11 += M1
        w[(k0+1)*n + (i0+1)] = 1  # C22 += M1
        add_term(u, v, w)
        
        # M2 = (c+d)e: c=A[i0+1,j0], d=A[i0+1,j0+1], e=B[j0,k0]
        u = make_u(); v = make_v(); w = make_w()
        u[(i0+1)*m + j0] = 1
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + k0] = 1
        w[k0*n + (i0+1)] = 1      # C21 += M2
        w[(k0+1)*n + (i0+1)] = -1 # C22 -= M2
        add_term(u, v, w)
        
        # M3 = a(f-h): a=A[i0,j0], f=B[j0,k0+1], h=B[j0+1,k0+1]
        u = make_u(); v = make_v(); w = make_w()
        u[i0*m + j0] = 1
        v[j0*p + (k0+1)] = 1
        v[(j0+1)*p + (k0+1)] = -1
        w[(k0+1)*n + i0] = 1      # C12 += M3
        w[(k0+1)*n + (i0+1)] = 1  # C22 += M3
        add_term(u, v, w)
        
        # M4 = d(g-e): d=A[i0+1,j0+1], g=B[j0+1,k0], e=B[j0,k0]
        u = make_u(); v = make_v(); w = make_w()
        u[(i0+1)*m + (j0+1)] = 1
        v[(j0+1)*p + k0] = 1
        v[j0*p + k0] = -1
        w[k0*n + i0] = 1      # C11 += M4
        w[k0*n + (i0+1)] = 1  # C21 += M4
        add_term(u, v, w)
        
        # M5 = (a+b)h: a=A[i0,j0], b=A[i0,j0+1], h=B[j0+1,k0+1]
        u = make_u(); v = make_v(); w = make_w()
        u[i0*m + j0] = 1
        u[i0*m + (j0+1)] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = -1      # C11 -= M5
        w[(k0+1)*n + i0] = 1   # C12 += M5
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f): c=A[i0+1,j0], a=A[i0,j0], e=B[j0,k0], f=B[j0,k0+1]
        u = make_u(); v = make_v(); w = make_w()
        u[(i0+1)*m + j0] = 1
        u[i0*m + j0] = -1
        v[j0*p + k0] = 1
        v[j0*p + (k0+1)] = 1
        w[(k0+1)*n + (i0+1)] = 1  # C22 += M6
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h): b=A[i0,j0+1], d=A[i0+1,j0+1], g=B[j0+1,k0], h=B[j0+1,k0+1]
        u = make_u(); v = make_v(); w = make_w()
        u[i0*m + (j0+1)] = 1
        u[(i0+1)*m + (j0+1)] = -1
        v[(j0+1)*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = 1  # C11 += M7
        add_term(u, v, w)
    
    # Add Strassen for the four 2x2x2 blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                if _time_left() < 5:
                    break
                strassen_block(i0, j0, k0)
    
    # Add standard terms for all (i,j,k) where j==4 or k==4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == 4 or k == 4:
                    if _time_left() < 2:
                        break
                    u = make_u(); v = make_v(); w = make_w()
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    add_term(u, v, w)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

