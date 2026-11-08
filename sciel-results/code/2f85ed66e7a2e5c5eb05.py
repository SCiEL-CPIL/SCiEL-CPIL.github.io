import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Rank-71 decomposition: 4x Strassen 2x2x2 + schoolbook remainder
    # Blocks: (0,0,0), (0,2,0), (0,0,2), (0,2,2) cover i in {0,1}, j in {0..3}, k in {0..3}
    # Remaining: 43 schoolbook terms
    # Total rank: 4*7 + 43 = 71
    
    R = 71
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        base = col
        # Strassen 2x2x2: 7 rank-1 terms
        # M1 = (A11+A22)(B11+B22)
        U[i0*m + j0][base] += 1
        U[(i0+1)*m + (j0+1)][base] += 1
        V[j0*p + k0][base] += 1
        V[(j0+1)*p + (k0+1)][base] += 1
        W[k0*n + i0][base] += 1
        W[(k0+1)*n + (i0+1)][base] += 1
        
        # M2 = (A21+A22)B11
        U[(i0+1)*m + j0][base+1] += 1
        U[(i0+1)*m + (j0+1)][base+1] += 1
        V[j0*p + k0][base+1] += 1
        W[k0*n + (i0+1)][base+1] += 1
        W[(k0+1)*n + (i0+1)][base+1] += -1
        
        # M3 = A11(B21-B11)
        U[i0*m + j0][base+2] += 1
        V[j0*p + (k0+1)][base+2] += 1
        V[(j0+1)*p + (k0+1)][base+2] += -1
        W[(k0+1)*n + i0][base+2] += 1
        W[(k0+1)*n + (i0+1)][base+2] += 1
        
        # M4 = A22(B12-B22)
        U[(i0+1)*m + (j0+1)][base+3] += 1
        V[(j0+1)*p + k0][base+3] += 1
        V[j0*p + k0][base+3] += -1
        W[k0*n + i0][base+3] += 1
        W[k0*n + (i0+1)][base+3] += 1
        
        # M5 = (A11+A12)B22
        U[i0*m + j0][base+4] += 1
        U[i0*m + (j0+1)][base+4] += 1
        V[(j0+1)*p + (k0+1)][base+4] += 1
        W[k0*n + i0][base+4] += -1
        W[(k0+1)*n + i0][base+4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][base+5] += 1
        U[i0*m + j0][base+5] += -1
        V[j0*p + k0][base+5] += 1
        V[j0*p + (k0+1)][base+5] += 1
        W[(k0+1)*n + (i0+1)][base+5] += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][base+6] += 1
        U[(i0+1)*m + (j0+1)][base+6] += -1
        V[(j0+1)*p + k0][base+6] += 1
        V[(j0+1)*p + (k0+1)][base+6] += 1
        W[k0*n + i0][base+6] += 1
        
        col += 7
    
    # Four 2x2x2 Strassen blocks
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
    
    # Mark covered positions
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    if time.time() - start_time > timeout:
        pass
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verify correctness
    def check(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # C[k*n+i, :] should equal sum over r of U[i*m+j, r] * V[j*p+k, r]
                    # But wait, the convention is W[k*n+i, r]
                    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
                    # 
                    # The tensor decomposition is:
                    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
                    # where T is the matrix multiplication tensor.
                    #
                    # For matrix multiplication C = A * B:
                    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
                    #
                    # So the tensor T_{i,j,k} = 1 if this is a valid term, else 0.
                    # Actually, T_{i,j,k} corresponds to the coefficient of A_{i,j} * B_{j,k} in C_{i,k}.
                    # So T_{i,j,k} = delta_{i,i} * delta_{k,k} ... no.
                    #
                    # The matrix multiplication tensor has T_{i,j,k} = 1 for all i,j,k in the appropriate ranges,
                    # and the decomposition is:
                    # sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1 for all i,j,k
                    #
                    # Wait, that's not right either. Let me think again.
                    #
                    # The bilinear form is: C_{i,k} = sum_j A_{i,j} B_{j,k}
                    #
                    # In tensor form, the multiplication tensor T satisfies:
                    # T(A,B)_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
                    # where T_{i,j,k} = 1.
                    #
                    # The tensor decomposition expresses T as:
                    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
                    #
                    # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    #
                    # Let's verify this.
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False, (i, j, k, val)
        return True, None
    
    ok, info = check(U, V, W)
    if ok:
        print(f"Rank: {U.shape[1]} - CORRECT")
    else:
        print(f"Rank: {U.shape[1]} - INCORRECT at {info}")
