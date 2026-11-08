import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Block definitions: (i0, j0, k0)
    # We place 4 disjoint 2x2x2 Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # Total rank = 4 * 7 (Strassen) + remaining schoolbook
    # Covered terms: 4 * 8 = 32
    # Total terms: 3 * 5 * 5 = 75
    # Schoolbook: 75 - 32 = 43
    # Total Rank: 28 + 43 = 71
    
    rank = 71
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    def add_strassen(i0, j0, k0, c):
        # 7 multiplications
        # 1
        U[i0*m+j0][c] += 1; U[(i0+1)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        
        # 2
        U[(i0+1)*m+j0][c+1] += 1; U[(i0+1)*m+(j0+1)][c+1] += 1
        V[j0*p+k0][c+1] += 1
        W[k0*n+(i0+1)][c+1] += 1; W[(k0+1)*n+(i0+1)][c+1] += -1
        
        # 3
        U[i0*m+j0][c+2] += 1
        V[j0*p+(k0+1)][c+2] += 1; V[(j0+1)*p+(k0+1)][c+2] += -1
        W[(k0+1)*n+i0][c+2] += 1; W[(k0+1)*n+(i0+1)][c+2] += 1
        
        # 4
        U[(i0+1)*m+(j0+1)][c+3] += 1
        V[(j0+1)*p+k0][c+3] += 1; V[j0*p+k0][c+3] += -1
        W[k0*n+i0][c+3] += 1; W[k0*n+(i0+1)][c+3] += 1
        
        # 5
        U[i0*m+j0][c+4] += 1; U[i0*m+(j0+1)][c+4] += 1
        V[(j0+1)*p+(k0+1)][c+4] += 1
        W[k0*n+i0][c+4] += -1; W[(k0+1)*n+i0][c+4] += 1
        
        # 6
        U[(i0+1)*m+j0][c+5] += 1; U[i0*m+j0][c+5] += -1
        V[j0*p+k0][c+5] += 1; V[j0*p+(k0+1)][c+5] += 1
        W[(k0+1)*n+(i0+1)][c+5] += 1
        
        # 7
        U[i0*m+(j0+1)][c+6] += 1; U[(i0+1)*m+(j0+1)][c+6] += -1
        V[(j0+1)*p+k0][c+6] += 1; V[(j0+1)*p+(k0+1)][c+6] += 1
        W[k0*n+i0][c+6] += 1
        
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True

    for i0, j0, k0 in blocks:
        add_strassen(i0, j0, k0, col)
        col += 7
        
    # Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verify
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_ref = A @ B
    
    C_calc = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        a_u = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                a_u[i,j] = u[i*m+j]
                
        b_v = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                b_v[j,k] = v[j*p+k]
                
        c_w = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                c_w[i,k] = w[k*n+i]
                
        C_calc += a_u @ b_v * c_w # This is wrong, it's outer product sum
        
    # Correct verification
    C_calc = np.zeros((n, p))
    for r in range(U.shape[1]):
        # u is vector of nm
        # v is vector of mp
        # w is vector of pn
        # Term is sum_{i,j,k} U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
        # = sum_{i,j} ( sum_k U[i,j,r] W[k,i,r] B[j,k] ) A[i,j] ? No.
        # It is: sum_r (u_r^T A v_r) w_r^T
        # where u_r is n x m, v_r is m x p, w_r is p x n.
        
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n) # W[k*n+i] -> row k, col i? 
        # W index is k*n+i. So w_mat[k,i] = W[k*n+i].
        
        # C += (A u_mat^T ? No)
        # Term is trace(W (A U V)^T)?
        # Standard: C = sum_r (A * u_r) (v_r * B) ... no.
        # C_{ik} += sum_j A_{ij} B_{jk} W_{ki} ? No.
        # Tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
        # C_{ik} = sum_j sum_l A_{il} B_{lk} T_{il k}? No.
        # C = A B.
        # (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # We want sum_r U_{ij} V_{jk} W_{ki} = A_{ij} B_{jk}.
        # So U_{ij} V_{jk} W_{ki} is the coefficient for the monomial A_{ij} B_{jk} in the expansion?
        # No, the decomposition is for the tensor of multiplication.
        # T_{ijk} = U_{i j r} V_{j k r} W_{k i r} ? No.
        # U shape (n*m, rank). Index i*m+j.
        # V shape (m*p, rank). Index j*p+k.
        # W shape (p*n, rank). Index k*n+i.
        
        # C_{ik} = sum_j sum_r (U_{im+j, r} V_{jp+k, r} W_{kn+i, r}) A_{ij} B_{jk}
        
        a_u = np.zeros((n, m))
        a_u.flat = U[:, r] # i*m+j
        
        b_v = np.zeros((m, p))
        b_v.flat = V[:, r] # j*p+k
        
        w_w = np.zeros((p, n))
        w_w.flat = W[:, r] # k*n+i -> w_w[k,i]
        
        # C += (A * a_u.T?) No.
        # Sum over j,k: A_{ij} B_{jk} U_{ij} V_{jk} W_{ki}
        # = sum_j A_{ij} U_{ij} ( sum_k B_{jk} V_{jk} W_{ki} )
        # Let M_{j i} = sum_k V_{jk} W_{ki}
        # Then C_{ik} += sum_j A_{ij} U_{ij} M_{j i}
        # This is not a simple matrix multiply.
        
        # Simpler check:
        # Reconstruct C
        c_term = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c_term[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        C_calc += c_term
        
    err = np.max(np.abs(C_calc - C_ref))
    print(f"Rank: {U.shape[1]}, Error: {err}")
