import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=3 (rows of A), m=5 (cols of A / rows of B), p=5 (cols of B)
    # A is 3x5, B is 5x5, C is 3x5.
    
    # Candidate Rank: 65
    # Structure:
    # Row 0 of A: 25 terms (all j,k)
    # Row 1 of A: 20 terms (j != k)
    # Row 2 of A: 20 terms (j != k)
    
    rank_candidate = 65
    U = np.zeros((n * m, rank_candidate), dtype=int)
    V = np.zeros((m * p, rank_candidate), dtype=int)
    W = np.zeros((p * n, rank_candidate), dtype=int)
    
    col = 0
    # Row 0
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
    # Row 1
    for j in range(m):
        for k in range(p):
            if j != k:
                U[1 * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + 1, col] = 1
                col += 1
                
    # Row 2
    for j in range(m):
        for k in range(p):
            if j != k:
                U[2 * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + 2, col] = 1
                col += 1

    # Verify
    # C_ik = sum_j A_ij B_jk
    # Using decomposition: C = W^T * (V^T * B * U^T)? No.
    # C_{ik} = sum_t U_{i,m+j?} ...
    # Standard: C_{ik} = sum_t W_{k*n+i, t} * (sum_j U_{i*m+j, t} * V_{j*p+k, t} * A_{ij} B_{jk}?)
    # No, the decomposition is:
    # M(A,B) = sum_t U_t(A) V_t(B) W_t
    # Where U_t is a vector, V_t is a vector.
    # C_{ik} = sum_t (sum_j U_{i*m+j, t} A_{ij}) * (sum_l V_{l*p+k, t} B_{lk}) * W_{k*n+i, t}?
    # Wait, the indices in the seed program:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Product: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]?
    # No, the tensor decomposition is:
    # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}?
    # The seed program uses:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This corresponds to term A[i,j] B[j,k] contributing to C[i,k].
    # So C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j] * V[j*p+k, c] B[j,k])?
    # No, it's bilinear.
    # C[i,k] = sum_c W[k*n+i, c] * ( (U_c . A_i) * (V_c . B_k) )?
    # Where U_c is a row of U, V_c is a row of V.
    # A_i is row i of A. B_k is column k of B?
    # V[j*p+k, c] indexes B[j,k]. So V_c is a vector of length m*p.
    # B is m x p. Flatten row-major: B[j,k] is at j*p+k.
    # So V_c . B_flat = sum_j V[j*p+k, c] B[j,k]? No, V is indexed by j*p+k.
    # So sum_{j,k} V[j*p+k, c] B[j,k].
    # Similarly U_c . A_flat = sum_{i,j} U[i*m+j, c] A[i,j].
    
    # Let's verify with random matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for c in range(rank_candidate):
        # Extract U_c and V_c
        u_vec = U[:, c]
        v_vec = V[:, c]
        
        # u_vec is length n*m. Reshape to n x m.
        u_mat = u_vec.reshape(n, m)
        # v_vec is length m*p. Reshape to m x p.
        v_mat = v_vec.reshape(m, p)
        
        # Term is (u_mat * A) ? No.
        # The decomposition is:
        # C = sum_c W_c * (U_c(A)) * (V_c(B))
        # Where U_c(A) is a scalar?
        # In the seed: U[i*m+j, c]=1.
        # So U_c(A) = sum_{i,j} U[i*m+j, c] A[i,j].
        # But U only has 1s in specific spots.
        # For a specific c, U[i*m+j, c] is 1 for exactly one (i,j)?
        # In schoolbook, yes.
        # In my candidate, for row 0, U[0*m+j, c]=1 for specific j.
        # So U_c(A) is just A[i,j] for that specific i,j?
        # No, U is n*m x rank.
        # For a fixed c, U[:, c] has non-zero entries.
        # In my construction, for c in row 0 block, U[i*m+j, c] is 1 for i=0, specific j.
        # So U_c(A) = A[0, j].
        # Similarly V_c(B) = B[j, k].
        # And W_c is 1 at W[k*n+0, c].
        # So C[0, k] += A[0, j] * B[j, k] * 1.
        
        # So the term is:
        # scalar_u = sum_{i,j} U[i*m+j, c] * A[i,j]
        # scalar_v = sum_{j,k} V[j*p+k, c] * B[j,k]
        # C += W[:, c] * (scalar_u * scalar_v)
        
        scalar_u = np.dot(U[:, c], A.flatten())
        scalar_v = np.dot(V[:, c], B.flatten())
        
        w_vec = W[:, c]
        # w_vec is length p*n. Reshape to p x n?
        # W[k*n+i, c]. Index is k*n+i.
        # So w_mat[k, i] = W[k*n+i, c].
        w_mat = w_vec.reshape(p, n)
        
        C_comp += w_mat.T * (scalar_u * scalar_v) # w_mat.T is n x p.
        
    if not np.allclose(C_true, C_comp, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
