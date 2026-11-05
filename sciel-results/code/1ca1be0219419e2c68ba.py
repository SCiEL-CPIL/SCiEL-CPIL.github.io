import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 110

    # Attempt Rank 30 solution
    # Split into 3 blocks of 2x3x2
    # Each block uses Rank 10 scheme
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for block in range(3):
        # A indices: i in {0,1}, j in {0,1,2} -> idx = i*3 + j
        # B indices: j in {0,1,2}, k in {2*block, 2*block+1} -> idx = j*6 + k
        # C indices: k in {2*block, 2*block+1}, i in {0,1} -> idx = k*2 + i
        
        a00 = 0; a01 = 1; a02 = 2
        a10 = 3; a11 = 4; a12 = 5
        
        b00 = 0 * 6 + 2 * block
        b01 = 0 * 6 + 2 * block + 1
        b10 = 1 * 6 + 2 * block
        b11 = 1 * 6 + 2 * block + 1
        b20 = 2 * 6 + 2 * block
        b21 = 2 * 6 + 2 * block + 1
        
        c00 = (2 * block) * 2 + 0
        c01 = (2 * block + 1) * 2 + 0
        c10 = (2 * block) * 2 + 1
        c11 = (2 * block + 1) * 2 + 1
        
        # Rank 10 Scheme for 2x3x2
        # M1 = (a00 + a10) * (b00 + b20)
        U[a00, col] = 1.0; U[a10, col] = 1.0
        V[b00, col] = 1.0; V[b20, col] = 1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        # M2 = (a00 - a10) * (b01 + b21)
        U[a00, col] = 1.0; U[a10, col] = -1.0
        V[b01, col] = 1.0; V[b21, col] = 1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        # M3 = (a01 + a11) * (b00 + b10)
        U[a01, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b10, col] = 1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        # M4 = (a01 - a11) * (b01 + b11)
        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b01, col] = 1.0; V[b11, col] = 1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        # M5 = (a02 + a12) * (b00 + b20)
        U[a02, col] = 1.0; U[a12, col] = 1.0
        V[b00, col] = 1.0; V[b20, col] = 1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        # M6 = (a02 - a12) * (b01 + b21)
        U[a02, col] = 1.0; U[a12, col] = -1.0
        V[b01, col] = 1.0; V[b21, col] = 1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        # M7 = (a00 + a02) * (b00 + b10)
        U[a00, col] = 1.0; U[a02, col] = 1.0
        V[b00, col] = 1.0; V[b10, col] = 1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        # M8 = (a00 - a02) * (b01 + b11)
        U[a00, col] = 1.0; U[a02, col] = -1.0
        V[b01, col] = 1.0; V[b11, col] = 1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        # M9 = (a10 + a12) * (b00 + b10)
        U[a10, col] = 1.0; U[a12, col] = 1.0
        V[b00, col] = 1.0; V[b10, col] = 1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        # M10 = (a10 - a12) * (b01 + b11)
        U[a10, col] = 1.0; U[a12, col] = -1.0
        V[b01, col] = 1.0; V[b11, col] = 1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1

    # Verify the decomposition
    try:
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C from U, V, W
        # C[k*n + i, :] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]? No.
        # The decomposition is T = sum U_r V_r W_r^T
        # C = sum_r (U_r . A) (V_r . B) W_r^T? No.
        # Standard tensor decomposition:
        # C_{ik} = sum_{j,r} A_{ij} B_{jk} W_{ik,r} U_{ij,r} V_{jk,r}
        # Let's check the convention.
        # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
        # C = sum_r outer( (U[:,r] reshaped to n,m) * A , (V[:,r] reshaped to m,p) * B ) ?
        # No, the tensor T has indices (i, j, k).
        # T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r} (if W is indexed by k,i).
        # The problem says W[k*n+i, r]. So W_{ki,r}.
        # So C_{ik} = sum_{j,r} A_{ij} B_{jk} (U_{ij,r} V_{jk,r} W_{ki,r})?
        # No, the decomposition is of the *multiplication tensor*.
        # The multiplication tensor T has T_{ijk} = 1 if A_{ij} B_{jk} contributes to C_{ik}.
        # Actually, C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor is sum_{ijk} A_{ij} B_{jk} C_{ki}^*?
        # The decomposition provides the coefficients.
        # C_{ik} = sum_r W_{ki,r} * (sum_j U_{ij,r} A_{ij}) * (sum_j V_{jk,r} B_{jk})?
        # Let's check the shape.
        # U_{ij,r} is a coefficient for A_{ij}.
        # V_{jk,r} is a coefficient for B_{jk}.
        # W_{ki,r} is a coefficient for C_{ik}.
        # So C_{ik} = sum_r W_{ki,r} ( sum_j U_{ij,r} A_{ij} ) ( sum_j V_{jk,r} B_{jk} )
        # Let's test this.
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            
            # u_vec * A is element-wise? No, it's the contraction.
            # sum_j U_{ij,r} A_{ij} is a vector of length n? No, sum over j.
            # Let a_r = sum_j U_{ij,r} A_{ij} for each i. This is a vector of length n.
            # Let b_r = sum_j V_{jk,r} B_{jk} for each k. This is a vector of length p.
            # Then C_{ik} += W_{ki,r} * a_r[i] * b_r[k]
            
            a_r = np.zeros(n)
            for i in range(n):
                a_r[i] = np.sum(U[i*m:(i+1)*m, r] * A[i, :])
                
            b_r = np.zeros(p)
            for k in range(p):
                b_r[k] = np.sum(V[k:m*p:m, r] * B[:, k]) # V is (m*p, rank). Index j*p+k.
                # V[j*p+k, r] * B[j,k]. Sum over j.
                # V[:, r] is length m*p. Reshape to (m, p).
                # v_mat = V[:, r].reshape(m, p)
                # b_r = v_mat.T @ B ? No.
                # sum_j V_{jk,r} B_{jk} = sum_j (V[:,r].reshape(m,p)[j,k]) * B[j,k]
                # This is element-wise product summed over j?
                # No, it's a scalar for each k?
                # sum_j V_{jk,r} B_{jk}. Yes, for fixed k, sum over j.
                # So b_r[k] = sum_j V[j*p+k, r] * B[j,k]
                
            # Now C_{ik} += W[k*n+i, r] * a_r[i] * b_r[k]
            w_mat = W[:, r].reshape(p, n)
            # C_{ik} = sum_r w_mat[k,i] * a_r[i] * b_r[k]
            # This is an outer product scaled by w_mat?
            # C += w_mat.T * a_r[:, None] * b_r[None, :] ?
            # w_mat is (p, n). w_mat.T is (n, p).
            # a_r is (n,). b_r is (p,).
            # C += w_mat.T * (a_r[:, None] * b_r[None, :])
            
            C_computed += w_mat.T * (a_r[:, None] * b_r[None, :])
            
        if not np.allclose(C_computed, C_expected):
            # Fallback to Rank 33
            rank = 33
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            # Part 1: A[:, 0] * B[0, :] (Rank 12)
            for k in range(p):
                for i in range(n):
                    U[i * m + 0, col] = 1.0
                    V[0 * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

            # Part 2: A[:, 1:3] * B[1:3, :] (Rank 21)
            for t in range(3):
                a00 = 1; a01 = 2; a10 = 4; a11 = 5
                b00 = p + 2*t; b01 = p + 2*t + 1; b10 = 2*p + 2*t; b11 = 2*p + 2*t + 1
                c00 = 4*t; c01 = 4*t+1; c10 = 4*t+2; c11 = 4*t+3
                
                U[a00, col] = 1.0; U[a11, col] = 1.0
                V[b00, col] = 1.0; V[b11, col] = 1.0
                W[c00, col] = 1.0; W[c11, col] = 1.0
                col += 1

                U[a10, col] = 1.0; U[a11, col] = 1.0
                V[b00, col] = 1.0
                W[c10, col] = 1.0; W[c11, col] = -1.0
                col += 1

                U[a00, col] = 1.0
                V[b01, col] = 1.0; V[b11, col] = -1.0
                W[c01, col] = 1.0; W[c11, col] = 1.0
                col += 1

                U[a11, col] = 1.0
                V[b10, col] = 1.0; V[b00, col] = -1.0
                W[c00, col] = 1.0; W[c10, col] = 1.0
                col += 1

                U[a00, col] = 1.0; U[a01, col] = 1.0
                V[b11, col] = 1.0
                W[c00, col] = -1.0; W[c01, col] = 1.0
                col += 1

                U[a10, col] = 1.0; U[a00, col] = -1.0
                V[b00, col] = 1.0; V[b01, col] = 1.0
                W[c11, col] = 1.0
                col += 1

                U[a01, col] = 1.0; U[a11, col] = -1.0
                V[b10, col] = 1.0; V[b11, col] = 1.0
                W[c00, col] = 1.0
                col += 1

    except Exception:
        # If anything goes wrong, return the known good Rank 33
        rank = 33
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        # ... (Copy Rank 33 logic here for safety)
        # Re-using the logic from above to avoid duplication errors
        col = 0
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        for t in range(3):
            a00 = 1; a01 = 2; a10 = 4; a11 = 5
            b00 = p + 2*t; b01 = p + 2*t + 1; b10 = 2*p + 2*t; b11 = 2*p + 2*t + 1
            c00 = 4*t; c01 = 4*t+1; c10 = 4*t+2; c11 = 4*t+3
            U[a00, col] = 1.0; U[a11, col] = 1.0
            V[b00, col] = 1.0; V[b11, col] = 1.0
            W[c00, col] = 1.0; W[c11, col] = 1.0
            col += 1
            U[a10, col] = 1.0; U[a11, col] = 1.0
            V[b00, col] = 1.0
            W[c10, col] = 1.0; W[c11, col] = -1.0
            col += 1
            U[a00, col] = 1.0
            V[b01, col] = 1.0; V[b11, col] = -1.0
            W[c01, col] = 1.0; W[c11, col] = 1.0
            col += 1
            U[a11, col] = 1.0
            V[b10, col] = 1.0; V[b00, col] = -1.0
            W[c00, col] = 1.0; W[c10, col] = 1.0
            col += 1
            U[a00, col] = 1.0; U[a01, col] = 1.0
            V[b11, col] = 1.0
            W[c00, col] = -1.0; W[c01, col] = 1.0
            col += 1
            U[a10, col] = 1.0; U[a00, col] = -1.0
            V[b00, col] = 1.0; V[b01, col] = 1.0
            W[c11, col] = 1.0
            col += 1
            U[a01, col] = 1.0; U[a11, col] = -1.0
            V[b10, col] = 1.0; V[b11, col] = 1.0
            W[c00, col] = 1.0
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
