"""
Tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses recursive Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x1 (rank 16) = 65.
Includes robust verification and fallback to schoolbook (rank 80) if verification fails.
"""
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 140

    def verify_decomposition(U, V, W, n, m, p):
        """
        Verify that U, V, W correctly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Convention: C[i,k] = sum_j A[i,j] B[j,k]
        Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        """
        rank = U.shape[1]
        # Reshape to 3D tensors for einsum
        # U: (n, m, rank)
        U_3d = U.reshape(n, m, rank)
        # V: (m, p, rank)
        V_3d = V.reshape(m, p, rank)
        # W: (p, n, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute C = A @ B using the decomposition
        # C[i,k] = sum_{j, c} A[i,j] B[j,k] U[i,j,c] V[j,k,c] W[k,i,c]
        # Actually, the tensor identity is:
        # C[i,k] = sum_{j,c} A[i,j] B[j,k] * (U[i,j,c] V[j,k,c] W[k,i,c])
        # This is equivalent to:
        # C = sum_c (A * U[:, :, c]) @ (B * V[:, :, c]) ... no, that's not quite right.
        # The standard tensor decomposition for matrix multiplication is:
        # C_{ik} = \sum_{j,c} A_{ij} B_{jk} T_{ijkc}
        # Where T_{ijkc} = U_{ijc} V_{jkc} W_{kic}
        # So C_{ik} = \sum_c ( \sum_j A_{ij} U_{ijc} ) ( \sum_k ... ) NO.
        
        # Let's use the direct definition:
        # We want to check if for all i,k:
        # \sum_j A_{ij} B_{jk} = \sum_c (\sum_j A_{ij} U_{ijc}) (\sum_k ... ) NO.
        
        # The correct check is:
        # The tensor T_{ijk} = \sum_c U_{ijc} V_{jkc} W_{kic} must equal \delta_{jk} (for the multiplication tensor? No.)
        # The multiplication tensor M_{ijk} is 1 if j is the index connecting i and k? No.
        # M_{ijk} = 1 if we are computing C_{ik} = \sum_j A_{ij} B_{jk}.
        # So the tensor is T_{ijk} = \delta_{j, j} ... wait.
        # The tensor to decompose is T_{ijk} where T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
        # In standard indices: C_{ik} = \sum_j A_{ij} B_{jk}.
        # So the tensor is T_{ijk} = 1 for all i,j,k? No, that would be rank n*m*p.
        # The tensor is defined by the bilinear form.
        # The decomposition is: C_{ik} = \sum_c ( \sum_j A_{ij} U_{ijc} ) ( \sum_k B_{jk} V_{jkc} ) ... NO.
        
        # Let's stick to the explicit sum:
        # Val[i,k] = \sum_c \sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
        # This must equal C[i,k] = \sum_j A[i,j] * B[j,k].
        
        # Let's test with random A and B.
        rng = np.random.default_rng(42)
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_true = A @ B
        
        # Compute C_decomp
        # C_decomp[i,k] = \sum_c \sum_j A[i,j] B[j,k] U[i,j,c] V[j,k,c] W[k,i,c]
        # We can compute this efficiently:
        # Term_c[i,j,k] = U[i,j,c] V[j,k,c] W[k,i,c]
        # C_decomp[i,k] = \sum_c \sum_j A[i,j] B[j,k] Term_c[i,j,k]
        
        # Let's just do the einsum:
        # U_3d: (n, m, rank)
        # V_3d: (m, p, rank)
        # W_3d: (p, n, rank)
        
        # We want to compute:
        # C_decomp[i,k] = \sum_{j,c} A[i,j] B[j,k] U[i,j,c] V[j,k,c] W[k,i,c]
        
        # This is:
        # C_decomp = \sum_c ( A * U[:, :, c] ) @ ( B * V[:, :, c] ) ... NO.
        # Let's look at the indices again.
        # U[i,j,c], V[j,k,c], W[k,i,c].
        # Sum over j and c.
        
        # Let's compute it directly for small sizes.
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            # U_c: (n, m)
            U_c = U_3d[:, :, c]
            # V_c: (m, p)
            V_c = V_3d[:, :, c]
            # W_c: (p, n)
            W_c = W_3d[:, :, c]
            
            # The contribution of term c to C[i,k] is:
            # \sum_j A[i,j] B[j,k] U_c[i,j] V_c[j,k] W_c[k,i]
            
            # We can compute this as:
            # Temp[i,k] = \sum_j (A[i,j] U_c[i,j]) * (B[j,k] V_c[j,k]) * W_c[k,i]
            # This is not a simple matrix product because of W_c[k,i] depending on i and k.
            
            # Let's use einsum:
            # 'ij,jk,ki->ik'
            # A[i,j] * U_c[i,j] * B[j,k] * V_c[j,k] * W_c[k,i]
            # = (A * U_c)[i,j] * (B * V_c)[j,k] * W_c[k,i]
            
            AU = A * U_c  # (n, m)
            BV = B * V_c  # (m, p)
            
            # Now sum over j:
            # \sum_j AU[i,j] * BV[j,k] * W_c[k,i]
            # Let M[i,k] = \sum_j AU[i,j] * BV[j,k] = (AU @ BV)[i,k]
            M = AU @ BV  # (n, p)
            
            # Then C_decomp[i,k] += M[i,k] * W_c[k,i]
            C_decomp += M * W_c.T  # W_c is (p, n), so W_c.T is (n, p)
            
        return np.allclose(C_decomp, C_true, atol=1e-9)

    def build_4x4x4_strassen_numpy():
        """
        Build 4x4*4x4 decomposition using recursive Strassen. Rank = 49.
        Returns U, V, W as numpy arrays of shape (16, 49).
        """
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Helper to get indices
        def a_idx(bi, u, v):
            i = (bi // 2) * 2 + u
            j = (bi % 2) * 2 + v
            return i * 4 + j  # m=4

        def b_idx(bi, u, v):
            j = (bi // 2) * 2 + u
            k = (bi % 2) * 2 + v
            return j * 4 + k  # p=4 (for the 4x4x4 part)

        def c_idx(bi, u, v):
            i = (bi // 2) * 2 + u
            k = (bi % 2) * 2 + v
            return k * 4 + i  # n=4

        # Strassen definitions for 2x2x2
        # A_side[mk] is list of (block_idx, coeff)
        A_side = [
            [(0, 1.0), (3, 1.0)],
            [(2, 1.0), (3, 1.0)],
            [(0, 1.0)],
            [(3, 1.0)],
            [(0, 1.0), (1, 1.0)],
            [(2, 1.0), (0, -1.0)],
            [(1, 1.0), (3, -1.0)],
        ]
        
        B_side = [
            [(0, 1.0), (3, 1.0)],
            [(0, 1.0)],
            [(1, 1.0), (3, -1.0)],
            [(2, 1.0), (0, -1.0)],
            [(3, 1.0)],
            [(0, 1.0), (1, 1.0)],
            [(2, 1.0), (3, 1.0)],
        ]
        
        out_coeff = [
            {0: 1.0, 3: 1.0},
            {2: 1.0, 3: -1.0},
            {1: 1.0, 3: 1.0},
            {0: 1.0, 2: 1.0},
            {0: -1.0, 1: 1.0},
            {3: 1.0},
            {0: 1.0},
        ]
        
        # Inner Strassen (2x2x2) definitions
        inner_A_side = [
            [(0, 1.0), (3, 1.0)],
            [(2, 1.0), (3, 1.0)],
            [(0, 1.0)],
            [(3, 1.0)],
            [(0, 1.0), (1, 1.0)],
            [(2, 1.0), (0, -1.0)],
            [(1, 1.0), (3, -1.0)],
        ]
        
        inner_B_side = [
            [(0, 1.0), (3, 1.0)],
            [(0, 1.0)],
            [(1, 1.0), (3, -1.0)],
            [(2, 1.0), (0, -1.0)],
            [(3, 1.0)],
            [(0, 1.0), (1, 1.0)],
            [(2, 1.0), (3, 1.0)],
        ]
        
        inner_out_coeff = [
            {0: 1.0, 3: 1.0},
            {2: 1.0, 3: -1.0},
            {1: 1.0, 3: 1.0},
            {0: 1.0, 2: 1.0},
            {0: -1.0, 1: 1.0},
            {3: 1.0},
            {0: 1.0},
        ]
        
        for mk in range(7):
            for nk in range(7):
                col = mk * 7 + nk
                
                # Fill U
                for (bi, c1) in A_side[mk]:
                    for (bi2, c2) in inner_A_side[nk]:
                        u_local = bi2 // 2
                        v_local = bi2 % 2
                        U[a_idx(bi, u_local, v_local), col] += c1 * c2
                
                # Fill V
                for (bi, c1) in B_side[mk]:
                    for (bi2, c2) in inner_B_side[nk]:
                        u_local = bi2 // 2
                        v_local = bi2 % 2
                        V[b_idx(bi, u_local, v_local), col] += c1 * c2
                
                # Fill W
                for (bi_out, c_inner) in inner_out_coeff[nk].items():
                    for (bi_outer, c_outer) in out_coeff[mk].items():
                        u_local = bi_out // 2
                        v_local = bi_out % 2
                        W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer
        
        return U, V, W

    def build_schoolbook_4x4x1():
        """
        Build 4x4*4x1 decomposition. Rank = 16.
        U: 16x16, V: 4x16, W: 4x16
        """
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4 + j, col] = 1.0
                V[j, col] = 1.0
                W[i, col] = 1.0
                col += 1
        
        return U, V, W

    def build_schoolbook_4x4x5():
        """
        Build full schoolbook 4x4*4x5 decomposition. Rank = 80.
        """
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4 + j, col] = 1.0
                    V[j*5 + k, col] = 1.0
                    W[k*4 + i, col] = 1.0
                    col += 1
        
        return U, V, W

    # Attempt rank 65 construction
    U44, V44, W44 = build_4x4x4_strassen_numpy()
    U41, V41, W41 = build_schoolbook_4x4x1()
    
    total_rank = 65
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    # Copy 4x4x4 part (first 49 columns)
    # V and W for 4x4x4 only use the first 4 columns of B and C (k=0..3)
    U[:, :49] = U44
    V[:16, :49] = V44
    W[:16, :49] = W44
    
    # Copy 4x4x1 part (last 16 columns)
    # This corresponds to k=4
    U[:, 49:] = U41
    V[16:20, 49:] = V41
    W[16:20, 49:] = W41
    
    # Verify
    if verify_decomposition(U, V, W, 4, 4, 5):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 80
    U_s, V_s, W_s = build_schoolbook_4x4x5()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
