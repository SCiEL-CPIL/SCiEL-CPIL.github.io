import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, num_tests=30, seed=42):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
            return False
    return True


def _build_strassen_2x2_2x2():
    """Return (U, V, W) for 2x2 * 2x2 = 2x2 using Strassen (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen for C = A B where A, B are 2x2
    # U: index = i*2+j
    U[0, 0] = 1; U[3, 0] = 1       # M1: a+d
    U[2, 1] = 1; U[3, 1] = 1       # M2: c+d
    U[0, 2] = 1                    # M3: a
    U[3, 3] = 1                    # M4: d
    U[0, 4] = 1; U[1, 4] = 1       # M5: a+b
    U[2, 5] = 1; U[0, 5] = -1      # M6: c-a
    U[1, 6] = 1; U[3, 6] = -1      # M7: b-d
    
    # V: index = j*2+k
    V[0, 0] = 1; V[3, 0] = 1       # M1: e+h
    V[0, 1] = 1                    # M2: e
    V[1, 2] = 1; V[3, 2] = -1      # M3: f-h
    V[2, 3] = 1; V[0, 3] = -1      # M4: g-e
    V[3, 4] = 1                    # M5: h
    V[0, 5] = 1; V[2, 5] = 1       # M6: e+g
    V[1, 6] = 1; V[3, 6] = 1       # M7: f+h
    
    # W: index = k*2+i
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1   # C00
    W[1, 1] = 1; W[1, 3] = 1                              # C10
    W[2, 2] = 1; W[2, 4] = 1                              # C01
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1   # C11
    
    return U, V, W


def _build_trivial(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m*p) + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    
    return U, V, W


def _round_to_half_int(matrix, tol=1e-6):
    """Round a matrix to the nearest half-integer (0, 0.5, 1, 1.5, ...)."""
    return np.round(matrix * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Build a dense "coefficient matrix" M of size (nmp) x R.
    # M[(i, j, k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want M to be the identity matrix (or at least have rank 1 in each (i,j,k) slot).
    # Actually, the condition is:
    # For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # And for all (i', j', k') != (i, j, k): sum_r U[i*m+j', r] V[j'*p+k', r] W[k'*n+i', r] = 0
    
    # Let's construct a large R decomposition using a "product" of Strassen and Trivial.
    # We can use a Kronecker-like structure.
    # A (2x4) * B (4x6).
    # Split A into 2x2 blocks: A = [A1, A2].
    # Split B into 2x3 blocks? No, B is 4x6. Split B into 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23]].
    # This gives the rank 42 construction.
    
    # To find a smaller rank, we can treat the 42 columns as vectors in R^(nmp) and find the minimal subset that spans the "identity" directions?
    # No, the tensor is not linear in that simple way.
    
    # Alternative: Use SVD to find the numerical rank of the "flattened" tensor.
    # But the tensor is not just a matrix.
    
    # Let's try a different approach: "Greedy Null Space Elimination".
    # Start with the rank 42 decomposition (U42, V42, W42).
    # We want to find a linear combination of the 42 terms that is zero.
    # i.e., find a vector c in R^42 such that sum_r c_r * T_r = 0, where T_r is the rank-1 tensor.
    # If such a c exists, we can project out the subspace spanned by c.
    
    # Let's build the 42-term decomposition first.
    rank_42 = 42
    U42 = np.zeros((n*m, rank_42), dtype=float)
    V42 = np.zeros((m*p, rank_42), dtype=float)
    W42 = np.zeros((p*n, rank_42), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2_2x2()
    
    for a_block in range(2):
        for b_block in range(3):
            base_r = (a_block * 3 + b_block) * 7
            a_col_offset = a_block * 2
            b_row_offset = a_block * 2
            b_col_offset = b_block * 2
            
            for r_s in range(7):
                r = base_r + r_s
                for i in range(2):
                    for j in range(2):
                        full_j = j + a_col_offset
                        U42[i * m + full_j, r] = U_s[i * 2 + j, r_s]
                
                for j in range(2):
                    for k in range(2):
                        full_j = j + b_row_offset
                        full_k = k + b_col_offset
                        V42[full_j * p + full_k, r] = V_s[j * 2 + k, r_s]
                
                for k in range(2):
                    for i in range(2):
                        full_k = k + b_col_offset
                        W42[full_k * n + i, r] = W_s[k * 2 + i, r_s]
                        
    # Now, let's check if we can reduce the rank.
    # We can form the "Hadamard product" matrix.
    # Let H be a matrix of size (nmp) x rank_42.
    # H[(i,j,k), r] = U42[i*m+j, r] * V42[j*p+k, r] * W42[k*n+i, r].
    # We want H to be the Identity matrix of size nmp (which is 48).
    # Wait, the condition is that for each (i,j,k), the dot product of the r-th column of H with the e_{(i,j,k)} vector is 1.
    # So H should be the Identity matrix (48 x 42).
    # If H is 48 x 42, it cannot be the identity.
    # However, the decomposition is valid if for all (i,j,k), sum_r ... = 1.
    # This means that the vector of 1s for each (i,j,k) is in the column space?
    
    # Let's check the rank of H. If rank(H) < 42, we can reduce the rank.
    # Actually, we can just SVD H.
    # H = U_h S V_h^T.
    # If some singular values are ~0, we can drop those columns.
    
    H = np.zeros((n*m*p, rank_42), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m*p) + j * p + k
                H[idx, :] = U42[i*m+j, :] * V42[j*p+k, :] * W42[k*n+i, :]
                
    # We want H to be the Identity matrix (48x48). But it's 48x42.
    # The condition is: For each (i,j,k), the sum over r of H[idx, r] * (something?)
    # No, the condition is exactly:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # This means that the vector H[idx, :] must be such that when we sum it (dot with ones?), no.
    # The sum is just the sum of the elements of H[idx, :]? No.
    # The sum is sum_r (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]).
    # This is exactly sum_r H[idx, r].
    # So we require sum_r H[idx, r] = 1 for all idx.
    # And for "orthogonality" (which is not required for matrix multiplication, only for the tensor identity)?
    # Wait, the tensor identity is:
    # T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
    # We want T_{ijk} = delta_{ij} delta_{jk} delta_{ki}? No.
    # We want T_{ijk} to be the matrix multiplication tensor.
    # T_{ijk} = 1 if (i,j,k) is a valid term in the sum?
    # No, the matrix multiplication tensor M_{2,4,6} has entries M_{i, j, k} = 1 if we are computing C_{ik} = sum_j A_{ij} B_{jk}.
    # So M_{i, j, k} is 1 for all i, j, k.
    # Wait, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor is T_{i, j, k} = 1 for all i, j, k.
    # So we require sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for ALL i, j, k.
    # This means that the vector H[idx, :] must sum to 1 for all idx.
    
    # Let's check if the 42-term decomposition satisfies this.
    # It should, by construction.
    
    # Can we reduce the rank?
    # We can try to find a linear dependency among the columns of H.
    # If H has rank < 42, we can reduce.
    
    U_h, S_h, V_h_t = np.linalg.svd(H, full_matrices=False)
    
    # Find the numerical rank
    tol = 1e-10
    num_rank = np.sum(S_h > tol)
    
    if num_rank < rank_42:
        # We can drop the last (rank_42 - num_rank) columns
        # But we need to recompute U, V, W from the reduced H.
        # This is tricky because H is a "flattened" tensor.
        # We can project U, V, W onto the subspace spanned by the first num_rank right singular vectors.
        
        V_h = V_h_t[:num_rank, :].T  # (42, num_rank)
        
        U_new = U42 @ V_h
        V_new = V42 @ V_h
        W_new = W42 @ V_h
        
        # Verify
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            # Try to round to half-integers
            U_r = _round_to_half_int(U_new)
            V_r = _round_to_half_int(V_new)
            W_r = _round_to_half_int(W_new)
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            else:
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If no reduction found, return the 42-term decomposition
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

# EVOLVE-BLOCK-END
