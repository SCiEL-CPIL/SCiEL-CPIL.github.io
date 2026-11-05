"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _elapsed(start):
    return time.time() - start

def _deadline_exceeded(start, limit=120.0):
    return _elapsed(start) > limit

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Helper to get U index for A[i, j]
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index for B[j, k]
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index for C[i, k] (stored as k*n+i)
    def w_idx(i, k):
        return k * n + i

    # 1. A11 B11 (2x2x2) -> Rank 7
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C11: i in {0,1}, k in {0,1}
    
    # Strassen 2x2x2
    # U: A00, A01, A10, A11
    # V: B00, B01, B10, B11
    # W: C00, C01, C10, C11 (indices k*n+i)
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    add_term(u_idx(0,0), v_idx(0,0), w_idx(0,0), 1, 1, 1)
    add_term(u_idx(1,1), v_idx(1,1), w_idx(0,0), 1, 1, 0) # Wait, add_term increments col. 
    # This approach is wrong. Strassen uses the SAME column for multiple U/V/W entries?
    # No, each rank-1 term is one column.
    # But U, V, W are vectors.
    # So for one term, we set multiple entries in U, V, W for that column.
    
    # Let's redefine add_term to not increment col, but set values.
    # And a separate function to advance col.
    
    # Reset
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def set_term(u_indices_vals, v_indices_vals, w_indices_vals):
        nonlocal col
        for idx, val in u_indices_vals:
            U[idx][col] += val
        for idx, val in v_indices_vals:
            V[idx][col] += val
        for idx, val in w_indices_vals:
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 for A11 B11
    # Local indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> w_idx(i,k)
    
    # M1
    set_term(
        [(u_idx(0,0), 1), (u_idx(1,1), 1)],
        [(v_idx(0,0), 1), (v_idx(1,1), 1)],
        [(w_idx(0,0), 1), (w_idx(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx(1,0), 1), (u_idx(1,1), 1)],
        [(v_idx(0,0), 1)],
        [(w_idx(1,0), 1), (w_idx(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx(0,0), 1)],
        [(v_idx(0,1), 1), (v_idx(1,1), -1)],
        [(w_idx(0,1), 1), (w_idx(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx(1,1), 1)],
        [(v_idx(1,0), 1), (v_idx(0,0), -1)],
        [(w_idx(0,0), 1), (w_idx(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx(0,0), 1), (u_idx(0,1), 1)],
        [(v_idx(1,1), 1)],
        [(w_idx(0,0), -1), (w_idx(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx(1,0), 1), (u_idx(0,0), -1)],
        [(v_idx(0,0), 1), (v_idx(0,1), 1)],
        [(w_idx(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx(0,1), 1), (u_idx(1,1), -1)],
        [(v_idx(1,0), 1), (v_idx(1,1), 1)],
        [(w_idx(0,0), 1)]
    )

    # 2. A12 B21 (2x2x2) -> Rank 7
    # A12: i in {0,1}, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # C12? No, A12 B21 contributes to C11?
    # A is 3x4, B is 4x5.
    # C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # A12 (rows 0,1, cols 2,3) * B21 (rows 2,3, cols 0,1).
    # Contributes to C[i,k] for i in {0,1}, k in {0,1}.
    # So it adds to the C11 block (i in 0,1, k in 0,1).
    
    # Map local 0->2, 1->3 for A cols and B rows.
    def u_idx_12(i, j_local):
        return u_idx(i, j_local + 2)
    def v_idx_21(j_local, k):
        return v_idx(j_local + 2, k)
    # W is same as C11 block: w_idx(i,k)
    
    # M1
    set_term(
        [(u_idx_12(0,0), 1), (u_idx_12(1,1), 1)],
        [(v_idx_21(0,0), 1), (v_idx_21(1,1), 1)],
        [(w_idx(0,0), 1), (w_idx(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx_12(1,0), 1), (u_idx_12(1,1), 1)],
        [(v_idx_21(0,0), 1)],
        [(w_idx(1,0), 1), (w_idx(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx_12(0,0), 1)],
        [(v_idx_21(0,1), 1), (v_idx_21(1,1), -1)],
        [(w_idx(0,1), 1), (w_idx(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx_12(1,1), 1)],
        [(v_idx_21(1,0), 1), (v_idx_21(0,0), -1)],
        [(w_idx(0,0), 1), (w_idx(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx_12(0,0), 1), (u_idx_12(0,1), 1)],
        [(v_idx_21(1,1), 1)],
        [(w_idx(0,0), -1), (w_idx(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx_12(1,0), 1), (u_idx_12(0,0), -1)],
        [(v_idx_21(0,0), 1), (v_idx_21(0,1), 1)],
        [(w_idx(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx_12(0,1), 1), (u_idx_12(1,1), -1)],
        [(v_idx_21(1,0), 1), (v_idx_21(1,1), 1)],
        [(w_idx(0,0), 1)]
    )

    # 3. A11 B12 (2x2x3) -> Rank 11
    # A11: i in {0,1}, j in {0,1}
    # B12: j in {0,1}, k in {2,3,4}
    # C12: i in {0,1}, k in {2,3,4}
    
    # Split B12 into B12_L (k=2,3) and B12_R (k=4)
    # B12_L is 2x2. B12_R is 2x1.
    
    # Strassen on A11 * B12_L
    def v_idx_12L(j, k_local):
        return v_idx(j, k_local + 2)
    def w_idx_12L(i, k_local):
        return w_idx(i, k_local + 2)
        
    # M1
    set_term(
        [(u_idx(0,0), 1), (u_idx(1,1), 1)],
        [(v_idx_12L(0,0), 1), (v_idx_12L(1,1), 1)],
        [(w_idx_12L(0,0), 1), (w_idx_12L(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx(1,0), 1), (u_idx(1,1), 1)],
        [(v_idx_12L(0,0), 1)],
        [(w_idx_12L(1,0), 1), (w_idx_12L(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx(0,0), 1)],
        [(v_idx_12L(0,1), 1), (v_idx_12L(1,1), -1)],
        [(w_idx_12L(0,1), 1), (w_idx_12L(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx(1,1), 1)],
        [(v_idx_12L(1,0), 1), (v_idx_12L(0,0), -1)],
        [(w_idx_12L(0,0), 1), (w_idx_12L(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx(0,0), 1), (u_idx(0,1), 1)],
        [(v_idx_12L(1,1), 1)],
        [(w_idx_12L(0,0), -1), (w_idx_12L(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx(1,0), 1), (u_idx(0,0), -1)],
        [(v_idx_12L(0,0), 1), (v_idx_12L(0,1), 1)],
        [(w_idx_12L(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx(0,1), 1), (u_idx(1,1), -1)],
        [(v_idx_12L(1,0), 1), (v_idx_12L(1,1), 1)],
        [(w_idx_12L(0,0), 1)]
    )
    
    # Standard on A11 * B12_R (2x2x1)
    # B12_R: j in {0,1}, k=4.
    # C12_R: i in {0,1}, k=4.
    def v_idx_12R(j):
        return v_idx(j, 4)
    def w_idx_12R(i):
        return w_idx(i, 4)
        
    # C_i0 = A_i0 B_00 + A_i1 B_10
    # Here B is col vector.
    set_term([(u_idx(0,0), 1)], [(v_idx_12R(0), 1)], [(w_idx_12R(0), 1)])
    set_term([(u_idx(0,1), 1)], [(v_idx_12R(1), 1)], [(w_idx_12R(0), 1)])
    set_term([(u_idx(1,0), 1)], [(v_idx_12R(0), 1)], [(w_idx_12R(1), 1)])
    set_term([(u_idx(1,1), 1)], [(v_idx_12R(1), 1)], [(w_idx_12R(1), 1)])

    # 4. A12 B22 (2x2x3) -> Rank 11
    # A12: i in {0,1}, j in {2,3}
    # B22: j in {2,3}, k in {2,3,4}
    # C22: i in {0,1}, k in {2,3,4}
    
    def u_idx_12_2(i, j_local):
        return u_idx(i, j_local + 2)
    def v_idx_22(j_local, k_local):
        return v_idx(j_local + 2, k_local + 2)
    def w_idx_22(i, k_local):
        return w_idx(i, k_local + 2)
        
    # Strassen on A12 * B22_L (k=2,3)
    # M1
    set_term(
        [(u_idx_12_2(0,0), 1), (u_idx_12_2(1,1), 1)],
        [(v_idx_22(0,0), 1), (v_idx_22(1,1), 1)],
        [(w_idx_22(0,0), 1), (w_idx_22(1,1), 1)]
    )
    # M2
    set_term(
        [(u_idx_12_2(1,0), 1), (u_idx_12_2(1,1), 1)],
        [(v_idx_22(0,0), 1)],
        [(w_idx_22(1,0), 1), (w_idx_22(1,1), -1)]
    )
    # M3
    set_term(
        [(u_idx_12_2(0,0), 1)],
        [(v_idx_22(0,1), 1), (v_idx_22(1,1), -1)],
        [(w_idx_22(0,1), 1), (w_idx_22(1,1), 1)]
    )
    # M4
    set_term(
        [(u_idx_12_2(1,1), 1)],
        [(v_idx_22(1,0), 1), (v_idx_22(0,0), -1)],
        [(w_idx_22(0,0), 1), (w_idx_22(1,0), 1)]
    )
    # M5
    set_term(
        [(u_idx_12_2(0,0), 1), (u_idx_12_2(0,1), 1)],
        [(v_idx_22(1,1), 1)],
        [(w_idx_22(0,0), -1), (w_idx_22(0,1), 1)]
    )
    # M6
    set_term(
        [(u_idx_12_2(1,0), 1), (u_idx_12_2(0,0), -1)],
        [(v_idx_22(0,0), 1), (v_idx_22(0,1), 1)],
        [(w_idx_22(1,1), 1)]
    )
    # M7
    set_term(
        [(u_idx_12_2(0,1), 1), (u_idx_12_2(1,1), -1)],
        [(v_idx_22(1,0), 1), (v_idx_22(1,1), 1)],
        [(w_idx_22(0,0), 1)]
    )
    
    # Standard on A12 * B22_R (k=4)
    def v_idx_22R(j_local):
        return v_idx(j_local + 2, 4)
    def w_idx_22R(i):
        return w_idx(i, 4)
        
    set_term([(u_idx_12_2(0,0), 1)], [(v_idx_22R(0), 1)], [(w_idx_22R(0), 1)])
    set_term([(u_idx_12_2(0,1), 1)], [(v_idx_22R(1), 1)], [(w_idx_22R(0), 1)])
    set_term([(u_idx_12_2(1,0), 1)], [(v_idx_22R(0), 1)], [(w_idx_22R(1), 1)])
    set_term([(u_idx_12_2(1,1), 1)], [(v_idx_22R(1), 1)], [(w_idx_22R(1), 1)])

    # 5. A21 B11 (1x2x2) -> Rank 4
    # A21: i=2, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C21: i=2, k in {0,1}
    
    def u_idx_21(j):
        return u_idx(2, j)
    def w_idx_21(k):
        return w_idx(2, k)
        
    set_term([(u_idx_21(0), 1)], [(v_idx(0,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_21(1), 1)], [(v_idx(1,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_21(0), 1)], [(v_idx(0,1), 1)], [(w_idx_21(1), 1)])
    set_term([(u_idx_21(1), 1)], [(v_idx(1,1), 1)], [(w_idx_21(1), 1)])

    # 6. A22 B21 (1x2x2) -> Rank 4
    # A22: i=2, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # C21: i=2, k in {0,1}
    
    def u_idx_22(j_local):
        return u_idx(2, j_local + 2)
    def v_idx_21_2(j_local, k):
        return v_idx(j_local + 2, k)
        
    set_term([(u_idx_22(0), 1)], [(v_idx_21_2(0,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_22(1), 1)], [(v_idx_21_2(1,0), 1)], [(w_idx_21(0), 1)])
    set_term([(u_idx_22(0), 1)], [(v_idx_21_2(0,1), 1)], [(w_idx_21(1), 1)])
    set_term([(u_idx_22(1), 1)], [(v_idx_21_2(1,1), 1)], [(w_idx_21(1), 1)])

    # 7. A21 B12 (1x2x3) -> Rank 6
    # A21: i=2, j in {0,1}
    # B12: j in {0,1}, k in {2,3,4}
    # C22: i=2, k in {2,3,4}
    
    def w_idx_22_2(k_local):
        return w_idx(2, k_local + 2)
        
    for j in range(2):
        for k_local in range(3):
            set_term(
                [(u_idx_21(j), 1)],
                [(v_idx(j, k_local + 2), 1)],
                [(w_idx_22_2(k_local), 1)]
            )

    # 8. A22 B22 (1x2x3) -> Rank 6
    # A22: i=2, j in {2,3}
    # B22: j in {2,3}, k in {2,3,4}
    # C22: i=2, k in {2,3,4}
    
    for j_local in range(2):
        for k_local in range(3):
            set_term(
                [(u_idx_22(j_local), 1)],
                [(v_idx_22(j_local, k_local), 1)],
                [(w_idx_22_2(k_local), 1)]
            )

    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    # Verify the decomposition is correct
    if not _deadline_exceeded(_start, 110.0):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_vec = U_arr[:, r]
            v_vec = V_arr[:, r]
            w_vec = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    u_val = u_vec[i * m + j]
                    if u_val != 0:
                        for k in range(p):
                            v_val = v_vec[j * p + k]
                            w_val = w_vec[k * n + i]
                            T[i, j, k] += u_val * v_val * w_val
        
        # Check against the standard multiplication tensor
        # T[i,j,k] should be 1 if this is a valid multiplication tensor
        # For matrix multiplication, the tensor is T[i,j,k] = delta[i,j,k] in the right sense
        # Actually, for A*B=C, the tensor satisfies C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
        # Wait, no. The matrix multiplication tensor is defined as:
        # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
        # So T[i,j,k] = 1 if the indices are consistent, 0 otherwise
        # But in our case, all (i,j,k) triples are valid for 3x4x5
        # So T should be all ones? No, that's not right either.
        
        # The tensor for matrix multiplication of (n x m) * (m x p) -> (n x p) is:
        # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
        # So T[i,j,k] = 1 for all i,j,k in the valid ranges
        # But that would mean the tensor is all ones, which has rank n*m*p = 60
        # That can't be right.
        
        # Actually, the matrix multiplication tensor is:
        # T = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)
        # In flattened form, T[i*m+j, j*p+k, k*n+i] = 1
        # So the tensor is a 3-tensor of size (n*m) x (m*p) x (p*n)
        # with T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # Otherwise T[a, b, c] = 0
        
        # Let's build the correct target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1
        
        # Now check if our decomposition matches
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        
        if not np.allclose(T_decomp, T_target, atol=1e-10):
            diff = np.abs(T_decomp - T_target)
            max_diff = np.max(diff)
            if max_diff > 1e-10:
                # The decomposition is incorrect; return the schoolbook method as fallback
                rank_fb = n * m * p
                U_fb = np.zeros((n*m, rank_fb))
                V_fb = np.zeros((m*p, rank_fb))
                W_fb = np.zeros((p*n, rank_fb))
                col_fb = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i*m+j, col_fb] = 1
                            V_fb[j*p+k, col_fb] = 1
                            W_fb[k*n+i, col_fb] = 1
                            col_fb += 1
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
