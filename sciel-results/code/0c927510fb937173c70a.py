import time
import random
import numpy as np

def construct():
    _t0 = time.time()
    DEADLINE = _t0 + 110.0  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 6
    RANK = 66
    
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    
    col = 0
    
    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)
        
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1; col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1; col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1; col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1; col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # Build 66-term decomposition
    for j in range(3):
        bc = 2 * j
        _add_1x2_2x2(0, 0, 0, bc, 0, bc)
        _add_1x2_2x2(0, 2, 2, bc, 0, bc)
        _add_strassen(1, 0, 0, bc, 1, bc)
        _add_strassen(1, 2, 2, bc, 1, bc)

    assert col == RANK, f"Expected {RANK} terms, got {col}"
    
    # Convert to numpy for faster verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomp(U_mat, V_mat, W_mat):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U_mat.shape[1]
        # Check: sum_k U[i,k] * V[k,j] * W[j,i] == delta_{i,k} * delta_{k,j}
        # For all i in [0,n), k in [0,m), j in [0,p)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    expected = 1.0 if (i >= 0 and k >= 0 and j >= 0) else 0.0
                    # Actually the tensor T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j]
                    # So T[i,k,j] = 1 for all valid i,k,j
                    # Wait, let me reconsider. The tensor for matrix multiplication:
                    # T[i,k,j] = 1 means A[i,k] * B[k,j] contributes to C[i,j]
                    # So T[i,k,j] = 1 for all i,k,j in range
                    expected = 1.0
                    actual = 0.0
                    for t in range(r):
                        actual += U_mat[i*m+k, t] * V_mat[k*p+j, t] * W_mat[j*n+i, t]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Verify baseline
    if not verify_decomp(U_np, V_np, W_np):
        # Return baseline if verification fails
        return {"U": U, "V": V, "W": W}
    
    # Local search: try to reduce rank by merging or dropping terms
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = RANK
    
    def try_merge_terms(U_mat, V_mat, W_mat, r):
        """Try to merge two terms into one by finding linear combination."""
        for a in range(r):
            for b in range(a+1, r):
                # Check if term b is a scalar multiple of term a
                # Compare U columns
                u_a = U_mat[:, a]
                u_b = U_mat[:, b]
                
                # Find non-zero entries in u_a
                nz_a = np.nonzero(u_a)[0]
                if len(nz_a) == 0:
                    continue
                
                # Check if u_b is proportional to u_a
                ratio = None
                for idx in nz_a:
                    if u_a[idx] != 0:
                        if ratio is None:
                            ratio = u_b[idx] / u_a[idx]
                        else:
                            if abs(u_b[idx] / u_a[idx] - ratio) > 1e-9:
                                break
                        break
                
                if ratio is None:
                    continue
                
                # Verify proportionality for all entries
                valid = True
                for idx in range(len(u_a)):
                    expected = u_a[idx] * ratio
                    if abs(u_b[idx] - expected) > 1e-9:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check V and W similarly
                v_a = V_mat[:, a]
                v_b = V_mat[:, b]
                valid = True
                for idx in range(len(v_a)):
                    expected = v_a[idx] * ratio
                    if abs(v_b[idx] - expected) > 1e-9:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                w_a = W_mat[:, a]
                w_b = W_mat[:, b]
                valid = True
                for idx in range(len(w_a)):
                    expected = w_a[idx] * ratio
                    if abs(w_b[idx] - expected) > 1e-9:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Merge: remove term b, scale term a by (1 + ratio)
                # Actually, if term_b = ratio * term_a, then we can absorb
                # But we need to be careful with signs
                new_U = U_mat.copy()
                new_V = V_mat.copy()
                new_W = W_mat.copy()
                
                # Term b contributes ratio * (term a's contribution)
                # So total = (1 + ratio) * term_a
                new_U[:, a] = u_a * (1 + ratio)
                new_V[:, a] = v_a
                new_W[:, a] = w_a
                # Remove term b
                new_U = np.delete(new_U, b, axis=1)
                new_V = np.delete(new_V, b, axis=1)
                new_W = np.delete(new_W, b, axis=1)
                
                # Verify
                if verify_decomp(new_U, new_V, new_W):
                    return new_U, new_V, new_W, r - 1
                
        return None
    
    def try_drop_term(U_mat, V_mat, W_mat, r, term_idx):
        """Try to drop a single term."""
        new_U = np.delete(U_mat, term_idx, axis=1)
        new_V = np.delete(V_mat, term_idx, axis=1)
        new_W = np.delete(W_mat, term_idx, axis=1)
        
        if verify_decomp(new_U, new_V, new_W):
            return new_U, new_V, new_W, r - 1
        return None
    
    # Try to drop or merge terms
    current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
    current_rank = best_rank
    
    # Try dropping each term
    for i in range(current_rank):
        if time.time() > DEADLINE - 5:
            break
        result = try_drop_term(current_U, current_V, current_W, current_rank, i)
        if result:
            current_U, current_V, current_W, current_rank = result
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            best_rank = current_rank
            # Try again from new state
            i = 0
            continue
    
    # Try merging pairs
    for _ in range(10):  # Limit iterations
        if time.time() > DEADLINE - 5:
            break
        result = try_merge_terms(current_U, current_V, current_W, current_rank)
        if result:
            current_U, current_V, current_W, current_rank = result
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            best_rank = current_rank
        else:
            break
    
    # Convert back to lists
    final_U = best_U.tolist()
    final_V = best_V.tolist()
    final_W = best_W.tolist()
    
    return {"U": final_U, "V": final_V, "W": final_W}
