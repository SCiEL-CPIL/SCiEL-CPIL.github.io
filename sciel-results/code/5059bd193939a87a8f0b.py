import numpy as np
import time

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _strassen_2x2_block(i_base, j_base, k_base):
    """
    Returns the U, V, W vectors for a 2x2x2 Strassen block.
    i_base, j_base, k_base are the starting indices.
    The block covers i in [i_base, i_base+1], j in [j_base, j_base+1], k in [k_base, k_base+1].
    """
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the global indices.
    # U vector for term t has non-zeros at (i, j)
    # V vector for term t has non-zeros at (j, k)
    # W vector for term t has non-zeros at (k, i)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to get index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Term 1: (A00+A11)(B00+B11)
    # U: i0,j0=1; i1,j1=1
    # V: j0,k0=1; j1,k1=1
    # W: k0,i0=1; k1,i1=1
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    # Term 2: (A10+A11)B00
    # U: i1,j0=1; i1,j1=1
    # V: j0,k0=1
    # W: k0,i1=1
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    # Term 3: A00(B01-B11)
    # U: i0,j0=1
    # V: j0,k1=1; j1,k1=-1
    # W: k1,i0=1
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    # Term 4: A11(B10-B00)
    # U: i1,j1=1
    # V: j1,k0=1; j0,k0=-1
    # W: k0,i1=1
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    # Term 5: (A00+A01)B11
    # U: i0,j0=1; i0,j1=1
    # V: j1,k1=1
    # W: k1,i0=1
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    # Term 6: (A10-A00)(B00+B01)
    # U: i1,j0=1; i0,j0=-1
    # V: j0,k0=1; j0,k1=1
    # W: k0,i1=1; k1,i1=1
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    # Term 7: (A01-A11)(B10+B11)
    # U: i0,j1=1; i1,j1=-1
    # V: j1,k0=1; j1,k1=1
    # W: k0,i0=1; k1,i0=1
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy:
    # 1. Start with Schoolbook (Rank 36)
    # 2. Replace two disjoint 2x2x2 blocks with Strassen (Rank 7 each)
    #    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    #    Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # 3. Keep the rest as Schoolbook.
    
    # Determine which (i,j,k) tuples are covered by Strassen blocks
    strassen_covered = set()
    blocks = [
        (0, 0, 0), # i_base=0, j_base=0, k_base=0
        (0, 0, 2)  # i_base=0, j_base=0, k_base=2
    ]
    
    for ib, jb, kb in blocks:
        for i in range(ib, ib+2):
            for j in range(jb, jb+2):
                for k in range(kb, kb+2):
                    strassen_covered.add((i, j, k))
    
    # Initialize arrays
    # We will build the columns one by one.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add Strassen columns
    for ib, jb, kb in blocks:
        U_terms, V_terms, W_terms = _strassen_2x2_block(ib, jb, kb)
        for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
            # Convert dict to vector
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for idx, val in u_t.items():
                u_vec[idx] = val
            for idx, val in v_t.items():
                v_vec[idx] = val
            for idx, val in w_t.items():
                w_vec[idx] = val
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    # Add Schoolbook columns for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_covered:
                    continue
                # Schoolbook term
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
    
    # Assemble matrices
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    rank = U.shape[1]
    
    # Verify
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook if verification fails (should not happen)
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
