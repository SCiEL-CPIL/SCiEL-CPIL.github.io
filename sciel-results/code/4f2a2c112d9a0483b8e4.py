import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 15]

def _build_target_tensor():
    """
    Construct the matrix multiplication tensor M_{2,2,15}.
    Indices:
    U: n*m = 4
    V: m*p = 30
    W: p*n = 30
    
    Convention:
    A is 2x2. Index (i, j) -> flat index i*m + j.
    B is 2x15. Index (j, k) -> flat index j*p + k.
    C is 2x15. Index (i, k) -> flat index k*n + i (as per task description W[k*n+i, r]).
    """
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """
    Verify if U, V, W decompose T exactly.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False
    rank = U.shape[1]
    
    # Efficiently compute the tensor reconstruction
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    diff = np.max(np.abs(T_recon - T))
    return diff < tol

def _get_seed_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U, V, W

def _construct_strassen_decomposition():
    """
    Construct the optimal rank-53 decomposition for 2x2x15 matrix multiplication.
    
    Strategy:
    The tensor $M_{2,2,15}$ can be decomposed using the Kronecker product structure.
    We decompose the dimension $p=15$ into $7 \times 2 + 1 \times 1$.
    The $2 \times 2 \times 2$ tensor has rank 7 (Strassen).
    The $2 \times 2 \times 1$ tensor has rank 4 (standard).
    Total rank = $7 \times 7 + 1 \times 4 = 49 + 4 = 53$.
    
    Indices:
    A is 2x2 (indices i, j)
    B is 2x15 (indices j, k). We map k to (block, local_k).
    C is 2x15 (indices i, k).
    
    For each block b in range(7) (size 2) and one block of size 1:
    We use the Strassen algorithm for the 2x2x2 part.
    
    Strassen 2x2x2:
    Inputs: A (2x2), B (2x2).
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A10)(B00+B10)
    M7 = (A01-A11)(B11+B01)
    
    Outputs:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Let's define the factor matrices U, V, W for the whole 2x2x15 problem.
    
    U dimension: 4 (i, j)
    V dimension: 30 (j, k)
    W dimension: 30 (k, i) -> Note: W index is k*n + i.
    
    We will build columns for each of the 7 Strassen blocks and 1 standard block.
    """
    n, m, p = DIMENSIONS
    # p = 15. We use 7 blocks of size 2 and 1 block of size 1.
    # Block 0: k in [0, 1]
    # Block 1: k in [2, 3]
    # ...
    # Block 6: k in [12, 13]
    # Block 7: k in [14, 14] (size 1)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3 (i*m+j)
    # B indices for a 2x2 block: 00, 01, 10, 11 -> flat: j*2 + local_k
    
    # M1: (A00+A11)(B00+B11)
    # A part: [1, 0, 0, 1]
    # B part: [1, 0, 0, 1] (relative to block)
    # C part: 
    # C00 += 1, C01 += 0, C10 += 0, C11 += 1
    # W indices: k*n + i.
    # For block b, local_k=0,1.
    # k_global = b*2 + local_k.
    # C00: i=0, j=0. k_global. W[k_global*2 + 0]
    # C11: i=1, j=1. k_global. W[k_global*2 + 1]
    
    def add_strassen_term(b, local_b_0, local_b_1, coeff_b0, coeff_b1, a_vec, c_offsets):
        # a_vec: length 4, values for A00, A01, A10, A11
        # c_offsets: dict mapping (i, local_k) to coeff
        # Actually, it's easier to just define the 7 terms explicitly.
        pass
        
    # Let's explicitly build the 7 Strassen terms for a generic 2x2x2 block.
    # Term 1: M1
    # A: A00 + A11 -> [1, 0, 0, 1]
    # B: B00 + B11 -> [1, 0, 0, 1] (indices j*2+lk: 0*2+0, 1*2+1 -> 0, 3)
    # C: C00 + C11
    # C00: i=0, lk=0,1. C11: i=1, lk=0,1.
    
    # We will generate the columns for one block, then shift them for each block.
    
    def generate_block_2x2x2():
        cols_U = []
        cols_V = []
        cols_W = []
        
        # Term 1: M1
        u = np.array([1, 0, 0, 1])
        v = np.array([1, 0, 0, 1]) # B00, B01, B10, B11 -> indices 0, 1, 2, 3
        # Wait, B is 2x2. Index j*p + k.
        # In a 2x2 block, j is 0 or 1. k is local 0 or 1.
        # Flat index in V for this block: j*2 + local_k.
        # B00: j=0, lk=0 -> 0
        # B01: j=0, lk=1 -> 1
        # B10: j=1, lk=0 -> 2
        # B11: j=1, lk=1 -> 3
        # So v = [1, 0, 0, 1] corresponds to B00 + B11.
        
        # W: C00 + C11
        # C00: i=0, lk=0,1. C11: i=1, lk=0,1.
        # W index: k*n + i.
        # For lk=0: k=0. W[0*2+0]=W0, W[0*2+1]=W1.
        # For lk=1: k=1. W[1*2+0]=W2, W[1*2+1]=W3.
        # C00 (i=0) is at W[0] and W[2].
        # C11 (i=1) is at W[1] and W[3].
        # So w = [1, 1, 1, 1] for the 4 entries in this block?
        # Let's check:
        # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i).
        # Term M1 contributes to:
        # (i=0, j=0, k=0) -> u=0, v=0, w=0
        # (i=1, j=1, k=0) -> u=3, v=3, w=1
        # (i=0, j=0, k=1) -> u=0, v=1, w=2
        # (i=1, j=1, k=1) -> u=3, v=3, w=3
        # 
        # My U vector [1,0,0,1] means u=0 and u=3 are active.
        # My V vector [1,0,0,1] means v=0 and v=3 are active.
        # My W vector needs to be 1 at w=0, w=1, w=2, w=3?
        # Let's check the tensor product:
        # Sum over r U[i,j,r] V[j,k,r] W[k,i,r]
        # For r=M1:
        # (i=0, j=0, k=0): U[0]=1, V[0]=1, W[0] should be 1.
        # (i=1, j=1, k=0): U[3]=1, V[3]=1, W[1] should be 1. (w = 0*2+1 = 1)
        # (i=0, j=0, k=1): U[0]=1, V[1]=1, W[2] should be 1. (w = 1*2+0 = 2)
        # (i=1, j=1, k=1): U[3]=1, V[3]=1, W[3] should be 1. (w = 1*2+1 = 3)
        # So W is [1, 1, 1, 1].
        
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(np.array([1, 1, 1, 1]))
        
        # Term 2: M2 = (A10+A11)B00
        # A: A10+A11 -> [0, 0, 1, 1] (indices 2, 3)
        # B: B00 -> [1, 0, 0, 0] (index 0)
        # C: C10
        # C10: i=1, j=0.
        # k=0: w = 0*2+1 = 1.
        # k=1: w = 1*2+1 = 3.
        # So W is [0, 1, 0, 1].
        u = np.array([0, 0, 1, 1])
        v = np.array([1, 0, 0, 0])
        w = np.array([0, 1, 0, 1])
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # Term 3: M3 = A00(B01-B11)
        # A: A00 -> [1, 0, 0, 0]
        # B: B01-B11 -> [0, 1, 0, -1] (indices 1, 3)
        # C: C01
        # C01: i=0, j=1.
        # k=0: w = 0*2+0 = 0.
        # k=1: w = 1*2+0 = 2.
        # So W is [1, 0, 1, 0].
        u = np.array([1, 0, 0, 0])
        v = np.array([0, 1, 0, -1])
        w = np.array([1, 0, 1, 0])
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # Term 4: M4 = A11(B10-B00)
        # A: A11 -> [0, 0, 0, 1]
        # B: B10-B00 -> [ -1, 0, 1, 0] (indices 0, 2)
        # C: C00 + C10
        # C00: i=0, j=0. k=0->w0, k=1->w2.
        # C10: i=1, j=0. k=0->w1, k=1->w3.
        # So W is [1, 1, 1, 1].
        u = np.array([0, 0, 0, 1])
        v = np.array([-1, 0, 1, 0])
        w = np.array([1, 1, 1, 1])
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # Term 5: M5 = (A00+A10)B11
        # A: A00+A10 -> [1, 0, 1, 0]
        # B: B11 -> [0, 0, 0, 1]
        # C: -C00 + C01
        # C00: i=0, j=0. k=0->w0, k=1->w2. Coeff -1.
        # C01: i=0, j=1. k=0->w0, k=1->w2. Coeff +1.
        # Wait, C00 and C01 both have i=0.
        # C00 is (i=0, j=0). C01 is (i=0, j=1).
        # In the tensor, U index distinguishes j.
        # Here U is [1, 0, 1, 0], so j=0 (u=0) and j=1 (u=2).
        # For u=0 (j=0): contributes to C00.
        # For u=2 (j=1): contributes to C01.
        # W index depends on k and i. i=0 for both.
        # k=0: w=0. k=1: w=2.
        # So for u=0, w=0 and w=2 should be -1.
        # For u=2, w=0 and w=2 should be +1.
        # But W is a single vector for the whole term.
        # This implies that the term M5 cannot be represented as a single rank-1 tensor if the W vector must be the same for all i,j?
        # No, the tensor is T[u, v, w].
        # The term is U_vec * V_vec * W_vec.
        # U_vec has non-zeros at u=0 and u=2.
        # V_vec has non-zero at v=3.
        # W_vec has non-zeros at w=0 and w=2.
        # The product will have non-zeros at:
        # (u=0, v=3, w=0), (u=0, v=3, w=2)
        # (u=2, v=3, w=0), (u=2, v=3, w=2)
        # With value U[u]*V[v]*W[w].
        # U[0]=1, U[2]=1. V[3]=1.
        # We want:
        # (0, 3, 0) -> -1 (C00, k=0)
        # (0, 3, 2) -> -1 (C00, k=1)
        # (2, 3, 0) -> +1 (C01, k=0)
        # (2, 3, 2) -> +1 (C01, k=1)
        # If W[0]=1 and W[2]=1, then:
        # (0,3,0) = 1*1*1 = 1. We want -1.
        # (2,3,0) = 1*1*1 = 1. We want +1.
        # This works if we flip the sign in U or V.
        # Let's set U[0] = -1, U[2] = 1.
        # Then (0,3,0) = -1*1*1 = -1.
        # (2,3,0) = 1*1*1 = 1.
        # Correct.
        u = np.array([-1, 0, 1, 0])
        v = np.array([0, 0, 0, 1])
        w = np.array([1, 0, 1, 0])
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # Term 6: M6 = (A11-A10)(B00+B10)
        # A: A11-A10 -> [0, 0, -1, 1]
        # B: B00+B10 -> [1, 0, 1, 0]
        # C: C11
        # C11: i=1, j=1.
        # k=0: w=1. k=1: w=3.
        # So W is [0, 1, 0, 1].
        u = np.array([0, 0, -1, 1])
        v = np.array([1, 0, 1, 0])
        w = np.array([0, 1, 0, 1])
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # Term 7: M7 = (A01-A11)(B11+B01)
        # A: A01-A11 -> [0, 1, 0, -1]
        # B: B11+B01 -> [0, 1, 0, 1]
        # C: C00
        # C00: i=0, j=0.
        # k=0: w=0. k=1: w=2.
        # So W is [1, 0, 1, 0].
        u = np.array([0, 1, 0, -1])
        v = np.array([0, 1, 0, 1])
        w = np.array([1, 0, 1, 0])
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        return cols_U, cols_V, cols_W

    def generate_block_2x2x1():
        # Standard 4 terms for 2x2x1.
        # A (2x2), B (2x1), C (2x1).
        # C00 = A00 B0
        # C01 = A01 B1
        # C10 = A10 B0
        # C11 = A11 B1
        
        cols_U = []
        cols_V = []
        cols_W = []
        
        # T1: A00 B0 -> C00
        u = np.array([1, 0, 0, 0])
        v = np.array([1, 0]) # B0, B1
        w = np.array([1, 0]) # C0, C1 (i=0, i=1). Wait, W index is k*n+i. k=0.
        # C00: i=0. w = 0*2+0 = 0.
        # C10: i=1. w = 0*2+1 = 1.
        # So w = [1, 0].
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # T2: A01 B1 -> C01
        u = np.array([0, 1, 0, 0])
        v = np.array([0, 1])
        w = np.array([1, 0]) # C01: i=0. w=0.
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # T3: A10 B0 -> C10
        u = np.array([0, 0, 1, 0])
        v = np.array([1, 0])
        w = np.array([0, 1]) # C10: i=1. w=1.
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        # T4: A11 B1 -> C11
        u = np.array([0, 0, 0, 1])
        v = np.array([0, 1])
        w = np.array([0, 1]) # C11: i=1. w=1.
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)
        
        return cols_U, cols_V, cols_W

    # Assemble
    # 7 blocks of 2x2x2
    for b in range(7):
        u_cols, v_cols, w_cols = generate_block_2x2x2()
        for r in range(7):
            U_full = np.zeros(4)
            V_full = np.zeros(30)
            W_full = np.zeros(30)
            
            U_full[:] = u_cols[r]
            
            # Shift V
            # Block b starts at k = b*2.
            # V index: j*p + k.
            # For this block, j is 0 or 1. k is b*2 + local_k.
            # local_k 0,1 correspond to indices 0,1 in the 4-element vector.
            # So V_full[j*15 + b*2 + local_k] = v_cols[r][j*2 + local_k]
            for j in range(2):
                for lk in range(2):
                    v_idx = j * 15 + (b * 2 + lk)
                    v_col_idx = j * 2 + lk
                    V_full[v_idx] = v_cols[r][v_col_idx]
                    
            # Shift W
            # W index: k*n + i.
            # k is b*2 + local_k. i is 0 or 1.
            # W_full[(b*2+lk)*2 + i] = w_cols[r][lk*2 + i]
            for lk in range(2):
                for i in range(2):
                    w_idx = (b * 2 + lk) * 2 + i
                    w_col_idx = lk * 2 + i
                    W_full[w_idx] = w_cols[r][w_col_idx]
                    
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
            
    # 1 block of 2x2x1 (k=14)
    u_cols, v_cols, w_cols = generate_block_2x2x1()
    for r in range(4):
        U_full = np.zeros(4)
        V_full = np.zeros(30)
        W_full = np.zeros(30)
        
        U_full[:] = u_cols[r]
        
        # Shift V
        # k = 14.
        # V index: j*15 + 14.
        for j in range(2):
            v_idx = j * 15 + 14
            v_col_idx = j
            V_full[v_idx] = v_cols[r][v_col_idx]
            
        # Shift W
        # k = 14.
        # W index: 14*2 + i = 28 + i.
        for i in range(2):
            w_idx = 14 * 2 + i
            w_col_idx = i
            W_full[w_idx] = w_cols[r][w_col_idx]
            
        U_list.append(U_full)
        V_list.append(V_full)
        W_list.append(W_full)
        
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # 1. Get seed decomposition (Rank 60)
    U_seed, V_seed, W_seed = _get_seed_decomposition()
    
    # 2. Construct Strassen decomposition (Rank 53)
    try:
        U_str, V_str, W_str = _construct_strassen_decomposition()
        
        # Verify
        if _verify_decomposition(U_str, V_str, W_str, T, tol=1e-6):
            return {
                "U": U_str.tolist(),
                "V": V_str.tolist(),
                "W": W_str.tolist()
            }
        else:
            # If verification fails, print debug info
            print("Strassen decomposition verification failed", file=sys.stderr)
            # Attempt a quick numerical cleanup?
            # Since it's integer based, it should be exact.
            # If it fails, fall back to seed.
            pass
    except Exception as e:
        print(f"Strassen construction failed: {e}", file=sys.stderr)
        
    # Fallback to seed
    return {
        "U": U_seed.tolist(),
        "V": V_seed.tolist(),
        "W": W_seed.tolist()
    }

if __name__ == "__main__":
    construct()
