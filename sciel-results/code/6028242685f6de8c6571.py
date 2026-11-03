import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 30
    W_dim = p * n  # 12
    
    # Build target tensor T of shape (U_dim, V_dim, W_dim)
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i
    target = np.zeros((U_dim, V_dim, W_dim), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1

    def verify(U, V, W):
        """
        Verify if U, V, W exactly decompose the target tensor.
        T_reconstructed[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        """
        # Using einsum for efficient computation
        # 'ur, vr, wr -> uvw'
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_rec, target)

    def schoolbook():
        """
        Standard schoolbook decomposition of rank n*m*p.
        """
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=int)
        V = np.zeros((V_dim, rank), dtype=int)
        W = np.zeros((W_dim, rank), dtype=int)
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

    # Initialize with schoolbook
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # --- Strassen-based Construction ---
    # We can use Strassen's algorithm on 2x2 blocks to reduce rank.
    # Dimensions: 2x5 and 5x6.
    # We can partition the middle dimension m=5 into (2, 2, 1) or similar?
    # Actually, Strassen works best when dimensions are powers of 2 or can be padded.
    # Here m=5. We can split j into {0,1}, {2,3}, {4}.
    # Split k into {0,1}, {2,3}, {4,5}.
    # Split i into {0,1}.
    
    # Let's try to apply Strassen on the 2x2 blocks of (i, j) and (j, k).
    # The tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # We can group terms.
    # Let's try a specific construction that exploits the 2x2 structure in i and pairs in j,k.
    
    # We will attempt to find a decomposition of rank < 60 using a heuristic search
    # initialized with Strassen-like structures or random perturbations.
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety

    # Strategy:
    # 1. Try a specific Strassen-based construction if we can derive one.
    # 2. If not, use local search from schoolbook or random starts to reduce rank.
    # Given the complexity of deriving a perfect Strassen for 2x5x6 manually in code without verification,
    # we will use a "Local Search" approach that tries to merge terms or find lower rank representations.
    
    # However, a simpler and more robust improvement is to use the known Strassen result for 2x2x2.
    # If we can map 2x5x6 to a problem solvable by Strassen + schoolbook remainder.
    
    # Let's try a direct search for rank 59, 58, etc. using "term merging" or "random perturbation".
    # A better approach for this specific task might be to check if a rank 54 (from inspiration 2) is valid.
    # Let's implement the rank 54 construction from inspiration 2 and verify it.
    
    def construct_strassen_rank_54():
        # Based on inspiration 2:
        # Split j into (0,1), (2,3), (4).
        # Split k into (0,1), (2,3), (4,5).
        # For each 2x2 block of j,k, use 7 Strassen products.
        # There are 2 j-blocks and 3 k-blocks? No, j has 5 elements.
        # j: 0,1 (block 1), 2,3 (block 2), 4 (remainder).
        # k: 0,1 (block 1), 2,3 (block 2), 4,5 (block 3).
        
        rank = 0
        U = []
        V = []
        W = []
        
        def add_term(u_vec, v_vec, w_vec):
            nonlocal rank
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            rank += 1
            
        # Helper to create sparse vectors
        def make_vec(dim, indices_vals):
            vec = [0] * dim
            for idx, val in indices_vals:
                vec[idx] += val
            return vec
        
        j_blocks = [(0, 1), (2, 3)]
        k_blocks = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                # Strassen 7 products for C[i,k] where i in {0,1}, k in {k0,k1}
                # Note: i is 0,1. j is j0,j1. k is k0,k1.
                
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                # U: A_0,j0, A_1,j1
                # V: B_j0,k0, B_j1,k1
                # W: C_0,k0, C_1,k1
                u = make_vec(U_dim, [(0*m+j0, 1), (1*m+j1, 1)])
                v = make_vec(V_dim, [(j0*p+k0, 1), (j1*p+k1, 1)])
                w = make_vec(W_dim, [(k0*n+0, 1), (k1*n+1, 1)])
                add_term(u, v, w)
                
                # P2 = (A_1,j0 + A_1,j1) B_j0,k0
                # U: A_1,j0, A_1,j1
                # V: B_j0,k0
                # W: C_1,k0, C_1,k1 (with sign -1 for k1? No, standard Strassen:
                # M2 = (A11 + A12) B11 -> C11 += M2, C12 -= M2? 
                # Let's use the standard Strassen formulas:
                # M1 = (A11+A22)(B11+B22)
                # M2 = (A21+A22)B11
                # M3 = A11(B21-B11)
                # M4 = A22(B12-B22)
                # M5 = (A11+A12)B22
                # M6 = (A12-A22)(B11+B12)
                # M7 = (A21-A11)(B21+B22)
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # My previous manual derivation in inspiration 2 might be slightly off or specific.
                # Let's stick to the standard Strassen mapping for 2x2.
                # A = [[A00, A01], [A10, A11]]
                # B = [[B00, B01], [B10, B11]]
                # C = [[C00, C01], [C10, C11]]
                
                # Mapping to our indices:
                # A00 = A[0, j0], A01 = A[0, j1]
                # A10 = A[1, j0], A11 = A[1, j1]
                # B00 = B[j0, k0], B01 = B[j0, k1]
                # B10 = B[j1, k0], B11 = B[j1, k1]
                # C00 = C[0, k0], C01 = C[0, k1]
                # C10 = C[1, k0], C11 = C[1, k1]
                
                # M1 = (A00+A11)(B00+B11)
                u1 = make_vec(U_dim, [(0*m+j0, 1), (1*m+j1, 1)])
                v1 = make_vec(V_dim, [(j0*p+k0, 1), (j1*p+k1, 1)])
                w1 = make_vec(W_dim, [(k0*n+0, 1), (k0*n+1, 1), (k1*n+0, 1), (k1*n+1, 1)]) # Wait, W needs to be linear.
                # Strassen adds M1 to C00, C11 and subtracts from C01, C10? No.
                # C00 += M1, C11 += M1.
                # So W vector for M1 should have 1 at C00 and C11?
                # But the term is rank 1: u * v * w.
                # If w has 1 at C00 and 1 at C11, then u*v*1 contributes to both.
                # Is that valid?
                # C00 gets contribution from M1. C11 gets contribution from M1.
                # Yes, if the same product M1 is added to both, we can use a single rank-1 term
                # where W has 1s at those positions.
                
                # Let's re-verify the Strassen decomposition structure.
                # The tensor is sum of 7 rank-1 terms.
                # Each term corresponds to one M_i.
                # The W vector for term i has non-zeros at the C positions that M_i contributes to.
                
                # M1 contributes to C00 (+1), C11 (+1).
                # M2 contributes to C10 (+1), C11 (-1).
                # M3 contributes to C00 (-1), C01 (+1).
                # M4 contributes to C00 (+1), C10 (+1).
                # M5 contributes to C00 (-1), C01 (+1).
                # M6 contributes to C11 (+1).
                # M7 contributes to C00 (+1).
                
                # Wait, let's check the standard formulas again.
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # So:
                # M1: C00(+1), C11(+1)
                # M2: C10(+1), C11(-1)
                # M3: C00(-1), C01(+1)
                # M4: C00(+1), C10(+1)
                # M5: C00(-1), C01(+1)
                # M6: C11(+1)
                # M7: C00(+1)
                
                # Let's build the 7 terms for this block.
                
                # M1
                u_m1 = make_vec(U_dim, [(0*m+j0, 1), (1*m+j1, 1)])
                v_m1 = make_vec(V_dim, [(j0*p+k0, 1), (j1*p+k1, 1)])
                w_m1 = make_vec(W_dim, [(k0*n+0, 1), (k1*n+1, 1)])
                add_term(u_m1, v_m1, w_m1)
                
                # M2
                u_m2 = make_vec(U_dim, [(1*m+j0, 1), (1*m+j1, 1)])
                v_m2 = make_vec(V_dim, [(j0*p+k0, 1)])
                w_m2 = make_vec(W_dim, [(k0*n+1, 1), (k1*n+1, -1)])
                add_term(u_m2, v_m2, w_m2)
                
                # M3
                u_m3 = make_vec(U_dim, [(0*m+j0, 1)])
                v_m3 = make_vec(V_dim, [(j1*p+k0, 1), (j0*p+k0, -1)])
                w_m3 = make_vec(W_dim, [(k0*n+0, -1), (k1*n+0, 1)])
                add_term(u_m3, v_m3, w_m3)
                
                # M4
                u_m4 = make_vec(U_dim, [(1*m+j1, 1)])
                v_m4 = make_vec(V_dim, [(j1*p+k1, 1), (j0*p+k1, -1)])
                w_m4 = make_vec(W_dim, [(k0*n+0, 1), (k0*n+1, 1)])
                add_term(u_m4, v_m4, w_m4)
                
                # M5
                u_m5 = make_vec(U_dim, [(0*m+j0, 1), (0*m+j1, 1)])
                v_m5 = make_vec(V_dim, [(j1*p+k1, 1)])
                w_m5 = make_vec(W_dim, [(k0*n+0, -1), (k1*n+0, 1)])
                add_term(u_m5, v_m5, w_m5)
                
                # M6
                u_m6 = make_vec(U_dim, [(0*m+j1, 1), (1*m+j1, -1)])
                v_m6 = make_vec(V_dim, [(j0*p+k0, 1), (j0*p+k1, 1)])
                w_m6 = make_vec(W_dim, [(k1*n+1, 1)])
                add_term(u_m6, v_m6, w_m6)
                
                # M7
                u_m7 = make_vec(U_dim, [(1*m+j0, 1), (0*m+j0, -1)])
                v_m7 = make_vec(V_dim, [(j1*p+k0, 1), (j1*p+k1, 1)])
                w_m7 = make_vec(W_dim, [(k0*n+0, 1)])
                add_term(u_m7, v_m7, w_m7)

        # Handle remainder j=4
        # A[0,4] and A[1,4] are separate.
        # For each k in 0..5:
        # C[0,k] += A[0,4] B[4,k]
        # C[1,k] += A[1,4] B[4,k]
        # This is 2 * 6 = 12 terms.
        
        for k in range(p):
            # Term for C[0,k]
            u_r0 = make_vec(U_dim, [(0*m+4, 1)])
            v_r0 = make_vec(V_dim, [(4*p+k, 1)])
            w_r0 = make_vec(W_dim, [(k*n+0, 1)])
            add_term(u_r0, v_r0, w_r0)
            
            # Term for C[1,k]
            u_r1 = make_vec(U_dim, [(1*m+4, 1)])
            v_r1 = make_vec(V_dim, [(4*p+k, 1)])
            w_r1 = make_vec(W_dim, [(k*n+1, 1)])
            add_term(u_r1, v_r1, w_r1)
            
        # Total rank: 2 j-blocks * 3 k-blocks * 7 + 12 = 42 + 12 = 54.
        U_np = np.array(U).T # Transpose to (dim, rank)
        V_np = np.array(V).T
        W_np = np.array(W).T
        
        return U_np, V_np, W_np

    U_str, V_str, W_str = construct_strassen_rank_54()
    if verify(U_str, V_str, W_str):
        best_U, best_V, best_W = U_str, V_str, W_str
        best_rank = U_str.shape[1]
    
    # Try to improve further using local search
    # We can try to reduce rank by 1, 2, etc.
    # This is a hard problem. We will attempt a simple random restart search for lower ranks.
    
    import random
    random.seed(42)
    
    # Try ranks from best_rank - 1 down to 50
    # Given time constraints, we will only try a few.
    
    for target_rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Simple heuristic: try to find a rank-1 term that can be removed or merged.
        # This is complex. Instead, we rely on the Strassen result which is known to be valid for 2x2.
        # Since 54 is a significant improvement over 60, and finding 53 or lower is research-level,
        # we will stick with 54 unless a very quick random search finds something better.
        
        # Quick random search for target_rank
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            U_try = np.zeros((U_dim, target_rank), dtype=int)
            V_try = np.zeros((V_dim, target_rank), dtype=int)
            W_try = np.zeros((W_dim, target_rank), dtype=int)
            
            # Initialize with random sparse integers
            for r in range(target_rank):
                # Pick a few non-zeros
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, U_dim - 1)
                    U_try[idx, r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, V_dim - 1)
                    V_try[idx, r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, W_dim - 1)
                    W_try[idx, r] = random.choice([-1, 1])
                    
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                found = True
                break
        
        if found:
            # If we found a lower rank, we could continue to search for even lower,
            # but given the difficulty, we might just break if we drop below 54.
            # However, the loop continues to lower ranks anyway.
            pass

    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
