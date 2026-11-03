import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_np(U_arr, V_arr, W_arr):
        """Fast vectorized verification using numpy."""
        # For each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] == 1
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_rows = U_arr[i_idx*m + j_idx, :]
        v_rows = V_arr[j_idx*p + k_idx, :]
        w_rows = W_arr[k_idx*n + i_idx, :]
        
        sums = np.einsum('ij,ij->i', u_rows * v_rows, w_rows)
        return np.all(np.abs(sums - 1.0) < 1e-9)

    def verify(U, V, W, rank):
        """Verify decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        return verify_np(U_arr, V_arr, W_arr)

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U_arr, V_arr, W_arr, rank):
        """Try to eliminate redundant terms using numpy for speed."""
        U_c = U_arr.copy()
        V_c = V_arr.copy()
        W_c = W_arr.copy()
        cur_rank = rank
        
        for _ in range(30):
            if time_left() < 3:
                break
            progress = False
            # Sort by sparsity (most sparse first)
            sparsity = np.sum(U_c[:, :cur_rank] == 0, axis=0) + \
                       np.sum(V_c[:, :cur_rank] == 0, axis=0) + \
                       np.sum(W_c[:, :cur_rank] == 0, axis=0)
            order = np.argsort(-sparsity)
            
            for term_idx in order:
                if time_left() < 3:
                    break
                if term_idx >= cur_rank:
                    continue
                # Save and zero out
                save_U = U_c[:, term_idx].copy()
                save_V = V_c[:, term_idx].copy()
                save_W = W_c[:, term_idx].copy()
                U_c[:, term_idx] = 0
                V_c[:, term_idx] = 0
                W_c[:, term_idx] = 0
                
                if verify_np(U_c, V_c, W_c):
                    # Remove the column
                    mask = np.ones(cur_rank, dtype=bool)
                    mask[term_idx] = False
                    U_c = U_c[:, mask]
                    V_c = V_c[:, mask]
                    W_c = W_c[:, mask]
                    cur_rank -= 1
                    progress = True
                    # Adjust order
                    order = order[order != term_idx]
                    order = np.where(order > term_idx, order - 1, order)
                else:
                    # Restore
                    U_c[:, term_idx] = save_U
                    V_c[:, term_idx] = save_V
                    W_c[:, term_idx] = save_W
            
            if not progress:
                break

        return U_c.tolist(), V_c.tolist(), W_c.tolist(), cur_rank

    def try_pairwise_elimination(U_arr, V_arr, W_arr, rank):
        """Try to eliminate two terms at once using numpy."""
        if time_left() < 8:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank
        
        U_c = U_arr.copy()
        V_c = V_arr.copy()
        W_c = W_arr.copy()
        cur_rank = rank
        max_pairs = min(2000, cur_rank * (cur_rank - 1) // 2)
        pairs_checked = 0
        
        for _ in range(5):
            if time_left() < 5 or pairs_checked >= max_pairs:
                break
            progress = False
            for idx1 in range(cur_rank):
                if time_left() < 5 or pairs_checked >= max_pairs:
                    break
                for idx2 in range(idx1 + 1, cur_rank):
                    if time_left() < 5 or pairs_checked >= max_pairs:
                        break
                    pairs_checked += 1
                    
                    # Save and zero out both
                    save_U1 = U_c[:, idx1].copy()
                    save_U2 = U_c[:, idx2].copy()
                    save_V1 = V_c[:, idx1].copy()
                    save_V2 = V_c[:, idx2].copy()
                    save_W1 = W_c[:, idx1].copy()
                    save_W2 = W_c[:, idx2].copy()
                    U_c[:, idx1] = 0
                    U_c[:, idx2] = 0
                    V_c[:, idx1] = 0
                    V_c[:, idx2] = 0
                    W_c[:, idx1] = 0
                    W_c[:, idx2] = 0
                    
                    if verify_np(U_c, V_c, W_c):
                        # Remove both columns
                        mask = np.ones(cur_rank, dtype=bool)
                        mask[idx1] = False
                        mask[idx2] = False
                        U_c = U_c[:, mask]
                        V_c = V_c[:, mask]
                        W_c = W_c[:, mask]
                        cur_rank -= 2
                        progress = True
                        # Restart
                        break
                    else:
                        # Restore
                        U_c[:, idx1] = save_U1
                        U_c[:, idx2] = save_U2
                        V_c[:, idx1] = save_V1
                        V_c[:, idx2] = save_V2
                        W_c[:, idx1] = save_W1
                        W_c[:, idx2] = save_W2
                if progress:
                    break
            
            if not progress:
                break

        return U_c.tolist(), V_c.tolist(), W_c.tolist(), cur_rank

    def build_split_middle():
        """
        Attempt the 'split-middle' decomposition.
        The idea: C = A B.
        Let A = [A_0, A_1] where A_0 is 2x5, A_1 is 2x5? No.
        Let's partition the inner dimension m=5 into m1=2, m2=3.
        A = [A_L, A_R] where A_L is 2x2, A_R is 2x3.
        B = [B_L; B_R] where B_L is 2x6, B_R is 3x6.
        C = A_L B_L + A_R B_R.
        
        Rank(A_L B_L) for 2x2 * 2x6.
        We can use Strassen for the 2x2 part of B_L?
        B_L is 2x6. Let B_L = [B_L1, B_L2] where both are 2x3?
        Or use the fact that 2x2 * 2x6 can be done in 7 * 3 = 21? No.
        For 2x2 * 2x6, we can view it as 3 independent 2x2 * 2x2 multiplications.
        Rank = 7 * 3 = 21.
        
        Rank(A_R B_R) for 2x3 * 3x6.
        This is 2x3 * 3x6.
        We can decompose 2x3 * 3x6.
        Standard rank is 2*3*6 = 36.
        Can we do better?
        2x3 * 3x6.
        Let's try to combine.
        
        Total rank = 21 + 36 = 57.
        This is worse than 54 from Strassen blocks.
        
        What if we partition m=5 into 1 and 4?
        A_L (2x1) * B_L (1x6) -> Rank 2*1*6 = 12? No, rank 2?
        A_L is 2x1 vector, B_L is 1x6 vector.
        C_L = A_L B_L is rank 1 matrix.
        Decomposition:
        For i in 0..1:
          For k in 0..5:
            Term: A_L[i,0] * B_L[0,k].
            This is 2*6 = 12 terms.
            But we can factor:
            U[i, t] = A_L[i,0]
            V[0, t] = B_L[0,k] ... wait.
            The tensor is T[i, j, k] = sum_j A[i,j] B[j,k].
            For j=0 (the 1x1 block):
            T[i, 0, k] = A[i,0] B[0,k].
            This is a rank 1 tensor in (i, k) for fixed j? No, it's a slice.
            The full tensor for j=0 is A[:,0] B[0,:].
            This is an outer product of a 2-vector and a 6-vector.
            Rank 1 in the tensor sense?
            T[i, 0, k] = u_i v_k.
            This can be represented by 1 term?
            U[i, 0, t] = u_i, V[0, 0, t] = v_k? No.
            V index is j*p+k. Here j=0. So V[0*p+k, t].
            If we use 1 term:
            U[i, 0, t] = A[i,0]
            V[0, 0, t] = 1? No, V depends on k.
            V[j*p+k, t].
            If V[0*p+k, t] = B[0,k], then V is not constant across k for a single term unless B[0,k] is constant.
            So we need 6 terms for the j=0 slice?
            Actually, the term is U[i, 0, t] * V[0, k, t] * W[k, i, t].
            If we set t=0..5:
            U[i, 0, t] = A[i, 0]
            V[0, t, t] = B[0, t] (diagonal in V for j=0)
            W[t, i, t] = 1
            Then sum_t A[i,0] B[0,t] 1 = A[i,0] B[0,t] for specific k=t.
            So yes, 6 terms for the 1x1 block.
            
            A_R (2x4) * B_R (4x6).
            Rank for 2x4 * 4x6.
            Standard 2*4*6 = 48.
            Total = 6 + 48 = 54.
            Same as before.
        """
        pass

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(3,4)], [(0,2),(1,3),(4,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,2),(3,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_np(U, V, W):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 15:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(
                np.array(U_list, dtype=float), np.array(V_list, dtype=float), 
                np.array(W_list, dtype=float), new_rank)
            
        # Try more elimination passes after pairwise
        if time_left() > 8 and new_rank < best_rank:
            U_list, V_list, W_list, new_rank2 = try_eliminate_terms(
                np.array(U_list, dtype=float), np.array(V_list, dtype=float), 
                np.array(W_list, dtype=float), new_rank)
            if new_rank2 < new_rank:
                new_rank = new_rank2
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
