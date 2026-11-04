import time
import numpy as np
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 3, 5, 7
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # All possible Strassen blocks: choose 2 from each dimension
    i_pairs = [(0,1), (0,2), (1,2)]
    j_pairs = list(combinations(range(m), 2))
    k_pairs = list(combinations(range(p), 2))
    
    # For a fixed i_pair, find maximum set of disjoint j-pairs and k-pairs
    def max_disjoint_pairs(pairs):
        """Find maximum number of disjoint pairs from a list of pairs."""
        best = 0
        best_set = set()
        # Try all subsets (small enough for j and k)
        for size in range(len(pairs), 0, -1):
            found = False
            for subset in combinations(pairs, size):
                used = set()
                ok = True
                for pair in subset:
                    if pair[0] in used or pair[1] in used:
                        ok = False
                        break
                    used.add(pair[0])
                    used.add(pair[1])
                if ok:
                    best = size
                    best_set = set(subset)
                    found = True
                    break
            if found:
                break
        return best, best_set
    
    # For each i_pair, find best j and k packings
    best_total_blocks = 0
    best_config = None
    
    for ip in i_pairs:
        if time.time() > deadline - 40:
            break
        # For this i_pair, we need disjoint j-pairs and k-pairs
        # The j and k packings are independent
        j_count, j_set = max_disjoint_pairs(j_pairs)
        k_count, k_set = max_disjoint_pairs(k_pairs)
        total = j_count * k_count
        
        if total > best_total_blocks:
            best_total_blocks = total
            best_config = (ip, list(j_set), list(k_set))
    
    # Build the blocks
    ip, j_list, k_list = best_config
    blocks = []
    for jp in j_list:
        for kp in k_list:
            blocks.append((ip, jp, kp))
    
    # Determine which triples are covered by Strassen blocks
    strassen_covered = set()
    for (ip, jp, kp) in blocks:
        for i in ip:
            for j in jp:
                for k in kp:
                    strassen_covered.add((i, j, k))
    
    # Schoolbook triples: all triples not covered by Strassen
    schoolbook_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    schoolbook_triples.append((i, j, k))
    
    rank = len(blocks) * 7 + len(schoolbook_triples)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Add Strassen blocks
    for (ip, jp, kp) in blocks:
        i0, i1 = ip
        j0, j1 = jp
        k0, k1 = kp
        
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0, col] += 1
        U[i1 * m + j1, col] += 1
        V[j0 * p + k0, col] += 1
        V[j1 * p + k1, col] += 1
        W[k0 * n + i0, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
        # M2: (A[i1,j0]+A[i1,j1])*(B[j0,k0])
        U[i1 * m + j0, col] += 1
        U[i1 * m + j1, col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + i1, col] += 1
        W[k1 * n + i1, col] -= 1
        col += 1
        
        # M3: (A[i0,j0])*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0, col] += 1
        V[j0 * p + k1, col] += 1
        V[j1 * p + k1, col] -= 1
        W[k1 * n + i0, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
        # M4: (A[i1,j1])*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1, col] += 1
        V[j1 * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + i1, col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j1])*(B[j1,k1])
        U[i0 * m + j0, col] += 1
        U[i0 * m + j1, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i0, col] += 1
        W[k0 * n + i0, col] -= 1
        col += 1
        
        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + k1, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1, col] += 1
        U[i1 * m + j1, col] -= 1
        V[j1 * p + k0, col] += 1
        V[j1 * p + k1, col] += 1
        W[k0 * n + i0, col] += 1
        col += 1
    
    # Add schoolbook terms
    for (i, j, k) in schoolbook_triples:
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
    
    # Column elimination: try to remove columns that are linear combinations of others
    # This is the key rank reduction step
    if time.time() < deadline - 30:
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        curr_rank = rank
        improved = True
        max_elim_rounds = 10
        
        for round_num in range(max_elim_rounds):
            if time.time() > deadline - 25:
                break
            if not improved:
                break
            improved = False
            
            # Try to eliminate each column
            for c in range(curr_rank):
                if time.time() > deadline - 25:
                    break
                
                # Check if column c can be expressed as a linear combination of others
                # Build the tensor contribution of column c
                # T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # We want to find coefficients alpha_d such that:
                # U[:,c] = sum_d alpha_d * U[:,d]
                # V[:,c] = sum_d alpha_d * V[:,d]  (same alpha)
                # W[:,c] = sum_d alpha_d * W[:,d]  (same alpha)
                
                # This is a system: for each (a,b) where U[a,c]!=0 and V[b,c]!=0:
                # U[a,c] * V[b,c] * W[.,c] = sum_d alpha_d * U[a,d] * V[b,d] * W[:,d]
                
                # Simpler approach: check if the rank-1 tensor for column c is in the span
                # of the other rank-1 tensors. This is expensive, so use a heuristic.
                
                # Heuristic: if U[:,c] is in the column space of U[:, other_cols],
                # and V[:,c] is in the column space of V[:, other_cols],
                # and W[:,c] is in the column space of W[:, other_cols],
                # then try to find consistent coefficients.
                
                other_cols = [d for d in range(curr_rank) if d != c]
                if not other_cols:
                    continue
                
                U_others = U_curr[:, other_cols]
                V_others = V_curr[:, other_cols]
                W_others = W_curr[:, other_cols]
                
                # Check if U[:,c] is in span of U_others
                U_c = U_curr[:, c:c+1]
                # Use QR decomposition to check
                try:
                    Q, R = np.linalg.qr(U_others)
                    # Project U_c onto Q
                    proj = Q @ (Q.T @ U_c)
                    if not np.allclose(proj, U_c, atol=1e-9):
                        continue
                except:
                    continue
                
                # Check if V[:,c] is in span of V_others
                V_c = V_curr[:, c:c+1]
                try:
                    Q, R = np.linalg.qr(V_others)
                    proj = Q @ (Q.T @ V_c)
                    if not np.allclose(proj, V_c, atol=1e-9):
                        continue
                except:
                    continue
                
                # Check if W[:,c] is in span of W_others
                W_c = W_curr[:, c:c+1]
                try:
                    Q, R = np.linalg.qr(W_others)
                    proj = Q @ (Q.T @ W_c)
                    if not np.allclose(proj, W_c, atol=1e-9):
                        continue
                except:
                    continue
                
                # All three are in their respective spans. Now find coefficients.
                # We need alpha such that:
                # U_c = U_others @ alpha
                # V_c = V_others @ alpha
                # W_c = W_others @ alpha
                
                # Solve least squares for each and check consistency
                try:
                    alpha_u = np.linalg.lstsq(U_others, U_c, rcond=None)[0]
                    alpha_v = np.linalg.lstsq(V_others, V_c, rcond=None)[0]
                    alpha_w = np.linalg.lstsq(W_others, W_c, rcond=None)[0]
                except:
                    continue
                
                # Check if all three give the same coefficients
                if (np.allclose(alpha_u, alpha_v, atol=1e-8) and 
                    np.allclose(alpha_v, alpha_w, atol=1e-8)):
                    # Column c is redundant! Remove it.
                    # Remove column c from all three matrices
                    mask = [d for d in range(curr_rank) if d != c]
                    U_curr = U_curr[:, mask]
                    V_curr = V_curr[:, mask]
                    W_curr = W_curr[:, mask]
                    curr_rank -= 1
                    improved = True
                    break  # Restart the loop since indices changed
        
        if curr_rank < rank:
            U, V, W = U_curr, V_curr, W_curr
            rank = curr_rank
    
    # Final verification
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    try:
        computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(computed, T, atol=1e-9):
            # Fallback to full schoolbook
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col_fb] = 1
                        V_fb[j * p + k, col_fb] = 1
                        W_fb[k * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
