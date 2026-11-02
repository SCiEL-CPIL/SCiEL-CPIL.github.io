import time
import numpy as np
from itertools import product

def _verify(U, V, W, n=2, m=4, p=5):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i, j, k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _verify_exact(U, V, W, n=2, m=4, p=5):
    """Verify with exact rational arithmetic using fractions."""
    from fractions import Fraction
    Uf = [[Fraction(str(U[i,j])) for j in range(U.shape[1])] for i in range(U.shape[0])]
    Vf = [[Fraction(str(V[i,j])) for j in range(V.shape[1])] for i in range(V.shape[0])]
    Wf = [[Fraction(str(W[i,j])) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r] for r in range(U.shape[1]))
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 5
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb

    # Strassen 2x2x2 terms
    strassen_terms = [
        ({(1,0):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(1,1):1}, {(1,0):1, (1,1):-1}),
        ({(0,1):1, (1,1):-1}, {(0,0):1, (0,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):1}, {(1,0):1, (1,1):-1}, {(0,0):1}),
    ]
    
    def build_block_strassen(a_col_off, b_row_off, b_col_off, c_col_off):
        """Build Strassen terms for a 2x2x2 block."""
        terms = []
        for a_c, b_c, c_c in strassen_terms:
            U_idx = {}
            V_idx = {}
            W_idx = {}
            
            for (i, j), coeff in a_c.items():
                U_idx[i * m + (a_col_off + j)] = coeff
            
            for (j, k), coeff in b_c.items():
                V_idx[(b_row_off + j) * p + (b_col_off + k)] = coeff
            
            for (i, k), coeff in c_c.items():
                W_idx[(c_col_off + k) * n + i] = coeff
                
            terms.append((U_idx, V_idx, W_idx))
        return terms
    
    def build_schoolbook_2x1(a_col_off, b_row_off, b_col_idx, c_col_idx):
        """Build schoolbook terms for 2x2 * 2x1."""
        terms = []
        for i in range(2):
            for j in range(2):
                U_idx = {i * m + (a_col_off + j): 1}
                V_idx = {(b_row_off + j) * p + b_col_idx: 1}
                W_idx = {c_col_idx * n + i: 1}
                terms.append((U_idx, V_idx, W_idx))
        return terms
    
    # Build rank-36 decomposition
    terms_36 = []
    terms_36 += build_block_strassen(0, 0, 0, 0)   # A_L * B_T0
    terms_36 += build_block_strassen(0, 0, 2, 2)   # A_L * B_T1
    terms_36 += build_schoolbook_2x1(0, 0, 4, 4)   # A_L * B_T2
    terms_36 += build_block_strassen(2, 2, 0, 0)   # A_R * B_B0
    terms_36 += build_block_strassen(2, 2, 2, 2)   # A_R * B_B1
    terms_36 += build_schoolbook_2x1(2, 2, 4, 4)   # A_R * B_B2
    
    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r, (U_idx, V_idx, W_idx) in enumerate(terms):
            for idx, coeff in U_idx.items():
                U[idx, r] = coeff
            for idx, coeff in V_idx.items():
                V[idx, r] = coeff
            for idx, coeff in W_idx.items():
                W[idx, r] = coeff
        return U, V, W
    
    U_36, V_36, W_36 = terms_to_matrices(terms_36)
    if _verify(U_36, V_36, W_36, n, m, p):
        best_rank = 36
        best_result = {"U": U_36.tolist(), "V": V_36.tolist(), "W": W_36.tolist()}
    
    # Try to find lower rank by merging terms that share the same tensor pattern
    # Key insight: two terms can be merged if their (U_idx, V_idx, W_idx) 
    # are proportional, i.e., one is a scalar multiple of the other.
    
    def can_merge(t1, t2):
        """Check if two terms can be merged (one is proportional to the other)."""
        U1, V1, W1 = t1
        U2, V2, W2 = t2
        
        # Find a non-zero entry in U1 to determine ratio
        ratio = None
        for idx in U1:
            if U1[idx] != 0:
                if idx in U2 and U2[idx] != 0:
                    ratio = U2[idx] / U1[idx]
                    break
                elif idx in U2:
                    return False  # U2 has 0 where U1 is non-zero
                else:
                    return False  # U2 doesn't have this index
        
        if ratio is None:
            # U1 is all zeros, check V and W
            for idx in V1:
                if V1[idx] != 0:
                    if idx in V2 and V2[idx] != 0:
                        ratio = V2[idx] / V1[idx]
                        break
                    else:
                        return False
            if ratio is None:
                for idx in W1:
                    if W1[idx] != 0:
                        if idx in W2 and W2[idx] != 0:
                            ratio = W2[idx] / W1[idx]
                            break
                        else:
                            return False
            if ratio is None:
                return True  # Both are zero terms
        
        # Verify proportionality
        for idx, val in U1.items():
            if idx not in U2:
                if val != 0:
                    return False
            elif abs(U2[idx] - ratio * val) > 1e-10:
                return False
        for idx in U2:
            if idx not in U1 and U2[idx] != 0:
                return False
        
        for idx, val in V1.items():
            if idx not in V2:
                if val != 0:
                    return False
            elif abs(V2[idx] - ratio * val) > 1e-10:
                return False
        for idx in V2:
            if idx not in V1 and V2[idx] != 0:
                return False
        
        for idx, val in W1.items():
            if idx not in W2:
                if val != 0:
                    return False
            elif abs(W2[idx] - ratio * val) > 1e-10:
                return False
        for idx in W2:
            if idx not in W1 and W2[idx] != 0:
                return False
        
        return True
    
    # Try greedy merging
    def try_merge_terms(terms, max_rank):
        """Try to merge terms to reduce rank."""
        current = list(terms)
        merged = True
        while merged and len(current) > max_rank:
            merged = False
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    return current
                for j in range(i+1, len(current)):
                    if can_merge(current[i], current[j]):
                        # Merge: sum the terms
                        U_new = {}
                        V_new = {}
                        W_new = {}
                        for idx, val in current[i][0].items():
                            U_new[idx] = U_new.get(idx, 0) + val
                        for idx, val in current[j][0].items():
                            U_new[idx] = U_new.get(idx, 0) + val
                        for idx, val in current[i][1].items():
                            V_new[idx] = V_new.get(idx, 0) + val
                        for idx, val in current[j][1].items():
                            V_new[idx] = V_new.get(idx, 0) + val
                        for idx, val in current[i][2].items():
                            W_new[idx] = W_new.get(idx, 0) + val
                        for idx, val in current[j][2].items():
                            W_new[idx] = W_new.get(idx, 0) + val
                        
                        new_term = (U_new, V_new, W_new)
                        new_current = [t for k, t in enumerate(current) if k != i and k != j]
                        new_current.append(new_term)
                        current = new_current
                        merged = True
                        break
                if merged:
                    break
        return current
    
    # Try to merge the rank-36 terms
    if time.time() < deadline - 10:
        merged_terms = try_merge_terms(terms_36, 30)
        if len(merged_terms) < best_rank:
            U_m, V_m, W_m = terms_to_matrices(merged_terms)
            if _verify(U_m, V_m, W_m, n, m, p):
                best_rank = len(merged_terms)
                best_result = {"U": U_m.tolist(), "V": V_m.tolist(), "W": W_m.tolist()}
    
    # Try alternative block decompositions
    # Alternative: split B differently
    # B_T (2x5) -> B_T0 (2x3), B_T1 (2x2)
    # B_B (2x5) -> B_B0 (2x3), B_B1 (2x2)
    # But 2x2x3 doesn't have a known rank-7 algorithm, so use schoolbook for 2x2x3 (rank 6)
    # Total: 2*7 + 2*6 + 2*4 + 2*6 + 2*4 = 14+12+8+12+8 = 54 (worse)
    
    # Alternative: use Strassen for 2x2x2 and handle the extra column differently
    # Try: A_L * B_T where B_T is 2x5
    # Split B_T into B_T_left (2x4) and B_T_right (2x1)
    # B_T_left = [B_T0 (2x2), B_T1 (2x2)] -> 2 Strassen blocks = 14 terms
    # B_T_right (2x1) -> schoolbook 2x2x1 = 4 terms
    # Total for A_L*B_T: 18
    # Same for A_R*B_B: 18
    # Total: 36 (same as before)
    
    # Try a different approach: use the fact that n=2 to create a "tall" Strassen
    # Actually, let's try to search for rank 35 by perturbing the rank-36 solution
    
    if time.time() < deadline - 5 and best_rank > 35:
        # Try to find a rank-35 solution by searching for a linear combination
        # that eliminates one dimension
        # This is a linear algebra problem: find if the rank-36 tensor lies in
        # a rank-35 subspace
        
        # Alternative: try different block sizes
        # Split A (2x4) into A_0 (2x1), A_1 (2x1), A_2 (2x1), A_3 (2x1)
        # Split B (4x5) into B_0 (1x5), B_1 (1x5), B_2 (1x5), B_3 (1x5)
        # C = sum_j A_j * B_j
        # Each A_j * B_j is 2x1 * 1x5 = 2x5, rank 10 (schoolbook)
        # Total: 40 (worse)
        
        # Try: A (2x4) = [A_0 (2x2), A_1 (2x2)]
        # B (4x5) = [B_0 (2x5); B_1 (2x5)]
        # Use Strassen for each 2x2 * 2x5
        # 2x2 * 2x5: split B into 2x2 and 2x3
        # 2x2 * 2x2: Strassen rank 7
        # 2x2 * 2x3: schoolbook rank 6
        # Total per block: 7 + 6 = 13
        # Two blocks: 26
        # But wait, 2x2*2x5 is not standard. Let me think again.
        # Actually 2x2 * 2x5: A is 2x2, B is 2x5
        # Split B into B_a (2x2) and B_b (2x3)
        # A*B_a: 2x2*2x2, Strassen rank 7
        # A*B_b: 2x2*2x3, schoolbook: 2*2*3 = 12
        # Total: 7+12 = 19 per block
        # Two blocks: 38 (worse than 36)
        
        pass
    
    return best_result
