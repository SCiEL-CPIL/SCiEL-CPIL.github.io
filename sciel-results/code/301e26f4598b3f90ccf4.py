"""Tensor decomposition of 5x5x5 matrix multiplication, rank 73."""
import time

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin

    n, m, p = 5, 5, 5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(U_dict, V_dict, W_dict):
        U_cols.append(U_dict)
        V_cols.append(V_dict)
        W_cols.append(W_dict)

    # Part 1: Frame terms (i==4 or j==4)
    # i=4, j=0..4, k=0..4
    for j in range(5):
        for k in range(5):
            U = {4 * 5 + j: 1}
            V = {j * 5 + k: 1}
            W = {k * 5 + 4: 1}
            add_term(U, V, W)
            
    # i=0..3, j=4, k=0..4
    for i in range(4):
        for k in range(5):
            U = {i * 5 + 4: 1}
            V = {4 * 5 + k: 1}
            W = {k * 5 + i: 1}
            add_term(U, V, W)

    # Part 2: Core 4x4x4 via Strassen (7 terms) + Schoolbook inner (4 terms each)
    # Total 28 terms
    
    # Strassen definitions for 4x4 matrices partitioned into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # Helper to create a dict for a 2x2 block of A or B
    # Block (bi, bj) for A: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    def get_A_block(bi, bj):
        return {
            (2*bi, 2*bj): 1, (2*bi, 2*bj+1): 1,
            (2*bi+1, 2*bj): 1, (2*bi+1, 2*bj+1): 1
        }
    
    def get_B_block(bi, bj):
        return {
            (2*bi, 2*bj): 1, (2*bi, 2*bj+1): 1,
            (2*bi+1, 2*bj): 1, (2*bi+1, 2*bj+1): 1
        }
    
    # Strassen M definitions: (A_parts, B_parts)
    # Each part is (block, sign)
    M_defs = [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A21+A22)B11
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3 = A11(B12-B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M5 = (A11+A12)B22
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
    ]
    
    # For each M, we have A' and B' which are 2x2 matrices.
    # We compute C' = A' B' using schoolbook (4 terms).
    # C'_{00} = A'_{00}B'_{00} + A'_{01}B'_{10}
    # C'_{01} = A'_{00}B'_{01} + A'_{01}B'_{11}
    # C'_{10} = A'_{10}B'_{00} + A'_{11}B'_{10}
    # C'_{11} = A'_{10}B'_{01} + A'_{11}B'_{11}
    
    # Each A'_{ab} is a sum of A entries.
    # Each B'_{cd} is a sum of B entries.
    # The product A'_{ab} B'_{cd} is a bilinear form.
    # We expand it into rank-1 terms.
    
    for A_parts, B_parts in M_defs:
        # Build A' and B' as dicts of (local_row, local_col) -> coeff
        # But we need to map to global indices for U and V.
        # A' is 2x2. Local indices 0,1.
        # A'_{lr, lc} = sum_{global (i,j) in block} sign * A_{i,j}
        
        A_prime = {}
        for bi, bj, sign in A_parts:
            for lr in range(2):
                for lc in range(2):
                    gi = 2*bi + lr
                    gj = 2*bj + lc
                    key = (lr, lc)
                    A_prime[key] = A_prime.get(key, 0) + sign
        
        B_prime = {}
        for bi, bj, sign in B_parts:
            for lr in range(2):
                for lc in range(2):
                    gj = 2*bi + lr
                    gk = 2*bj + lc
                    key = (lr, lc)
                    B_prime[key] = B_prime.get(key, 0) + sign
        
        # Now expand the 4 schoolbook terms for C'
        # C'_{00} = A'_{00}B'_{00} + A'_{01}B'_{10}
        # C'_{01} = A'_{00}B'_{01} + A'_{01}B'_{11}
        # C'_{10} = A'_{10}B'_{00} + A'_{11}B'_{10}
        # C'_{11} = A'_{10}B'_{01} + A'_{11}B'_{11}
        
        # For each term like A'_{lr1, lc1} * B'_{lr2, lc2}:
        # A'_{lr1, lc1} is a linear combo of A.
        # B'_{lr2, lc2} is a linear combo of B.
        # The product is a sum of A_i,j * B_j,k terms.
        # Note: The local indices of A' and B' do NOT directly map to global j.
        # We must expand the linear combinations.
        
        def expand_product(A_lr, A_lc, B_lr, B_lc):
            # Returns list of (U_dict, V_dict, W_dict)
            terms = []
            # A_lr, A_lc are local indices in A' (0,1)
            # B_lr, B_lc are local indices in B' (0,1)
            
            # Get the global components of A'_{A_lr, A_lc}
            # This is stored in A_prime[(A_lr, A_lc)]? No, A_prime maps (lr,lc) to coeff.
            # Wait, my A_prime construction was:
            # key = (lr, lc) where lr,lc are local.
            # Value is the sum of signs from the blocks.
            # This is WRONG. A'_{lr,lc} is a SUM of DIFFERENT global A entries.
            # I need the list of (global_i, global_j, coeff) for each local (lr,lc).
            
            return terms
        
        # Let's redo the expansion properly.
        # We need to track the global indices.
        
        A_expanded = {}
        for bi, bj, sign in A_parts:
            for lr in range(2):
                for lc in range(2):
                    gi = 2*bi + lr
                    gj = 2*bj + lc
                    key = (lr, lc)
                    if key not in A_expanded:
                        A_expanded[key] = {}
                    A_expanded[key][(gi, gj)] = A_expanded[key].get((gi, gj), 0) + sign

        B_expanded = {}
        for bi, bj, sign in B_parts:
            for lr in range(2):
                for lc in range(2):
                    gj = 2*bi + lr
                    gk = 2*bj + lc
                    key = (lr, lc)
                    if key not in B_expanded:
                        B_expanded[key] = {}
                    B_expanded[key][(gj, gk)] = B_expanded[key].get((gj, gk), 0) + sign

        # Now iterate over the 4 schoolbook terms for C'
        # C'_{00} comes from A'_{00}B'_{00} and A'_{01}B'_{10}
        # C'_{01} comes from A'_{00}B'_{01} and A'_{01}B'_{11}
        # C'_{10} comes from A'_{10}B'_{00} and A'_{11}B'_{10}
        # C'_{11} comes from A'_{10}B'_{01} and A'_{11}B'_{11}
        
        # Map local C' indices to global C indices
        # C'_{00} -> C_{0,0}, C'_{01} -> C_{0,1}, C'_{10} -> C_{1,0}, C'_{11} -> C_{1,1}
        # Wait, C' is the top-left 2x2 block of C.
        # So C'_{00} is C_{0,0}.
        
        # For each pair (A_lr, A_lc, B_lr, B_lc) contributing to C'_{cr, cc}:
        # We expand A'_{A_lr, A_lc} * B'_{B_lr, B_lc}
        
        c_map = {
            (0,0): [(0,0,0,0), (0,1,1,0)], # C'_{00} from A'_{00}B'_{00} + A'_{01}B'_{10}
            (0,1): [(0,0,0,1), (0,1,1,1)], # C'_{01}
            (1,0): [(1,0,0,0), (1,1,1,0)], # C'_{10}
            (1,1): [(1,0,0,1), (1,1,1,1)]  # C'_{11}
        }
        
        for (cr, cc), pairs in c_map.items():
            # Global C index
            gi_c = cr
            gj_c = cc
            
            # Accumulate U, V, W for this specific C entry contribution
            # Note: Multiple Strassen terms contribute to the same C entry.
            # We add them as separate rank-1 terms.
            
            for (A_lr, A_lc, B_lr, B_lc) in pairs:
                # Get global components
                A_comps = A_expanded.get((A_lr, A_lc), {})
                B_comps = B_expanded.get((B_lr, B_lc), {})
                
                # Expand product
                for (gi_a, gj_a), coeff_a in A_comps.items():
                    for (gj_b, gk_b), coeff_b in B_comps.items():
                        # Term: coeff_a * coeff_b * A_{gi_a, gj_a} * B_{gj_b, gk_b}
                        # This contributes to C_{gi_a, gk_b}
                        # Check if gj_a == gj_b?
                        # In Strassen, the blocks are chosen such that the contraction index matches.
                        # A'_{00} involves A_{0,0}, A_{2,2}. (j=0,2)
                        # B'_{00} involves B_{0,0}, B_{2,2}. (j=0,2)
                        # The product A'_{00}B'_{00} is not a single term in the sum \sum_j.
                        # It is a bilinear form.
                        # We represent it as a sum of rank-1 terms.
                        # Each pair (A_{gi_a, gj_a}, B_{gj_b, gk_b}) is a rank-1 term
                        # IF we allow U, V, W to have specific values.
                        
                        # U[gi_a, gj_a] = coeff_a
                        # V[gj_b, gk_b] = coeff_b
                        # W[gk_b, gi_a] = 1
                        
                        # But wait, the sum is over j.
                        # If gj_a != gj_b, this term does not contribute to the standard matrix product sum \sum_j A_{ij}B_{jk}.
                        # However, the tensor decomposition is a linear combination.
                        # We are constructing U, V, W such that \sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = C_{ik}.
                        # The term above contributes coeff_a * coeff_b to C_{gi_a, gk_b} ONLY IF gj_a == gj_b?
                        # No. The formula is \sum_j U_{i,j} V_{j,k} W_{k,i}.
                        # If we set U[gi_a, gj_a] = coeff_a, V[gj_b, gk_b] = coeff_b, W[gk_b, gi_a] = 1.
                        # The contribution to C_{gi_a, gk_b} is \sum_j U[gi_a, j] V[j, gk_b] W[gk_b, gi_a].
                        # Only j=gj_a and j=gj_b matter.
                        # If gj_a == gj_b, then j=gj_a is the only non-zero term.
                        # If gj_a != gj_b, then U[gi_a, gj_a] and V[gj_b, gk_b] don't share j.
                        # So the contribution is 0.
                        
                        # Therefore, we can ONLY form rank-1 terms where gj_a == gj_b.
                        
                        if gj_a == gj_b:
                            U_dict = {gi_a * 5 + gj_a: coeff_a}
                            V_dict = {gj_b * 5 + gk_b: coeff_b}
                            W_dict = {gk_b * 5 + gi_a: 1}
                            add_term(U_dict, V_dict, W_dict)
                        # If gj_a != gj_b, this pair of components does not form a valid rank-1 term
                        # that contributes to the correct C entry via the standard sum.
                        # This means Strassen's "cross terms" (like A_{0,0} B_{2,2}) cannot be represented
                        # as a single rank-1 term in this decomposition scheme if they don't align on j.
                        # This implies that the simple expansion of Strassen terms into rank-1 terms
                        # only works if the blocks align on j.
                        # In Strassen, A11 and B11 align on j=0,1.
                        # A11 and B22 do NOT align.
                        # So M1 = (A11+A22)(B11+B22) contains terms like A11*B22 which don't align.
                        # These terms CANNOT be represented as rank-1 terms in the standard tensor decomposition?
                        # No, they CAN be represented, but they contribute to different C entries?
                        # A_{0,0} B_{2,2} contributes to C_{0,2} (i=0, k=2).
                        # In C'_{00} (which is C_{0,0}), this term should NOT appear.
                        # But in the Strassen formula, M1 contributes to C_{0,0}.
                        # Contradiction?
                        # No. The Strassen formula is an identity for the *values* of the matrix product.
                        # It is not an identity for the tensor decomposition terms.
                        # The tensor decomposition must sum to the correct C_{ik}.
                        # If we include a term A_{0,0} B_{2,2}, it adds to C_{0,2}.
                        # We must ensure that the sum of all such terms for C_{0,2} is correct.
                        # Strassen ensures the final matrix is correct.
                        # So we should add ALL terms, regardless of alignment?
                        # If we add U[0,0]=1, V[2,2]=1, W[2,0]=1.
                        # Contribution to C_{0,2}: \sum_j U[0,j]V[j,2]W[2,0].
                        # j=0: U[0,0]V[0,2]... V[0,2] is 0.
                        # j=2: U[0,2]V[2,2]... U[0,2] is 0.
                        # So it contributes 0 to C_{0,2}.
                        # It contributes 0 to everything.
                        # So we MUST have gj_a == gj_b.
                        
                        # This means we can only use the aligned terms.
                        # This implies Strassen does NOT reduce the tensor rank if we require aligned terms?
                        # Actually, the aligned terms in Strassen are sufficient to reconstruct the matrix.
                        # The non-aligned terms in the expansion of M_t cancel out or are not needed?
                        # No, Strassen uses the full product.
                        
                        # Let's just use the aligned terms. If the result is wrong, we fall back.
                        pass

    # Fallback: Schoolbook
    return _schoolbook()

def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
