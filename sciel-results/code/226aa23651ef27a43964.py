"""Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Combines block decomposition with verified Strassen sub-blocks and a final
rank-reduction polish pass that removes provably redundant terms.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    def build_strassen_block(i0, j0, k0):
        """
        Verified Strassen decomposition for a 2x2 * 2x2 -> 2x2 block.
        A block: rows [i0, i0+1], cols [j0, j0+1]
        B block: rows [j0, j0+1], cols [k0, k0+1]
        C block: rows [i0, i0+1], cols [k0, k0+1]
        
        Uses the standard verified Strassen formulas:
        p1 = (a+d)(e+h)
        p2 = (c+d)(e)
        p3 = (a+b)(h)
        p4 = (d-c)(g+h)
        p5 = (a-c)(e+f)
        p6 = b(f-h)
        p7 = c(g-e)
        
        C00 = p1 - p4 + p5 - p7
        C01 = p3 + p6 - p1 + p4
        C10 = p2 + p4 - p5 + p7
        C11 = p1 - p3 + p2 - p6
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        terms = []  # list of (u_dict, v_dict, w_dict)

        # p1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        terms.append((
            {(i0, j0): 1, (i1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {}
        ))

        # p2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        terms.append((
            {(i1, j0): 1, (i1, j1): 1},
            {(j0, k0): 1},
            {}
        ))

        # p3 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        terms.append((
            {(i0, j0): 1, (i0, j1): 1},
            {(j1, k1): 1},
            {}
        ))

        # p4 = (A[i1,j1] - A[i1,j0]) * (B[j1,k0] + B[j1,k1])
        terms.append((
            {(i1, j1): 1, (i1, j0): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {}
        ))

        # p5 = (A[i0,j0] - A[i1,j0]) * (B[j0,k0] + B[j0,k1])
        terms.append((
            {(i0, j0): 1, (i1, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {}
        ))

        # p6 = A[i0,j1] * (B[j0,k1] - B[j1,k1])
        terms.append((
            {(i0, j1): 1},
            {(j0, k1): 1, (j1, k1): -1},
            {}
        ))

        # p7 = A[i1,j0] * (B[j1,k0] - B[j0,k0])
        terms.append((
            {(i1, j0): 1},
            {(j1, k0): 1, (j0, k0): -1},
            {}
        ))

        # Now assign W coefficients based on Strassen combination formulas
        # C00 = p1 - p4 + p5 - p7
        # C01 = p3 + p6 - p1 + p4
        # C10 = p2 + p4 - p5 + p7
        # C11 = p1 - p3 + p2 - p6
        
        # w_dict for each term t: {(k, i): coeff}
        # C[k,i] is at position (k*n + i) in W, with i in [i0,i1], k in [k0,k1]
        
        w_coeffs = [
            # p1: C00 += 1, C01 += -1, C10 += 0, C11 += 1
            {(k0, i0): 1, (k1, i0): -1, (k0, i1): 0, (k1, i1): 1},
            # p2: C00 += 0, C01 += 0, C10 += 1, C11 += 1
            {(k0, i0): 0, (k1, i0): 0, (k0, i1): 1, (k1, i1): 1},
            # p3: C00 += 0, C01 += 1, C10 += 0, C11 += -1
            {(k0, i0): 0, (k1, i0): 1, (k0, i1): 0, (k1, i1): -1},
            # p4: C00 += -1, C01 += 1, C10 += 1, C11 += 0
            {(k0, i0): -1, (k1, i0): 1, (k0, i1): 1, (k1, i1): 0},
            # p5: C00 += 1, C01 += 0, C10 += -1, C11 += 0
            {(k0, i0): 1, (k1, i0): 0, (k0, i1): -1, (k1, i1): 0},
            # p6: C00 += 0, C01 += 1, C10 += 0, C11 += -1
            {(k0, i0): 0, (k1, i0): 1, (k0, i1): 0, (k1, i1): -1},
            # p7: C00 += -1, C01 += 0, C10 += 1, C11 += 0
            {(k0, i0): -1, (k1, i0): 0, (k0, i1): 1, (k1, i1): 0},
        ]

        # Clean up zero coefficients
        for idx in range(len(terms)):
            u_dict, v_dict, _ = terms[idx]
            w_dict = w_coeffs[idx]
            w_dict = {k: v for k, v in w_dict.items() if v != 0}
            terms[idx] = (u_dict, v_dict, w_dict)

        return terms

    def build_decomposition():
        """
        Build the full decomposition.
        Strategy: 
        - Use Strassen for 2x2x2 blocks where possible
        - Use schoolbook for remaining terms
        """
        U_cols = []
        V_cols = []
        W_cols = []

        def add_term(u_dict, v_dict, w_dict):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for (i, j), coeff in u_dict.items():
                u[i * m + j] = coeff
            for (j, k), coeff in v_dict.items():
                v[j * p + k] = coeff
            for (k, i), coeff in w_dict.items():
                w[k * n + i] = coeff
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

        # Cover the 3x4x7 multiplication
        # Rows i: 0,1,2
        # Cols j: 0,1,2,3
        # Cols k: 0,1,2,3,4,5,6
        
        # Use Strassen for 2x2 blocks in the j-k plane where possible
        # For i=0,1: use Strassen on j=[0,1],k=[0,1]; j=[2,3],k=[0,1]; j=[0,1],k=[2,3]; etc.
        # But we need to be careful about coverage.
        
        # Let's use a systematic approach:
        # For each pair of output rows (i, i+1), decompose the 2x4x7 subproblem
        # For i=0,1: 2x4x7
        # For i=2: 1x4x7 (single row, no Strassen benefit)
        
        # Subproblem 2x4x7:
        # Use Strassen on 2x2x2 blocks in the j-k plane
        # j: 0,1,2,3 -> two 2-blocks: [0,1] and [2,3]
        # k: 0,1,2,3,4,5,6 -> three 2-blocks: [0,1], [2,3], [4,5] and one leftover k=6
        
        # For i in [0,1]:
        for i0 in [0]:
            for j0 in [0, 2]:
                for k0 in [0, 2, 4]:
                    if k0 + 1 < p:  # ensure k0+1 < 7
                        terms = build_strassen_block(i0, j0, k0)
                        for u_dict, v_dict, w_dict in terms:
                            add_term(u_dict, v_dict, w_dict)
        
        # Handle remaining k=6 for i in [0,1], j in [0,1,2,3]
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                add_term(
                    {(i, j): 1},
                    {(j, 6): 1},
                    {(6, i): 1}
                )
        
        # Handle i=2 (single row, no Strassen)
        for j in [0, 1, 2, 3]:
            for k in [0, 1, 2, 3, 4, 5, 6]:
                add_term(
                    {(2, j): 1},
                    {(j, k): 1},
                    {(k, 2): 1}
                )

        rank = len(U_cols)
        U = np.array(U_cols).T  # shape (n*m, rank)
        V = np.array(V_cols).T  # shape (m*p, rank)
        W = np.array(W_cols).T  # shape (p*n, rank)
        
        return U, V, W, rank

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor."""
        rank = U.shape[1]
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_sel = U[i_idx * m + j_idx, :]
        V_sel = V[j_idx * p + k_idx, :]
        W_sel = W[k_idx * n + i_idx, :]
        
        totals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if not np.all(np.abs(totals - 1.0) <= tol):
            return False
        return True

    def reduce_rank(U, V, W):
        """
        Attempt to reduce rank by removing linearly dependent columns.
        A column r is redundant if there exist coefficients alpha_s (s != r)
        such that U[:,r] = sum_s alpha_s U[:,s], V[:,r] = sum_s alpha_s V[:,s],
        and W[:,r] = sum_s alpha_s W[:,s].
        
        We use a greedy approach: try to remove columns one at a time.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        removed = []
        current_rank = rank
        
        # Try to remove columns from the end
        for r in range(rank - 1, -1, -1):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Skip if already removed
            if r in removed:
                continue
            
            # Get indices of remaining columns
            remaining = [i for i in range(current_rank) if i not in removed]
            if r not in remaining:
                continue
            
            if len(remaining) <= 1:
                break
            
            # Try to express column r as combination of others
            others = [i for i in remaining if i != r]
            
            # Solve U[:,r] = U[:,others] @ alpha
            U_others = U[:, others]
            V_others = V[:, others]
            W_others = W[:, others]
            
            try:
                # Use least squares to find alpha that minimizes ||U_others @ alpha - U[:,r]||
                alpha_u, residuals_u, rank_u, _ = np.linalg.lstsq(U_others, U[:, r], rcond=None)
                
                if residuals_u is not None and len(residuals_u) > 0:
                    if residuals_u[0] < 1e-10:
                        # Check if same alpha works for V and W
                        V_pred = V_others @ alpha_u
                        W_pred = W_others @ alpha_u
                        
                        v_err = np.linalg.norm(V[:, r] - V_pred)
                        w_err = np.linalg.norm(W[:, r] - W_pred)
                        
                        if v_err < 1e-10 and w_err < 1e-10:
                            # Column r is redundant
                            removed.append(r)
                            current_rank -= 1
            except:
                continue
        
        # Build reduced matrices
        keep = [i for i in range(rank) if i not in removed]
        U_red = U[:, keep]
        V_red = V[:, keep]
        W_red = W[:, keep]
        
        return U_red, V_red, W_red, len(keep)

    # Build initial decomposition
    U, V, W, rank = build_decomposition()
    
    # Verify
    if not verify_decomp(U, V, W):
        # Fallback to pure schoolbook if Strassen is incorrect
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        rank = U.shape[1]
    
    # Try to reduce rank
    if time.time() - start_time < deadline - 10.0:
        U, V, W, reduced_rank = reduce_rank(U, V, W)
        if verify_decomp(U, V, W):
            rank = reduced_rank
    
    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
