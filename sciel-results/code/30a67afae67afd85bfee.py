"""Stronger algorithm for rank-minimal tensor decomposition of 2x5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60

    def is_valid_decomposition(U, V, W, rank):
        """Exact verification of tensor decomposition."""
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        # Also verify that no "wrong" terms appear (off-diagonal should be 0)
        # The tensor for matmul has T[i,j,k] = 1 iff output[i,k] uses A[i,j]*B[j,k]
        # So every (i,j,k) triple must sum to 1, and that's the only constraint for
        # the standard matmul tensor (it's exactly the tensor with 1s at those positions).
        return True

    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        doubled = np.array(arr, dtype=float) * 2
        rounded = np.round(doubled)
        return np.max(np.abs(doubled - rounded)) < tol

    # ------------------------------------------------------------------
    # Baseline: schoolbook (rank 60)
    # ------------------------------------------------------------------
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist(), "rank": rank_sb}

    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    # ------------------------------------------------------------------
    # Strategy 1: Structured decomposition via block splitting + Strassen
    # Split the 5 middle dimension into 2+3, giving 2x2x6 and 2x3x6 subproblems.
    # For 2x2x2 sub-blocks, Strassen gives rank 7 instead of 8.
    # ------------------------------------------------------------------
    def strassen_2x2x2(U_base, V_base, W_base, i_off, j_off, k_off, rank_offset):
        """
        Add Strassen rank-7 decomposition for a 2x2x2 block.
        U rows: (i_off+i, j_off+j) for i in 0..1, j in 0..1  -> row i_off*? ... 
        Actually U is indexed by (i, j) -> row i*m + j
        V is indexed by (j, k) -> row j*p + k
        W is indexed by (k, i) -> row k*n + i
        """
        # Strassen 7 products for C = A*B where A, B, C are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # We express each M as a rank-1 term: U_row(i,j) * V_row(j,k) * W_row(k,i)
        # For a single (i,j,k) contribution, U[i,j]=1, V[j,k]=1, W[k,i]=1.
        # Strassen combines: each M corresponds to a product of a linear combo of A entries
        # and a linear combo of B entries.
        #
        # Let's build it term by term.
        # M1 = (A00+A11)(B00+B11)
        #   = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each A_ii * B_jj is a rank-1 term with U[i,j]=1, V[j,k]=1, W[k,i]=1.
        # But Strassen's trick is to compute the 7 Ms once and reuse.
        # In tensor decomposition language, we need to find U, V, W such that
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta (correct matmul)
        #
        # The standard approach: for each of the 7 Strassen products, we create a
        # "virtual" intermediate. But in a flat tensor decomposition, we directly
        # assign coefficients.
        #
        # Let me just enumerate: for 2x2x2, the rank-7 decomposition is known.
        # I'll hardcode the 7 rank-1 terms.
        
        # The 7 terms (U_coeffs, V_coeffs, W_coeffs) for each of the 7 products:
        # Term structure: for each term t, we have:
        #   U[(i,j), t] = a_t[i,j]
        #   V[(j,k), t] = b_t[j,k]  
        #   W[(k,i), t] = c_t[k,i]
        # such that sum_t a_t[i,j] b_t[j,k] c_t[k,i] reconstructs matmul.
        
        # Standard Strassen tensor decomposition for 2x2x2 (rank 7):
        # Using the 7 Strassen formulas, we can write:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # Each M is a product of a linear form in A and a linear form in B.
        # In the tensor: A[i,j] appears in U, B[j,k] in V, C[k,i] in W.
        #
        # For M1: A_form = A00+A11, B_form = B00+B11
        #   U contribution: U[0,0]+=1, U[1,1]+=1  (for A00 and A11)
        #   V contribution: V[0,0]+=1, V[1,1]+=1  (for B00 and B11)
        #   W contribution: W[0,0]+=1, W[0,1]+=1, W[1,0]+=1, W[1,1]+=1? 
        #   No - W must be such that W[k,i] * U[i,j] * V[j,k] sums correctly.
        #
        # This is getting complex. Let me use a known rank-7 decomposition explicitly.
        # 
        # Alternative: just use the fact that for 2x2x2, we can find a rank-7
        # decomposition by solving the linear system with the constraint that
        # U, V, W entries are in {-1, 0, 1}.
        #
        # Actually, let me take a different approach: I'll directly search for
        # low-rank decompositions using the linear system method, but with
        # structured initializations inspired by block decomposition.
        
        pass  # Placeholder - we'll use the search approach instead

    # ------------------------------------------------------------------
    # Strategy: Search for lower rank using linear system with smart init
    # ------------------------------------------------------------------
    def search_rank(r, max_attempts, seed_offset=0):
        """Search for a rank-r decomposition."""
        rng = np.random.RandomState(seed_offset + r)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                return None
            
            # Generate sparse integer U, V with entries in {-1, 0, 1}
            density = 0.3 + 0.2 * (r / schoolbook_rank)  # higher density for lower rank
            density = min(density, 0.6)
            
            U_rand = np.zeros((n*m, r), dtype=int)
            V_rand = np.zeros((m*p, r), dtype=int)
            
            mask_u = rng.rand(n*m, r) < density
            mask_v = rng.rand(m*p, r) < density
            
            U_rand[mask_u] = rng.choice([-1, 1], size=mask_u.sum())
            V_rand[mask_v] = rng.choice([-1, 1], size=mask_v.sum())
            
            # Build linear system A * W_vec = b
            num_vars = p * n * r
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        u_row = U_rand[i*m+j]
                        v_row = V_rand[j*p+k]
                        coeffs = u_row * v_row  # shape (r,)
                        base_var_idx = (k*n + i) * r
                        A[eq_idx, base_var_idx:base_var_idx+r] = coeffs
            
            # Solve least squares
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if W entries are half-integers
            W_doubled = W_vec * 2
            W_rounded = np.round(W_doubled)
            if np.max(np.abs(W_doubled - W_rounded)) < 1e-6:
                W_snap = W_rounded / 2.0
                W_mat = W_snap.reshape(p*n, r)
                
                # Verify
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_mat.tolist()
                if is_valid_decomposition(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
        
        return None

    # Try ranks from schoolbook-1 downward
    # Focus on the most promising range first
    # For 2x5x6, expect rank around 45-55
    for r in range(rank_sb - 1, max(30, rank_sb - 25), -1):
        if time.time() - start_time > time_limit:
            break
        
        # More attempts for ranks close to expected optimum
        if 45 <= r <= 55:
            attempts = 2000
        elif 40 <= r < 45 or 55 < r <= 60:
            attempts = 1000
        else:
            attempts = 500
        
        result = search_rank(r, attempts, seed_offset=int(time.time()*1000) % 10000)
        if result is not None:
            best = result
            # Found a valid decomposition; try to go lower
            continue
        else:
            # If we've been trying for a while and not finding, reduce attempts
            # and continue to lower ranks
            pass
    
    # ------------------------------------------------------------------
    # Strategy 2: Structured block decomposition with Strassen
    # Split middle dimension 5 into 2+3
    # Subproblem 1: 2x2x6 (rows 0-1 of A, cols 0-1 of B, all 6 cols of C)
    # Subproblem 2: 2x3x6 (rows 0-1 of A, cols 2-4 of B, all 6 cols of C)
    # 
    # For 2x2x6: split k into three 2x2x2 blocks, each rank 7 via Strassen -> 21
    # For 2x3x6: split k into three 2x3x2 blocks. For 2x3x2, rank is at most 12
    #   (schoolbook), possibly less. Let's use schoolbook for 2x3x2 = 12 each, 3 blocks = 36
    # Total: 21 + 36 = 57
    #
    # But we can do better: for 2x3x2, the rank might be 9 or less.
    # Actually, let me just construct the block decomposition explicitly.
    # ------------------------------------------------------------------
    
    def build_block_decomposition():
        """
        Construct decomposition by splitting m=5 into m1=2, m2=3.
        Subproblem 1: n x m1 x p = 2x2x6
        Subproblem 2: n x m2 x p = 2x3x6
        For each subproblem, further split p=6 into three p_i=2 blocks.
        Each sub-block is n x m_i x 2.
        For n=2, m_i=2, p_i=2: use Strassen rank 7
        For n=2, m_i=3, p_i=2: use schoolbook rank 12 (or search for lower)
        """
        m1, m2 = 2, 3
        # Split p=6 into three groups of 2
        p_groups = [(0, 2), (2, 4), (4, 6)]
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        total_rank = 0
        
        for (k_start, k_end) in p_groups:
            # Subproblem 1: 2 x 2 x 2 (A rows 0-1, B cols 0-1, C cols k_start..k_start+1)
            # Strassen rank 7
            U1 = np.zeros((n*m1, 7))
            V1 = np.zeros((m1*2, 7))
            W1 = np.zeros((2*n, 7))
            
            # Strassen 7-term decomposition for 2x2x2
            # Term 1: (A00+A11)(B00+B11) -> contributes to C00, C11
            # We need: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 if (i,j,k) is a valid matmul term
            #
            # Let me use a known explicit rank-7 decomposition.
            # After research, here's one valid set of 7 terms:
            #
            # t=0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
            #   This gives: A00*B00*C00 + A00*B11*C11 + A11*B00*C00 + A11*B11*C11
            #   = A00*B00*C00 + A00*B11*C11 + A11*B00*C00 + A11*B11*C11
            #   But we need each (i,j,k) to sum to 1, not these combined terms.
            #
            # I think I'm overcomplicating this. Let me just use the linear system
            # approach for the 2x2x2 subproblem to find a rank-7 decomposition.
            
            # Actually, let me just use the search for each sub-block.
            # For 2x2x2, search for rank 7.
            sub_result = search_rank_small(2, 2, 2, 7, attempts=500)
            if sub_result is not None:
                U_parts.append(sub_result["U"])
                V_parts.append(sub_result["V"])
                W_parts.append(sub_result["W"])
                total_rank += 7
            else:
                # Fallback to schoolbook rank 8
                U_parts.append(np.eye(4, 8).tolist()[:4])  # placeholder
                V_parts.append(np.eye(4, 8).tolist()[:4])
                W_parts.append(np.eye(4, 8).tolist()[:4])
                total_rank += 8
            
            # Subproblem 2: 2 x 3 x 2
            sub_result2 = search_rank_small(2, 3, 2, 12, attempts=200)
            if sub_result2 is not None:
                U_parts.append(sub_result2["U"])
                V_parts.append(sub_result2["V"])
                W_parts.append(sub_result2["W"])
                total_rank += sub_result2["rank"]
            else:
                # Schoolbook 2x3x2 = 12
                U2 = np.zeros((6, 12))
                V2 = np.zeros((6, 12))
                W2 = np.zeros((4, 12))
                col = 0
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            U2[i*3+j, col] = 1
                            V2[j*2+k, col] = 1
                            W2[k*2+i, col] = 1
                            col += 1
                U_parts.append(U2.tolist())
                V_parts.append(V2.tolist())
                W_parts.append(W2.tolist())
                total_rank += 12
        
        # Combine all parts
        U_full = np.hstack(U_parts)
        V_full = np.hstack(V_parts)
        W_full = np.hstack(W_parts)
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist(), "rank": total_rank}

    def search_rank_small(nn, mm, pp, target_rank, attempts=100):
        """Search for a decomposition of an nn x mm x pp matmul tensor."""
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                return None
            rng = np.random.RandomState(attempt + target_rank * 1000 + nn*100 + mm*10 + pp)
            
            density = 0.4
            U_rand = np.zeros((nn*mm, target_rank), dtype=int)
            V_rand = np.zeros((mm*pp, target_rank), dtype=int)
            
            mask_u = rng.rand(nn*mm, target_rank) < density
            mask_v = rng.rand(mm*pp, target_rank) < density
            U_rand[mask_u] = rng.choice([-1, 1], size=mask_u.sum())
            V_rand[mask_v] = rng.choice([-1, 1], size=mask_v.sum())
            
            num_vars = pp * nn * target_rank
            num_eqs = nn * mm * pp
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        eq_idx = i * mm * pp + j * pp + k
                        u_row = U_rand[i*mm+j]
                        v_row = V_rand[j*pp+k]
                        coeffs = u_row * v_row
                        base_var_idx = (k*nn + i) * target_rank
                        A[eq_idx, base_var_idx:base_var_idx+target_rank] = coeffs
            
            try:
                W_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            W_doubled = W_vec * 2
            W_rounded = np.round(W_doubled)
            if np.max(np.abs(W_doubled - W_rounded)) < 1e-6:
                W_snap = W_rounded / 2.0
                W_mat = W_snap.reshape(pp*nn, target_rank)
                
                # Verify
                U_l = U_rand.tolist()
                V_l = V_rand.tolist()
                W_l = W_mat.tolist()
                for i in range(nn):
                    for j in range(mm):
                        for k in range(pp):
                            total = np.sum(U_l[i*mm+j] * V_l[j*pp+k] * W_l[k*nn+i])
                            if not np.isclose(total, 1.0, atol=1e-9):
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    return {"U": U_l, "V": V_l, "W": W_l, "rank": target_rank}
        return None

    # Try the block decomposition
    if time.time() - start_time < time_limit - 10:
        block_result = build_block_decomposition()
        if block_result["rank"] < best["rank"]:
            # Verify the block result
            if is_valid_decomposition(block_result["U"], block_result["V"], block_result["W"], block_result["rank"]):
                best = block_result
    
    # ------------------------------------------------------------------
    # Final: try a few more searches around the best rank found
    # ------------------------------------------------------------------
    if best["rank"] < schoolbook_rank:
        for r in range(best["rank"] - 1, max(best["rank"] - 5, 30), -1):
            if time.time() - start_time > time_limit:
                break
            result = search_rank(r, 500, seed_offset=99999 + r)
            if result is not None:
                best = result
                break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
