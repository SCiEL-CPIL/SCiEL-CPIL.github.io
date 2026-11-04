import numpy as np
import time
from fractions import Fraction
import itertools
from scipy.optimize import linprog

def construct():
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n  # 6, 9, 6
    
    start = time.time()
    DEADLINE = 140  # safety margin
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def verify_exact(U, V, W):
        """Verify with exact rational arithmetic."""
        rank = U.shape[1]
        for mat in [U, V, W]:
            doubled = mat * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-6):
                return False
        Uf = np.array([[Fraction(int(round(U[i,r]*2)), 2) for r in range(rank)] for i in range(nm)], dtype=object)
        Vf = np.array([[Fraction(int(round(V[i,r]*2)), 2) for r in range(rank)] for i in range(mp)], dtype=object)
        Wf = np.array([[Fraction(int(round(W[i,r]*2)), 2) for r in range(rank)] for i in range(pn)], dtype=object)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Uf[a,r] * Vf[b,r] * Wf[c,r]
                    expected = int(T[a,b,c])
                    if total != expected:
                        return False
        return True
    
    def to_lists(mat):
        return mat.tolist()
    
    # ============================================================
    # STRATEGY: Systematic block-partitioned algorithm design
    # ============================================================
    
    # We'll try multiple structural approaches and return the best (lowest rank) verified solution.
    
    best_rank = 18  # schoolbook
    best_result = None
    
    # --- Approach 1: Direct Strassen hybrid (rank 17 baseline) ---
    def build_strassen_hybrid():
        """
        Split A (2x3) into A_L (2x2) + A_R (2x1)
        Split B (3x3) into B_T (2x3) + B_B (1x3)
        
        C = A_L * B_T + A_R * B_B
        
        For A_L * B_T (2x2 * 2x3):
          Split B_T into B_TL (2x2) + B_TR (2x1)
          A_L * B_TL: Strassen 2x2*2x2 -> rank 7
          A_L * B_TR: schoolbook 2x2*2x1 -> rank 4
          
        For A_R * B_B (2x1 * 1x3): schoolbook -> rank 1 (outer product)
        
        Total: 7 + 4 + 1 = 12? Wait, let me recount.
        Actually A_R * B_B where A_R is 2x1 and B_B is 1x3 gives a 2x3 result.
        Each entry C[i,k] += A_R[i,0] * B_B[0,k], that's 2*3=6 terms but they share
        the same structure. As a tensor rank-1 term: U has 2 nonzeros, V has 3 nonzeros,
        W has 6 nonzeros. That's 1 rank term? No, it's actually rank 2 (one per row of A_R).
        
        Let me be more careful. A_R is 2x1: entries at U indices 2,5.
        B_B is 1x3: entries at V indices 6,7,8.
        Result C is 2x3: all entries.
        
        Term 1: U[2]=1, V[6]=1, W[0]=1 (C[0,0])
        Term 2: U[2]=1, V[7]=1, W[1]=1 (C[0,1])  
        Term 3: U[2]=1, V[8]=1, W[2]=1 (C[0,2])
        Term 4: U[5]=1, V[6]=1, W[3]=1 (C[1,0])
        Term 5: U[5]=1, V[7]=1, W[4]=1 (C[1,1])
        Term 6: U[5]=1, V[8]=1, W[5]=1 (C[1,2])
        
        That's 6 rank-1 terms, not 1! Unless we can factor.
        Actually: (a0 * b0 + a1 * b1) style... no.
        
        Wait: C[i,k] = A_R[i,0] * B_B[0,k]
        This is an outer product of column vector A_R (dim 2) and row vector B_B (dim 3).
        As a tensor: sum over r of U[i,r] V[k,r] W[i,k,r]... 
        
        Hmm, the issue is that in our tensor formulation, each rank-1 term is
        U[a] * V[b] * W[c]. For the outer product A_R * B_B:
        T[a,b,c] = A_R[a] * B_B[b] * delta[c == f(a,b)]
        
        Actually no. Let me reconsider.
        
        The multiplication tensor for 2x1 * 1x3 -> 2x3:
        T[i*m+j, j*p+k, k*n+i] where the only nonzero is when j=0 (the single column/row).
        Wait, A_R is column 2 of A, so j=2. B_B is row 2 of B, so j=2.
        T[i*3+2, 2*3+k, k*2+i] = 1 for i in {0,1}, k in {0,1,2}.
        That's 6 entries: (2,6,0), (2,7,1), (2,8,2), (5,6,3), (5,7,4), (5,8,5).
        
        Can this be done in rank < 6? 
        Rank of a 2x1x3 tensor (with all entries 1 in the right positions)...
        Actually this is just the outer product structure. The minimal rank of the 
        2x3 matrix (all ones) viewed as a tensor T[i,k] = 1 is rank 1 if we could
        write it as u[i]*v[k]*w[i,k]... but w depends on both i and k.
        
        In our tensor formulation T[a,b,c], the entries are:
        (2,6,0)=1, (2,7,1)=1, (2,8,2)=1, (5,6,3)=1, (5,7,4)=1, (5,8,5)=1
        
        This is a 6x9x6 tensor with 6 nonzeros. The minimal rank... 
        Each rank-1 term U[a]V[b]W[c] can cover at most one of these 6 entries
        if we require exactness (since the supports would overlap incorrectly).
        Actually no - a rank-1 term with U[2]=1, V[6]=1, W[0]=1 covers (2,6,0).
        But could a single term cover multiple? Only if the pattern matches.
        
        For example, if U[2]=1, U[5]=1, V[6]=1, V[7]=1, W[0]=1, W[1]=1...
        Then T[2,6,0] += 1*1*1 = 1, T[2,7,1] += 1*1*1 = 1, T[5,6,3] += 1*1*W[5,3]...
        We'd need W[5,3] to also be 1, but then T[2,6,3] would get 1*1*1=1 (wrong, should be 0).
        
        So we need to be careful. The minimal rank for this sub-problem is actually 2:
        Term 1: U[2]=1, V[6]=1, V[7]=1, V[8]=1, W[0]=1, W[1]=1, W[2]=1
                -> covers (2,6,0),(2,7,1),(2,8,2)
        Term 2: U[5]=1, V[6]=1, V[7]=1, V[8]=1, W[3]=1, W[4]=1, W[5]=1
                -> covers (5,6,3),(5,7,4),(5,8,5)
        
        Yes! Rank 2 for the A_R * B_B part.
        
        So total: 7 (Strassen) + 4 (schoolbook 2x2*2x1) + 2 (outer product) = 13?
        
        Wait, but I need to double-check the 2x2*2x1 part.
        A_L is 2x2 (indices 0,1,3,4 in U), B_TR is 2x1 (indices 2,5 in V), result is 2x1 (indices 4,5 in W).
        T entries: (0,2,4), (1,5,4), (3,2,5), (4,5,5)
        Can this be rank 4? Or can we do better?
        
        Actually for 2x2 * 2x1 -> 2x1, the standard schoolbook is rank 4.
        But maybe rank 3? The tensor has 4 nonzeros: (0,2,4),(1,5,4),(3,2,5),(4,5,5).
        
        Hmm, this is getting complex. Let me just build the explicit rank-17 and verify,
        then try to find improvements through systematic search.
        """
        
        U = np.zeros((6, 17))
        V = np.zeros((9, 17))
        W = np.zeros((6, 17))
        
        # Part 1: Strassen 2x2*2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # U indices: 0,1,3,4 (A_L)
        # V indices: 0,1,3,4 (B_TL)  
        # W indices: 0,1,2,3 (C_TL)
        
        # m1: (a00+a11)(b00+b11)
        U[0,0]=1; U[4,0]=1
        V[0,0]=1; V[4,0]=1
        W[0,0]=1; W[3,0]=1
        
        # m2: (a10+a11)b00
        U[3,1]=1; U[4,1]=1
        V[0,1]=1
        W[1,1]=1; W[3,1]=-1
        
        # m3: a00(b01-b11)
        U[0,2]=1
        V[1,2]=1; V[4,2]=-1
        W[0,2]=1; W[2,2]=1
        
        # m4: a11(b10-b00)
        U[4,3]=1
        V[3,3]=1; V[0,3]=-1
        W[0,3]=1; W[1,3]=1
        
        # m5: (a00+a01)b11
        U[0,4]=1; U[1,4]=1
        V[4,4]=1
        W[0,4]=-1; W[2,4]=1
        
        # m6: (a10-a00)(b00+b01)
        U[3,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # m7: (a01-a11)(b10+b11)
        U[1,6]=1; U[4,6]=-1
        V[3,6]=1; V[4,6]=1
        W[0,6]=1; W[2,6]=1
        
        # Part 2: Schoolbook 2x2*2x1 for A[0:2,0:2] * B[0:2,2] -> C[0:2,2]
        # U indices: 0,1,3,4
        # V indices: 2,5
        # W indices: 4,5
        
        # a00*b02
        U[0,7]=1; V[2,7]=1; W[4,7]=1
        # a01*b12
        U[1,8]=1; V[5,8]=1; W[4,8]=1
        # a10*b02
        U[3,9]=1; V[2,9]=1; W[5,9]=1
        # a11*b12
        U[4,10]=1; V[5,10]=1; W[5,10]=1
        
        # Part 3: Outer product A[0:2,2] * B[2:3,0:3] -> C[0:2,0:3]
        # U indices: 2,5
        # V indices: 6,7,8
        # W indices: 0,1,2,3,4,5
        
        # Row 0 of A_R (index 2) times B_B
        U[2,11]=1
        V[6,11]=1; V[7,11]=1; V[8,11]=1
        W[0,11]=1; W[1,11]=1; W[2,11]=1
        
        # Row 1 of A_R (index 5) times B_B
        U[5,12]=1
        V[6,12]=1; V[7,12]=1; V[8,12]=1
        W[3,12]=1; W[4,12]=1; W[5,12]=1
        
        # That's only 13 terms used (0-12). We have 17 columns.
        # Wait, I miscounted. Let me recount: 7 + 4 + 2 = 13.
        # But the standard construction gives 17. Let me check if 13 is valid.
        
        # Actually, I think I made an error. Let me verify this 13-term construction.
        # The issue: in Part 3, I'm using the same V entries (6,7,8) for both terms.
        # That's fine since they're in different rank slots.
        
        return U[:, :13], V[:, :13], W[:, :13]
    
    U13, V13, W13 = build_strassen_hybrid()
    if verify(U13, V13, W13):
        print("Rank 13 verified (floating point)")
        if verify_exact(U13, V13, W13):
            print("Rank 13 verified (exact)")
            best_rank = 13
            best_result = (U13, V13, W13)
    else:
        print("Rank 13 failed verification")
    
    # --- Approach 2: Try rank 12 by merging Strassen terms ---
    # Can we reduce the Strassen 2x2*2x2 from rank 7 to rank 6?
    # No, Strassen's 7 is optimal for 2x2*2x2 over any field.
    
    # Can we reduce the 2x2*2x1 part from rank 4 to rank 3?
    # The tensor for 2x2*2x1 has 4 nonzeros. Let's check if rank 3 suffices.
    # Actually, let's try a different partition.
    
    # --- Approach 3: Alternative partition ---
    # Split A (2x3) into A_L (2x1) + A_M (2x1) + A_R (2x1)
    # Split B (3x3) into B_T (1x3) + B_M (1x3) + B_B (1x3)
    # C = A_L*B_T + A_M*B_M + A_R*B_B
    # Each term is 2x1 * 1x3 -> 2x3, rank 2 each.
    # Total: 2+2+2 = 6? That seems too good.
    
    # Wait, that's not right. The issue is that in matrix multiplication,
    # C[i,k] = sum_j A[i,j] * B[j,k]. If we split by j:
    # C = sum_j (column j of A) * (row j of B)
    # Each term is a 2x1 * 1x3 = 2x3 outer product, rank 2.
    # 3 terms * rank 2 = rank 6.
    
    # But wait - can we share rank terms across different j values?
    # The tensor entries for j=0: (0,0,0),(1,0,1),(0,3,2),(1,3,3) [using i*3+j, j*3+k, k*2+i]
    # Hmm, let me recompute.
    
    # For j=0: A indices 0,1 (i=0,1; j=0), B indices 0,1,2 (j=0; k=0,1,2)
    # C indices: k*2+i for i in {0,1}, k in {0,1,2}: 0,1,2,3,4,5
    # T entries: (0,0,0),(1,0,1),(0,1,2),(1,1,3),(0,2,4),(1,2,5)
    
    # For j=1: A indices 1,4 (i=0,1; j=1), B indices 3,4,5 (j=1; k=0,1,2)
    # T entries: (1,3,0),(4,3,1),(1,4,2),(4,4,3),(1,5,4),(4,5,5)
    
    # For j=2: A indices 2,5 (i=0,1; j=2), B indices 6,7,8 (j=2; k=0,1,2)
    # T entries: (2,6,0),(5,6,1),(2,7,2),(5,7,3),(2,8,4),(5,8,5)
    
    # Each j-group has 6 entries. The minimal rank for each 2x1x3 sub-tensor is 2.
    # If we can't share between groups, total rank = 6.
    # But can we share? The supports are disjoint in the U and V indices,
    # so no sharing is possible without creating spurious nonzero entries.
    
    # Therefore rank 6 is achievable! Let me construct it.
    
    def build_rank_6():
        """
        C = sum_j (col_j(A)) * (row_j(B))
        For each j, use 2 rank-1 terms (one per row of A).
        Total: 3 * 2 = 6 rank-1 terms.
        """
        U = np.zeros((6, 6))
        V = np.zeros((9, 6))
        W = np.zeros((6, 6))
        
        col = 0
        for j in range(m):  # j = 0, 1, 2
            for i in range(n):  # i = 0, 1
                # U index: i*m + j
                # V indices: j*p + k for k = 0, 1, 2
                # W indices: k*n + i for k = 0, 1, 2
                
                U[i*m + j, col] = 1
                for k in range(p):
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                col += 1
        
        return U, V, W
    
    U6, V6, W6 = build_rank_6()
    if verify(U6, V6, W6):
        print("Rank 6 verified (floating point)")
        if verify_exact(U6, V6, W6):
            print("Rank 6 verified (exact)")
            best_rank = 6
            best_result = (U6, V6, W6)
    else:
        print("Rank 6 failed verification")
        # Debug: show which entries are wrong
        T_approx = np.einsum('ar,br,cr->abc', U6, V6, W6)
        diff = np.abs(T_approx - T)
        bad = np.argwhere(diff > 1e-9)
        print(f"  {len(bad)} entries wrong")
        for idx in bad[:10]:
            print(f"  T[{tuple(idx)}] = {T[tuple(idx)]}, approx = {T_approx[tuple(idx)]}")
    
    if time.time() - start > DEADLINE:
        if best_result is not None:
            U, V, W = best_result
            return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
        # Fallback schoolbook
        U_sb = np.zeros((6, 18))
        V_sb = np.zeros((9, 18))
        W_sb = np.zeros((6, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": to_lists(U_sb), "V": to_lists(V_sb), "W": to_lists(W_sb)}
    
    # --- Approach 4: Try to beat rank 6 ---
    # Is rank 6 minimal? The tensor has 18 nonzeros.
    # Each rank-1 term U[a]V[b]W[c] can contribute to multiple entries.
    # But due to the structure, it's hard to share.
    
    # Let's try rank 5 with a systematic search over small coefficient sets.
    # We'll use the fact that any valid decomposition must satisfy:
    # For each (a,b,c) with T[a,b,c]=1: sum_r U[a,r]V[b,r]W[c,r] = 1
    # For each (a,b,c) with T[a,b,c]=0: sum_r U[a,r]V[b,r]W[c,r] = 0
    
    # This is a system of polynomial equations. Let's try a randomized search
    # with integer coefficients in {-2,-1,0,1,2}.
    
    def random_search(rank, n_trials, coeff_set):
        """Randomized search for a valid decomposition."""
        import random
        random.seed(42)
        
        for trial in range(n_trials):
            if time.time() - start > DEADLINE - 10:
                break
            
            U_r = np.zeros((nm, rank))
            V_r = np.zeros((mp, rank))
            W_r = np.zeros((pn, rank))
            
            for r in range(rank):
                # Generate sparse random vectors
                nnz_u = random.randint(1, 3)
                nnz_v = random.randint(1, 3)
                nnz_w = random.randint(1, 3)
                
                u_idx = random.sample(range(nm), nnz_u)
                v_idx = random.sample(range(mp), nnz_v)
                w_idx = random.sample(range(pn), nnz_w)
                
                for idx in u_idx:
                    U_r[idx, r] = random.choice(coeff_set)
                for idx in v_idx:
                    V_r[idx, r] = random.choice(coeff_set)
                for idx in w_idx:
                    W_r[idx, r] = random.choice(coeff_set)
            
            if verify(U_r, V_r, W_r):
                if verify_exact(U_r, V_r, W_r):
                    return U_r, V_r, W_r
        
        return None
    
    # Try rank 5
    if best_rank > 5:
        print("Trying rank 5 search...")
        result5 = random_search(5, 5000, [-2, -1, 1, 2])
        if result5 is not None:
            print("Found rank 5!")
            best_rank = 5
            best_result = result5
    
    # Try rank 4
    if best_rank > 4:
        print("Trying rank 4 search...")
        result4 = random_search(4, 3000, [-2, -1, 1, 2])
        if result4 is not None:
            print("Found rank 4!")
            best_rank = 4
            best_result = result4
    
    # --- Approach 5: Structured search for rank 5 ---
    # Based on the structure, try specific patterns.
    # The key insight: we need 18 nonzero tensor entries covered by rank terms.
    # With rank 5, each term must cover on average 18/5 = 3.6 entries.
    
    # Let's try a more structured approach: fix the U and V matrices,
    # then solve for W (which is linear in W).
    
    def structured_search(rank):
        """
        Fix U and V, solve for W via linear system.
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        For fixed U,V: for each c, W[c,:] is determined by:
        sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all (a,b)
        This is a linear system in W[c,0..rank-1].
        """
        import random
        random.seed(123)
        
        for trial in range(2000):
            if time.time() - start > DEADLINE - 5:
                break
            
            # Generate random sparse U, V
            U_r = np.zeros((nm, rank))
            V_r = np.zeros((mp, rank))
            
            for r in range(rank):
                # U: pick 2-3 nonzero entries
                nnz_u = random.randint(2, min(4, nm))
                u_idx = random.sample(range(nm), nnz_u)
                for idx in u_idx:
                    U_r[idx, r] = random.choice([-1, 1])
                
                # V: pick 2-3 nonzero entries
                nnz_v = random.randint(2, min(4, mp))
                v_idx = random.sample(range(mp), nnz_v)
                for idx in v_idx:
                    V_r[idx, r] = random.choice([-1, 1])
            
            # Build linear system for W
            # For each c in range(pn):
            #   For each (a,b) pair: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # This gives nm*mp equations in rank unknowns per c.
            
            W_r = np.zeros((pn, rank))
            valid = True
            
            for c in range(pn):
                # Build A_matrix: (nm*mp) x rank
                A_mat = np.zeros((nm * mp, rank))
                b_vec = np.zeros(nm * mp)
                
                eq_idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for r in range(rank):
                            A_mat[eq_idx, r] = U_r[a, r] * V_r[b, r]
                        b_vec[eq_idx] = T[a, b, c]
                        eq_idx += 1
                
                # Solve A_mat @ W[c,:] = b_vec
                try:
                    sol, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    # Check if solution is exact
                    if np.allclose(A_mat @ sol, b_vec, atol=1e-9):
                        W_r[c, :] = sol
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Check if W has integer/half-integer entries
                if np.allclose(W_r * 2, np.round(W_r * 2), atol=1e-6):
                    W_r = np.round(W_r * 2) / 2
                    if verify(U_r, V_r, W_r):
                        if verify_exact(U_r, V_r, W_r):
                            return U_r, V_r, W_r
        
        return None
    
    if best_rank > 5:
        print("Structured search for rank 5...")
        result5 = structured_search(5)
        if result5 is not None:
            print("Found rank 5 via structured search!")
            best_rank = 5
            best_result = result5
    
    if best_rank > 4:
        print("Structured search for rank 4...")
        result4 = structured_search(4)
        if result4 is not None:
            print("Found rank 4 via structured search!")
            best_rank = 4
            best_result = result4
    
    # --- Final: return best result ---
    if best_result is not None:
        U, V, W = best_result
        print(f"Returning rank {best_rank} solution")
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
    
    # Fallback: rank 18 schoolbook
    print("Falling back to rank 18")
    U_sb = np.zeros((6, 18))
    V_sb = np.zeros((9, 18))
    W_sb = np.zeros((6, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": to_lists(U_sb), "V": to_lists(V_sb), "W": to_lists(W_sb)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
