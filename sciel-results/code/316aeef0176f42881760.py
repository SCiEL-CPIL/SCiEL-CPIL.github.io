"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 10.0
    
    n, m, p = DIMENSIONS
    
    def make_rank_55():
        """
        Rank 55 decomposition for 2x3 x 3x10.
        Strategy:
        - j=0 (first column of A, first row of B): Naive 2x10 = 20 terms.
        - j=1,2 (last 2 cols of A, last 2 rows of B): Strassen-like 2x2 x 2x10.
          Split B into 5 blocks of 2 columns.
          Each 2x2 x 2x2 block uses 7 terms (Strassen).
          5 blocks * 7 = 35 terms.
        Total = 20 + 35 = 55.
        """
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        # A[i,0] * B[0,k] -> C[i,k]
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
            
        # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
        # A2 is 2x2, B2 is 2x10
        # Split B2 into 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # Strassen indices for U (A2 entries)
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            # Strassen indices for V (B2 block entries)
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # W indices for outputs
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1
            W[w_11][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11][col] += 1
            V[v_12][col] += 1
            V[v_22][col] += -1
            W[w_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22][col] += 1
            V[v_21][col] += 1
            V[v_11][col] += -1
            W[w_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11][col] += 1
            U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1
            W[w_01][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21][col] += 1
            U[u_11][col] += -1
            V[v_11][col] += 1
            V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12][col] += 1
            U[u_22][col] += -1
            V[v_21][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1
            
        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(decomp, tol=1e-9):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        rank = U.shape[1]
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def tensor_slice_lower_bound():
        """
        Analytically determine if the rank can drop below 55 by fixing the middle index j.
        
        The multiplication tensor T for 2x3 * 3x10 has entries T[i,j,k] = 1 for all i,j,k.
        We can slice the tensor by fixing j.
        
        Slice j=0: T[i,0,k] = 1 for i in {0,1}, k in {0..9}.
        This is a 2x10 matrix of 1s. The rank of this matrix is 1 (since all rows are identical).
        However, the tensor rank is not the same as the matrix rank of a slice.
        
        A better lower bound technique:
        Consider the support of the tensor. The tensor is supported on all (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0..9}.
        
        Let's look at the slices for each j.
        For a rank-r decomposition, the tensor can be written as sum_{c=1}^r u_c v_c w_c.
        
        Consider the sub-tensor for j=1 and j=2.
        Let A2 be the 2x2 matrix A[:, 1:3] and B2 be the 2x10 matrix B[1:3, :].
        The sub-tensor corresponding to j=1,2 is the multiplication tensor for 2x2 * 2x10.
        
        It is known that the rank of the 2x2xn multiplication tensor is at least 3n for n>=2?
        Actually, for 2x2x2, the rank is 7.
        For 2x2xn, the rank is known to be 3n+1 for n even? No.
        
        Let's use the border rank or a simpler slicing argument.
        
        Consider the 2x2x10 sub-problem.
        If we can prove that the 2x2x10 tensor requires rank at least 35, and the 2x1x10 part requires rank at least 20,
        and if these parts are "independent" in a way that prevents sharing, then the total rank is at least 55.
        
        However, terms can potentially be shared.
        
        Let's try a different slicing argument.
        Fix i=0. The slice T[0,j,k] is a 3x10 matrix of 1s. Rank 1.
        Fix i=1. The slice T[1,j,k] is a 3x10 matrix of 1s. Rank 1.
        
        This doesn't give a strong lower bound.
        
        Let's look at the structure of the rank-55 solution.
        It uses 20 terms for j=0 and 35 terms for j=1,2.
        The terms for j=0 only involve U indices (i,0), V indices (0,k), W indices (k,i).
        The terms for j=1,2 involve U indices (i,1), (i,2), V indices (1,k), (2,k), W indices (k,i).
        
        Notice that the U, V, W indices used in the j=0 part are disjoint from those used in the j=1,2 part?
        U: j=0 uses col 0,1. j=1,2 uses col 1,2. Overlap on col 1? No, U index is (i,j).
        U indices for j=0: (0,0), (1,0).
        U indices for j=1,2: (0,1), (0,2), (1,1), (1,2).
        These are disjoint sets of U indices.
        
        V indices for j=0: (0,0)..(0,9).
        V indices for j=1,2: (1,0)..(1,9), (2,0)..(2,9).
        Disjoint.
        
        W indices for j=0: (k,0), (k,1) for k=0..9.
        W indices for j=1,2: (k,0), (k,1) for k=0..9.
        These are the SAME W indices!
        
        So the W factors are shared. This means a term in the j=0 part could potentially be "merged" or "cancelled" with a term in the j=1,2 part if they use the same W factor.
        
        However, the U and V factors are disjoint.
        Let's say we have a term c in the j=0 part: u_c (in U[:, 0:2]) * v_c (in V[0:10, :]) * w_c (in W).
        And a term d in the j=1,2 part: u_d (in U[:, 2:6]) * v_d (in V[10:30, :]) * w_d (in W).
        
        If w_c = w_d, and we try to combine them, we would need u_c v_c + u_d v_d to produce the correct tensor entries.
        But u_c is non-zero only in the first two rows of U, and u_d is non-zero only in the last four rows.
        Similarly for v.
        
        The tensor entry T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        
        For j=0: T[i,0,k] = sum_{c in j=0 terms} U[i,0,c] V[0,k,c] W[k,i,c].
        For j=1: T[i,1,k] = sum_{c in j=1,2 terms} U[i,1,c] V[1,k,c] W[k,i,c].
        
        Since the U and V supports are disjoint between the j=0 group and the j=1,2 group, the sums are over disjoint sets of terms effectively (if we partition the terms by which j they support).
        
        Wait, a single term c has a specific U vector, V vector, W vector.
        If a term c has U non-zero at (0,0) and (0,1), it contributes to both j=0 and j=1 parts.
        
        In the rank-55 construction, the terms are strictly separated.
        Terms 0-19 only have non-zero U at (i,0) and V at (0,k).
        Terms 20-54 only have non-zero U at (i,1) or (i,2) and V at (1,k) or (2,k).
        
        Could we construct a lower rank solution where some terms have mixed j-support?
        
        Let's check the lower bound for the 2x2x10 tensor.
        The rank of the 2x2xn tensor is known to be 3n+1 for n even?
        For n=2, rank 7 = 3*2+1.
        For n=4, rank 13? Or 14?
        Actually, it is known that rank(2x2xn) >= 3n.
        If rank(2x2x10) >= 30, and rank(2x1x10) >= 20, total >= 50? This is weak.
        
        Let's look at the specific structure.
        The 2x1x10 tensor is just a diagonal tensor (if we view it as mapping A[:,0] and B[0,:] to C).
        It is equivalent to 10 copies of 2x1x1.
        Rank of 2x1x1 is 2. So 10*2 = 20.
        
        The 2x2x10 tensor.
        We know rank(2x2x2) = 7.
        Can we do better than 5*7=35 for 2x2x10?
        If we treat it as 5 blocks of 2x2x2, we get 35.
        Is it possible to do 34?
        
        According to literature, the rank of 2x2xn is 3n+1 for n even?
        If n=10, rank = 31?
        If so, total rank could be 20 + 31 = 51.
        
        However, we must ensure the W factors can be shared or aligned.
        In the 2x2xn decomposition, the W factors correspond to the output C.
        In the 2x1xn decomposition, the W factors also correspond to C.
        Since both output to the same C matrix, the W factors are in the same space.
        
        If the 2x2x10 tensor can be decomposed with rank 31, and the 2x1x10 with rank 20, can we combine them?
        The U and V spaces are disjoint (different j indices).
        So we can just take the direct sum of the decompositions.
        Total rank = 20 + 31 = 51.
        
        So the question is: What is the rank of the 2x2x10 multiplication tensor?
        
        Known results:
        rank(2x2xn) = 3n+1 for n even?
        Let's check n=2: 7. 3*2+1 = 7. Correct.
        n=4: 13?
        If rank(2x2x10) = 31, then we can achieve rank 51.
        
        Let's try to construct a rank 31 decomposition for 2x2x10.
        Or at least search for one.
        
        Since we have a time budget, we should try to find a decomposition for the 2x2x10 part with rank < 35.
        If we find rank r2 for 2x2x10, the total rank is 20 + r2.
        
        Let's implement a search for the 2x2x10 part.
        We can use the known Strassen decomposition for 2x2x2 as a building block.
        
        Actually, there is a known algorithm for 2x2xn with rank 3n+1.
        It involves a "cyclic" structure or overlapping blocks.
        
        Let's try to search for a rank 34 or 33 or 32 decomposition for 2x2x10.
        If we find one, we return 20 + r2.
        
        We will use a local search or random restarts on the 2x2x10 sub-problem.
        
        The 2x2x10 tensor T' has entries T'[i, j, k] = 1 for i in {0,1}, j in {0,1}, k in {0..9}.
        We need U' (2x2 x r), V' (2x10 x r), W' (10x2 x r) such that sum_c U'[i,j,c] V'[j,k,c] W'[k,i,c] = 1.
        
        Note: The indices for U' are (i, j') where j' is 0 or 1 (corresponding to original j=1,2).
        V' indices are (j', k).
        W' indices are (k, i).
        
        Let's search for r from 34 down to 30.
        """
        pass

    def search_2x2x10_rank(target_rank, time_limit):
        """
        Search for a rank-target_rank decomposition of the 2x2x10 tensor.
        Uses structured patterns based on overlapping Strassen blocks.
        Returns (U', V', W') or None.
        U' is 4 x target_rank, V' is 20 x target_rank, W' is 20 x target_rank.
        Indices: U'[i*2+j, c], V'[j*10+k, c], W'[k*2+i, c] for i,j in {0,1}, k in 0..9.
        """
        import random
        import numpy as np
        
        n2, m2, p2 = 2, 2, 10
        U_size = n2 * m2  # 4
        V_size = m2 * p2  # 20
        W_size = p2 * n2  # 20
        
        def verify_2x2x10(U, V, W, tol=1e-9):
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            r = Ua.shape[1]
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        total = 0.0
                        for c in range(r):
                            total += Ua[i * m2 + j, c] * Va[j * p2 + k, c] * Wa[k * n2 + i, c]
                        if abs(total - 1.0) > tol:
                            return False
            return True
        
        def strassen_block_2x2x2(k0):
            """Return 7 (U, V, W) contributions for block k0..k0+1."""
            terms = []
            # M1 = (a00+a11)(b00+b11)
            terms.append(([1,0,0,1], [1,0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0], [1,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0]))
            # Let me use a cleaner representation: list of (u_vec, v_vec, w_vec)
            # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # V indices: j*10+k, so row j=0: 0-9, row j=1: 10-19
            # W indices: k*2+i, so col i=0: even, col i=1: odd
            # For block k0, k1=k0+1:
            # V[0,k0]=k0, V[0,k1]=k0+1, V[1,k0]=10+k0, V[1,k1]=10+k0+1
            # W[k0,0]=2*k0, W[k0,1]=2*k0+1, W[k1,0]=2*k0+2, W[k1,1]=2*k0+3
            
            v00 = k0
            v01 = k0 + 1
            v10 = 10 + k0
            v11 = 10 + k0 + 1
            w00 = 2 * k0
            w01 = 2 * k0 + 1
            w10 = 2 * k0 + 2
            w11 = 2 * k0 + 3
            
            def make_vec(size, indices):
                vec = [0] * size
                for idx, val in indices:
                    vec[idx] = val
                return vec
            
            # M1 = (a00+a11)(b00+b11) -> C[0,0]+=1, C[1,1]+=1
            u1 = make_vec(4, [(0,1),(3,1)])
            v1 = make_vec(20, [(v00,1),(v11,1)])
            w1 = make_vec(20, [(w00,1),(w11,1)])
            terms.append((u1, v1, w1))
            
            # M2 = (a10+a11)b00 -> C[1,0]+=1, C[1,1]-=1
            u2 = make_vec(4, [(2,1),(3,1)])
            v2 = make_vec(20, [(v00,1)])
            w2 = make_vec(20, [(w10,1),(w11,-1)])
            terms.append((u2, v2, w2))
            
            # M3 = a00(b01-b11) -> C[0,1]+=1, C[1,1]+=1
            u3 = make_vec(4, [(0,1)])
            v3 = make_vec(20, [(v01,1),(v11,-1)])
            w3 = make_vec(20, [(w01,1),(w11,1)])
            terms.append((u3, v3, w3))
            
            # M4 = a11(b10-b00) -> C[0,0]+=1, C[1,0]+=1
            u4 = make_vec(4, [(3,1)])
            v4 = make_vec(20, [(v10,1),(v00,-1)])
            w4 = make_vec(20, [(w00,1),(w10,1)])
            terms.append((u4, v4, w4))
            
            # M5 = (a00+a01)b11 -> C[0,0]-=1, C[0,1]+=1
            u5 = make_vec(4, [(0,1),(1,1)])
            v5 = make_vec(20, [(v11,1)])
            w5 = make_vec(20, [(w00,-1),(w01,1)])
            terms.append((u5, v5, w5))
            
            # M6 = (a10-a00)(b00+b01) -> C[1,1]+=1
            u6 = make_vec(4, [(2,1),(0,-1)])
            v6 = make_vec(20, [(v00,1),(v01,1)])
            w6 = make_vec(20, [(w11,1)])
            terms.append((u6, v6, w6))
            
            # M7 = (a01-a11)(b10+b11) -> C[0,0]+=1
            u7 = make_vec(4, [(1,1),(3,-1)])
            v7 = make_vec(20, [(v10,1),(v11,1)])
            w7 = make_vec(20, [(w00,1)])
            terms.append((u7, v7, w7))
            
            return terms
        
        # Try structured search: overlapping Strassen blocks
        # Pattern: use 3n+1 = 31 terms by having overlapping blocks
        # Blocks at positions: 0,2,4,6,8 (5 blocks of 2) but with sharing
        # Alternative: blocks at 0,2,4,6,8 with some terms shared between adjacent blocks
        
        # Let's try a different approach: use the known 2x2xn decomposition
        # For n=10, try rank 31, 32, 33, 34
        
        # Structured pattern: 3n+1 terms
        # The idea is to have a "chain" of Strassen-like operations
        # where consecutive 2x2 blocks share one term
        
        # For now, let's do a targeted random search with restarts
        # initialized near the Strassen solution and perturbed
        
        start_time = time.time()
        deadline = start_time + time_limit
        
        best_result = None
        
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            # Start from the 35-term Strassen solution and try to merge/remove terms
            U_list = []
            V_list = []
            W_list = []
            
            for b in range(5):
                k0 = 2 * b
                terms = strassen_block_2x2x2(k0)
                for u, v, w in terms:
                    U_list.append(u[:])
                    V_list.append(v[:])
                    W_list.append(w[:])
            
            # Now try to perturb and see if we can reduce rank
            # This is a simplified approach - in practice we'd need more sophisticated methods
            # For this iteration, we'll just verify the 35-term solution works
            # and note that finding rank < 35 requires more advanced techniques
            
            # Actually, let's try a different structured pattern
            # Use 3n+1 = 31 terms with a cyclic structure
            if target_rank == 31:
                # Try the known 3n+1 pattern
                # This requires a specific construction that I'll implement
                pass
            
            # For now, return None to indicate no improvement found
            # The rank-55 solution remains the best known
        
        return None

    # Try to find a lower rank decomposition
    # First, verify the rank-55 construction
    result_55 = make_rank_55()
    if result_55 is None:
        return make_schoolbook()
    
    if not verify(result_55):
        return make_schoolbook()
    
    # Now try to improve by searching for rank 54 or 53
    # Strategy: start from rank-55 and try to merge terms
    import random
    
    def try_reduce_rank(base_decomp, target_rank, time_limit):
        """Try to reduce the rank of a decomposition by merging terms."""
        U = [row[:] for row in base_decomp["U"]]
        V = [row[:] for row in base_decomp["V"]]
        W = [row[:] for row in base_decomp["W"]]
        
        start_time = time.time()
        deadline = start_time + time_limit
        
        while len(U) > target_rank:
            if time.time() > deadline - 20:
                break
            
            # Pick two random terms to try to merge
            i, j = random.sample(range(len(U)), 2)
            
            # Try to find a single term that replaces terms i and j
            # Use the fact that if U_i * V_i * W_i + U_j * V_j * W_j can be 
            # written as U_new * V_new * W_new, we can merge them
            # This is generally not possible, but we can try local search
            
            # Simple heuristic: try averaging
            u_avg = [(a + b) / 2 for a, b in zip(U[i], U[j])]
            v_avg = [(a + b) / 2 for a, b in zip(V[i], V[j])]
            w_avg = [(a + b) / 2 for a, b in zip(W[i], W[j])]
            
            # Remove i and j, add new term
            new_U = [t[:] for t in U if t is not U[i] and t is not U[j]]
            new_V = [t[:] for t in V if t is not V[i] and t is not V[j]]
            new_W = [t[:] for t in W if t is not W[i] and t is not W[j]]
            new_U.append(u_avg)
            new_V.append(v_avg)
            new_W.append(w_avg)
            
            if verify({"U": new_U, "V": new_V, "W": new_W}):
                U, V, W = new_U, new_V, new_W
            else:
                # Try a few random perturbations
                found = False
                for _ in range(10):
                    u_pert = [a + random.uniform(-0.1, 0.1) for a in u_avg]
                    v_pert = [a + random.uniform(-0.1, 0.1) for a in v_avg]
                    w_pert = [a + random.uniform(-0.1, 0.1) for a in w_avg]
                    test_U = [t[:] for t in U if t is not U[i] and t is not U[j]]
                    test_V = [t[:] for t in V if t is not V[i] and t is not V[j]]
                    test_W = [t[:] for t in W if t is not W[i] and t is not W[j]]
                    test_U.append(u_pert)
                    test_V.append(v_pert)
                    test_W.append(w_pert)
                    if verify({"U": test_U, "V": test_V, "W": test_W}):
                        U, V, W = test_U, test_V, test_W
                        found = True
                        break
                if not found:
                    # Try a different pair
                    continue
        
        if len(U) <= target_rank and verify({"U": U, "V": V, "W": W}):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try rank 54
    remaining_time = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining_time > 30:
        result_54 = try_reduce_rank(result_55, 54, remaining_time - 20)
        if result_54 is not None:
            return result_54
    
    return result_55

# EVOLVE-BLOCK-END
