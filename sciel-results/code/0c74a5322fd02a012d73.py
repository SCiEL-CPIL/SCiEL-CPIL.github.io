"""Strassen-based rank-44 decomposition for 2x3x8 matrix multiplication."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback (rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Check valid entries equal 1
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    acc = 0
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        if u_val:
                            v_val = v_row[c]
                            if v_val:
                                acc += u_val * v_val * w_row[c]
                    if acc != 1:
                        return False
        # Check invalid entries are 0
        # Valid (a_idx, b_idx, c_idx) triples are those where:
        # a_idx = i*m+a, b_idx = a'*p+k, c_idx = k'*n+i' 
        # and a == a', k == k', i == i' (i.e., the indices are consistent)
        for a_idx in range(n * m):
            for b_idx in range(m * p):
                for c_idx in range(p * n):
                    # Determine if this is a valid triple
                    i1 = a_idx // m
                    a1 = a_idx % m
                    a2 = b_idx // p
                    k1 = b_idx % p
                    k2 = c_idx // n
                    i2 = c_idx % n
                    if a1 == a2 and k1 == k2 and i1 == i2:
                        continue  # valid, already checked
                    # Invalid: must be 0
                    u_row = U[a_idx]
                    v_row = V[b_idx]
                    w_row = W[c_idx]
                    acc = 0
                    for c in range(rank):
                        u_val = u_row[c]
                        if u_val:
                            v_val = v_row[c]
                            if v_val:
                                acc += u_val * v_val * w_row[c]
                    if acc != 0:
                        return False
        return True

    # Strassen 2x2x2 decomposition vectors
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        # Check valid entries equal 1
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        # Check invalid entries are 0
        for a_idx in range(4):
            for b_idx in range(4):
                for c_idx in range(4):
                    i1 = a_idx // 2
                    a1 = a_idx % 2
                    a2 = b_idx // 2
                    k1 = b_idx % 2
                    k2 = c_idx // 2
                    i2 = c_idx % 2
                    if a1 == a2 and k1 == k2 and i1 == i2:
                        continue
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][a_idx]
                        v_val = S_V[r][b_idx]
                        w_val = S_W[r][c_idx]
                        acc += u_val * v_val * w_val
                    if acc != 0:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 candidate using direct sum structure
    # Part 1: 2x2x8 block via 4 Strassen 2x2x2 sub-blocks (rank 28)
    # Part 2: 2x1x8 schoolbook (rank 16)
    if not _elapsed_ok():
        return _schoolbook()
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen on 4 column blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8) schoolbook
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Try to reduce rank via random perturbation + SVD truncation
    best_rank = rank
    best_candidate = candidate

    def _try_reduce(U_mat, V_mat, W_mat, target_rank):
        """Try to find a rank-target_rank decomposition matching the tensor."""
        # Build the tensor as a 3D array T[i*m+a, a*p+k, k*n+i]
        # Actually: T[a_idx, b_idx, c_idx] where
        # a_idx = i*m+a (0..5), b_idx = a*p+k (0..23), c_idx = k*n+i (0..15)
        # T[a_idx, b_idx, c_idx] = sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,c_idx]
        # We need: for all i,a,k: sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] == 1
        # and for all other (a_idx, b_idx, c_idx) the sum should be 0 (but we only check the valid ones)
        
        # The key insight: the tensor is determined by the 48 entries T[i,a,k] = 1
        # We can try to find a lower rank decomposition by solving a system
        
        # Approach: random perturbation to break degeneracies, then SVD
        # The tensor can be viewed as a matrix of size (n*m) x (m*p*p*n) in mode-1 vs mode-2+3
        # But that's not quite right for rank.
        
        # Better: use the fact that if we have a valid rank-R decomposition,
        # we can try to find a rank-r < R one by:
        # 1. Randomly perturb the factor matrices slightly
        # 2. Check if the perturbed system has a solution with fewer terms
        # This is complex. Instead, let's try a direct approach:
        
        # For the 2x3x8 case, let's try to see if rank 43 or lower works
        # by checking if one of the rank-44 columns is linearly dependent
        # in a tensor sense.
        
        # Simpler approach: try to eliminate one column at a time
        # by checking if column c can be expressed as a combination of others
        # in the tensor sense.
        
        # Actually, the most practical approach: use the structure.
        # The rank-44 decomposition has:
        # - 28 columns from 4 Strassen 2x2x2 blocks (for A[0:2,0:2] * B[0:2,0:8])
        # - 16 columns from schoolbook (for A[0:2,2] * B[2,0:8])
        
        # Can we do better? Let's try mixing the Strassen and schoolbook parts.
        # The last row of A (index 2) multiplies the last row of B (index 2).
        # What if we include row 2 of A in the Strassen blocks?
        
        # Alternative: use a 2x3x2 block decomposition with rank < 6
        # For 2x3x2: schoolbook is 12. Can we do better?
        # 2x3x2 tensor: T[i,a,k] = 1 for valid (i,a,k)
        # This is equivalent to computing C[i,k] = sum_a A[i,a]*B[a,k] for i=0,1; k=0,1
        # The rank of this tensor... let's think.
        # Actually 2x3x2: we can use Strassen-like ideas.
        # Split B into B[:,0] and B[:,1].
        # C[:,0] = A * B[:,0], C[:,1] = A * B[:,1]
        # If we could compute both with shared intermediate products...
        # Let's just try random search for lower rank.
        
        return None

    # Direct approach: try to find rank < 44 by random search
    # Use the structure: the tensor T has entries T[i,a,k] = delta (1 if valid)
    # We need min rank such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Let's try a greedy/iterative approach:
    # Start with the rank-44 solution, try to merge columns
    
    # Actually, let's try a completely different structural approach:
    # 2x3x8 = 2x3x2 + 2x3x2 + 2x3x2 + 2x3x2 (4 blocks of 2x3x2)
    # If we can find a rank-5 decomposition for 2x3x2, total rank = 20
    # Rank of 2x3x2 tensor: it's the border rank / tensor rank of the multiplication
    # For 2x3x2, the tensor is:
    # T[i,a,k] where i∈{0,1}, a∈{0,1,2}, k∈{0,1}
    # T = sum over valid (i,a,k) of e_i ⊗ e_a ⊗ e_k
    # This is a 2x3x2 tensor with all 12 entries being 1? No!
    # T[i,a,k] = 1 only when the indices are "valid" for matrix mult.
    # In our convention: T[i*m+a, a*p+k, k*n+i] but for the full tensor
    # Actually the tensor M_{i,a,k} = 1 for all valid (i,a,k) in the multiplication.
    # For 2x3x2: M[i,a,k] = 1 for i=0,1; a=0,1,2; k=0,1 → all 12 entries are 1.
    # Wait no. The matrix multiplication tensor has M[i,a,k] = 1 for all i,a,k
    # because (AB)[i,k] = sum_a A[i,a] B[a,k], so each (i,a,k) triple contributes.
    # So the 2x3x2 tensor is all-ones? That can't be right for rank.
    
    # Let me re-examine. The tensor for matrix multiplication C = AB where
    # A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_a A[i,a] B[a,k]
    # The tensor T has T[i,a,k] = 1 for all i,a,k (it's the identity in the a-sum)
    # So T is just the all-ones tensor of shape (n,m,p)!
    # The rank of the all-ones tensor of shape (n,m,p) is max(n,m,p) if we allow
    # arbitrary entries... no wait.
    
    # The all-ones tensor 1_{n×m×p} has tensor rank = max(n,m,p) when entries can be arbitrary?
    # No. The tensor rank of the all-ones tensor is actually min(n,m,p) ... no.
    # For a 2×3×2 all-ones tensor: rank?
    # T = sum_r u_r ⊗ v_r ⊗ w_r where all entries of T are 1.
    # If we set u_r = e_{i_r}, v_r = e_{a_r}, w_r = e_{k_r}, we need to cover all 12 entries.
    # But we can use non-basis vectors.
    # T[i,a,k] = sum_r u_r[i] v_r[a] w_r[k] = 1 for all i,a,k
    # This means the matrix T_{i,(a,k)} has rank ≤ r for each fixed i... 
    # T_{i,(a,k)} = u_r[i] * (v_r ⊗ w_r)[a,k]
    # So T_i = sum_r u_r[i] * (v_r ⊗ w_r)
    # Each T_i is a 3×2 matrix. T_i = u_r[i] * M_r where M_r = v_r ⊗ w_r (rank-1 3×2 matrix)
    # So each T_i is a sum of r rank-1 matrices, but also T_i = c_i * (sum of rank-1s weighted)
    # Actually T_i[i,a,k] = sum_r u_r[i] v_r[a] w_r[k]
    # For the all-ones case: T_i = u_r[i] * (sum over r of v_r[a]w_r[k])... no
    # T_i[a,k] = sum_r u_r[i] v_r[a] w_r[k]
    # Let S = sum_r (v_r ⊗ w_r) ⊗ u_r (this is the full tensor)
    # T_i = sum_r u_r[i] (v_r ⊗ w_r)
    # So T_0 and T_1 are both in the span of {v_r ⊗ w_r}
    # T_0 = sum_r u_r[0] M_r, T_1 = sum_r u_r[1] M_r
    # where M_r = v_r ⊗ w_r is a 3×2 rank-1 matrix.
    # T_0 = T_1 = all-ones 3×2 matrix J_{3×2}.
    # So J = sum_r u_r[0] M_r = sum_r u_r[1] M_r
    # This means sum_r (u_r[0] - u_r[1]) M_r = 0
    # If u_r[0] ≠ u_r[1] for some r, there's a linear dependency among M_r.
    # The space of 3×2 matrices has dimension 6. Each M_r is rank-1.
    # We need J = sum_r c_r M_r (with c_r = u_r[0]) and also J = sum_r d_r M_r (with d_r = u_r[1])
    # and sum_r (c_r - d_r) M_r = 0.
    # Minimal r: J is a 3×2 matrix of rank 1 (since all entries are 1, rank(J)=1).
    # J = 1_3 ⊗ 1_2 (outer product of all-ones vectors).
    # So J itself is rank-1! Thus M_1 = J, c_1 = 1, d_1 = 1, r=1 works?
    # Check: T[i,a,k] = u_1[i] v_1[a] w_1[k] = 1*1*1 = 1. Yes!
    # But wait, that gives rank 1 for the 2×3×2 tensor? That can't be right for matrix mult.
    
    # I think I'm confusing myself. The matrix multiplication tensor is NOT all-ones.
    # Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 8 matrix."
    # The multiplication tensor M for (n,m,p) = (2,3,8) has:
    # M[i,a,k] = 1 if and only if the indices form a valid triple for the multiplication.
    # In standard matrix mult: C[i,k] = sum_a A[i,a] B[a,k]
    # So M[i,a,k] = 1 for ALL valid (i,a,k) where i∈[0,n), a∈[0,m), k∈[0,p).
    # That means M is indeed the all-ones tensor of shape (2,3,8)!
    # And its rank would be... let me think again.
    # M[i,a,k] = 1 for all i,a,k.
    # M = sum_r u_r ⊗ v_r ⊗ w_r
    # M[i,a,k] = sum_r u_r[i] v_r[a] w_r[k] = 1
    # This is the same as before. For 2×3×8:
    # M_i[a,k] = sum_r u_r[i] v_r[a] w_r[k] = 1 for all a,k
    # So M_i = 1_3 ⊗ 1_8 = J_{3×8} (all-ones 3×8 matrix) for both i=0 and i=1.
    # J_{3×8} has rank 1 (as a matrix). So M_i = c_i * (rank-1 matrix)
    # We can write J_{3×8} = 1_3 ⊗ 1_8
    # So M_0 = u_1[0] * (v_1 ⊗ w_1) = 1_3 ⊗ 1_8
    # M_1 = u_1[1] * (v_1 ⊗ w_1) = 1_3 ⊗ 1_8
    # This requires u_1[0] = u_1[1] and v_1 ⊗ w_1 = 1_3 ⊗ 1_8
    # So v_1 = 1_3, w_1 = 1_8, u_1 = [1, 1]
    # Rank 1?! That can't be right.
    
    # OH WAIT. I think the issue is that the tensor entries are NOT all 1.
    # Let me re-read the validation code:
    # for i in range(n): for a in range(m): for k in range(p):
    #   acc = sum_r U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
    #   if acc != 1: return False
    # So yes, the constraint is that for each valid (i,a,k), the sum equals 1.
    # And there are no constraints on invalid (i,a,k) triples because they don't exist
    # in the index ranges. The indices i*m+a, a*p+k, k*n+i are all valid indices.
    # So the tensor IS all-ones on its support, and the support is the full (n,m,p) grid.
    # So the tensor is the all-ones tensor of shape (2,3,8).
    
    # But that contradicts the fact that schoolbook rank is 48 = 2*3*8.
    # Unless... the tensor rank of the all-ones tensor of shape (n,m,p) is NOT 1.
    # Let me reconsider. The all-ones tensor J_{n×m×p} has J[i,a,k] = 1.
    # J = sum_r u_r ⊗ v_r ⊗ w_r means J[i,a,k] = sum_r u_r[i]v_r[a]w_r[k] = 1.
    # For n=m=p=2: J[i,a,k] = 1 for all 8 entries.
    # Can we write this as sum of r rank-1 tensors?
    # J_0[a,k] = sum_r u_r[0] v_r[a] w_r[k] = 1 for all a,k → J_0 = J_{2×2} (all ones)
    # J_1[a,k] = sum_r u_r[1] v_r[a] w_r[k] = 1 for all a,k → J_1 = J_{2×2}
    # J_{2×2} has matrix rank 1. So J_0 = c_0 * M where M is rank-1, J_1 = c_1 * M
    # We need c_0 * M = J_{2×2} and c_1 * M = J_{2×2}, so c_0 = c_1.
    # M = J_{2×2} = 1_2 ⊗ 1_2. So v_r ⊗ w_r terms sum to 1_2 ⊗ 1_2.
    # With r=1: u_1 = [c,c], v_1 = 1_2, w_1 = 1_2, c=1. Works!
    # So rank of 2×2×2 all-ones tensor is 1?? But Strassen says rank 7 for 2×2×2 matmul.
    
    # I MUST be misunderstanding the tensor. Let me look at the Strassen verification:
    # S_U[r][i*2+a], S_V[r][a*2+k], S_W[r][k*2+i]
    # The indices are: U indexed by (i,a) → i*2+a, V by (a,k) → a*2+k, W by (k,i) → k*2+i
    # And the check is sum_r S_U[r][i*2+a] * S_V[r][a*2+k] * S_W[r][k*2+i] = 1
    # This is the same as what I described. So the Strassen decomposition satisfies
    # sum_r U[i,a,r] V[a,k,r] W[k,i,r] = 1 for all i,a,k.
    # And the all-ones tensor would have rank 1 by my argument. Contradiction!
    
    # The resolution: the all-ones tensor of shape (2,2,2) does NOT have rank 1.
    # Let me recheck. J[i,a,k] = 1. J = sum_r u_r[i] v_r[a] w_r[k].
    # J_0 = J_{2×2} = [[1,1],[1,1]]. This matrix has rank 1.
    # J_0 = sum_r u_r[0] (v_r ⊗ w_r). Each v_r ⊗ w_r is rank-1 2×2.
    # J_0 is rank 1, so it's a single rank-1 matrix. So sum_r u_r[0] M_r = M_0 (rank 1)
    # This doesn't force r=1. It forces the combination to be rank 1.
    # Similarly J_1 = sum_r u_r[1] M_r = M_0 (same rank-1 matrix, since J_0=J_1).
    # So u_r[0] M_r and u_r[1] M_r both sum to M_0.
    # If r=1: u_1[0] M_1 = M_0, u_1[1] M_1 = M_0 → u_1[0]=u_1[1], M_1=M_0. Works.
    # So rank IS 1 for 2×2×2 all-ones. But Strassen rank is 7 for 2×2×2 MATMULT.
    
    # THE KEY INSIGHT: The matrix multiplication tensor is NOT the all-ones tensor!
    # I've been making an error. Let me re-examine.
    # In matrix multiplication C = AB, the (i,k) entry is C[i,k] = sum_a A[i,a] B[a,k].
    # The "multiplication tensor" T satisfies C[i,k] = sum_{i,a,k} T[i,a,k] A[i,a] B[a,k]
    # For this to work, T[i,a,k] must be 1 when the indices match, i.e.,
    # T[i,a,k] = 1 for all i,a,k (since every (i,a,k) triple contributes to C[i,k]).
    # So T IS the all-ones tensor. But then its rank is 1, not 7.
    
    # I think the issue is that the tensor rank here is NOT the standard tensor rank.
    # Or perhaps I'm wrong about the all-ones tensor having rank 1.
    # Let me verify with a concrete example. 2×2×2 all-ones tensor.
    # Can I write J = u_1 ⊗ v_1 ⊗ w_1 with J[i,a,k]=1?
    # u_1[i] v_1[a] w_1[k] = 1 for all i,a,k.
    # Set i=0,a=0,k=0: u_1[0]v_1[0]w_1[0] = 1
    # Set i=1,a=0,k=0: u_1[1]v_1[0]w_1[0] = 1 → u_1[1]=u_1[0]
    # Set i=0,a=1,k=0: u_1[0]v_1[1]w_1[0] = 1 → v_1[1]=v_1[0]
    # Set i=0,a=0,k=1: u_1[0]v_1[0]w_1[1] = 1 → w_1[1]=w_1[0]
    # So u_1=[c,c], v_1=[d,d], w_1=[e,e] with cde=1. Yes, rank 1!
    # So the all-ones 2×2×2 tensor has rank 1.
    
    # This means the "multiplication tensor" in this problem is NOT the all-ones tensor.
    # There must be additional structure I'm missing.
    # Let me re-read the problem statement: "output factor follows the official trace convention W[k*n+i,r]"
    # Maybe the tensor is not simply all-ones. Perhaps the indices in the validation
    # are structured differently.
    
    # Looking at the validation again:
    # for i in range(n): for a in range(m): for k in range(p):
    #   u_row = U[i * m + a]  # index i*m+a
    #   v_row = V[a * p + k]  # index a*p+k  
    #   w_row = W[k * n + i]  # index k*n+i
    #   acc = sum_r u_row[r] * v_row[r] * w_row[r]
    #   assert acc == 1
    # 
    # This checks that for each (i,a,k), the inner product of U[i*m+a], V[a*p+k], W[k*n+i] is 1.
    # This is exactly the all-ones tensor condition. So either:
    # 1. The all-ones tensor really does have rank 1 (and Strassen is for something else)
    # 2. I'm making an error.
    
    # Let me test: for 2×2×2, is there a rank-1 decomposition?
    # U = [[1,1],[1,1],[1,1],[1,1]] (4 rows, 1 col) → U[i*2+a] = 1
    # V = [[1,1],[1,1],[1,1],[1,1]] (4 rows, 1 col) → V[a*2+k] = 1
    # W = [[1,1],[1,1],[1,1],[1,1]] (4 rows, 1 col) → W[k*2+i] = 1
    # Then U[i*2+a]*V[a*2+k]*W[k*2+i] = 1*1*1 = 1. ✓
    # So rank 1 works for 2×2×2!
    
    # But the Strassen code has rank 7. Why?
    # OH! I bet the issue is that the factor matrices must have the CORRECT SHAPES.
    # U has shape (n*m, rank) = (4, rank), V has shape (m*p, rank) = (4, rank), W has shape (p*n, rank) = (4, rank).
    # For rank 1: U is 4×1, V is 4×1, W is 4×1.
    # U[i*m+a] for i=0,1 and a=0,1: U[0], U[1], U[2], U[3]
    # V[a*p+k] for a=0,1 and k=0,1: V[0], V[1], V[2], V[3]
    # W[k*n+i] for k=0,1 and i=0,1: W[0], W[1], W[2], W[3]
    # If U=[1,1,1,1]^T, V=[1,1,1,1]^T, W=[1,1,1,1]^T, then
    # For (i,a,k)=(0,0,0): U[0]*V[0]*W[0] = 1 ✓
    # For (i,a,k)=(0,0,1): U[0]*V[1]*W[2] = 1*1*1 = 1 ✓
    # For (i,a,k)=(0,1,0): U[1]*V[2]*W[0] = 1*1*1 = 1 ✓
    # All work! So rank 1 is valid for 2×2×2.
    
    # This means the problem as stated has a trivial solution of rank 1 for any (n,m,p)!
    # U[i*m+a] = 1 for all, V[a*p+k] = 1 for all, W[k*n+i] = 1 for all.
    # Then sum_r (over r=1) U*V*W = 1*1*1 = 1 for all (i,a,k).
    
    # Wait, but the score is 1.0909 = 52/48? No, score = seed/value where seed is the
    # schoolbook rank 48. So value = 48/1.0909 ≈ 44. So the current program achieves rank 44.
    # If rank 1 were valid, the score would be 48/1 = 48.
    
    # So rank 1 must NOT be valid. Let me re-examine.
    # The issue must be that the factor matrices have specific shapes that prevent this.
    # U has shape (n*m, rank) = (6, rank) for n=2,m=3.
    # U[i*m+a] for i∈{0,1}, a∈{0,1,2}: U[0],U[1],U[2],U[3],U[4],U[5]
    # V has shape (m*p, rank) = (24, rank).
    # V[a*p+k] for a∈{0,1,2}, k∈{0..7}: V[0..23]
    # W has shape (p*n, rank) = (16, rank).
    # W[k*n+i] for k∈{0..7}, i∈{0,1}: W[0..15]
    
    # For rank 1: U is 6×1, V is 24×1, W is 16×1.
    # U[i*m+a] = u_{i,a}, V[a*p+k] = v_{a,k}, W[k*n+i] = w_{k,i}
    # Need: u_{i,a} * v_{a,k} * w_{k,i} = 1 for ALL i,a,k.
    # From (i,a,k)=(0,0,0): u_{0,0} v_{0,0} w_{0,0} = 1
    # From (i,a,k)=(0,0,1): u_{0,0} v_{0,1} w_{1,0} = 1
    # From (i,a,k)=(0,1,0): u_{0,1} v_{1,0} w_{0,0} = 1
    # From (i,a,k)=(0,1,1): u_{0,1} v_{1,1} w_{1,0} = 1
    # ...
    # This is a system of 48 equations. Let's see if it's consistent.
    # u_{0,0} v_{0,0} w_{0,0} = 1  ... (1)
    # u_{0,0} v_{0,1} w_{1,0} = 1  ... (2)
    # u_{0,1} v_{1,0} w_{0,0} = 1  ... (3)
    # u_{0,1} v_{1,1} w_{1,0} = 1  ... (4)
    # u_{1,0} v_{0,0} w_{0,1} = 1  ... (5)
    # u_{1,0} v_{0,1} w_{1,1} = 1  ... (6)
    # u_{1,1} v_{1,0} w_{0,1} = 1  ... (7)
    # u_{1,1} v_{1,1} w_{1,1} = 1  ... (8)
    # ...and so on for a=2.
    
    # From (1) and (2): v_{0,0} w_{0,0} = v_{0,1} w_{1,0}
    # From (3) and (4): v_{1,0} w_{0,0} = v_{1,1} w_{1,0}
    # From (5) and (6): v_{0,0} w_{0,1} = v_{0,1} w_{1,1}
    # From (7) and (8): v_{1,0} w_{0,1} = v_{1,1} w_{1,1}
    
    # Let's try to find a solution. Set w_{0,0}=w_{0,1}=w_{1,0}=w_{1,1}=1.
    # Then v_{0,0}=v_{0,1} (from first eq) and v_{1,0}=v_{1,1} (from second).
    # And u_{0,0} v_{0,0} = 1, u_{0,1} v_{1,0} = 1, u_{1,0} v_{0,0} = 1, u_{1,1} v_{1,0} = 1
    # So u_{0,0} = u_{1,0} = 1/v_{0,0} and u_{0,1} = u_{1,1} = 1/v_{1,0}
    # Now check a=2:
    # u_{0,2} v_{2,0} w_{0,0} = 1 → u_{0,2} v_{2,0} = 1
    # u_{0,2} v_{2,1} w_{1,0} = 1 → u_{0,2} v_{2,1} = 1 → v_{2,0}=v_{2,1}
    # Similarly v_{2,0}=v_{2,1}=v_{2,2}=...=v_{2,7}
    # u_{0,2} = 1/v_{2,0}, u_{1,2} = 1/v_{2,0}
    
    # So a solution exists! u_{i,a} = 1/v_{a,0}, v_{a,k} = v_{a,0} for all k, w_{k,i} = 1.
    # Let me verify: u_{i,a} v_{a,k} w_{k,i} = (1/v_{a,0}) * v_{a,0} * 1 = 1. ✓
    # So rank 1 IS valid for 2×3×8!
    
    # But the score is 44, not 1. So something is wrong with my understanding.
    # Let me re-read the problem: "integer/half-integer entries"
    # The entries must be integers or half-integers! That's the constraint I was missing.
    # With that constraint, rank 1 may not work because we need v_{a,0} to divide 1
    # in a way that all entries are integers or half-integers.
    # u_{i,a} = 1/v_{a,0}. For this to be integer or half-integer, v_{a,0} must be
    # 1, 2, 1/2, etc. And v_{a,k} = v_{a,0} for all k.
    # If v_{a,0} = 1: u_{i,a} = 1, v_{a,k} = 1, w_{k,i} = 1. All integers. ✓
    # So rank 1 with all-ones IS valid with integer entries!
    
    # I'm confused. Let me just test the all-ones rank-1 solution against the validator.
    
    def _test_rank1():
        rank1 = 1
        U1 = [[1] for _ in range(n * m)]
        V1 = [[1] for _ in range(m * p)]
        W1 = [[1] for _ in range(p * n)]
        cand = {"U": U1, "V": V1, "W": W1}
        return _validate(cand)

    if _test_rank1():
        return {"U": [[1] for _ in range(n*m)], "V": [[1] for _ in range(m*p)], "W": [[1] for _ in range(p*n)]}

    # OK so if rank 1 doesn't pass validation, my understanding is wrong.
    # Let me proceed with the rank-44 solution and try to reduce it.
    return candidate

# EVOLVE-BLOCK-END
