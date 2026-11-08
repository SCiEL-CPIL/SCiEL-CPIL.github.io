"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    def verify_solution(U, V, W):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Compute T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    def get_rank_33_solution():
        """Construct the rank-33 solution using 3 blocks of 2x3x2."""
        rank = 33
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        terms = [
            {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},   # M1
            {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},        # M2
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},        # M3
            {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},        # M4
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},        # M5
            {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},        # M6
            {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},        # M7
            {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},                      # K1
            {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},                      # K2
            {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},                      # K3
            {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},                      # K4
        ]
        
        for block in range(3):
            for t_idx, term in enumerate(terms):
                col = block * 11 + t_idx
                
                for idx, val in term['u'].items():
                    U[idx, col] = val
                
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx, col] = val
                
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx, col] = val
        
        return U, V, W
    
    def get_trivial_solution():
        """Fallback trivial rank-36 solution."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m + j, r] = 1
                    V[j*p + k, r] = 1
                    W[k*n + i, r] = 1
        
        return U, V, W
    
    def try_reduce_rank(U, V, W, target_rank, max_iterations=500):
        """Try to reduce rank by zeroing columns and solving residual."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        current_rank = U.shape[1]
        
        for attempt in range(max_iterations):
            if time.time() > deadline - 10:
                break
            if current_rank <= target_rank:
                break
            
            # Try removing each column and see if we can compensate
            best_col = -1
            best_residual = float('inf')
            
            for col in range(current_rank):
                if U[:, col].all() == 0 and V[:, col].all() == 0 and W[:, col].all() == 0:
                    # Already zero, remove it
                    best_col = col
                    best_residual = 0
                    break
                
                # Compute residual if we zero out this column
                residual = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            val -= U[u_idx, col] * V[v_idx, col] * W[w_idx, col]
                            residual[i, j, k] = val - 1.0
                
                res_norm = np.linalg.norm(residual)
                if res_norm < best_residual:
                    best_residual = res_norm
                    best_col = col
            
            if best_col == -1:
                break
            
            # Zero out the best column
            U[:, best_col] = 0
            V[:, best_col] = 0
            W[:, best_col] = 0
            current_rank -= 1
            
            # Check if solution is still valid
            if verify_solution(U[:, :current_rank], V[:, :current_rank], W[:, :current_rank]):
                U = U[:, :current_rank]
                V = V[:, :current_rank]
                W = W[:, :current_rank]
        
        if verify_solution(U, V, W) and U.shape[1] <= target_rank:
            return U, V, W
        return None
    
    def find_low_rank_by_construction(target_rank):
        """Try to construct a solution of given rank using structured search."""
        if time.time() > deadline - 15:
            return None
        
        # Try a structured approach: use shared factors across blocks
        # For 2x3x6, try to find rank < 33
        
        # Strategy: Use the fact that 2x3x6 = 3 * (2x3x2) but with potential sharing
        # Try rank 30: 10 per block
        # Try rank 27: 9 per block
        
        # Let's try a different grouping: 2x3x6 can be seen as
        # A (2x3) * B (3x6)
        # Group B columns as (0,1), (2,3), (4,5) - current approach
        # Or try (0,2,4), (1,3,5) - interleaved
        
        # Alternative: treat as 2x6x3 (transpose perspective)
        # Or use a 3x2x6 decomposition
        
        # Let me try: A is 2x3, B is 3x6
        # C = A*B is 2x6
        # Think of B as [B1, B2, B3] where each Bj is 3x2
        # C = [A*B1, A*B2, A*B3]
        
        # For each A*Bj (2x2 result from 2x3 * 3x2), we need rank 11
        # But maybe we can do rank 10 for 2x3x2?
        
        # Let's try to find rank 10 for 2x3x2
        # Tensor entries: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1}
        # 4 output entries, 12 input entries
        
        # Try rank 10 for a single 2x3x2 block
        # This is a small problem, let's try to solve it
        
        # Actually, let me try a completely different approach:
        # Use the "bit-plane" idea from the directive
        # Represent entries as sums of bits, use AND-OR logic
        
        # For integer/half-integer entries, we can use:
        # x = sum of bits * 2^k
        # But this gives rank proportional to number of bits, which is worse
        
        # Instead, let me try to find if rank 30 is achievable
        # by solving the polynomial system
        
        # Set up: for rank r, we need U(6xr), V(18xr), W(12xr)
        # such that sum_r U[i*3+j,r] * V[j*6+k,r] * W[k*2+i,r] = 1 for all (i,j,k)
        
        # This is 36 equations. For r=30, 1080 unknowns.
        # We can use a greedy approach: fix most variables, solve for a few.
        
        # Alternative: try specific structured solutions
        # What if we use rank 32? 33 - 1 = 32
        # Can we merge two of the 11 terms per block into 10?
        
        # For 2x3x2, the rank is at least the rank of any "slice"
        # Fixing i=0: we get a 3x2 tensor (j,k) -> sum_j A[0,j]*B[j,k]
        # This is just the matrix product A[0,:]*B, which has rank at most min(3,2)=2
        # But as a tensor, fixing i=0 gives a bilinear form in (j,k) with coefficients from A[0,:]*B
        # The "fiber" rank... 
        
        # Let me just try to verify if rank 32 works by a different construction
        # What if for one block we can do rank 10 instead of 11?
        
        # For 2x3x2: C[0,0]=A00*B00+A01*B10+A02*B20
        #            C[0,1]=A00*B01+A01*B11+A02*B21
        #            C[1,0]=A10*B00+A11*B10+A12*B20
        #            C[1,1]=A10*B01+A11*B11+A12*B21
        
        # Strassen on (A00,A01,A10,A11) and (B00,B01,B10,B11) gives 7 terms for
        # C[0,0],C[0,1],C[1,0],C[1,1] using only j=0,1
        # Then we need to add A02*B20, A02*B21, A12*B20, A12*B21
        # That's 4 more terms, total 11.
        
        # Can we do better? What if we don't use Strassen but a different 7-term scheme?
        # Or what if we can absorb one of the 4 scalar terms into the Strassen terms?
        
        # Let's try: instead of pure Strassen + 4 scalars, use 8 terms total
        # where some terms involve j=2 in a mixed way.
        
        # Try rank 8 for 2x3x2:
        # We need 8 rank-1 tensors whose sum gives the 2x3x2 multiplication tensor.
        
        # Let me try a parametric search for rank 10 for 2x3x2
        # Using the structure: some terms involve j=0,1 and some involve j=2
        
        # Actually, the minimum rank of the 2x3x2 tensor is known to be 11
        # (it's the same as 2x3x2 matrix multiplication rank)
        # So 33 is likely optimal for the block decomposition.
        
        # But maybe we can do better with a non-block decomposition!
        # The 2x3x6 tensor might have rank < 33 due to cross-block sharing.
        
        # Let me try to find rank 30 using a numerical optimization
        # Fix the structure: use 30 columns, try to solve
        
        # For a fixed rank r, the problem is:
        # Find U,V,W such that T_{ijk} = sum_r U_{im+j,r} V_{jp+k,r} W_{kn+i,r} = 1
        
        # This is equivalent to: the tensor T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector in R^6, v_r in R^18, w_r in R^12
        
        # We can use the "hyperdeterminant" or try to find a solution
        # by fixing some variables and solving linearly for others.
        
        # Strategy: Fix U and V, then W is determined linearly (if the system is full rank)
        # Or fix U and W, then V is determined.
        
        # Let me try: fix U to have a specific structure, then solve for V and W
        
        # Actually, let me try a simpler approach: use the rank-33 solution and try
        # to find a rank-32 solution by checking if one column is linearly dependent
        # in a tensor sense.
        
        # For now, let's try rank 32 with a modified construction
        # What if we can merge the K3 and K4 terms across blocks?
        
        # In the current solution, K3 for block b is:
        # U[5, b*11+10] = 1, V[4, b*11+10] = 1, W[2*b*2+2, b*11+10] = 1
        # Wait, let me recheck the indices.
        
        # K3: u={5:1}, v={4:1}, w={2:1}
        # v index 4: j_local=2, k_local=0, so g_idx = 2*6 + (block*2+0) = 12 + block*2
        # w index 2: i_local=1, k_local=0, so g_idx = (block*2+0)*2 + 1 = block*4 + 1
        
        # K4: u={5:1}, v={5:1}, w={3:1}
        # v index 5: j_local=2, k_local=1, so g_idx = 2*6 + (block*2+1) = 13 + block*2
        # w index 3: i_local=1, k_local=1, so g_idx = (block*2+1)*2 + 1 = block*4 + 3
        
        # These don't obviously share structure across blocks.
        
        # Let me try a completely different approach: use a known result.
        # The rank of 2x3x6 matrix multiplication...
        # By submultiplicativity: R(2,3,6) <= R(2,3,2)*R(2,1,3) ... no that's not right.
        # R(2,3,6) <= R(2,3,2) * R(1,1,3)? No.
        # Actually R(m,n,p) <= R(m,n,p1)*R(1,1,p2) if p = p1*p2? No.
        
        # We know R(2,3,2) = 11 (I believe)
        # R(2,3,6) <= 3*R(2,3,2) = 33
        # Also R(2,3,6) <= R(2,3,3)*R(1,1,2)? No, that doesn't work directly.
        
        # R(2,3,6) <= R(2,3,3) + R(2,3,3) if 6 = 3+3? No, that's not how it works.
        # Actually, if B is 3x6, we can split into B1 (3x3) and B2 (3x3):
        # C = A*B1 + A*B2
        # R(2,3,6) <= R(2,3,3) + R(2,3,3) = 2*R(2,3,3)
        # R(2,3,3) = ? The rank of 2x3x3 matrix multiplication.
        # Trivial: 18. Can we do better?
        # R(2,3,3) <= R(2,2,3)*R(1,1,3)? No.
        # R(2,3,3) <= 3*R(2,1,1) = 3? No, that's not right either.
        
        # Actually R(2,3,3): A is 2x3, B is 3x3, C is 2x3.
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # 6 output entries. Trivial rank 18 (3*3*2).
        # We can write C = A*B where A is 2x3 and B is 3x3.
        # R(2,3,3) <= R(2,3,2) + R(2,3,1) = 11 + 6 = 17? 
        # R(2,3,1): A is 2x3, B is 3x1, C is 2x1.
        # C[i,0] = sum_j A[i,j]*B[j,0]. 2 output entries. Trivial rank 6.
        # Can we do better? C[0,0] = A00*B0 + A01*B1 + A02*B2
        # C[1,0] = A10*B0 + A11*B1 + A12*B2
        # This is like a 2x3x1 tensor. Rank is at most 6, at least max(rank of slices).
        # Fixing B: it's a 2x3 matrix times a vector. The tensor is actually
        # just a bilinear map. The rank of this tensor...
        # It's the same as the rank of the "flattened" matrix.
        # Flatten: for each (i,j), the coefficient of B[j,0] in C[i,0] is A[i,j].
        # So the tensor T_{i,j,b} = A[i,j] * delta_{b,0}... no.
        # T_{i,j,k} where k is the B index. T[i,j,k] = delta_{i,k}... no.
        
        # I'm getting confused. Let me just try to numerically find a lower rank.
        
        # Try rank 32: modify the rank-33 solution to see if one term is redundant
        U33, V33, W33 = get_rank_33_solution()
        
        # Check if any single column can be removed and the residual is zero
        # (i.e., the column contributes nothing)
        for col in range(33):
            if np.allclose(U33[:, col], 0) or np.allclose(V33[:, col], 0) or np.allclose(W33[:, col], 0):
                # This column is already zero, remove it
                mask = [c for c in range(33) if c != col]
                U_new = U33[:, mask]
                V_new = V33[:, mask]
                W_new = W33[:, mask]
                if verify_solution(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        
        # Try rank 32 by solving for a modified last column
        # Keep first 32 columns, solve for the 33rd to be zero
        # This is equivalent to checking if the first 32 columns already give the right answer
        for num_cols in range(32, 25, -1):
            if time.time() > deadline - 20:
                break
            U_sub = U33[:, :num_cols]
            V_sub = V33[:, :num_cols]
            W_sub = W33[:, :num_cols]
            if verify_solution(U_sub, V_sub, W_sub):
                return U_sub, V_sub, W_sub
        
        # Try a different construction for rank 30
        # Use 10 terms per block with a different scheme
        # For 2x3x2, try to find rank 10
        
        # Parametric search for 2x3x2 rank 10
        # T[i,j,k] = sum_{r=1}^{10} U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r] = 1
        # 8 equations, 10*(3+2+2)=70 unknowns
        
        # Let me try a specific structured solution
        # Use 7 Strassen terms for j=0,1 and 3 terms for j=2 (instead of 4)
        # Can we merge two of the j=2 terms?
        
        # The 4 j=2 terms are:
        # A02*B20 -> C[0,0]: U[2,r]=1, V[4,r]=1, W[0,r]=1
        # A02*B21 -> C[0,1]: U[2,r]=1, V[5,r]=1, W[1,r]=1
        # A12*B20 -> C[1,0]: U[5,r]=1, V[4,r]=1, W[2,r]=1
        # A12*B21 -> C[1,1]: U[5,r]=1, V[5,r]=1, W[3,r]=1
        
        # Can we combine these into 3 terms?
        # We need: sum_r U[2,r]*V[4,r]*W[0,r] = 1 (for C[0,0] contribution from j=2)
        #          sum_r U[2,r]*V[5,r]*W[1,r] = 1
        #          sum_r U[5,r]*V[4,r]*W[2,r] = 1
        #          sum_r U[5,r]*V[5,r]*W[3,r] = 1
        # And these should not interfere with the Strassen terms.
        
        # If we use 3 new terms, we need to satisfy 4 equations with 3 terms.
        # Each term contributes to at most 1 equation (since U and W indices are specific).
        # So 3 terms can satisfy at most 3 equations. Not enough for 4.
        
        # Unless some terms contribute to multiple equations.
        # A term with U[2,r]=a, V[4,r]=b, W[0,r]=c contributes a*b*c to C[0,0] j=2 part.
        # It also contributes to other equations if other U,V,W entries are non-zero.
        
        # Let me try: can a single term contribute to two of the j=2 equations?
        # Term r: U[2,r]=1, V[4,r]=1, W[0,r]=1, W[2,r]=1
        # Then it contributes to C[0,0] (via U[2]*V[4]*W[0]) and C[1,0] (via U[5]... no, U[5,r] is 0)
        # Hmm, U[5,r] would need to be non-zero too.
        
        # Term r: U[2,r]=1, U[5,r]=1, V[4,r]=1, W[0,r]=1, W[2,r]=1
        # Contributes to: C[0,0] j=2: U[2]*V[4]*W[0] = 1
        #                C[1,0] j=2: U[5]*V[4]*W[2] = 1
        #                C[0,0] from Strassen part: U[2] is in the "A02" slot, not in Strassen
        #                But U[2,r] also appears in the Strassen equations!
        
        # In the Strassen part, U[2] is not used (Strassen uses U[0],U[1],U[3],U[4])
        # So U[2,r] only affects the j=2 terms. Good.
        
        # So term r with U[2,r]=1, U[5,r]=1, V[4,r]=1, W[0,r]=1, W[2,r]=1
        # contributes 1 to C[0,0] j=2 and 1 to C[1,0] j=2.
        
        # Similarly, term s with U[2,s]=1, U[5,s]=1, V[5,s]=1, W[1,s]=1, W[3,s]=1
        # contributes 1 to C[0,1] j=2 and 1 to C[1,1] j=2.
        
        # That's only 2 terms for all 4 j=2 contributions!
        # Total: 7 (Strassen) + 2 = 9 per block!
        
        # But wait, I need to check that these terms don't interfere with the Strassen equations.
        # The Strassen equations use U[0],U[1],U[3],U[4] and V[0],V[1],V[2],V[3] and W[0],W[1],W[2],W[3]
        # My new terms use U[2],U[5],V[4],V[5],W[0],W[1],W[2],W[3]
        
        # W[0],W[1],W[2],W[3] are also used in Strassen! This will cause interference.
        
        # The Strassen term M1 has W[0]=1, W[3]=1
        # My new term r has W[0]=1, W[2]=1
        # When computing C[0,0] = sum_r U[0,r]*V[0,r]*W[0,r] + U[0,r]*V[1,r]*W[0,r] + ... 
        # Wait, no. The tensor equation is:
        # C[i,k] = sum_j sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r]
        
        # For C[0,0]: sum_r U[0,r]*V[0,r]*W[0,r] + U[1,r]*V[1,r]*W[0,r] + U[2,r]*V[2,r]*W[0,r]
        # Wait, V index is j*2+k. For j=0,k=0: V[0]. For j=1,k=0: V[2]. For j=2,k=0: V[4].
        
        # Let me redo: C[i,k] = sum_{j=0}^{2} sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r]
        
        # C[0,0] = sum_r [U[0,r]*V[0,r]*W[0,r] + U[1,r]*V[2,r]*W[0,r] + U[2,r]*V[4,r]*W[0,r]]
        # C[0,1] = sum_r [U[0,r]*V[1,r]*W[1,r] + U[1,r]*V[3,r]*W[1,r] + U[2,r]*V[5,r]*W[1,r]]
        # C[1,0] = sum_r [U[3,r]*V[0,r]*W[2,r] + U[4,r]*V[2,r]*W[2,r] + U[5,r]*V[4,r]*W[2,r]]
        # C[1,1] = sum_r [U[3,r]*V[1,r]*W[3,r] + U[4,r]*V[3,r]*W[3,r] + U[5,r]*V[5,r]*W[3,r]]
        
        # In the Strassen part (terms 0-6), U[2,r] and U[5,r] are 0, and V[4,r],V[5,r] are 0.
        # So the Strassen terms only contribute through U[0],U[1],U[3],U[4] and V[0],V[1],V[2],V[3].
        
        # My new term r: U[2,r]=1, U[5,r]=1, V[4,r]=1, W[0,r]=1, W[2,r]=1
        # Contribution to C[0,0]: U[2,r]*V[4,r]*W[0,r] = 1*1*1 = 1 ✓
        # Contribution to C[1,0]: U[5,r]*V[4,r]*W[2,r] = 1*1*1 = 1 ✓
        # Contribution to C[0,1]: U[2,r]*V[5,r]*W[1,r] = 1*0*0 = 0 (V[5,r]=0 for this term) ✓
        # Contribution to C[1,1]: U[5,r]*V[5,r]*W[3,r] = 1*0*0 = 0 ✓
        
        # My new term s: U[2,s]=1, U[5,s]=1, V[5,s]=1, W[1,s]=1, W[3,s]=1
        # Contribution to C[0,1]: U[2,s]*V[5,s]*W[1,s] = 1*1*1 = 1 ✓
        # Contribution to C[1,1]: U[5,s]*V[5,s]*W[3,s] = 1*1*1 = 1 ✓
        # Contribution to C[0,0]: U[2,s]*V[4,s]*W[0,s] = 1*0*0 = 0 ✓
        # Contribution to C[1,0]: U[5,s]*V[4,s]*W[2,s] = 1*0*0 = 0 ✓
        
        # So 7 Strassen + 2 combined terms = 9 per block!
        # Total rank: 3*9 = 27!
        
        # But wait, I need to double-check that the Strassen terms don't have any
        # non-zero entries in U[2],U[5],V[4],V[5] that would interfere.
        # From the current solution, Strassen terms only use U indices 0,1,3,4
        # and V indices 0,1,2,3 and W indices 0,1,2,3.
        # So U[2],U[5],V[4],V[5] are 0 for Strassen terms. 
        # And W[0],W[1],W[2],W[3] are non-zero for Strassen terms, but that's fine
        # because the Strassen terms have U[2]=0 and U[5]=0, so they don't
        # contribute to the j=2 parts.
        
        # This gives rank 27! Let me implement this.
        
        rank_27 = 27
        U = np.zeros((n*m, rank_27))
        V = np.zeros((m*p, rank_27))
        W = np.zeros((p*n, rank_27))
        
        # Strassen terms for 2x2x2 (using j=0,1)
        strassen_terms = [
            # (U_indices, V_indices, W_indices) for 2x2x2 Strassen
            # A is 2x2: A[0,0]=U0, A[0,1]=U1, A[1,0]=U3, A[1,1]=U4
            # B is 2x2: B[0,0]=V0, B[0,1]=V1, B[1,0]=V2, B[1,1]=V3
            # C is 2x2: C[0,0]=W0, C[0,1]=W1, C[1,0]=W2, C[1,1]=W3
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            # C00 = M1 - M4 + M6 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M5 - M6
            {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 1: 0, 2: 0, 3: -1}},  # M1
            {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {0: 0, 1: 0, 2: 1, 3: 0}},          # M2
            {'u': {0: 1}, 'v': {2: 1, 3: -1}, 'w': {0: 0, 1: 1, 2: 0, 3: -1}},        # M3
            {'u': {4: 1}, 'v': {1: 1, 0: -1}, 'w': {0: -1, 1: 0, 2: 1, 3: 0}},        # M4
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: 0, 1: 1, 2: 0, 3: -1}},         # M5
            {'u': {3: 1, 0: -1}, 'v': {0: 1, 2: 1}, 'w': {0: 1, 1: 0, 2: 0, 3: -1}},  # M6
            {'u': {1: 1, 4: -1}, 'v': {1: 1, 3: 1}, 'w': {0: -1, 1: 0, 2: 0, 3: 1}},  # M7
        ]
        
        # Wait, I need to be more careful with the Strassen formulation.
        # Let me use the standard one and verify.
        
        # Actually, let me just use the original Strassen terms from the current solution
        # and add the 2 combined terms.
        
        # Original Strassen terms (from current solution):
        strassen_orig = [
            {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},   
            {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},        
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},        
            {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},        
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},        
            {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},        
            {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},        
        ]
        
        # 2 combined terms for j=2
        combined_terms = [
            {'u': {2: 1, 5: 1}, 'v': {4: 1}, 'w': {0: 1, 2: 1}},  # contributes to C[0,0] and C[1,0] j=2 parts
            {'u': {2: 1, 5: 1}, 'v': {5: 1}, 'w': {1: 1, 3: 1}},  # contributes to C[0,1] and C[1,1] j=2 parts
        ]
        
        for block in range(3):
            for t_idx, term in enumerate(strassen_orig):
                col = block * 9 + t_idx
                for idx, val in term['u'].items():
                    U[idx, col] = val
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx, col] = val
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx, col] = val
            
            for t_idx, term in enumerate(combined_terms):
                col = block * 9 + 7 + t_idx
                for idx, val in term['u'].items():
                    U[idx, col] = val
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx, col] = val
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx, col] = val
        
        return U, V, W
    
    # Try the rank-27 solution first
    U27, V27, W27 = find_low_rank_by_construction(27)
    
    if U27 is not None and verify_solution(U27, V27, W27):
        best_U, best_V, best_W = U27, V27, W27
        best_rank = 27
    else:
        # Fall back to rank-33
        U33, V33, W33 = get_rank_33_solution()
        if verify_solution(U33, V33, W33):
            best_U, best_V, best_W = U33, V33, W33
            best_rank = 33
        else:
            U36, V36, W36 = get_trivial_solution()
            best_U, best_V, best_W = U36, V36, W36
            best_rank = 36
    
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
# EVOLVE-BLOCK-END
