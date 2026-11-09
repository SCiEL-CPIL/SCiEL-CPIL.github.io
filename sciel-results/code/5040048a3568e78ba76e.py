import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if i,j,k correspond to valid indices
    # Actually, the tensor for C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for bilinear form)
    # Wait, let me reconsider. The matrix multiplication tensor has entries:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _groebner_search(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition of rank target_rank using Groebner basis
    over a finite field. Returns (U, V, W) if found, else None.
    """
    # Use a prime field for computation
    # We need a prime large enough to have good properties
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for prime in primes:
        if time.time() > deadline - 30:
            break
        
        # The number of variables is 3 * (n*m + m*p + p*n) * target_rank which is huge
        # Instead, we'll use a randomized approach: fix U and V to random values,
        # then solve for W, and check if the decomposition works.
        # This is a heuristic based on the algebraic structure.
        
        # Better approach: use the fact that for a valid decomposition,
        # the tensor must be in the span of rank-1 tensors.
        # We'll try to find a low-rank approximation and then verify.
        
        # For now, this is a placeholder that will be refined.
        # The actual Groebner basis computation would be extremely expensive
        # for this problem size, so we use a randomized heuristic instead.
        
        rng = np.random.RandomState(prime)
        
        # Try random initializations and check if they can be completed
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Generate random U and V over the reals (we'll verify exactly)
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            
            # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of linear equations in W if we fix U and V
            # But it's not linear in W because W appears multiplied by U*V
            # Actually, for fixed U and V, each equation is linear in W[k*n+i, r]
            
            # Let's set up the linear system
            # For each (i,j,k): sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # This is linear in W!
            
            # Build the system
            num_equations = n * m * p
            num_unknowns = p * n * target_rank
            
            # Coefficient matrix A_sys where A_sys[eq, unknown] = coefficient
            # eq index: (i, j, k) -> i * m * p + j * p + k
            # unknown index: (k, i, r) -> (k * n + i) * target_rank + r
            
            A_sys = np.zeros((num_equations, num_unknowns))
            b_sys = np.ones(num_equations)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(target_rank):
                            unk_idx = (k * n + i) * target_rank + r
                            A_sys[eq_idx, unk_idx] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
            
            # Solve the linear system
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                # Check if the solution is exact
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    continue
                W_rand = W_flat.reshape(p * n, target_rank)
            except:
                continue
            
            # Verify the decomposition
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But for matrix multiplication, the lower bound is higher.
    # Known results: for 4x4x4, the rank is at most 49 (Strassen)
    # For 4x4x9, we can use Strassen for the 4x4x4 part and schoolbook for the rest
    # That gives 49 + 4*4*5 = 49 + 80 = 129
    
    # First, try the Strassen-based decomposition (rank 129)
    # Then try to improve further
    
    # Strassen for 4x4x4
    # The 4x4x4 tensor can be decomposed using Strassen(2x2) tensor Strassen(2x2)
    # This gives rank 7*7 = 49
    
    # For 4x4x9, we split the 9 columns into 4 + 5
    # Use Strassen for the 4x4x4 part (rank 49)
    # Use schoolbook for the 4x4x5 part (rank 80)
    # Total rank: 129
    
    # Build the Strassen 4x4x4 decomposition
    def _strassen_2x2_decomposition():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        U = [
            [1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1]
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0]
        ]
        W = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1]
        ]
        return U, V, W
    
    def _kron_strassen_4x4x4():
        """Build 4x4x4 decomposition using Kronecker product of two Strassen 2x2x2."""
        U2, V2, W2 = _strassen_2x2_decomposition()
        
        # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors
        # U_4[i1*i2 + i1', j1*j2 + j1'] = U2[i1, j1] * U2[i1', j1']
        # Similarly for V and W
        
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1 * 7 + r2
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U4[i * 4 + j][r] = U2[i1][j1] * U2[i2][j2]
                                V4[j * 4 + r1 * 2 + r2 // 7 if False else j * 4 + (r1 * 7 + r2) // 4 if False else 0][r] = 0  # Placeholder
                                
        # Let me redo this more carefully
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1 * 7 + r2
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                k = j1 * 2 + j2  # For 2x2, k = j in the second dimension
                                
                                U4[i * 4 + j][r] = U2[i1][j1] * U2[i2][j2]
                                # V index: for 2x2x2, V is indexed by (j, k) where j is input col, k is output col
                                # In 4x4x4, V is indexed by (j, k) where j in [0,4), k in [0,4)
                                # j = j1*2 + j2, k = j1*2 + j2 (since it's square)
                                # Wait, for matrix multiplication A*B where A is 4x4 and B is 4x4:
                                # C[i,k] = sum_j A[i,j] * B[j,k]
                                # So V should be indexed by (j, k)
                                j_v = j1 * 2 + j2
                                k_v = j1 * 2 + j2  # This is wrong, k should be independent
                                
        # I'm making this too complicated. Let me use a different approach.
        # The 4x4x4 matrix multiplication tensor can be written as the tensor product
        # of two 2x2x2 matrix multiplication tensors.
        # T_4[i,j,k] = T_2[i1,j1,k1] * T_2[i2,j2,k2] where i=i1*2+i2, j=j1*2+j2, k=k1*2+k2
        # But for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
        # So T_4[i,j,k] = 1 = T_2[i1,j1,k1] * T_2[i2,j2,k2] only if both are 1
        # This works because T_2[i,j,k] = 1 for all valid i,j,k in 2x2x2
        
        # The decomposition: if T_2 = sum_r u2_r ⊗ v2_r ⊗ w2_r
        # Then T_4 = sum_{r1,r2} (u2_{r1} ⊗ u2_{r2}) ⊗ (v2_{r1} ⊗ v2_{r2}) ⊗ (w2_{r1} ⊗ w2_{r2})
        
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for i2 in range(2):
                        i = i1 * 2 + i2
                        U4[i * 4 + (i1 * 2 + i2)][r] = U2[i1][r1] * U2[i2][r2]  # This is wrong indexing
        
        # Let me be very explicit
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            for i2 in range(2):
                                for j2 in range(2):
                                    for k2 in range(2):
                                        i = i1 * 2 + i2
                                        j = j1 * 2 + j2
                                        k = k1 * 2 + k2
                                        # The tensor entry is T[i,j,k] = 1
                                        # The decomposition gives: sum_{r1,r2} U4[i,j,r1,r2] * V4[j,k,r1,r2] * W4[k,i,r1,r2]
                                        # We need this to equal 1
                                        # Using Kronecker: U4[i,j,r] = U2[i1,j1,r1] * U2[i2,j2,r2]
                                        # V4[j,k,r] = V2[j1,k1,r1] * V2[j2,k2,r2]
                                        # W4[k,i,r] = W2[k1,i1,r1] * W2[k2,i2,r2]
                                        
                                        # But U2, V2, W2 are for 2x2x2 where:
                                        # sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] = 1 for all i,j,k in {0,1}
                                        
                                        # So the product gives:
                                        # sum_{r1,r2} [U2[i1,j1,r1]*U2[i2,j2,r2]] * [V2[j1,k1,r1]*V2[j2,k2,r2]] * [W2[k1,i1,r1]*W2[k2,i2,r2]]
                                        # = (sum_{r1} U2[i1,j1,r1]*V2[j1,k1,r1]*W2[k1,i1,r1]) * (sum_{r2} U2[i2,j2,r2]*V2[j2,k2,r2]*W2[k2,i2,r2])
                                        # = 1 * 1 = 1
                                        # This is correct!
                                        
                                        # Now fill in the values
                                        U4[i * 4 + j][r] = U2[i1][j1] * U2[i2][j2]
                                        V4[j * 4 + k][r] = V2[j1][k1] * V2[j2][k2]
                                        W4[k * 4 + i][r] = W2[k1][i1] * W2[k2][i2]
        
        return U4, V4, W4
    
    # Build the full 4x4x9 decomposition
    U4_4x4, V4_4x4, W4_4x4 = _kron_strassen_4x4x4()
    
    # Verify the 4x4x4 part
    if not _verify_decomposition(U4_4x4, V4_4x4, W4_4x4, 4, 4, 4):
        # Fall back to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Now add the 4x4x5 schoolbook part
    # The 4x4x9 tensor: C[i,k] = sum_j A[i,j] * B[j,k] for k in [0,9)
    # Split k into k in [0,4) and k in [4,9)
    # For k in [0,4): use Strassen (rank 49)
    # For k in [4,9): use schoolbook (rank 4*4*5 = 80)
    
    rank_strassen = 49
    rank_schoolbook = 4 * 4 * 5
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Copy Strassen part (for k in [0,4))
    for i in range(16):
        for r in range(rank_strassen):
            U[i][r] = U4_4x4[i][r]
    
    for j in range(16):
        for r in range(rank_strassen):
            # V is indexed by (j, k) where j in [0,4), k in [0,9)
            # For the Strassen part, k is in [0,4)
            # V[j, k] for k in [0,4) comes from V4_4x4[j, k]
            # But V4_4x4 is for 4x4x4, so V4_4x4[j, k] for j in [0,4), k in [0,4)
            # In our V matrix, row index is j*9 + k
            for k in range(4):
                V[j * 9 + k][r] = V4_4x4[j][k]
    
    for k in range(4):
        for i in range(4):
            for r in range(rank_strassen):
                W[k * 4 + i][r] = W4_4x4[k * 4 + i][r]
    
    # Add schoolbook part (for k in [4,9))
    col = rank_strassen
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U[i * 4 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    # Verify the full decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
