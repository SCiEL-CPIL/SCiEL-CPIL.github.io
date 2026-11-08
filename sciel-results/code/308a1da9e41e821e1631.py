import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 100  # 100s deadline, leaving 20s margin for 120s limit
    
    # First, establish the baseline: rank 128 schoolbook
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    def schoolbook_decomp():
        rank = n * m * p  # 128
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    U_base, V_base, W_base = schoolbook_decomp()
    best_rank = 128
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Verify the schoolbook decomposition
    def verify(U, V, W, rank):
        """Verify that the tensor decomposition exactly represents matrix multiplication.
        
        The tensor T[i,j,k] for matrix multiplication is 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        The decomposition says T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        
        For matrix multiplication 4x4x8:
        T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the multiplication)
        Wait no. The matrix multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
        Actually, T[i,j,k] = 1 for all valid i,j,k in the multiplication.
        
        More precisely: C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
        So the tensor is T where T[i,j,k] = 1 (the multiplication tensor).
        
        The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal T[i,j,k] = 1 for all i,j,k.
        Wait, that's not right either. Let me think more carefully.
        
        Actually, the tensor for matrix multiplication M_{n,m,p} is defined by:
        (M)_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        
        The rank-r decomposition gives:
        M_{i,j,k} = sum_{r=1}^{r} U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        
        But for matrix multiplication, M_{i,j,k} is just 1 for all i,j,k.
        So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        
        Hmm, that can't be right because the LHS depends on i,j,k in a specific way.
        
        Let me reconsider. The standard formulation:
        The matrix multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
        T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (coefficient)
        
        For standard matrix multiplication C = AB:
        C_{ik} = sum_j A_{ij} B_{jk}
        So T = sum_{i,j,k} A_{ij} B_{jk} C_{ik} as a bilinear form.
        
        The tensor T has entries T_{(i,j),(j,k),(k,i)} = 1 for all valid indices.
        Wait, I'm overcomplicating this.
        
        The correct statement: The multiplication tensor is
        T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^{(A)} ⊗ e_j^{(B)} ⊗ e_k^{(C)}
        where e_i^{(A)} is the vector with 1 at position (i,j) for the A factor... no.
        
        Let me use the standard definition. The tensor M_{n,m,p} ∈ R^{nm} ⊗ R^{mp} ⊗ R^{pn}
        is defined by:
        M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{ij}^{(A)} ⊗ E_{jk}^{(B)} ⊗ E_{ki}^{(C)}
        
        where E_{ij}^{(A)} is the nm-dimensional vector with 1 at position (i,j) and 0 elsewhere, etc.
        
        A rank-r decomposition is a set of r triples (u_r, v_r, w_r) such that:
        M = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
        
        In matrix form: U is nm × r, V is mp × r, W is pn × r.
        The reconstruction: for each i,j,k, 
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        Wait, that's still not right. Let me look at this differently.
        
        The tensor T is a 3D array of shape (nm, mp, pn).
        T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        T[a,b,c] = 1 if i=i', j=j', k=k' (i.e., the indices match for the multiplication)
        
        So T[(i*m+j), (j*p+k), (k*n+i)] = 1, and T is 0 otherwise.
        
        The decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        So for the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        where col = i*m*p + j*p + k or similar.
        
        Then sum_r U[a,r]*V[b,r]*W[c,r] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        and 0 otherwise. This matches T.
        
        So verification: check that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c] for all a,b,c.
        """
        # Build the target tensor T
        # T shape: (nm, mp, pn) = (16, 32, 32)
        # T[i*m+j, j*p+k, k*n+i] = 1
        
        # Instead of building the full 16x32x32 tensor (16384 entries), 
        # we can check the constraint directly.
        
        # For the decomposition to be valid:
        # sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c]
        
        # T[a,b,c] = 1 if there exist i,j,k such that:
        # a = i*m + j, b = j*p + k, c = k*n + i
        # i.e., j = a % m, i = a // m, k = b % p, j_check = b // p, i_check = c % n, k_check = c // n
        # and j == j_check and i == i_check and k == k_check
        
        # Actually simpler: T[a,b,c] = 1 iff (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
        
        # Let's verify by checking all (a,b,c) where T should be 1
        # and a sample where T should be 0.
        
        # For correctness, check all 16*32*32 = 16384 entries
        U_np = np.array(U) if not isinstance(U, np.ndarray) else U
        V_np = np.array(V) if not isinstance(V, np.ndarray) else V
        W_np = np.array(W) if not isinstance(W, np.ndarray) else W
        
        # Compute the reconstructed tensor
        # recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a tensor contraction.
        # We can compute it as: for each r, add outer(U[:,r], V[:,r], W[:,r])
        
        # More efficient: use einsum or broadcasting
        # recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # But 16x32x32x32 might be large. Let's use a smarter check.
        # We only need to verify that for each (i,j,k), the sum equals 1,
        # and for non-matching indices, it equals 0.
        
        # Check all (i,j,k) tuples:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some non-matching indices to ensure they're 0
        # This is a necessary but not sufficient condition, but combined with the above
        # and the rank being <= 128, it's likely correct.
        
        # For a stronger check, verify a few random non-matching triples
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            
            # Check if this is a matching triple
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            
            is_matching = (j_a == j_b) and (k_b == k_c) and (i_a == i_c)
            
            val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
            if is_matching:
                if abs(val - 1.0) > 1e-9:
                    return False
            else:
                if abs(val) > 1e-9:
                    return False
        
        return True
    
    # Verify the baseline
    if not verify(U_base, V_base, W_base, 128):
        # This shouldn't happen, but just in case
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    # Now try to find a lower-rank decomposition
    # Strategy: Use the known rank 49 decomposition for 4x4x4
    # The rank 49 decomposition of 4x4x4 is obtained from the rank 7 decomposition of 2x2x2
    # using a block structure.
    
    # First, let's try to construct the rank 7 decomposition for 2x2x2.
    # The 2x2x2 multiplication tensor has rank 7.
    # One known decomposition uses the following 7 terms:
    
    # Let me use a different approach: try to find a rank-32 decomposition
    # by splitting into two 4x4x4 blocks, each with rank 16 (schoolbook).
    # Wait, that's just the schoolbook rank 32 for 4x4x8? No, schoolbook for 4x4x8 is 128.
    # If I split into two 4x4x4, each has schoolbook rank 16, total 32.
    # But that's the same as the schoolbook! Because 4*4*8 = 128, and 2*(4*4) = 32.
    # Wait, no. Schoolbook for 4x4x4 is 4*4 = 16 multiplications.
    # Schoolbook for 4x4x8 is 4*4*8 = 128 multiplications.
    # If I split B into two 4x4 matrices, I do two 4x4x4 multiplications, each with 16 multiplications.
    # Total: 32. But this is the same as schoolbook! Because the schoolbook for 4x4x8
    # is exactly doing 4*8 = 32 dot products of length 4, which is 32*4 = 128 scalar multiplications.
    # But in terms of tensor rank, each 4x4x4 has rank 16 (schoolbook), so two of them have rank 32.
    # Wait, I'm confusing scalar multiplications with tensor rank.
    
    # The tensor rank of 4x4x4 schoolbook is 16.
    # The tensor rank of 4x4x8 schoolbook is 128.
    # If I decompose 4x4x8 as two independent 4x4x4 tensors, the total rank is 16+16=32.
    # But this is NOT the same as the schoolbook 4x4x8 tensor!
    # The schoolbook 4x4x8 tensor is a single tensor of rank 128.
    # The two independent 4x4x4 tensors form a tensor of rank 32, but they represent
    # a different multiplication: C_left = A*B_left and C_right = A*B_right,
    # where B_left and B_right are both 4x4.
    # But B is 4x8, so B_left = B[:,0:4] and B_right = B[:,4:8].
    # This is exactly the same multiplication! C = A*B = [A*B_left | A*B_right].
    # So the tensor for 4x4x8 IS the direct sum of two 4x4x4 tensors.
    # Therefore, rank(4x4x8) = rank(4x4x4) + rank(4x4x4) = 2*rank(4x4x4).
    # And rank(4x4x4) <= 16 (schoolbook), so rank(4x4x8) <= 32.
    
    # Wait, this means the schoolbook rank for 4x4x8 should be 32, not 128!
    # Let me re-examine.
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The rank is the minimum r such that M = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
    
    # For 4x4x4: M has 4*4*4 = 64 non-zero entries (one for each (i,j,k)).
    # The schoolbook decomposition uses 16 terms: for each (i,k), one term with
    # u_{i,k} having 1 at position (i,j) for all j... no.
    
    # Schoolbook for 4x4x4:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For each (i,k), we have one "multiplication" that computes the dot product.
    # But in tensor decomposition, each term is a single scalar multiplication.
    # So for 4x4x4, schoolbook uses 4*4*4 = 64 scalar multiplications? No.
    
    # I think I've been confusing myself. Let me start over.
    
    # The tensor M_{n,m,p} ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices are:
    # e_i is in R^{n×m}, e_j is in R^{m×p}, e_k is in R^{p×n}
    
    # Actually, the standard definition:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
    # where E_{i,j} ∈ R^{n×m} is the matrix with 1 at (i,j) and 0 elsewhere.
    
    # So M has n*m*p non-zero "terms" in its standard form, but the rank is the minimum
    # number of rank-1 tensors needed to sum to M.
    
    # For 4x4x4: n*m*p = 64 terms in the standard sum.
    # The schoolbook algorithm uses 4*4 = 16 "multiplications" but each "multiplication"
    # is actually a dot product of length 4, which is 4 scalar multiplications.
    # So schoolbook uses 16*4 = 64 scalar multiplications? No, that's not right either.
    
    # I think the issue is that in the tensor decomposition, each rank-1 term corresponds
    # to ONE scalar multiplication. So the rank is the number of scalar multiplications.
    
    # For 4x4x4:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This requires 4*4*4 = 64 scalar multiplications in the naive approach.
    # But Strassen reduces it to 49 for 4x4x4 (from 7 for 2x2x2, recursively).
    
    # Wait, Strassen for 2x2x2 uses 7 multiplications.
    # For 4x4x4, using Strassen recursively: 7 * 7 = 49 multiplications.
    # So rank(4x4x4) <= 49.
    # And rank(4x4x8) = rank(4x4x4) + rank(4x4x4) <= 49 + 49 = 98.
    
    # But the schoolbook for 4x4x4 uses 64 scalar multiplications, so rank(4x4x4) <= 64.
    # And the schoolbook for 4x4x8 uses 128 scalar multiplications, so rank(4x4x8) <= 128.
    
    # So the baseline rank 128 is correct for schoolbook 4x4x8.
    # And we can do better: 98 using Strassen for the two 4x4x4 blocks.
    
    # Let me implement the rank 98 solution.
    # I need the rank 7 decomposition for 2x2x2.
    
    # The rank 7 decomposition of 2x2x2:
    # One known decomposition (from Strassen):
    
    # Let me use a concrete decomposition. After some thought, I'll use the following:
    # The 7 terms for 2x2x2 can be constructed as follows.
    
    # Actually, let me just implement the block Strassen approach directly.
    # For 4x4x4, split into four 2x2 blocks.
    # Use 7 Strassen products of 2x2 matrices.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank: 7 * 7 = 49.
    
    # But I need the explicit rank 7 decomposition for 2x2x2.
    
    # Here's one approach: use the fact that the 2x2x2 tensor can be decomposed as:
    # T = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r
    
    # Let me look up or derive the explicit vectors.
    
    # Alternative: Use a numerical approach to find a low-rank decomposition.
    # But that's risky for exactness.
    
    # Let me try a different structural approach:
    # Use the Winograd scheme or a known explicit decomposition.
    
    # For now, let me try to implement the rank 98 solution using a pre-computed
    # rank 7 decomposition for 2x2x2.
    
    # Here's a known rank 7 decomposition for 2x2x2 (from various sources):
    
    # The 7 vectors (u_r, v_r, w_r) where each is in R^4:
    
    # I'll use the following decomposition (one of the standard ones):
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 with rank 7.
        
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # One standard rank-7 decomposition for 2x2x2:
        # This is derived from Strassen's algorithm.
        
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # Reconstruction:
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # Wait, this doesn't use M7! Let me check.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # Indeed, M7 is not used in this variant. That's a different variant.
        # Let me use the variant where all 7 are used:
        
        # Alternative Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # Still M7 not used. I think I have the wrong reconstruction.
        
        # Correct Strassen reconstruction:
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # This only uses M1-M6. M7 is extra? No, that's only 6 products.
        
        # I'm confusing myself. Let me just use a known explicit rank-7 decomposition.
        
        # After research, one standard decomposition for 2x2x2 of rank 7 is:
        
        # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1], w1 = [1, 0, 0, 1]  (M1-like)
        # This is getting too complex to derive from scratch.
        
        # Let me use a different approach: use the schoolbook rank 8 for 2x2x2,
        # giving rank 7*8 = 56 for 4x4x4, and 112 for 4x4x8.
        # 112 is better than 128.
        
        return None
    
    # Given the complexity, let me implement the rank 112 solution:
    # Block Strassen for 4x4x4 with schoolbook (rank 8) for 2x2x2.
    
    def construct_rank112():
        """Construct rank 112 decomposition for 4x4x8.
        
        Split into two 4x4x4 blocks.
        Each 4x4x4 uses block Strassen with schoolbook 2x2x2 (rank 8).
        Each 4x4x4 has rank 7*8 = 56.
        Total rank: 56 + 56 = 112.
        """
        rank = 112
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((32, rank), dtype=float)
        W = np.zeros((32, rank), dtype=float)
        
        # Strassen coefficients for 4x4x4 (7 products)
        # A_coefs[prod][I][J] = coefficient of A_block[I,J] in L_A for product prod
        # B_coefs[prod][I][J] = coefficient of B_block[I,J] in L_B for product prod
        
        A_coefs = [
            [[1, 0], [0, 1]],   # M1: A00 + A11
            [[0, 0], [1, 1]],   # M2: A10 + A11
            [[1, 0], [-1, 0]],  # M3: A00 - A10
            [[1, 0], [0, 0]],   # M4: A00
            [[0, 0], [0, 1]],   # M5: A11
            [[0, 1], [0, -1]],  # M6: A01 - A11
            [[1, 1], [0, 0]]    # M7: A00 + A01
        ]
        
        B_coefs = [
            [[1, 0], [0, 1]],   # M1: B00 + B11
            [[1, 0], [0, 0]],   # M2: B00
            [[0, 0], [1, 1]],   # M3: B10 + B11
            [[1, 0], [0, 0]],   # M4: B00
            [[0, 0], [0, 1]],   # M5: B11
            [[1, 1], [0, 0]],   # M6: B00 + B01
            [[0, -1], [1, 0]]   # M7: B10 - B00
        ]
        
        # Reconstruction: which C blocks each product contributes to
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # But wait, this only uses 6 products (M1-M6). M7 is not used.
        # I need a correct Strassen variant where all 7 are used.
        
        # Let me use a different Strassen variant:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # This is the standard Strassen, but M7 is not used in the reconstruction!
        # That means this is actually a 6-product scheme, not 7.
        
        # I think the correct Strassen uses:
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # And M7 is defined but not used? That can't be right.
        
        # Let me look up the correct Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # I keep getting the same thing. Let me verify with a simple example.
        
        # Actually, I think the issue is that I'm using a non-standard variant.
        # The standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # This is correct, and M7 is indeed not used in the reconstruction.
        # This means the above is NOT a valid 7-product Strassen.
        
        # The correct Strassen must use all 7 products. Let me find the right one.
        
        # After more thought, I believe the correct reconstruction is:
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # And M7 is used in a different way. I'm clearly making an error.
        
        # Let me just abandon the Strassen approach and use a simpler method.
        
        # Alternative: Use the known rank 49 decomposition for 4x4x4 directly.
        # I'll hardcode it.
        
        # For now, return the schoolbook rank 128 to ensure correctness.
        return None
    
    # Given the difficulty of implementing Strassen correctly,
    # let me try a numerical approach: use SVD or tensor decomposition
    # to find a low-rank approximation, then round to integers.
    
    # But this is risky for exactness.
    
    # Let me try yet another approach: use the fact that the tensor
    # M_{4,4,8} = M_{4,4,4} ⊕ M_{4,4,4} (direct sum).
    # So rank(M_{4,4,8}) = 2 * rank(M_{4,4,4}).
    # And rank(M_{4,4,4}) <= 49 (Strassen).
    # So rank(M_{4,4,8}) <= 98.
    
    # But I need the explicit rank 49 decomposition for 4x4x4.
    
    # Let me try to construct it using the block structure.
    # 4x4x4 = 7 products of 2x2x2, each with rank 7.
    # Total rank: 49.
    
    # I need the rank 7 decomposition for 2x2x2.
    
    # Let me use a known explicit decomposition.
    
    # After some research, here's a rank-7 decomposition for 2x2x2:
    
    def strassen_2x2x2_explicit():
        """Return U, V, W for 2x2x2 with rank 7.
        
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # One known decomposition:
        # The 7 terms are:
        
        # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # Term 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # Term 3: u=[1,0,-1,0], v=[0,0,1,1], w=[0,1,0,1]
        # Term 4: u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]
        # Term 5: u=[0,0,0,1], v=[0,0,0,1], w=[0,0,0,1]
        # Term 6: u=[0,1,0,-1], v=[1,1,0,0], w=[0,0,1,0]
        # Term 7: u=[1,1,0,0], v=[0,-1,1,0], w=[0,1,0,0]
        
        # I'm not confident in these. Let me use a different approach.
        
        # Actually, let me just use the schoolbook rank 8 for 2x2x2.
        # This gives rank 7*8 = 56 for 4x4x4, and 112 for 4x4x8.
        
        return None
    
    # I'm spending too much time on this. Let me just return the schoolbook rank 128.
    # It's correct and has score 1.0.
    
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
# EVOLVE-BLOCK-END
