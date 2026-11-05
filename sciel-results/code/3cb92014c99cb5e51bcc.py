"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition via dependency elimination."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    def idx_A(i, j):
        return i * m + j

    def idx_B(j, k):
        return j * p + k

    def idx_C(i, k):
        return k * n + i  # W[k*n+i] per convention

    # Build schoolbook terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((idx_A(i, j), idx_B(j, k), idx_C(i, k), 1))

    # Each term is a rank-1 tensor: e_u ⊗ e_v ⊗ e_w
    # Flatten to a vector of length nm*mp*pn = 20*25*20 = 10000
    # The multiplication tensor T has T[u,v,w] = 1 iff (u,v,w) corresponds to a valid
    # (i,j,k) with u=idx_A(i,j), v=idx_B(j,k), w=idx_C(i,k)
    # Actually, the tensor we want to decompose is:
    # T[u,v,w] = 1 if there exists (i,j,k) such that u=idx_A(i,j), v=idx_B(j,k), w=idx_C(i,k)
    # Since each (i,j,k) maps uniquely, T is just the sum of 100 rank-1 tensors.
    # But we can also express T as a linear combination of fewer rank-1 tensors if they're dependent.
    
    # The tensor T as a 3D array:
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for (u, v, w, c) in terms:
        T[u, v, w] += c

    # Flatten T
    T_flat = T.flatten()  # length 10000

    # Each schoolbook term t_r = e_{u_r} ⊗ e_{v_r} ⊗ e_{w_r}
    # Its flat representation is a vector with 1 at position (u_r * mp * pn + v_r * pn + w_r)
    # We want to find the minimum subset of these 100 vectors that still spans T_flat.
    # This is the "minimum covering" problem, but since each vector has exactly one nonzero,
    # and T_flat has exactly 100 nonzeros (each = 1), the only way to represent T_flat
    # as a sum of rank-1 tensors is to use all 100 (since each rank-1 tensor has exactly
    # one nonzero position, and T has 100 distinct nonzero positions).
    
    # Wait - that means schoolbook IS minimal for this specific tensor structure!
    # Because each rank-1 term e_u ⊗ e_v ⊗ e_w has exactly one nonzero entry,
    # and T has 100 nonzero entries at distinct positions, we need at least 100 terms.
    
    # BUT: we can use rank-1 tensors with MULTIPLE nonzeros if we allow the U, V, W
    # matrices to have entries other than 0/1. A rank-1 term is U[:,r] ⊗ V[:,r] ⊗ W[:,r],
    # which can have many nonzeros if U[:,r], V[:,r], W[:,r] have multiple nonzeros.
    
    # So the question is: can we find rank-1 tensors (with non-binary U,V,W columns)
    # that sum to T with fewer than 100 terms?
    
    # This is the tensor rank problem. For 4x5x5 matmul, the known lower bound is 100
    # (since the "border rank" equals the rank for matmul tensors in general, and the
    # support size gives a lower bound). Actually, the support size argument:
    # T has 100 nonzero entries. A rank-1 tensor U[:,r]⊗V[:,r]⊗W[:,r] can have at most
    # (nnz(U[:,r]))*(nnz(V[:,r]))*(nnz(W[:,r])) nonzeros. But more importantly,
    # the rank of a tensor is at least the maximum over all "slices" of the rank of that slice.
    
    # For matmul tensor M_{n,m,p}: T[i,j,k] = δ_{i+k, j} (in some indexing).
    # The rank of the n×m×p matmul tensor is known to be at least n*m*p / max(n,m,p) in some cases,
    # but the exact rank is unknown for most sizes.
    
    # For 4×5×5: the trivial upper bound is 100. Lower bounds from various methods
    # (e.g., support size, slice rank) may give less than 100.
    
    # Slice rank lower bound: For a 3-tensor of size a×b×c, the slice rank is at least
    # abc / max(a,b,c). For 4×5×5: 100/5 = 20. That's a weak bound.
    
    # A better approach: use the fact that matmul tensor T satisfies:
    # For any fixed i, the slice T[i,:,:] is the m×p matrix with T[i,j,k] = 1 iff j=k (identity-like)
    # Wait, let me re-derive. T[u,v,w] where u=idx_A(i,j)=i*m+j, v=idx_B(j,k)=j*p+k, w=idx_C(i,k)=k*n+i.
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # For fixed i: T[i*m+j, j*p+k, k*n+i] = 1. The w index is k*n+i, which varies with k.
    # This is complex. Let me just try a numerical approach.
    
    # Strategy: Try to find a low-rank decomposition numerically.
    # We'll use an iterative approach: start with random rank-r decomposition and try to
    # minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||.
    
    # But first, let's check: is rank 100 actually achievable with a better structure?
    # The answer from tensor decomposition literature: the rank of the 4×5×5 matmul tensor
    # is exactly 100 (it's not known to be less, and for "thin" matmul tensors, the rank
    # often equals the number of nonzero entries).
    
    # However, let's try a different structural approach: use the block structure.
    # 4×5×5 = (2+2)×(2+3)×(2+3). We can use a block decomposition.
    
    # Actually, let me try the most promising approach: use Strassen for the 4×4×4 sub-problem
    # (which reduces 64 terms to 49) and handle the remaining 36 terms optimally.
    # The remaining terms involve A[0:4,4:5], A[2:4,0:5], etc.
    
    # Let me be more careful. The 4×5×5 multiplication:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k] for i=0..3, k=0..4
    # = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    # The first part is 4×4×5 (A[0:4,0:4] * B[0:4,0:5] -> C[0:4,0:5])
    # The second part is 4×1×5 (A[0:4,4] * B[4,0:5] -> C[0:4,0:5])
    
    # For 4×4×5: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # This is like multiplying a 4×4 matrix by a 4×5 matrix.
    # We can view B as two 4×3 and 4×2 blocks? No, 4×5 = 4×(3+2) or 4×(2+3).
    # Let's do B = [B_left(4×3) | B_right(4×2)]
    # Then C = A * B_left (4×3) + A * B_right (4×2)
    
    # For A(4×4) * B_left(4×3): rank ≤ 4*4*3 = 48, but we can use Strassen-like methods.
    # Actually, 4×4×3 matmul: we can split into 2×2 blocks in A and B.
    # A = [[A00, A01], [A10, A11]] where each is 2×2
    # B_left = [[B00, B01, B02], [B10, B11, B12]] where B00,B10 are 2×1, etc.
    # This gets complicated.
    
    # Let me try a completely different approach: use the known result that
    # rank(4×5×5 matmul) might be reducible. Let me just implement the best known
    # decomposition and see what rank we get.
    
    # Best approach: Use Strassen on the 4×4×4 core (rank 49) + handle the 5th column/row.
    # 4×5×5 = 4×4×5 + 4×1×5 (splitting B into B[:,0:4] and B[:,4])
    # 4×4×5 = 4×4×4 + 4×4×1 (splitting B into B[:,0:4] and B[:,4])
    # Wait, that's circular. Let me think differently.
    
    # 4×5×5: A is 4×5, B is 5×5, C is 4×5.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # Split: C = A[:,:4] * B[:4,:] + A[:,4:5] * B[4:5,:]
    # First term: 4×4×5 matmul (A1 is 4×4, B1 is 4×5)
    # Second term: 4×1×5 matmul (A2 is 4×1, B2 is 1×5) - this is just 4*5=20 terms (outer product)
    
    # For 4×4×5: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Split B into B[:,0:4] (4×4) and B[:,4:5] (4×1):
    # C[:,0:4] = A[:,:4] * B[:4,:4] (4×4×4, rank 49 via Strassen)
    # C[:,4:5] = A[:,:4] * B[:4,4:5] (4×4×1, rank 16 via schoolbook)
    
    # Total: 49 + 16 + 20 = 85 terms.
    
    # Let's implement this!
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Clear and rebuild
    terms = []
    
    # Part 1: 4×4×4 core using Strassen (rank 49)
    # A_core = A[0:4, 0:4], B_core = B[0:4, 0:4], C_core = C[0:4, 0:4]
    # Using two-level Strassen: 7 multiplications of 2×2×2, each with Strassen (rank 7)
    
    # Strassen 2×2×2 (rank 7):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 - M4 + M6 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 + M7
    
    def strassen_2x2(A_u, B_v, C_w):
        """Add 7 rank-1 terms for 2x2x2 Strassen.
        A_u[i][j] = U index for A[i,j]
        B_v[j][k] = V index for B[j,k]  
        C_w[i][k] = W index for C[i,k]
        """
        # We need to create intermediate "virtual" indices for the Strassen combinations.
        # But we can't create new U/V/W rows - we must express everything in terms of original indices.
        # 
        # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # Each of these is a scalar product contributing to c00 and c11.
        # 
        # The issue: in the tensor decomposition, each rank-1 term is U[:,r]⊗V[:,r]⊗W[:,r].
        # For M1, the "multiplication" (a00+a11)*(b00+b11) produces a 2x2 result.
        # To express this as rank-1 terms, we need to be careful.
        #
        # Actually, the standard way: each "multiplication" in Strassen at the 2x2 level
        # is itself a 2x2x2 tensor that needs rank 7 (or 8 for schoolbook).
        # But we're at the scalar level! The 2x2 matrices are already decomposed into scalars.
        #
        # Let me reconsider. At the top level, we have 4x4x4. We split into 2x2 blocks.
        # Each "block multiplication" like (A00+A11)*(B00+B11) is a 2x2*2x2 product.
        # To compute this 2x2*2x2 product, we need rank 7 (Strassen) or 8 (schoolbook) rank-1 terms.
        # Each such rank-1 term involves specific entries of A00, A11, B00, B11.
        #
        # For the 2x2*2x2 Strassen, the 7 multiplications are scalar multiplications:
        # m1 = (A00[0,0]+A11[0,0]+A00[1,1]+A11[1,1]) * (B00[0,0]+B11[0,0]+B00[1,1]+B11[1,1])
        # Wait no. In 2x2 Strassen, the "matrices" are 2x2, and the 7 multiplications are of 2x2 matrices.
        # If we're already at the 2x2 level and want to go to 1x1, we use Strassen again.
        #
        # Let me just use schoolbook for the inner 2x2x2 (rank 8) to keep it simple.
        # 7 * 8 = 56 for the 4x4x4 core.
        # Total: 56 + 16 + 20 = 92.
        
        # Actually, let me implement two-level Strassen properly.
        # Level 1: 4x4x4 -> 7 products of 2x2x2
        # Level 2: each 2x2x2 -> 7 products of 1x1x1 (scalar multiplications)
        # Total: 7*7 = 49 scalar multiplications.
        #
        # Each scalar multiplication m = x * y contributes one rank-1 term.
        # The rank-1 term has U, V, W vectors that are linear combinations of basis vectors.
        #
        # For the full decomposition, we collect all 49 scalar multiplications and
        # determine their U, V, W coefficient vectors.
        
        pass  # Will implement below
    
    # Let me implement the two-level Strassen by tracking coefficients.
    # Each scalar multiplication in the final decomposition is of the form:
    # (sum of A entries with coefficients) * (sum of B entries with coefficients)
    # contributing to (sum of C entries with coefficients)
    
    # I'll represent the decomposition as a list of (u_coeffs, v_coeffs, w_coeffs)
    # where u_coeffs is a dict {A_index: coeff}, etc.
    
    def make_strassen_decomposition():
        """Build the 4x4x4 Strassen decomposition as coefficient vectors."""
        # Returns list of (u_dict, v_dict, w_dict)
        # where u_dict maps A[i,j] index -> coefficient, etc.
        
        # Strassen 4x4x4:
        # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
        # Split B into 2x2 blocks: B00, B01, B10, B11 (each 2x2)
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 + M7
        # where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mk is a 2x2x2 product. We'll use Strassen for each.
        # A 2x2x2 Strassen has 7 scalar multiplications.
        # Each scalar mult is (linear combo of 4 A-entries) * (linear combo of 4 B-entries)
        # contributing to (linear combo of 4 C-entries).
        
        # For a 2x2 matrix X with entries X[0,0], X[0,1], X[1,0], X[1,1]:
        # Strassen scalar multiplications:
        # s1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
        # s2 = (X[1,0]+X[1,1]) * Y[0,0]
        # s3 = X[0,0] * (Y[0,1]-Y[1,1])
        # s4 = X[1,1] * (Y[1,0]-Y[0,0])
        # s5 = (X[0,0]+X[0,1]) * Y[1,1]
        # s6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
        # s7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
        # Result:
        # Z[0,0] = s1 - s4 + s6 - s7
        # Z[0,1] = s3 + s5
        # Z[1,0] = s2 + s4
        # Z[1,1] = s1 - s3 + s2 + s7
        
        # For Mk = X * Y where X and Y are 2x2 blocks (each entry is a linear combo of A or B entries):
        # The 7 scalar multiplications each produce a term.
        # The result Z is a 2x2 matrix that gets added to appropriate C blocks.
        
        # Let me represent each 2x2 block as a 2x2 array of coefficient dicts.
        # A00[i][j] = {(idx_A(0+i, 0+j): 1}
        # etc.
        
        def block_A(r, c):
            """2x2 block of A starting at row r, col c. Returns 2x2 list of dicts."""
            return [[{idx_A(r+i, c+j): 1} for j in range(2)] for i in range(2)]
        
        def block_B(r, c):
            """2x2 block of B starting at row r, col c."""
            return [[{idx_B(r+i, c+j): 1} for j in range(2)] for i in range(2)]
        
        def block_C(r, c):
            """2x2 block of C starting at row r, col c."""
            return [[{idx_C(r+i, c+j): 1} for j in range(2)] for i in range(2)]
        
        def add_dicts(d1, d2, sign=1):
            """Add sign*d2 to d1, return new dict."""
            result = dict(d1)
            for k, v in d2.items():
                result[k] = result.get(k, 0) + sign * v
            return result
        
        def scale_dict(d, s):
            return {k: v * s for k, v in d.items()}
        
        def mat_add(X, Y, sign=1):
            """Add sign*Y to X (2x2 matrices of dicts)."""
            return [[add_dicts(X[i][j], Y[i][j], sign) for j in range(2)] for i in range(2)]
        
        def strassen_2x2(X, Y):
            """Compute Strassen decomposition of X*Y where X, Y are 2x2 matrices of dicts.
            Returns list of (u_dict, v_dict, w_dict) for the 7 scalar multiplications.
            """
            # X[i][j] and Y[i][j] are dicts (coefficient vectors)
            # s1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
            x1 = add_dicts(X[0][0], X[1][1])
            y1 = add_dicts(Y[0][0], Y[1][1])
            
            x2 = add_dicts(X[1][0], X[1][1])
            y2 = Y[0][0]
            
            x3 = X[0][0]
            y3 = add_dicts(Y[0][1], Y[1][1], -1)
            
            x4 = X[1][1]
            y4 = add_dicts(Y[1][0], Y[0][0], -1)
            
            x5 = add_dicts(X[0][0], X[0][1])
            y5 = Y[1][1]
            
            x6 = add_dicts(X[1][0], X[0][0], -1)
            y6 = add_dicts(Y[0][0], Y[0][1])
            
            x7 = add_dicts(X[0][1], X[1][1], -1)
            y7 = add_dicts(Y[1][0], Y[1][1])
            
            # Now, the result Z has:
            # Z[0,0] = s1 - s4 + s6 - s7
            # Z[0,1] = s3 + s5
            # Z[1,0] = s2 + s4
            # Z[1,1] = s1 - s3 + s2 + s7
            #
            # Each s_r is a "scalar" that is actually a product of two linear combinations.
            # In the tensor decomposition, s_r * (C contribution) means:
            # The rank-1 term has U = x_r (coeff vector for A), V = y_r (coeff vector for B),
            # and W = (coeff vector for C) where the C coeff is determined by which Z entries s_r contributes to.
            #
            # s1 contributes to Z[0,0] with +1 and Z[1,1] with +1
            # s2 contributes to Z[1,0] with +1 and Z[1,1] with +1
            # s3 contributes to Z[0,1] with +1 and Z[1,1] with -1
            # s4 contributes to Z[0,0] with -1 and Z[1,0] with +1
            # s5 contributes to Z[0,1] with +1
            # s6 contributes to Z[0,0] with +1
            # s7 contributes to Z[0,0] with -1 and Z[1,1] with +1
            
            # For each s_r, the W coefficient vector is the sum of the C-block contributions.
            # But wait - the C block entries are themselves dicts (coefficient vectors for C indices).
            # So W for s_r is a dict that combines the C contributions.
            
            # Actually, I need to be more careful. The result of the 2x2 Strassen is a 2x2 matrix Z
            # where each Z[i][j] is a dict (linear combo of C entries).
            # The 7 scalar multiplications produce "intermediate" values s1..s7.
            # Z[i][j] = sum_r coeff_{r,ij} * s_r
            #
            # In the tensor decomposition, each s_r corresponds to ONE rank-1 term:
            # U[:,r] = x_r (the A coefficient vector)
            # V[:,r] = y_r (the B coefficient vector)
            # W[:,r] = sum_j coeff_{r,ij} * C_block[i][j] (the C coefficient vector)
            
            # Let me compute W for each s_r:
            w1 = {}  # s1 -> Z[0,0] (+1) and Z[1,1] (+1)
            # Z[0,0] and Z[1,1] are dicts. w1 = Z[0,0] + Z[1,1]
            # But Z is the result, not yet computed. Let me think differently.
            
            # The C block for this 2x2x2 product will be added to specific C positions.
            # Let's say the C block is C_block (2x2 matrix of dicts).
            # Then:
            # C_block[0][0] gets contribution s1 - s4 + s6 - s7
            # C_block[0][1] gets contribution s3 + s5
            # C_block[1][0] gets contribution s2 + s4
            # C_block[1][1] gets contribution s1 - s3 + s2 + s7
            #
            # For rank-1 term s_r:
            # W_coeff = sum over (i,j) where s_r appears in C_block[i][j] of sign * C_block[i][j]
            
            # s1: appears in C_block[0][0] (+1) and C_block[1][1] (+1)
            # s2: appears in C_block[1][0] (+1) and C_block[1][1] (+1)
            # s3: appears in C_block[0][1] (+1) and C_block[1][1] (-1)
            # s4: appears in C_block[0][0] (-1) and C_block[1][0] (+1)
            # s5: appears in C_block[0][1] (+1)
            # s6: appears in C_block[0][0] (+1)
            # s7: appears in C_block[0][0] (-1) and C_block[1][1] (+1)
            
            # So W for s_r is a linear combination of the C_block entries.
            # But C_block entries are dicts, so W is also a dict.
            
            # I'll pass C_block as a parameter.
            # Actually, let me restructure: the function returns (x_r, y_r, w_r) for each r,
            # where w_r depends on C_block.
            
            # Let me just return the x and y vectors, and compute w separately.
            return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7)]
        
        # Now build the 4x4x4 Strassen:
        A00 = block_A(0, 0)
        A01 = block_A(0, 2)
        A10 = block_A(2, 0)
        A11 = block_A(2, 2)
        
        B00 = block_B(0, 0)
        B01 = block_B(0, 2)
        B10 = block_B(2, 0)
        B11 = block_B(2, 2)
        
        C00 = block_C(0, 0)
        C01 = block_C(0, 2)
        C10 = block_C(2, 0)
        C11 = block_C(2, 2)
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1) and C11 (+1)
        # Wait, no. In the 4x4 Strassen:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 + M7
        #
        # Each Mk is a 2x2 matrix. So M1 contributes to C00 and C11.
        # Specifically, M1 (a 2x2 matrix) is added to C00 (top-left 2x2 of C)
        # and also to C11 (bottom-right 2x2 of C).
        #
        # For the tensor decomposition, when we compute M1 = X*Y (2x2x2 Strassen),
        # the 7 scalar multiplications each produce a rank-1 term.
        # The W coefficient for each term includes contributions from both C00 and C11.
        
        # Let me define a helper that, given X, Y (2x2 matrices of dicts) and a list of
        # (C_block, sign) pairs, computes the 7 rank-1 terms.
        
        def strassen_terms(X, Y, c_blocks):
            """c_blocks is a list of (C_block_2x2, sign) pairs.
            Returns list of (u_dict, v_dict, w_dict) for 7 terms.
            """
            xy_pairs = strassen_2x2(X, Y)
            
            # W contribution pattern for each s_r:
            # s1: C[0][0] (+1), C[1][1] (+1)
            # s2: C[1][0] (+1), C[1][1] (+1)
            # s3: C[0][1] (+1), C[1][1] (-1)
            # s4: C[0][0] (-1), C[1][0] (+1)
            # s5: C[0][1] (+1)
            # s6: C[0][0] (+1)
            # s7: C[0][0] (-1), C[1][1] (+1)
            
            w_patterns = [
                [(0,0,1), (1,1,1)],   # s1
                [(1,0,1), (1,1,1)],   # s2
                [(0,1,1), (1,1,-1)],  # s3
                [(0,0,-1), (1,0,1)],  # s4
                [(0,1,1)],            # s5
                [(0,0,1)],            # s6
                [(0,0,-1), (1,1,1)],  # s7
            ]
            
            results = []
            for r in range(7):
                x_r, y_r = xy_pairs[r]
                w_r = {}
                for (c_block, sign) in c_blocks:
                    for (ci, cj, w_sign) in w_patterns[r]:
                        # c_block[ci][cj] is a dict
                        for k, v in c_block[ci][cj].items():
                            w_r[k] = w_r.get(k, 0) + sign * w_sign * v
                # Remove zero entries
                w_r = {k: v for k, v in w_r.items() if abs(v) > 1e-15}
                results.append((x_r, y_r, w_r))
            return results
        
        all_terms = []
        
        # M1 = (A00+A11)(B00+B11)
        X1 = mat_add(A00, A11)
        Y1 = mat_add(B00, B11)
        all_terms.extend(strassen_terms(X1, Y1, [(C00, 1), (C11, 1)]))
        
        # M2 = (A10+A11)B00
        X2 = mat_add(A10, A11)
        Y2 = B00
        all_terms.extend(strassen_terms(X2, Y2, [(C10, 1), (C11, 1)]))
        
        # M3 = A00(B01-B11)
        X3 = A00
        Y3 = mat_add(B01, B11, -1)
        all_terms.extend(strassen_terms(X3, Y3, [(C01, 1), (C11, -1)]))
        
        # M4 = A11(B10-B00)
        X4 = A11
        Y4 = mat_add(B10, B00, -1)
        all_terms.extend(strassen_terms(X4, Y4, [(C00, -1), (C10, 1)]))
        
        # M5 = (A00+A01)B11
        X5 = mat_add(A00, A01)
        Y5 = B11
        all_terms.extend(strassen_terms(X5, Y5, [(C01, 1)]))
        
        # M6 = (A10-A00)(B00+B01)
        X6 = mat_add(A10, A00, -1)
        Y6 = mat_add(B00, B01)
        all_terms.extend(strassen_terms(X6, Y6, [(C00, 1)]))
        
        # M7 = (A01-A11)(B10+B11)
        X7 = mat_add(A01, A11, -1)
        Y7 = mat_add(B10, B11)
        all_terms.extend(strassen_terms(X7, Y7, [(C00, -1), (C11, 1)]))
        
        return all_terms
    
    core_terms = make_strassen_decomposition()
    # core_terms has 49 entries (7*7)
    # Each entry is (u_dict, v_dict, w_dict)
    
    # Part 2: 4×4×1 (A[0:4,0:4] * B[0:4,4] -> C[0:4,4])
    # Schoolbook: 4*4*1 = 16 terms
    for i in range(4):
        for j in range(4):
            u_dict = {idx_A(i, j): 1}
            v_dict = {idx_B(j, 4): 1}
            w_dict = {idx_C(i, 4): 1}
            core_terms.append((u_dict, v_dict, w_dict))
    
    # Part 3: 4×1×5 (A[0:4,4] * B[4,0:5] -> C[0:4,0:5])
    # Schoolbook: 4*1*5 = 20 terms
    for i in range(4):
        for k in range(5):
            u_dict = {idx_A(i, 4): 1}
            v_dict = {idx_B(4, k): 1}
            w_dict = {idx_C(i, k): 1}
            core_terms.append((u_dict, v_dict, w_dict))
    
    # Total: 49 + 16 + 20 = 85 terms
    rank = len(core_terms)
    
    # Now convert to U, V, W matrices
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(core_terms):
        for idx, coeff in u_dict.items():
            U[idx, r] = coeff
        for idx, coeff in v_dict.items():
            V[idx, r] = coeff
        for idx, coeff in w_dict.items():
            W[idx, r] = coeff
    
    # Verify correctness
    if time.time() < deadline:
        # Reconstruct T from U, V, W
        T_recon = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[idx_A(i, j), idx_B(j, k), idx_C(i, k)] = 1
        
        if not np.allclose(T_recon, T_target, atol=1e-10):
            # Fallback to schoolbook
            terms = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms.append((idx_A(i, j), idx_B(j, k), idx_C(i, k), 1))
            rank = len(terms)
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
                U[u_idx, r] = coeff
                V[v_idx, r] = coeff
                W[w_idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
