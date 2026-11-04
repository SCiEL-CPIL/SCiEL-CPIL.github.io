"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: split the 4x8 matrix B into two 4x4 blocks.
    # B_left = B[:, 0:4], B_right = B[:, 4:8]
    # C_left = A @ B_left (4x4), C_right = A @ B_right (4x4)
    # Total rank = rank(4x4*4x4) * 2
    #
    # For each 4x4*4x4, use Strassen's algorithm with rank 7.
    # Total rank = 14.
    #
    # However, Strassen requires careful construction of U, V, W matrices.
    # Let me build it properly.
    #
    # For a single 4x4 * 4x4 multiplication using Strassen:
    # We partition A, B into 2x2 blocks of 2x2 matrices:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M is a 2x2 matrix product, which itself can be done with Strassen
    # or schoolbook (rank 4). If we use schoolbook for the 2x2 products:
    # Each M requires 4 scalar multiplications.
    # 7 M's * 4 = 28 scalar multiplications per 4x4*4x4.
    # Total: 28 * 2 = 56 for the full 4x4*4x8.
    #
    # But we can do better. Let me use Strassen recursively on the 2x2 blocks too.
    # Each 2x2*2x2 with Strassen needs rank 3.
    # 7 * 3 = 21 per 4x4*4x4.
    # Total: 21 * 2 = 42.
    #
    # Let me implement the recursive Strassen approach.
    #
    # Actually, let me build the tensor decomposition directly.
    #
    # The matmul tensor T[i,j,k] for 4x4*4x4 has a decomposition of rank 49 (known optimal),
    # but implementing that is very complex. Let me use the Strassen-based rank 42 approach.
    #
    # I'll construct U, V, W by tracing through the Strassen computation graph.
    #
    # For each scalar multiplication in the Strassen recursion, we create one rank-1 term.
    #
    # Let me define a helper that builds the decomposition for 2x2*2x2 using Strassen.
    
    def strassen_2x2_decomp():
        """Return U, V, W lists for 2x2*2x2 Strassen decomposition.
        U has shape (4, rank), V has shape (4, rank), W has shape (4, rank)
        where U[i,j] indexes A[i,j], V[i,j] indexes B[i,j], W[i,j] indexes C[i,j]
        with row-major indexing: idx = i*2+j
        """
        U = []
        V = []
        W = []
        
        # Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a scalar product of two linear combinations.
        # For each M, we need to express it as sum of rank-1 terms.
        # Since each M is a product of two linear forms in A and B respectively,
        # it's already a single rank-1 term in the tensor decomposition.
        
        # M1: (A00+A11) * (B00+B11)
        # U coeffs: A00:1, A11:1, others 0
        # V coeffs: B00:1, B11:1, others 0
        # W coeffs: C00:+1, C11:+1
        U.append([1, 0, 0, 1])  # A00=1, A01=0, A10=0, A11=1
        V.append([1, 0, 0, 1])  # B00=1, B01=0, B10=0, B11=1
        W.append([1, 0, 0, 1])  # C00=1, C01=0, C10=0, C11=1
        
        # M2: (A10+A11) * B00
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 1, 0])  # C10=1
        
        # M3: A00 * (B01-B11)
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        W.append([0, 1, 0, 0])  # C01=1
        
        # M4: A11 * (B10-B00)
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])  # B10=1, B00=-1 -> wait, B10-B00 means B00:-1, B10:1
        # Actually V should be: B00=-1, B10=1
        V.append([0, 0, 1, -1])  # This is wrong order. Let me fix.
        # V indices: B00=0, B01=1, B10=2, B11=3
        # B10 - B00: V[2]=1, V[0]=-1
        V[-1] = [-1, 0, 1, 0]
        W.append([1, 0, 1, 0])  # C00=1, C10=1
        
        # M5: (A00+A01) * B11
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([-1, 1, 0, 0])  # C00=-1, C01=1
        
        # M6: (A10-A00) * (B00+B01)
        U.append([-1, 0, 1, 0])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 0, 1])  # C11=1
        
        # M7: (A01-A11) * (B10+B11)
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([1, 0, 0, 0])  # C00=1
        
        return U, V, W
    
    # Now for 4x4*4x4, partition into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    #
    # Strassen M's are 2x2 matrix products.
    # M1 = (A11+A22) * (B11+B22)
    # etc.
    
    # Let me build the 4x4*4x4 decomposition by composing the 2x2 Strassen.
    # This is getting complex. Let me just use schoolbook for the 2x2 sub-problems.
    # That gives rank 28 per 4x4*4x4, total 56.
    
    # Actually, let me just implement a clean 56-rank decomposition.
    # For each 4x4*4x4 block, use Strassen with schoolbook 2x2 sub-products.
    
    def strassen_4x4_with_schoolbook_2x2():
        """Build U, V, W for 4x4*4x4 using Strassen with schoolbook 2x2.
        U: (16, rank), V: (16, rank), W: (16, rank)
        Row-major: A[i,j] -> index i*4+j
        """
        U = []
        V = []
        W = []
        
        # 7 Strassen M's, each a 2x2*2x2 schoolbook (4 multiplications)
        # Total 28 multiplications.
        
        # For each M, we have 4 scalar multiplications.
        # For each scalar multiplication, we create one rank-1 term.
        
        # M1 = (A11+A22)*(B11+B22)
        # A11+A22 is a 2x2 matrix. B11+B22 is a 2x2 matrix.
        # (A11+A22)_{ab} * (B11+B22)_{cd} for a,c in {0,1}, b,d in {0,1}
        # This contributes to C11 and C22 (the top-left and bottom-right 2x2 blocks)
        
        # Let me index the 4x4 matrices with block indices.
        # A block (bi, bj) where bi,bj in {0,1} is the 2x2 submatrix at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
        # Within a block, element (a,b) where a,b in {0,1}
        # Global index: row = 2*bi + a, col = 2*bj + b, flat index = row*4+col
        
        # Strassen formulas:
        # M1 = (A00+A11) * (B00+B11)
        # M2 = (A10+A11) * B00
        # M3 = A00 * (B01-B11)
        # M4 = A11 * (B10-B00)
        # M5 = (A00+A01) * B11
        # M6 = (A10-A00) * (B00+B01)
        # M7 = (A01-A11) * (B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each M, iterate over all 4 scalar products in the 2x2*2x2.
        
        # Helper: add a rank-1 term for scalar product
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            """u_coeffs, v_coeffs, w_coeffs are dicts mapping flat index -> coeff"""
            u = [0.0] * 16
            v = [0.0] * 16
            w = [0.0] * 16
            for idx, c in u_coeffs.items():
                u[idx] += c
            for idx, c in v_coeffs.items():
                v[idx] += c
            for idx, c in w_coeffs.items():
                w[idx] += c
            U.append(u)
            V.append(v)
            W.append(w)
        
        # Block (bi, bj) element (a, b) flat index in 4x4:
        # row = 2*bi + a, col = 2*bj + b, flat = row*4 + col
        def flat_idx(bi, bj, a, b):
            row = 2*bi + a
            col = 2*bj + b
            return row * 4 + col
        
        # For a 2x2 matrix product X*Y where X,Y are 2x2:
        # X_{ab} * Y_{bc} contributes to Z_{ac}
        # We need to map the block-level indices to flat indices.
        
        # M1: (A00+A11) * (B00+B11)
        # A00+A11: element (a,b) = A[0,0,a,b] + A[1,1,a,b]
        # B00+B11: element (b,c) = B[0,0,b,c] + B[1,1,b,c]
        # Result contributes to C00 (top-left 2x2) and C11 (bottom-right 2x2)
        # C00_{a,c} += M1_{a,c}, C11_{a,c} += M1_{a,c}
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    # A coefficients: A[0,0,a,b] and A[1,1,a,b]
                    u = {flat_idx(0,0,a,b): 1.0, flat_idx(1,1,a,b): 1.0}
                    # B coefficients: B[0,0,b,c] and B[1,1,b,c]
                    v = {flat_idx(0,0,b,c): 1.0, flat_idx(1,1,b,c): 1.0}
                    # W: C[0,0,a,c] += 1, C[1,1,a,c] += 1
                    w = {flat_idx(0,0,a,c): 1.0, flat_idx(1,1,a,c): 1.0}
                    add_term(u, v, w)
        
        # M2: (A10+A11) * B00
        # A10+A11: element (a,b) = A[1,0,a,b] + A[1,1,a,b]
        # B00: element (b,c) = B[0,0,b,c]
        # Result contributes to C10 (bottom-left 2x2)
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    u = {flat_idx(1,0,a,b): 1.0, flat_idx(1,1,a,b): 1.0}
                    v = {flat_idx(0,0,b,c): 1.0}
                    w = {flat_idx(1,0,a,c): 1.0}
                    add_term(u, v, w)
        
        # M3: A00 * (B01-B11)
        # A00: element (a,b) = A[0,0,a,b]
        # B01-B11: element (b,c) = B[0,1,b,c] - B[1,1,b,c]
        # Result contributes to C01 (top-right 2x2)
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    u = {flat_idx(0,0,a,b): 1.0}
                    v = {flat_idx(0,1,b,c): 1.0, flat_idx(1,1,b,c): -1.0}
                    w = {flat_idx(0,1,a,c): 1.0}
                    add_term(u, v, w)
        
        # M4: A11 * (B10-B00)
        # A11: element (a,b) = A[1,1,a,b]
        # B10-B00: element (b,c) = B[1,0,b,c] - B[0,0,b,c]
        # Result contributes to C00 and C10
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    u = {flat_idx(1,1,a,b): 1.0}
                    v = {flat_idx(1,0,b,c): 1.0, flat_idx(0,0,b,c): -1.0}
                    w = {flat_idx(0,0,a,c): 1.0, flat_idx(1,0,a,c): 1.0}
                    add_term(u, v, w)
        
        # M5: (A00+A01) * B11
        # A00+A01: element (a,b) = A[0,0,a,b] + A[0,1,a,b]
        # B11: element (b,c) = B[1,1,b,c]
        # Result contributes to C00 (-1) and C01 (+1)
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    u = {flat_idx(0,0,a,b): 1.0, flat_idx(0,1,a,b): 1.0}
                    v = {flat_idx(1,1,b,c): 1.0}
                    w = {flat_idx(0,0,a,c): -1.0, flat_idx(0,1,a,c): 1.0}
                    add_term(u, v, w)
        
        # M6: (A10-A00) * (B00+B01)
        # A10-A00: element (a,b) = A[1,0,a,b] - A[0,0,a,b]
        # B00+B01: element (b,c) = B[0,0,b,c] + B[0,1,b,c]
        # Result contributes to C11 (bottom-right 2x2)
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    u = {flat_idx(1,0,a,b): 1.0, flat_idx(0,0,a,b): -1.0}
                    v = {flat_idx(0,0,b,c): 1.0, flat_idx(0,1,b,c): 1.0}
                    w = {flat_idx(1,1,a,c): 1.0}
                    add_term(u, v, w)
        
        # M7: (A01-A11) * (B10+B11)
        # A01-A11: element (a,b) = A[0,1,a,b] - A[1,1,a,b]
        # B10+B11: element (b,c) = B[1,0,b,c] + B[1,1,b,c]
        # Result contributes to C00 (top-left 2x2)
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    u = {flat_idx(0,1,a,b): 1.0, flat_idx(1,1,a,b): -1.0}
                    v = {flat_idx(1,0,b,c): 1.0, flat_idx(1,1,b,c): 1.0}
                    w = {flat_idx(0,0,a,c): 1.0}
                    add_term(u, v, w)
        
        return U, V, W
    
    # Now for 4x4*4x8, split into two 4x4*4x4 problems.
    # Left half: B columns 0-3, C columns 0-3
    # Right half: B columns 4-7, C columns 4-7
    
    U_left, V_left, W_left = strassen_4x4_with_schoolbook_2x2()
    U_right, V_right, W_right = strassen_4x4_with_schoolbook_2x2()
    
    # For the right half, we need to shift the B and C indices by 4 in the column direction.
    # A is the same.
    
    # V_right: B indices. Original B[bi,bj,b,c] where bi is row-block, bj is col-block.
    # In the full 4x8 B, the right half has columns 4-7.
    # Block (bi, bj) in the 4x4 sub-problem maps to global B row-block bi, col-block (bj+1)
    # So flat index: row = 2*bi + a, col = 2*(bj+1) + b = 2*bj+2+b, flat = row*4+col
    # Wait, B is 4x8, so flat index = row*8 + col
    
    # Let me reconsider the indexing.
    # U: (n*m, rank) = (16, rank), index = i*4+j for A[i,j]
    # V: (m*p, rank) = (32, rank), index = j*8+k for B[j,k]
    # W: (p*n, rank) = (32, rank), index = k*4+i for C[i,k]
    
    # For the left 4x4*4x4:
    # A[i,j] -> U index i*4+j
    # B[j,k] where k in 0..3 -> V index j*8+k
    # C[i,k] where k in 0..3 -> W index k*4+i
    
    # For the right 4x4*4x4:
    # A[i,j] -> U index i*4+j (same)
    # B[j,k] where k in 4..7 -> V index j*8+k
    # C[i,k] where k in 4..7 -> W index k*4+i
    
    # So I need to remap the V and W indices for the right half.
    
    def remap_V(V_list):
        """Remap V indices from 4x4 B (16 entries) to 4x8 B (32 entries).
        Original V index: j*4+k for B[j,k] in 4x4
        New V index: j*8+k for B[j,k] in 4x8 (left half, k in 0..3)
        """
        V_new = []
        for v in V_list:
            v_new = [0.0] * 32
            for j in range(4):
                for k in range(4):
                    old_idx = j * 4 + k
                    new_idx = j * 8 + k
                    v_new[new_idx] += v[old_idx]
            V_new.append(v_new)
        return V_new
    
    def remap_V_right(V_list):
        """Remap V indices from 4x4 B to right half of 4x8 B.
        Original V index: j*4+k for B[j,k] in 4x4
        New V index: j*8+(k+4) for B[j,k+4] in 4x8
        """
        V_new = []
        for v in V_list:
            v_new = [0.0] * 32
            for j in range(4):
                for k in range(4):
                    old_idx = j * 4 + k
                    new_idx = j * 8 + (k + 4)
                    v_new[new_idx] += v[old_idx]
            V_new.append(v_new)
        return V_new
    
    def remap_W(W_list):
        """Remap W indices from 4x4 C (16 entries) to 4x8 C (32 entries).
        Original W index: k*4+i for C[i,k] in 4x4
        New W index: k*4+i for C[i,k] in 4x8 (left half, k in 0..3)
        """
        W_new = []
        for w in W_list:
            w_new = [0.0] * 32
            for k in range(4):
                for i in range(4):
                    old_idx = k * 4 + i
                    new_idx = k * 4 + i
                    w_new[new_idx] += w[old_idx]
            W_new.append(w_new)
        return W_new
    
    def remap_W_right(W_list):
        """Remap W indices from 4x4 C to right half of 4x8 C.
        Original W index: k*4+i for C[i,k] in 4x4
        New W index: (k+4)*4+i for C[i,k+4] in 4x8
        """
        W_new = []
        for w in W_list:
            w_new = [0.0] * 32
            for k in range(4):
                for i in range(4):
                    old_idx = k * 4 + i
                    new_idx = (k + 4) * 4 + i
                    w_new[new_idx] += w[old_idx]
            W_new.append(w_new)
        return W_new
    
    def remap_U(U_list):
        """U stays the same shape (16, rank) -> (16, rank)."""
        return U_list
    
    # Combine left and right
    U_total = U_left + U_right
    V_total = remap_V(V_left) + remap_V_right(V_right)
    W_total = remap_W(W_left) + remap_W_right(W_right)
    
    rank = len(U_total)
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
