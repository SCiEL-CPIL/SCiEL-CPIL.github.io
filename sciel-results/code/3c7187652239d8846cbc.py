import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        # Check target entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1) > 1e-9:
                        return False
        
        # Check non-target entries (should be 0)
        # We can just check all entries not in the target set
        target_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target_set.add((a, b, c))
                    
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_set:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    def kron_decomposition():
        """
        Construct a rank-28 decomposition using the Kronecker product structure.
        
        The multiplication (2x4) * (4x4) -> (2x4) can be viewed as:
        A = [A0 | A1] where A0, A1 are 2x2
        B = [[B00, B01], [B10, B11]] where Bxy are 2x2
        C = [C0 | C1] where C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11
        
        This is equivalent to two independent 2x2 matrix multiplications for each output block,
        but the inputs are shared in a specific way.
        
        Actually, a cleaner view:
        C[i, k] = sum_j A[i, j] B[j, k]
        
        Let's split indices into high/low bits.
        i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
        
        Let j = 2*j1 + j0, k = 2*k1 + k0
        Then C[i, 2*k1+k0] = sum_{j1=0,1} sum_{j0=0,1} A[i, 2*j1+j0] B[2*j1+j0, 2*k1+k0]
        
        This looks like a tensor product of two 2x2 multiplications if we reshape A and B appropriately.
        
        Let's define the rank-1 terms for a 2x2 matmul.
        There are 8 rank-1 terms for a 2x2 matmul? No, schoolbook is 4.
        Wait, the rank of 2x2 matmul is 7 (Strassen).
        
        If we use Strassen for each 2x2 block, we get 4 blocks * 7 terms = 28 terms.
        But the blocks are not independent because A is shared.
        
        Let's look at the Kronecker product approach more carefully.
        The tensor T_{ijk} = delta_{ij} delta_{jk} ... no.
        
        Let's stick to the 4-block Strassen approach but implement it correctly using Kronecker products of the 2x2 Strassen factors.
        
        Strassen 2x2 factors:
        U_2x2, V_2x2, W_2x2 of shape (4, 7).
        
        We want to combine these to handle the 2x4 x 4x4 case.
        
        Let's define the mapping for the 4 blocks:
        Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
        
        For each block, we have a 2x2 matmul.
        We can define permutation matrices or just build the U, V, W by iterating over the 7 Strassen terms for each block.
        
        Let's generate the 2x2 Strassen factors first.
        """
        # Standard Strassen factors for 2x2
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # U (A part), V (B part), W (C part)
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        # M4 = A11(B00-B10) -> C00 += M4, C10 += M4
        # M5 = (A00+A01)B11 -> C01 += M5, C00 -= M5
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
        
        # Let's write them as vectors of length 4 (for A, B, C)
        # A indices: 0,1,2,3
        # B indices: 0,1,2,3
        # C indices: 0,1,2,3
        
        U2 = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [1, 0, 1, 0], # M6 (A10-A00) -> [ -1, 0, 1, 0 ]? No, A10 is idx 2, A00 is idx 0. So [-1, 0, 1, 0]
            [0, 1, 0, -1], # M7 (A01-A11) -> A01 is idx 1, A11 is idx 3. So [0, 1, 0, -1]
        ]
        # Fix M6: A10 - A00. A10 is row 1 col 0 -> idx 2. A00 is row 0 col 0 -> idx 0.
        # So vector is [-1, 0, 1, 0].
        U2[5] = [-1, 0, 1, 0]
        
        V2 = [
            [1, 0, 0, 1], # M1 (B00+B11)
            [1, 0, 0, 0], # M2 (B00)
            [0, 0, 1, -1], # M3 (B10-B11) -> B10 idx 2, B11 idx 3
            [1, 0, -1, 0], # M4 (B00-B10) -> B00 idx 0, B10 idx 2
            [0, 0, 0, 1], # M5 (B11)
            [1, 1, 0, 0], # M6 (B00+B01)
            [0, 0, 1, 1], # M7 (B10+B11)
        ]
        
        W2 = [
            [1, 0, 0, 1], # M1 (C00+C11)
            [0, 0, 1, -1], # M2 (C10-C11)
            [0, 1, 0, 1], # M3 (C01+C11)
            [1, 0, 1, 0], # M4 (C00+C10)
            [-1, 1, 0, 0], # M5 (C01-C00)
            [0, 0, 0, 1], # M6 (C11)
            [1, 0, 0, -1], # M7 (C00-C11)
        ]
        
        # Verify 2x2 Strassen
        # T = sum U2[r] * V2[r] * W2[r]
        # Let's just trust the standard formulas.
        
        # Now build the 2x4 x 4x4 decomposition.
        # Rank = 4 blocks * 7 terms = 28.
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)] # 8 rows
        V = [[0.0] * rank for _ in range(m * p)] # 16 rows
        W = [[0.0] * rank for _ in range(p * n)] # 8 rows
        
        # Blocks:
        # Block 0: A cols 0,1; B rows 0,1 cols 0,1; C cols 0,1
        # Block 1: A cols 2,3; B rows 2,3 cols 0,1; C cols 0,1
        # Block 2: A cols 0,1; B rows 0,1 cols 2,3; C cols 2,3
        # Block 3: A cols 2,3; B rows 2,3 cols 2,3; C cols 2,3
        
        # Mapping:
        # A global index: i * m + j = i * 4 + j
        # B global index: j * p + k = j * 4 + k
        # C global index: k * n + i = k * 2 + i
        
        # For a block with:
        # A col offset ac_off (0 or 2)
        # B row offset br_off (0 or 2)
        # B col offset bc_off (0 or 2)
        # C col offset cc_off (0 or 2)
        
        # Local indices 0,1,2,3 correspond to:
        # Local 0: (0,0)
        # Local 1: (0,1)
        # Local 2: (1,0)
        # Local 3: (1,1)
        
        # A global: i * 4 + (ac_off + lc) where i is local row (0 or 1)
        # B global: (br_off + lr) * 4 + (bc_off + lc) where lr is local row, lc is local col
        # C global: (cc_off + lc) * 2 + i where i is local row
        
        blocks = [
            (0, 0, 0, 0), # Block 0
            (2, 2, 0, 0), # Block 1
            (0, 0, 2, 2), # Block 2
            (2, 2, 2, 2), # Block 3
        ]
        
        col = 0
        for ac_off, br_off, bc_off, cc_off in blocks:
            for r in range(7):
                # U part
                for li in range(4):
                    coeff = U2[r][li]
                    if coeff != 0:
                        local_row = li // 2
                        local_col = li % 2
                        global_a = local_row * 4 + (ac_off + local_col)
                        U[global_a][col] += coeff
                        
                # V part
                for lj in range(4):
                    coeff = V2[r][lj]
                    if coeff != 0:
                        local_row = lj // 2
                        local_col = lj % 2
                        global_b = (br_off + local_row) * 4 + (bc_off + local_col)
                        V[global_b][col] += coeff
                        
                # W part
                for lk in range(4):
                    coeff = W2[r][lk]
                    if coeff != 0:
                        local_row = lk // 2
                        local_col = lk % 2
                        global_c = (cc_off + local_col) * 2 + local_row
                        W[global_c][col] += coeff
                        
                col += 1
                
        return {"U": U, "V": V, "W": W}

    if time.time() < deadline - 10:
        kron = kron_decomposition()
        if verify(kron["U"], kron["V"], kron["W"], 28):
            best = kron
            best_rank = 28
        else:
            # Fallback to schoolbook if verification fails (should not happen)
            pass

    return best
# EVOLVE-BLOCK-END
