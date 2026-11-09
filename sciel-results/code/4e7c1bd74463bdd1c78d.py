"""Matrix multiplication 4x4 by 4x8 - rank-minimal exact tensor decomposition.

Strategy: Use the polynomial ring F[x]/(x^2) representation.
A 4x8 matrix B = B0 + x*B1 where B0, B1 are 4x4.
A 4x8 matrix C = C0 + x*C1 where C0, C1 are 4x4.
The multiplication C = A*B becomes:
  C0 = A*B0
  C1 = A*B1

We use the known rank-49 bilinear algorithm for 4x4*4x4 (nested Strassen)
for each of the two products, giving total rank 98.

However, we can do better by observing that the A-factor is shared.
Actually, the tensor rank of the combined problem may be less than 2*49.

But the simplest correct improvement over schoolbook (128) is to use
the Strassen-based 4x4*4x4 decomposition twice: rank 98.

Even better: we can try to use a single 4x4*4x4 algorithm over F[x]/(x^2).
The rank of 4x4*4x4 over F[x]/(x^2) is at most 2*49 = 98, but could be less.
For now, we implement the 2x49 = 98 construction which is guaranteed correct.

Actually, let's think more carefully. The problem is:
  T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [0,8)
  C[i,k] = sum_j A[i,j] * B[j,k]

If we write B[j,k] = B0[j, k%4] + x*B1[j, k%4] (i.e., B = [B0 | B1]),
and C[i,k] = C0[i, k%4] + x*C1[i, k%4], then:
  C0[i,k'] = sum_j A[i,j] * B0[j,k']  for k' in [0,4)
  C1[i,k'] = sum_j A[i,j] * B1[j,k']  for k' in [0,4)

These are two independent 4x4*4x4 multiplications sharing the same A.
The tensor for each is the 4x4*4x4 multiplication tensor.
The total tensor is the direct sum of two 4x4*4x4 tensors.
The rank of a direct sum is at most the sum of ranks: 49 + 49 = 98.

So rank 98 is achievable. Let's implement this cleanly.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 130.0

    def _build_strassen_4x4():
        """
        Build U, V, W for 4x4*4x4 matrix multiplication using nested Strassen.
        Returns (U, V, W) as numpy arrays with shapes (16, 49), (16, 49), (16, 49).
        
        Uses 2x2 block Strassen at the outer level, and Strassen at the inner level.
        Total rank = 7 * 7 = 49.
        """
        # Inner 2x2*2x2 Strassen decomposition
        u_inner = np.zeros((4, 7))
        v_inner = np.zeros((4, 7))
        w_inner = np.zeros((4, 7))
        
        # Strassen 2x2*2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # M1
        u_inner[0, 0] = 1; u_inner[3, 0] = 1
        v_inner[0, 0] = 1; v_inner[3, 0] = 1
        w_inner[0, 0] = 1; w_inner[1, 0] = 1; w_inner[2, 0] = 1; w_inner[3, 0] = 1
        
        # M2
        u_inner[2, 1] = 1; u_inner[3, 1] = 1
        v_inner[0, 1] = 1
        w_inner[2, 1] = 1; w_inner[3, 1] = -1
        
        # M3
        u_inner[0, 2] = 1
        v_inner[1, 2] = 1; v_inner[3, 2] = -1
        w_inner[1, 2] = 1; w_inner[3, 2] = 1
        
        # M4
        u_inner[3, 3] = 1
        v_inner[2, 3] = 1; v_inner[0, 3] = -1
        w_inner[0, 3] = 1; w_inner[2, 3] = 1
        
        # M5
        u_inner[0, 4] = 1; u_inner[1, 4] = 1
        v_inner[3, 4] = 1
        w_inner[0, 4] = -1; w_inner[1, 4] = 1
        
        # M6
        u_inner[2, 5] = 1; u_inner[0, 5] = -1
        v_inner[0, 5] = 1; v_inner[1, 5] = 1
        w_inner[3, 5] = 1
        
        # M7
        u_inner[1, 6] = 1; u_inner[3, 6] = -1
        v_inner[2, 6] = 1; v_inner[3, 6] = 1
        w_inner[0, 6] = 1
        
        # Outer Strassen for 4x4*4x4 with 2x2 blocks
        # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
        # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
        # C blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
        
        # Block indices in flattened form (i*4+j for A, j*4+k for B, k*4+i for C)
        def _block_A(br, bc):
            return [(br*2+dr)*4 + (bc*2+dc) for dr in range(2) for dc in range(2)]
        
        def _block_B(br, bc):
            return [(br*2+dr)*4 + (bc*2+dc) for dr in range(2) for dc in range(2)]
        
        def _block_C(br, bc):
            return [(bc*2+dc)*4 + (br*2+dr) for dr in range(2) for dc in range(2)]
        
        A11 = _block_A(0, 0)
        A12 = _block_A(0, 1)
        A21 = _block_A(1, 0)
        A22 = _block_A(1, 1)
        
        B11 = _block_B(0, 0)
        B12 = _block_B(0, 1)
        B21 = _block_B(1, 0)
        B22 = _block_B(1, 1)
        
        C11 = _block_C(0, 0)
        C12 = _block_C(0, 1)
        C21 = _block_C(1, 0)
        C22 = _block_C(1, 1)
        
        # Outer Strassen products:
        # P1 = (A11+A22)*(B11+B22)
        # P2 = (A21+A22)*B11
        # P3 = A11*(B12-B22)
        # P4 = A22*(B21-B11)
        # P5 = (A11+A12)*B22
        # P6 = (A21-A11)*(B11+B12)
        # P7 = (A12-A22)*(B21+B22)
        
        # Result:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        coeff_A = [
            [1, 0, 0, 1],   # P1
            [0, 0, 1, 1],   # P2
            [1, 0, 0, 0],   # P3
            [0, 0, 0, 1],   # P4
            [1, 1, 0, 0],   # P5
            [-1, 0, 1, 0],  # P6
            [0, 1, 0, -1],  # P7
        ]
        
        coeff_B = [
            [1, 0, 0, 1],   # P1
            [1, 0, 0, 0],   # P2
            [0, 1, 0, -1],  # P3
            [0, 0, 1, -1],  # P4
            [0, 0, 0, 1],   # P5
            [1, 1, 0, 0],   # P6
            [0, 0, 1, 1],   # P7
        ]
        
        result_coeff = [
            [1, 0, 0, 0],   # P1 -> C11
            [0, 0, 1, -1],  # P2 -> C21, C22
            [0, 1, 0, 1],   # P3 -> C12, C22
            [1, 0, 1, 0],   # P4 -> C11, C21
            [-1, 1, 0, 0],  # P5 -> C11, C12
            [0, 0, 0, 1],   # P6 -> C22
            [1, 0, 0, 0],   # P7 -> C11
        ]
        
        A_blocks = [A11, A12, A21, A22]
        B_blocks = [B11, B12, B21, B22]
        C_blocks = [C11, C12, C21, C22]
        
        inner_rank = 7
        total_rank = 7 * inner_rank
        
        U = np.zeros((16, total_rank))
        V = np.zeros((16, total_rank))
        W = np.zeros((16, total_rank))
        
        for t in range(7):
            for c in range(inner_rank):
                col = t * inner_rank + c
                
                # A-side
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        u_coeff = u_inner[inner_idx, c]
                        if u_coeff == 0:
                            continue
                        for b in range(4):
                            a_coeff = coeff_A[t][b]
                            if a_coeff == 0:
                                continue
                            U[A_blocks[b][inner_idx], col] += u_coeff * a_coeff
                
                # B-side
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        v_coeff = v_inner[inner_idx, c]
                        if v_coeff == 0:
                            continue
                        for b in range(4):
                            b_coeff = coeff_B[t][b]
                            if b_coeff == 0:
                                continue
                            V[B_blocks[b][inner_idx], col] += v_coeff * b_coeff
                
                # C-side
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        w_coeff = w_inner[inner_idx, c]
                        if w_coeff == 0:
                            continue
                        for rb in range(4):
                            r_coeff = result_coeff[t][rb]
                            if r_coeff == 0:
                                continue
                            W[C_blocks[rb][inner_idx], col] += w_coeff * r_coeff
        
        return U, V, W, total_rank
    
    def _verify_4x4(U, V, W, rank):
        """Verify 4x4*4x4 decomposition."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    def _verify_full(U, V, W, rank):
        """Verify 4x4*4x8 decomposition."""
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * 4 + j, c] * V[j * 8 + k, c] * W[k * 4 + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    def _to_fraction_list(mat):
        """Convert numpy array to list of Fractions."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                h = round(val * 2) / 2.0
                new_row.append(Fraction(h).limit_denominator(2))
            result.append(new_row)
        return result
    
    def _verify_exact(U_list, V_list, W_list, rank):
        """Exact verification using Fractions."""
        if len(U_list) != 16 or len(V_list) != 32 or len(W_list) != 32:
            return False
        for row in U_list:
            if len(row) != rank:
                return False
        for row in V_list:
            if len(row) != rank:
                return False
        for row in W_list:
            if len(row) != rank:
                return False
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U_list[i * 4 + j][c] * V_list[j * 8 + k][c] * W_list[k * 4 + i][c]
                    if total != Fraction(1):
                        return False
        return True
    
    # Build the 4x4*4x4 Strassen decomposition (rank 49)
    U44, V44, W44, rank44 = _build_strassen_4x4()
    
    # Verify it
    ok, info = _verify_4x4(U44, V44, W44, rank44)
    
    if not ok:
        # Fallback to schoolbook if Strassen construction has a bug
        rank = 128
        U = np.zeros((16, rank))
        V = np.zeros((32, rank))
        W = np.zeros((32, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U[i * 4 + j, col] = 1
                    V[j * 8 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return {"U": _to_fraction_list(U), "V": _to_fraction_list(V), "W": _to_fraction_list(W)}
    
    # Build the 4x4*4x8 decomposition using two 4x4*4x4 decompositions
    # B = [B0 | B1] where B0, B1 are 4x4
    # C = [C0 | C1] where C0 = A*B0, C1 = A*B1
    
    total_rank = 2 * rank44  # 98
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((32, total_rank))
    W_full = np.zeros((32, total_rank))
    
    # First half: C0 = A * B0
    # U: same for both (A is shared)
    U_full[:, :rank44] = U44
    U_full[:, rank44:] = U44
    
    # V for B0: rows 0..15 (j*8+k for j in 0..3, k in 0..3)
    # V44 has shape (16, rank44) with indices j*4+k' for j,k' in 0..3
    # In the full V, B0[j,k] is at index j*8+k for k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j * 8 + k, :rank44] = V44[j * 4 + k, :]
    
    # V for B1: rows 16..31 (j*8+k for j in 0..3, k in 4..7)
    # B1[j,k'] is at index j*8+(k'+4)
    for j in range(4):
        for k in range(4):
            V_full[j * 8 + (k + 4), rank44:] = V44[j * 4 + k, :]
    
    # W for C0: rows 0..15 (k*4+i for k in 0..3, i in 0..3)
    W_full[:16, :rank44] = W44
    
    # W for C1: rows 16..31 (k*4+i for k in 4..7, i in 0..3)
    for k in range(4):
        for i in range(4):
            W_full[(k + 4) * 4 + i, rank44:] = W44[k * 4 + i, :]
    
    # Verify the full decomposition
    ok_full, info_full = _verify_full(U_full, V_full, W_full, total_rank)
    
    if ok_full:
        # Convert to fractions and verify exactly
        U_frac = _to_fraction_list(U_full)
        V_frac = _to_fraction_list(V_full)
        W_frac = _to_fraction_list(W_full)
        
        if _verify_exact(U_frac, V_frac, W_frac, total_rank):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # If fraction verification fails (shouldn't happen with integer entries),
    # fall back to schoolbook
    rank = 128
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i * 4 + j, col] = 1
                V[j * 8 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
    return {"U": _to_fraction_list(U), "V": _to_fraction_list(V), "W": _to_fraction_list(W)}
# EVOLVE-BLOCK-END
