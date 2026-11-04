<<<<<<< SEARCH
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Strategy 1: Block diagonal with 2x2x2 Strassen for 4x4x4 part + schoolbook for 4x4x3
    # 4x4x4 part: rank 49 (Strassen^2)
    # 4x4x3 part: rank 4*4*3 = 48 schoolbook
    # Total: 97
    # Better: use 2x2x2 Strassen for 4x4x4 and 2x2x3 for 4x4x3
    # 2x2x3: rank 2*2*3 = 12 schoolbook, or can we do better?
    # Actually for 2x2x3, the minimum rank is 7 (same as 2x2x2 since 3>=2)
    # Wait, no. 2x2x3 means C is 2x2, A is 2x2, B is 2x3. 
    # Actually the problem is: A is 4x4, B is 4x7.
    # We can split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2
    # A*B1: 4x4 * 4x4 -> 4x4, rank 49 with Strassen
    # A*B2: 4x4 * 4x3 -> 4x3, rank ?
    
    # For 4x4 * 4x3:
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B2 = [[B11, B12], [B21, B22]] where B11,B21 are 2x2, B12,B22 are 2x1
    # C1 = A11*B11 + A12*B21 (2x2)
    # C2 = A11*B12 + A12*B22 (2x1)
    # C3 = A21*B11 + A22*B21 (2x2)
    # C4 = A21*B12 + A22*B22 (2x1)
    
    # Each 2x2*2x2 -> 2x2: rank 7
    # Each 2x2*2x1 -> 2x1: rank ? (2x2*2x1 is like multiplying 2x2 by a 2x1 vector)
    # Actually 2x2 * 2x1: A (2x2) * b (2x1) = c (2x1)
    # c1 = a11*b1 + a12*b2
    # c2 = a21*b1 + a22*b2
    # This can be done with 4 products (schoolbook) or can we do better?
    # The tensor for 2x2*2x1 has rank 4 (it's just matrix-vector multiplication)
    # Actually, let's think: the multiplication tensor for (2,2,1) has 8 entries.
    # T[a,b,c] where a in {0,1}, b in {0,1}, c in {0,1}
    # T[0,0,0]=1 (a11*b1 -> c1)
    # T[0,1,0]=1 (a12*b2 -> c1)
    # T[1,0,1]=1 (a21*b1 -> c2)
    # T[1,1,1]=1 (a22*b2 -> c2)
    # This is 4 disjoint rank-1 terms, so rank 4.
    
    # So 4x4*4x3:
    # C1 = A11*B11 + A12*B21: two 2x2*2x2 products, rank 7+7=14
    # C2 = A11*B12 + A12*B22: two 2x2*2x1 products, rank 4+4=8
    # C3 = A21*B11 + A22*B21: two 2x2*2x2 products, rank 7+7=14
    # C4 = A21*B12 + A22*B22: two 2x2*2x1 products, rank 4+4=8
    # Total for 4x4*4x3: 14+8+14+8 = 44
    
    # Total: 49 + 44 = 93
    
    # Let's build this decomposition
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 4x4x4 part (indices for A: 0-15, B: 0-27 (first 16 are 4x4), W: 0-27 (first 16 are 4x4))
    # Actually let me be more careful with indexing.
    # A: 4x4, index a = i*4+j, i,j in 0..3
    # B: 4x7, index b = j*7+k, j in 0..3, k in 0..6
    # C: 7x4, index c = k*4+i, k in 0..6, i in 0..3
    
    # Split B into B_left (4x4, k=0..3) and B_right (4x3, k=4..6)
    # B_left: b = j*7+k, k in 0..3
    # B_right: b = j*7+k, k in 4..6
    
    # C_left: c = k*4+i, k in 0..3
    # C_right: c = k*4+i, k in 4..6
    
    # Part 1: A * B_left -> C_left (4x4 * 4x4 -> 4x4)
    # Use Strassen^2, rank 49
    U2, V2, W2 = _strassen_2x2x2()
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U2[:, r1], U2[:, r2])
            V_444[:, idx] = np.kron(V2[:, r1], V2[:, r2])
            W_444[:, idx] = np.kron(W2[:, r1], W2[:, r2])
    
    # Map to full B and C indices
    U_part1 = np.zeros((16, 49))
    V_part1 = np.zeros((28, 49))
    W_part1 = np.zeros((28, 49))
    
    for a in range(16):
        U_part1[a, :] = U_444[a, :]
    
    for j in range(4):
        for k in range(4):
            V_part1[j*7+k, :] = V_444[j*4+k, :]
    
    for k in range(4):
        for i in range(4):
            W_part1[k*4+i, :] = W_444[k*4+i, :]
    
    U_parts.append(U_part1)
    V_parts.append(V_part1)
    W_parts.append(W_part1)
    
    # Part 2: A * B_right -> C_right (4x4 * 4x3 -> 4x3)
    # Split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B_right = [[B11, B12], [B21, B22]] where B11,B21 are 2x2 (cols 4,5 of B), B12,B22 are 2x1 (col 6 of B)
    # Wait, B_right is 4x3. Columns 4,5,6 of B.
    # B_right[i, k'] where k' in 0..2 corresponds to B[i, 4+k']
    # So B_right has columns: col 0 (k=4), col 1 (k=5), col 2 (k=6)
    
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B_right top-left 2x2: B_right[0:2, 0:2] = B[0:2, 4:6]
    # B_right top-right 2x1: B_right[0:2, 2:3] = B[0:2, 6:7]
    # B_right bottom-left 2x2: B_right[2:4, 0:2] = B[2:4, 4:6]
    # B_right bottom-right 2x1: B_right[2:4, 2:3] = B[2:4, 6:7]
    
    # C_right: 4x3
    # C_right[0:2, 0:2] = A11*B11 + A12*B21 (2x2)
    # C_right[0:2, 2:3] = A11*B12 + A12*B22 (2x1)
    # C_right[2:4, 0:2] = A21*B11 + A22*B21 (2x2)
    # C_right[2:4, 2:3] = A21*B12 + A22*B22 (2x1)
    
    # Each 2x2*2x2: rank 7
    # Each 2x2*2x1: rank 4
    
    # Total: 7+4+7+4 = 22? Wait, I said 44 before. Let me recount.
    # C1 (2x2) = A11*B11 + A12*B21: This is two separate 2x2*2x2 products.
    # A11*B11: rank 7
    # A12*B21: rank 7
    # So C1 needs rank 14.
    
    # C2 (2x1) = A11*B12 + A12*B22: two 2x2*2x1 products.
    # A11*B12: rank 4
    # A12*B22: rank 4
    # So C2 needs rank 8.
    
    # C3 (2x2) = A21*B11 + A22*B21: rank 14
    # C4 (2x1) = A21*B12 + A22*B22: rank 8
    
    # Total: 14+8+14+8 = 44. OK.
    
    # Now build the decomposition for 4x4*4x3 -> 4x3
    U_part2 = np.zeros((16, 44))
    V_part2 = np.zeros((28, 44))
    W_part2 = np.zeros((28, 44))
    
    col_offset = 0
    
    # C1: A11*B11 + A12*B21
    # A11*B11: A indices 0-3, B indices (j*7+k for j in 0,1 and k in 4,5)
    # B11 = B[0:2, 4:6], so b = j*7+k, j in 0,1, k in 4,5
    # W: C_right[0:2, 0:2] = C[0:2, 4:6], c = k*4+i, k in 4,5, i in 0,1
    
    # A11*B11: 2x2*2x2, rank 7
    for r in range(7):
        # U: A11 indices
        U_part2[0, col_offset] = U2[0, r]
        U_part2[1, col_offset] = U2[1, r]
        U_part2[2, col_offset] = U2[2, r]
        U_part2[3, col_offset] = U2[3, r]
        # V: B11 indices
        V_part2[0*7+4, col_offset] = V2[0, r]
        V_part2[0*7+5, col_offset] = V2[1, r]
        V_part2[1*7+4, col_offset] = V2[2, r]
        V_part2[1*7+5, col_offset] = V2[3, r]
        # W: C_right[0:2, 0:2]
        W_part2[4*4+0, col_offset] = W2[0, r]
        W_part2[4*4+1, col_offset] = W2[1, r]
        W_part2[5*4+0, col_offset] = W2[2, r]
        W_part2[5*4+1, col_offset] = W2[3, r]
        col_offset += 1
    
    # A12*B21: A indices 4-7, B indices (j*7+k for j in 2,3 and k in 4,5)
    # B21 = B[2:4, 4:6], so b = j*7+k, j in 2,3, k in 4,5
    # W: C_right[0:2, 0:2]
    for r in range(7):
        U_part2[4, col_offset] = U2[0, r]
        U_part2[5, col_offset] = U2[1, r]
        U_part2[6, col_offset] = U2[2, r]
        U_part2[7, col_offset] = U2[3, r]
        V_part2[2*7+4, col_offset] = V2[0, r]
        V_part2[2*7+5, col_offset] = V2[1, r]
        V_part2[3*7+4, col_offset] = V2[2, r]
        V_part2[3*7+5, col_offset] = V2[3, r]
        W_part2[4*4+0, col_offset] = W2[0, r]
        W_part2[4*4+1, col_offset] = W2[1, r]
        W_part2[5*4+0, col_offset] = W2[2, r]
        W_part2[5*4+1, col_offset] = W2[3, r]
        col_offset += 1
    
    # C2: A11*B12 + A12*B22
    # B12 = B[0:2, 6:7], b = j*7+6, j in 0,1
    # B22 = B[2:4, 6:7], b = j*7+6, j in 2,3
    # W: C_right[0:2, 2:3] = C[0:2, 6:7], c = 6*4+i, i in 0,1
    
    # A11*B12: 2x2*2x1, rank 4
    for r in range(4):
        # Schoolbook for 2x2*2x1
        # c1 = a11*b1 + a12*b2
        # c2 = a21*b1 + a22*b2
        # r=0: a11*b1 -> c1
        if r == 0:
            U_part2[0, col_offset] = 1
            V_part2[0*7+6, col_offset] = 1
            W_part2[6*4+0, col_offset] = 1
        elif r == 1:
            U_part2[1, col_offset] = 1
            V_part2[0*7+6, col_offset] = 1
            W_part2[6*4+1, col_offset] = 1
        elif r == 2:
            U_part2[2, col_offset] = 1
            V_part2[1*7+6, col_offset] = 1
            W_part2[6*4+0, col_offset] = 1
        else:
            U_part2[3, col_offset] = 1
            V_part2[1*7+6, col_offset] = 1
            W_part2[6*4+1, col_offset] = 1
        col_offset += 1
    
    # A12*B22: 2x2*2x1, rank 4
    for r in range(4):
        if r == 0:
            U_part2[4, col_offset] = 1
            V_part2[2*7+6, col_offset] = 1
            W_part2[6*4+0, col_offset] = 1
        elif r == 1:
            U_part2[5, col_offset] = 1
            V_part2[2*7+6, col_offset] = 1
            W_part2[6*4+1, col_offset] = 1
        elif r == 2:
            U_part2[6, col_offset] = 1
            V_part2[3*7+6, col_offset] = 1
            W_part2[6*4+0, col_offset] = 1
        else:
            U_part2[7, col_offset] = 1
            V_part2[3*7+6, col_offset] = 1
            W_part2[6*4+1, col_offset] = 1
        col_offset += 1
    
    # C3: A21*B11 + A22*B21
    # A21: A indices 8-11, B11: b = j*7+k, j in 0,1, k in 4,5
    # W: C_right[2:4, 0:2] = C[2:4, 4:6], c = k*4+i, k in 4,5, i in 2,3
    
    for r in range(7):
        U_part2[8, col_offset] = U2[0, r]
        U_part2[9, col_offset] = U2[1, r]
        U_part2[10, col_offset] = U2[2, r]
        U_part2[11, col_offset] = U2[3, r]
        V_part2[0*7+4, col_offset] = V2[0, r]
        V_part2[0*7+5, col_offset] = V2[1, r]
        V_part2[1*7+4, col_offset] = V2[2, r]
        V_part2[1*7+5, col_offset] = V2[3, r]
        W_part2[4*4+2, col_offset] = W2[0, r]
        W_part2[4*4+3, col_offset] = W2[1, r]
        W_part2[5*4+2, col_offset] = W2[2, r]
        W_part2[5*4+3, col_offset] = W2[3, r]
        col_offset += 1
    
    # A22*B21: A indices 12-15, B21: b = j*7+k, j in 2,3, k in 4,5
    for r in range(7):
        U_part2[12, col_offset] = U2[0, r]
        U_part2[13, col_offset] = U2[1, r]
        U_part2[14, col_offset] = U2[2, r]
        U_part2[15, col_offset] = U2[3, r]
        V_part2[2*7+4, col_offset] = V2[0, r]
        V_part2[2*7+5, col_offset] = V2[1, r]
        V_part2[3*7+4, col_offset] = V2[2, r]
        V_part2[3*7+5, col_offset] = V2[3, r]
        W_part2[4*4+2, col_offset] = W2[0, r]
        W_part2[4*4+3, col_offset] = W2[1, r]
        W_part2[5*4+2, col_offset] = W2[2, r]
        W_part2[5*4+3, col_offset] = W2[3, r]
        col_offset += 1
    
    # C4: A21*B12 + A22*B22
    # A21: A indices 8-11, B12: b = j*7+6, j in 0,1
    # W: C_right[2:4, 2:3] = C[2:4, 6:7], c = 6*4+i, i in 2,3
    
    for r in range(4):
        if r == 0:
            U_part2[8, col_offset] = 1
            V_part2[0*7+6, col_offset] = 1
            W_part2[6*4+2, col_offset] = 1
        elif r == 1:
            U_part2[9, col_offset] = 1
            V_part2[0*7+6, col_offset] = 1
            W_part2[6*4+3, col_offset] = 1
        elif r == 2:
            U_part2[10, col_offset] = 1
            V_part2[1*7+6, col_offset] = 1
            W_part2[6*4+2, col_offset] = 1
        else:
            U_part2[11, col_offset] = 1
            V_part2[1*7+6, col_offset] = 1
            W_part2[6*4+3, col_offset] = 1
        col_offset += 1
    
    # A22*B22: A indices 12-15, B22: b = j*7+6, j in 2,3
    for r in range(4):
        if r == 0:
            U_part2[12, col_offset] = 1
            V_part2[2*7+6, col_offset] = 1
            W_part2[6*4+2, col_offset] = 1
        elif r == 1:
            U_part2[13, col_offset] = 1
            V_part2[2*7+6, col_offset] = 1
            W_part2[6*4+3, col_offset] = 1
        elif r == 2:
            U_part2[14, col_offset] = 1
            V_part2[3*7+6, col_offset] = 1
            W_part2[6*4+2, col_offset] = 1
        else:
            U_part2[15, col_offset] = 1
            V_part2[3*7+6, col_offset] = 1
            W_part2[6*4+3, col_offset] = 1
        col_offset += 1
    
    U_parts.append(U_part2)
    V_parts.append(V_part2)
    W_parts.append(W_part2)
    
    # Combine
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    # Verify
    recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
    valid = np.allclose(recon, T, atol=1e-10)
    
    if valid:
        rank = U_combined.shape[1]
        # Try to improve with ALS on the full tensor
        if time.time() < deadline - 10.0:
            als_result = _als_peel(T, dim_u, dim_v, dim_w, deadline - 5.0, rank - 5)
            if als_result is not None:
                r_als = len(als_result["U"][0])
                if r_als < rank:
                    return als_result
        return {
            "U": U_combined.tolist(),
            "V": V_combined.tolist(),
            "W": W_combined.tolist()
        }
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
=======
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Build the 93-rank baseline decomposition
    U_base, V_base, W_base = _build_93_rank(n, m, p)
    
    recon = np.einsum('ak,bk,ck->abc', U_base, V_base, W_base)
    valid = np.allclose(recon, T, atol=1e-10)
    
    if not valid:
        return _schoolbook(n, m, p)
    
    base_rank = U_base.shape[1]
    
    # Structural change: Try to find a lower-rank decomposition using
    # a "tensor completion" approach - treat the factor matrices as unknowns
    # and solve for a lower-rank solution using iterative refinement.
    
    best_result = {
        "U": U_base.tolist(),
        "V": V_base.tolist(),
        "W": W_base.tolist()
    }
    best_rank = base_rank
    
    # Try rank reduction via ALS with exact verification
    if time.time() < deadline - 20.0:
        for target_rank in range(base_rank - 1, max(base_rank - 30, 40), -1):
            if time.time() > deadline - 10.0:
                break
            
            # Try to find a rank-target_rank decomposition
            result = _als_exact(T, dim_u, dim_v, dim_w, target_rank, deadline - 5.0)
            if result is not None:
                r_found = len(result["U"][0])
                if r_found < best_rank:
                    best_rank = r_found
                    best_result = result
                    # If we found a good one, try to go lower
                    continue
            # If we can't find rank target_rank, stop trying lower
            break
    
    return best_result
>>>>>>> REPLACE

Now let me add the new helper functions:

