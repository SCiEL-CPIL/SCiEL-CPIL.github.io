import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def _verify(U, V, W, n, m, p):
    """Check that U, V, W decompose the matrix multiplication tensor."""
    R = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i * m + j, j * p + k, k * n + i] = 1
    return np.allclose(T, Expected, atol=1e-9)


def _trivial_decomposition(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}


def _strassen_222():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_r U[:,r] outer V[:,r] outer W[:,r] = T_{2,2,2}
    
    Indices: a = i*2+j, b = j*2+k, c = k*2+i
    """
    # Strassen's 7 products for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Row/col conventions:
    # U rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # V rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # W rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A10+A11)B00 -> C10 += M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1
    
    # M3: A00(B10-B11) -> C01 += M3
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1
    
    # M4: A11(B01-B00) -> C11 += M4
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[3, 3] += 1
    
    # M5: (A00+A01)B11 -> C01 += M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[1, 4] += 1
    
    # M6: (A10-A00)(B00+B01) -> C10 += M6
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W


def _trivial_nm_p(n, m, p):
    """Trivial decomposition for n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return U, V, W


def _block_decomposition(n, m, p):
    """
    Try block decomposition: split m into m1+m2, p into p1+p2, etc.
    For 2x4x7: try splitting m=4 into 2+2, p=7 into 3+4 or 2+5, etc.
    Use Strassen for the 2x2x2 blocks and trivial for the rest.
    
    Strategy: Split the middle dimension m into two parts m1, m2.
    A is n x (m1+m2) = [A1 | A2]
    B is (m1+m2) x p = [[B1],[B2]]
    C = A1*B1 + A2*B2
    
    For each block product (n x m1 x p) and (n x m2 x p), we can use
    various decompositions. If both are 2x2x? we can use Strassen-like.
    
    Let's try: m=4 = 2+2, p=7.
    Block 1: 2x2x7, Block 2: 2x2x7
    For 2x2x7, we can split p=7 into 2+2+2+1 or use other methods.
    
    Actually, let's think differently.
    The tensor T_{2,4,7} can be seen as:
    T = sum_j (e_j in A-dim) outer (e_j in B-dim) with the k coupling.
    
    A better approach: use the structure that n=2.
    C = A @ B where A is 2x4, B is 4x7, C is 2x7.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i=0: C[0,k] = sum_j A[0,j]*B[j,k]
    For i=1: C[1,k] = sum_j A[1,j]*B[j,k]
    
    We can write this as:
    C[0,k] = sum_j A[0,j]*B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k]
    
    Let's try to use the fact that we have 2 rows in A and C.
    Define:
    S0 = A[0,:] (row 0 of A, 4 elements)
    S1 = A[1,:] (row 1 of A, 4 elements)
    
    C[0,k] = S0 . B[:,k]
    C[1,k] = S1 . B[:,k]
    
    We need 7*2 = 14 output entries (C[0,0..6], C[1,0..6]).
    
    Alternative: group by k.
    For each k, C[:,k] = A @ B[:,k] (a 2-vector = A times a 4-vector)
    
    Let me try a specific block construction:
    Split m=4 into groups of 2: {j=0,1} and {j=2,3}
    For group {0,1}: contributes to C[i,k] via A[i,0]*B[0,k] + A[i,1]*B[1,k]
    For group {2,3}: contributes to C[i,k] via A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    For each group of 2 j-values and all 7 k-values, we have a 2x2x7 sub-problem.
    Actually it's a 2 (i) x 2 (j in group) x 7 (k) sub-tensor.
    
    For the 2x2x7 part:
    C_sub[i,k] = A_sub[i,0]*B_sub[0,k] + A_sub[i,1]*B_sub[1,k]
    
    This is like multiplying a 2x2 by a 2x7.
    
    We can use Strassen-like for the 2x2 part:
    For 2x2x2: rank 7 (Strassen)
    For 2x2x7: we can split k into 2+2+2+1
    2x2x2: rank 7 (Strassen)
    2x2x2: rank 7
    2x2x2: rank 7
    2x2x1: rank 4 (trivial, since 2*2*1=4)
    Total for 2x2x7: 7+7+7+4 = 25
    
    Two such groups: 25+25 = 50
    
    But wait, we need to be careful about how the indices map.
    
    Let me implement this:
    m=4 split into {0,1} and {2,3}
    For group g (g=0: j in {0,1}, g=1: j in {2,3}):
      p=7 split into {0,1}, {2,3}, {4,5}, {6}
      
      For the 2x2x2 sub-blocks (i in {0,1}, j_local in {0,1}, k_local in {0,1}):
        Use Strassen: rank 7 each
      For the 2x2x1 sub-block (i in {0,1}, j_local in {0,1}, k_local={0}):
        Use trivial: rank 4
    
    Total rank: 2 groups * (3*7 + 4) = 2 * 25 = 50
    """
    n, m, p = 2, 4, 7
    
    # We'll build the decomposition manually
    # Total rank estimate: 50
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    r_offset = 0
    
    # For each group of j (2 groups of 2)
    for j_group in range(2):
        j_base = j_group * 2  # 0 or 2
        # j indices in this group: j_base, j_base+1
        
        # Split p=7 into 3 groups of 2 and 1 group of 1
        k_groups = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for k_start, k_end in k_groups:
            k_len = k_end - k_start
            if k_len == 2:
                # 2x2x2 sub-problem: use Strassen
                U_s, V_s, W_s = _strassen_222()
                # Map indices
                for r in range(7):
                    for ia in range(4):  # 0..3 -> i*2+j_local
                        # ia corresponds to (i, j_local) where i=ia//2, j_local=ia%2
                        i = ia // 2
                        j_local = ia % 2
                        j = j_base + j_local
                        a_idx = i * m + j
                        U[a_idx, r_offset + r] += U_s[ia, r]
                        
                    for ib in range(4):  # 0..3 -> j_local*2+k_local
                        # ib corresponds to (j_local, k_local)
                        j_local = ib // 2
                        k_local = ib % 2
                        j = j_base + j_local
                        k = k_start + k_local
                        b_idx = j * p + k
                        V[b_idx, r_offset + r] += V_s[ib, r]
                        
                    for ic in range(4):  # 0..3 -> k_local*2+i
                        # ic corresponds to (k_local, i)
                        k_local = ic // 2
                        i = ic % 2
                        k = k_start + k_local
                        c_idx = k * n + i
                        W[c_idx, r_offset + r] += W_s[ic, r]
                
                r_offset += 7
                
            elif k_len == 1:
                # 2x2x1 sub-problem: trivial
                # i in {0,1}, j_local in {0,1}, k_local = 0
                k = k_start
                for i in range(2):
                    for j_local in range(2):
                        j = j_base + j_local
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        U[a_idx, r_offset] = 1
                        V[b_idx, r_offset] = 1
                        W[c_idx, r_offset] = 1
                        r_offset += 1
    
    return {"U": U, "V": V, "W": W}


def _block_decomposition_v2(n, m, p):
    """
    Alternative block decomposition for 2x4x7.
    
    Let's think about it differently.
    C = A @ B, A is 2x4, B is 4x7, C is 2x7.
    
    We can write:
    C = [A0 A1 A2 A3] @ [[B0],[B1],[B2],[B3]]
      = A0@B0 + A1@B1 + A2@B2 + A3@B3
    
    where A0..A3 are 2x1 columns of A, B0..B3 are 1x7 rows of B.
    Each A_i @ B_i is a 2x7 outer product.
    
    The outer product of a 2-vector and a 7-vector has rank 1 as a tensor,
    but we need to compute it. Actually, each term A[i,j]*B[j,k] is already
    a rank-1 tensor in the full tensor sense.
    
    So the trivial decomposition has 4*2*7 = 56 terms... wait no.
    Actually the trivial has n*m*p = 2*4*7 = 56 terms.
    
    Let me try a different blocking:
    Split A into 2x2 blocks: A = [A1 | A2] where A1, A2 are 2x2.
    Split B into 2x7 blocks: B = [B1; B2] where B1, B2 are 2x7.
    C = A1@B1 + A2@B2
    
    For A1@B1: 2x2 @ 2x7. 
    We can split B1 into columns: B1 = [B1_0 B1_1 B1_2 B1_3 B1_4 B1_5 B1_6]
    Each B1_i is 2x1.
    A1 @ B1_i is 2x1 = A1 @ (column vector).
    
    For a fixed 2x1 vector v, A1@v can be computed with... well, it's just
    matrix-vector multiplication. As a tensor, A1@v has entries (A1@v)[i] = sum_j A1[i,j]*v[j].
    
    This is getting complicated. Let me just try the first approach and verify.
    """
    return _block_decomposition(n, m, p)


def _try_alternative_split():
    """
    Try splitting differently.
    
    For 2x4x7, what if we split p=7 into 4+3?
    Or split m=4 into 1+3?
    
    Let's try: m=4 = 1+3, p=7 stays.
    Block 1: 2x1x7 (rank 14 trivial)
    Block 2: 2x3x7 (rank 42 trivial)
    Total: 56 - no improvement.
    
    What about using Strassen on the 2x2 part and handling the rest?
    
    Actually, let me try a completely different approach.
    Since n=2, we can write:
    C[0,k] = sum_j A[0,j]*B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k]
    
    Define for each k:
    d[k] = A[0,k] + A[1,k]  (but A only has 4 columns, not 7)
    
    Hmm, let me think about this more carefully.
    
    Actually, here's a cleaner idea:
    Let A be 2x4. Write A = [A_L | A_R] where A_L, A_R are 2x2.
    Let B be 4x7. Write B = [B_L; B_R] where B_L, B_R are 2x7.
    
    C = A_L @ B_L + A_R @ B_R
    
    For A_L @ B_L: this is 2x2 @ 2x7.
    We can compute this by splitting B_L into 7 columns, each 2x1.
    For column c (2x1 vector v): A_L @ v gives a 2x1 result.
    
    The tensor for A_L @ B_L has entries T[i, j, k] where:
    i in {0,1} (row of A_L and C)
    j in {0,1} (col of A_L = row of B_L)  
    k in {0,6} (col of B_L and C)
    
    T[i, j, k] = 1 if (i,j) in A_L and (j,k) in B_L and (i,k) in C
    
    Wait, that's not right. The tensor T_{2,4,7}[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i.
    
    For the block A_L@B_L (j in {0,1}):
    T[i, j, k] for i in {0,1}, j in {0,1}, k in {0,6}
    This is a 2x2x7 tensor.
    
    Rank of 2x2x7 tensor: at most 14 (trivial), can we do better?
    For 2x2x2: rank 7 (Strassen)
    For 2x2x7: split k into 2+2+2+1 -> 7+7+7+4 = 25? No wait.
    
    Hmm, 2x2x7 as a standalone tensor.
    The trivial rank is 2*2*7 = 28.
    Using Strassen on the 2x2 part: for each pair of k values, we get a 2x2x2.
    k pairs: (0,1), (2,3), (4,5), (6)
    3 groups of 2x2x2 (rank 7 each) + 1 group of 2x2x1 (rank 4)
    Total: 3*7 + 4 = 25.
    
    Two such blocks: 2*25 = 50.
    
    This matches my earlier calculation. Let me verify the index mapping is correct.
    """
    pass


def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with trivial
    trivial = _trivial_decomposition(n, m, p)
    best_rank = trivial["U"].shape[1]
    best_result = trivial
    
    # Try block decomposition (should give rank 50)
    if time.time() - start_time < deadline - 10:
        block = _block_decomposition(n, m, p)
        if _verify(block["U"], block["V"], block["W"], n, m, p):
            if block["U"].shape[1] < best_rank:
                best_rank = block["U"].shape[1]
                best_result = block
        else:
            # Debug: print what's wrong
            U, V, W = block["U"], block["V"], block["W"]
            T = np.einsum('ar,br,cr->abc', U, V, W)
            Expected = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Expected[i * m + j, j * p + k, k * n + i] = 1
            diff = T - Expected
            if np.abs(diff).max() > 1e-9:
                # Find where the differences are
                indices = np.where(np.abs(diff) > 1e-9)
                # Don't print in production, but useful for debugging
                pass
    
    # Try another approach: direct 2x4x7 decomposition
    # C = A@B, A is 2x4, B is 4x7, C is 2x7
    # Use the fact that we can express some products more efficiently
    
    # Approach: for each k in 0..6, C[:,k] = A @ B[:,k]
    # A @ B[:,k] is a 2-vector = [sum_j A[0,j]*B[j,k], sum_j A[1,j]*B[j,k]]
    
    # Let's try to use a shared computation.
    # Define:
    # s0 = A[0,0]+A[1,0], s1 = A[0,1]+A[1,1], s2 = A[0,2]+A[1,2], s3 = A[0,3]+A[1,3]
    # d0 = A[0,0]-A[1,0], d1 = A[0,1]-A[1,1], d2 = A[0,2]-A[1,2], d3 = A[0,3]-A[1,3]
    # t0 = B[0,0]+B[1,0], t1 = B[0,1]+B[1,1], t2 = B[0,2]+B[1,2], t3 = B[0,3]+B[1,3]
    # etc.
    
    # This is essentially trying to find a lower-rank decomposition.
    # For now, let's just go with the block approach.
    
    if time.time() - start_time > deadline - 5:
        # Safety: return best found
        return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}, Sum diff: {diff.sum()}")
        # Show some non-zero differences
        idx = np.where(diff > 1e-9)
        for ii in range(min(10, len(idx[0]))):
            a, b, c = idx[0][ii], idx[1][ii], idx[2][ii]
            print(f"  T[{a},{b},{c}] = {T[a,b,c]}, Expected = {Expected[a,b,c]}")
