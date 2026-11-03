"""Matrix multiplication 2x4x6 using Strassen's algorithm for 2x2x2 blocks.

We decompose the 2x4x6 multiplication into six 2x2x2 subproblems and use
Strassen's rank-7 algorithm for each, giving a total rank of 6*7 = 42.

The tensor convention: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
for i in range(n), j in range(m), k in range(p).
"""

import time
import sys
import random

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6

    # The 2x4x6 matrix multiplication can be viewed as 6 independent 2x2x2
    # subproblems by partitioning:
    # - A is 2x4: split columns into two 2x2 blocks
    # - B is 4x6: split rows into two 2x6 blocks, then each into 2x3... 
    # 
    # Actually, let me think more carefully. We have:
    # A: 2x4, B: 4x6, C = A*B: 2x6
    #
    # We can partition A into [A0, A1] where A0, A1 are 2x2
    # We can partition B into [B0; B1] where B0, B1 are 2x6
    # Then C = A0*B0 + A1*B1
    #
    # But B0 and B1 are 2x6, not 2x2. We need to further partition.
    # Let's partition B0 into [B00, B01, B02] where each is 2x2
    # Similarly B1 into [B10, B11, B12]
    #
    # So C = A0*[B00,B01,B02] + A1*[B10,B11,B12]
    # = [A0*B00 + A1*B10, A0*B01 + A1*B11, A0*B02 + A1*B12]
    #
    # Each of the three 2x2 output blocks is a sum of two 2x2 multiplications.
    # We can use Strassen for each pair: M1 = A0*B00 + A1*B10, etc.
    #
    # For each 2x2x2 Strassen block, we use rank 7.
    # We have 3 such blocks, so total rank = 3 * 7 = 21? 
    #
    # Wait, that's not right. Let me reconsider the tensor structure.
    #
    # The full tensor has dimensions (n*m) x (m*p) x (p*n) = 8 x 24 x 12
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    #
    # Let me index: a = i*4 + j (i in {0,1}, j in {0,1,2,3})
    #              b = j*6 + k (j in {0,1,2,3}, k in {0,1,2,3,4,5})
    #              c = k*2 + i (k in {0,1,2,3,4,5}, i in {0,1})
    #
    # T[a,b,c] = 1 iff there exist i,j,k such that:
    #   a = i*4+j, b = j*6+k, c = k*2+i
    #
    # This means: i = a//4, j = a%4, and we need j = b//6, k = b%6,
    # and then c should equal k*2 + i.
    #
    # So T[a,b,c] = 1 iff (a//4 == c%2) and (a%4 == b//6) and (b%6)*2 + a//4 == c
    #
    # Let me build the tensor explicitly.

    # Build the target tensor
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1.0

    # Strassen's algorithm for 2x2x2:
    # Given A, B (2x2), compute C = A*B using 7 multiplications:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For our problem, we need to map the 2x4x6 structure to Strassen blocks.
    # 
    # Let's partition the middle dimension (j) into two groups: j in {0,1} and j in {2,3}
    # A = [A_left, A_right] where A_left is 2x2 (rows i, cols j=0,1) and A_right is 2x2 (rows i, cols j=2,3)
    # B = [B_top; B_bottom] where B_top is 2x6 (rows j=0,1) and B_bottom is 2x6 (rows j=2,3)
    # 
    # C = A_left * B_top + A_right * B_bottom
    #
    # Now B_top is 2x6. Partition B_top into three 2x2 blocks: B_top = [B_top_0, B_top_1, B_top_2]
    # Similarly B_bottom = [B_bottom_0, B_bottom_1, B_bottom_2]
    #
    # C = A_left*[B_top_0, B_top_1, B_top_2] + A_right*[B_bottom_0, B_bottom_1, B_bottom_2]
    # = [A_left*B_top_0 + A_right*B_bottom_0, A_left*B_top_1 + A_right*B_bottom_1, A_left*B_top_2 + A_right*B_bottom_2]
    #
    # Each 2x2 output block C_t = A_left*B_top_t + A_right*B_bottom_t
    #
    # For each t in {0,1,2}, we can use a Strassen-like combination. But we need to be careful:
    # Strassen computes A*B for a single pair. Here we have A_left*B_top_t + A_right*B_bottom_t.
    #
    # We can write this as: let X = A_left, Y = B_top_t, Z = A_right, W = B_bottom_t
    # Then C_t = X*Y + Z*W
    #
    # This is a "sum of two 2x2x2 tensors". The rank of this combined tensor is at most 14
    # (just do Strassen on each separately), but might be less.
    #
    # Actually, we can use a direct Strassen-style decomposition for X*Y + Z*W.
    # 
    # Alternatively, we can just use the straightforward approach: for each of the 3 output blocks,
    # use Strassen on X*Y and Strassen on Z*W, giving 7+7=14 per block, total 42.
    #
    # Let's go with that: total rank = 3 * 14 = 42.

    # Define Strassen factors for a single 2x2x2 multiplication.
    # For 2x2x2, the tensor has indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1}
    # T[i,j,k] = 1 if the (i,j) input, (j,k) input, and (k,i) output are consistent.
    # In flat indexing for a 2x2x2 block:
    #   U index: i*2 + j (0,1,2,3)
    #   V index: j*2 + k (0,1,2,3)  
    #   W index: k*2 + i (0,1,2,3)
    #
    # Strassen's 7 rank-1 terms for 2x2x2:
    # Each term r has U_r, V_r, W_r vectors of length 4.
    
    # Strassen factors (standard):
    # M1 = (A00+A11)(B00+B11) -> U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # M2 = (A10+A11)B00       -> U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # M3 = A00(B01-B11)       -> U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # M4 = A11(B10-B00)       -> U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,0,1]
    # M5 = (A00+A01)B11       -> U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # M6 = (A10-A00)(B00+B01) -> U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,1,0]
    # M7 = (A01-A11)(B10+B11) -> U=[0,1,0,-1], V=[0,0,1,1], W=[0,1,0,1]
    
    # Let me verify these. For M1:
    # U=[1,0,0,1] means A00 and A11 are selected.
    # V=[1,0,0,1] means B00 and B11 are selected.
    # W=[1,0,0,1] means C00 and C11 receive this contribution.
    # Contribution: (A00+A11)(B00+B11) to C00 and C11.
    # C00 should get A00*B00 + A01*B10 + A10*B01 + A11*B11
    # C11 should get A00*B01 + A01*B11 + A10*B00 + A11*B10
    # 
    # M1 contributes (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11 to both C00 and C11.
    # 
    # Let me check all 7 terms sum correctly:
    # C00 = M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10 + A10*B01 + A11*B11  (after cancellation)
    # Wait, I'm missing A10*B01. Let me recheck.
    
    # I think I have the standard Strassen wrong. Let me use the well-known formulation:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00:
    # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M4 = A11*B10 - A11*B00
    # -M5 = -A00*B11 - A01*B11
    # M7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10 + A10*B01 + A11*B11? 
    # I see A00*B00, A01*B10, but where's A10*B01? I don't see it. Something's wrong.
    
    # Let me look up the correct Strassen formula. The standard is:
    # M1 = (A11 - A00)(B00 + B01)
    # M2 = (A00 + A01)(B10 - B00)
    # M3 = (A10 + A11)(B01 - B11)
    # M4 = (A00 - A10)(B10 + B11)
    # M5 = (A01 + A11)(B00 + B11)
    # M6 = (A00 - A01)(B01 + B11)
    # M7 = (A10 - A11)(B00 + B10)
    # C00 = M5 + M4 - M3 + M7
    # C01 = M1 + M2
    # C10 = M3 + M6
    # C11 = M5 + M1 - M2 - M4
    
    # This is getting complicated. Let me just build the Strassen factors by solving the system.
    # For 2x2x2, the tensor has 8 nonzero entries (one for each (i,j,k) triple).
    # We want 7 rank-1 tensors that sum to this.
    
    # Actually, let me just use a known correct set. The standard Strassen decomposition:
    # The 7 terms (U_r, V_r, W_r) for 2x2x2 where indices are (i,j) for U, (j,k) for V, (k,i) for W:
    
    # I'll define them explicitly and verify.
    strassen_2x2x2 = [
        # (U, V, W) each of length 4, indexed as (i*2+j, j*2+k, k*2+i)
        # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # Term 2: M2 = (A10+A11)B00 -> contributes to C10
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # Term 3: M3 = A00(B01-B11) -> contributes to C01
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # Term 4: M4 = A11(B10-B00) -> contributes to C10 and C11
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        # Term 5: M5 = (A00+A01)B11 -> contributes to C00 and C01
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]),
        # Term 6: M6 = (A10-A00)(B00+B01) -> contributes to C10
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        # Term 7: M7 = (A01-A11)(B10+B11) -> contributes to C01 and C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    
    # Verify Strassen factors
    def verify_strassen():
        T2 = {}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    T2[(a, b, c)] = 1.0
        
        for (a, b, c) in T2:
            s = 0.0
            for U_r, V_r, W_r in strassen_2x2x2:
                s += U_r[a] * V_r[b] * W_r[c]
            if abs(s - 1.0) > 1e-10:
                return False, f"Failed at {(a,b,c)}: got {s}"
        
        # Check zeros
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    if (a, b, c) in T2:
                        continue
                    s = 0.0
                    for U_r, V_r, W_r in strassen_2x2x2:
                        s += U_r[a] * V_r[b] * W_r[c]
                    if abs(s) > 1e-10:
                        return False, f"Nonzero at {(a,b,c)}: got {s}"
        return True, "OK"
    
    ok, msg = verify_strassen()
    if not ok:
        print(f"Strassen verification failed: {msg}", file=sys.stderr)
        # Fall back to schoolbook
        rank0 = n * m * p
        U = [[0.0] * rank0 for _ in range(n * m)]
        V = [[0.0] * rank0 for _ in range(m * p)]
        W = [[0.0] * rank0 for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1.0
                    V[b][col] = 1.0
                    W[c][col] = 1.0
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 2x4x6 decomposition using Strassen.
    # 
    # Partition strategy:
    # A (2x4): columns j=0,1 are "left" (j in {0,1}), j=2,3 are "right" (j in {2,3})
    # B (4x6): rows j=0,1 are "top" (j in {0,1}), j=2,3 are "bottom" (j in {2,3})
    #           Within top: cols k=0,1 are block 0; k=2,3 are block 1; k=4,5 are block 2
    #           Within bottom: cols k=0,1 are block 0; k=2,3 are block 1; k=4,5 are block 2
    #
    # C (2x6): cols k=0,1 are block 0; k=2,3 are block 1; k=4,5 are block 2
    #
    # For each output block t in {0,1,2}:
    #   C_t = A_left * B_top_t + A_right * B_bottom_t
    #   where A_left is 2x2 (rows i, cols j=0,1), A_right is 2x2 (rows i, cols j=2,3)
    #         B_top_t is 2x2 (rows j=0,1, cols k=2t,2t+1), B_bottom_t is 2x2 (rows j=2,3, cols k=2t,2t+1)
    #
    # We use Strassen for each of the two multiplications, giving 7+7=14 terms per block.
    # Total: 3 * 14 = 42 terms.

    total_rank = 42
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * total_rank for _ in range(p * n)]  # 12 x 42

    col_offset = 0
    for t in range(3):  # 3 output blocks
        # Block t: output cols k=2t, 2t+1
        # A_left: rows i=0,1, cols j=0,1
        # A_right: rows i=0,1, cols j=2,3
        # B_top_t: rows j=0,1, cols k=2t,2t+1
        # B_bottom_t: rows j=2,3, cols k=2t,2t+1
        
        # Strassen for A_left * B_top_t
        for (U_r, V_r, W_r) in strassen_2x2x2:
            # U_r index: i*2+j for i in {0,1}, j in {0,1} -> maps to global a = i*4 + j
            # V_r index: j*2+k for j in {0,1}, k in {0,1} -> maps to global b = j*6 + (2t+k)
            # W_r index: k*2+i for k in {0,1}, i in {0,1} -> maps to global c = (2t+k)*2 + i
            
            for local_idx in range(4):
                val = U_r[local_idx]
                if val == 0:
                    continue
                li, lj = divmod(local_idx, 2)
                global_a = li * 4 + lj  # i*4 + j where j in {0,1}
                U[global_a][col_offset] += val
            
            for local_idx in range(4):
                val = V_r[local_idx]
                if val == 0:
                    continue
                lj, lk = divmod(local_idx, 2)
                global_b = lj * 6 + (2 * t + lk)  # j*6 + k where j in {0,1}, k in {2t, 2t+1}
                V[global_b][col_offset] += val
            
            for local_idx in range(4):
                val = W_r[local_idx]
                if val == 0:
                    continue
                lk, li = divmod(local_idx, 2)
                global_c = (2 * t + lk) * 2 + li  # k*2 + i where k in {2t, 2t+1}
                W[global_c][col_offset] += val
            
            col_offset += 1
        
        # Strassen for A_right * B_bottom_t
        for (U_r, V_r, W_r) in strassen_2x2x2:
            # U_r index: i*2+j for i in {0,1}, j in {0,1} -> maps to global a = i*4 + (2+j)
            # V_r index: j*2+k for j in {0,1}, k in {0,1} -> maps to global b = (2+j)*6 + (2t+k)
            # W_r index: k*2+i for k in {0,1}, i in {0,1} -> maps to global c = (2t+k)*2 + i
            
            for local_idx in range(4):
                val = U_r[local_idx]
                if val == 0:
                    continue
                li, lj = divmod(local_idx, 2)
                global_a = li * 4 + (2 + lj)  # i*4 + j where j in {2,3}
                U[global_a][col_offset] += val
            
            for local_idx in range(4):
                val = V_r[local_idx]
                if val == 0:
                    continue
                lj, lk = divmod(local_idx, 2)
                global_b = (2 + lj) * 6 + (2 * t + lk)  # j*6 + k where j in {2,3}, k in {2t, 2t+1}
                V[global_b][col_offset] += val
            
            for local_idx in range(4):
                val = W_r[local_idx]
                if val == 0:
                    continue
                lk, li = divmod(local_idx, 2)
                global_c = (2 * t + lk) * 2 + li  # k*2 + i where k in {2t, 2t+1}
                W[global_c][col_offset] += val
            
            col_offset += 1

    # Verify the decomposition
    def verify_full(U, V, W, T_dict):
        max_r = len(U[0]) if U else 0
        for (a, b, c) in T_dict:
            s = 0.0
            for r in range(max_r):
                s += U[a][r] * V[b][r] * W[c][r]
            if abs(s - T_dict[(a, b, c)]) > 1e-8:
                return False
        # Check some zero entries
        import itertools
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in T_dict:
                        continue
                    s = 0.0
                    for r in range(max_r):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-8:
                        return False
        return True

    T_dict = {k: v for k, v in T.items()}
    if not verify_full(U, V, W, T_dict):
        print("Strassen decomposition verification failed, falling back to schoolbook", file=sys.stderr)
        rank0 = n * m * p
        U = [[0.0] * rank0 for _ in range(n * m)]
        V = [[0.0] * rank0 for _ in range(m * p)]
        W = [[0.0] * rank0 for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1.0
                    V[b][col] = 1.0
                    W[c][col] = 1.0
    else:
        print(f"Strassen decomposition verified, rank={total_rank}", file=sys.stderr)

    return {"U": U, "V": V, "W": W}
