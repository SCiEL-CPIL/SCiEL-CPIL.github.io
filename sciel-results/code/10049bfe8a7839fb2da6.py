import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # We'll use 4 Strassen blocks (2x2x2 each) + schoolbook for remaining terms
    # Strassen blocks at (i0, j0, k0):
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    # Now try to improve by finding linear dependencies among the columns
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the tensor as a matrix: each entry (i,j,k) corresponds to a row
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find if any column is a linear combination of others
    
    # For each triple (i,j,k), we have:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a 3-tensor. To find rank reductions, we can look for
    # linear dependencies in the "flattened" views.
    
    # Alternative approach: try to find if we can merge some schoolbook terms
    # with Strassen terms by finding linear combinations.
    
    # Let's try a different approach: use the fact that for small matrices,
    # we can sometimes find better decompositions.
    
    # For 3x5x5, the trivial rank is 75. With 4 Strassen blocks (28) + 43 schoolbook = 71.
    # Can we do better?
    
    # Let's try to see if we can use overlapping Strassen blocks or other tricks.
    
    # Actually, let's try to find if any of the schoolbook terms can be expressed
    # as linear combinations of the Strassen terms.
    
    # For each uncovered (i,j,k), check if the vector (U[i*m+j,:], V[j*p+k,:], W[k*n+i,:])
    # can be formed from existing columns.
    
    # This is complex. Let's instead try a different block arrangement.
    
    # Try different block placements to see if we can cover more terms.
    
    best_rank = total_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try all possible placements of 4 non-overlapping 2x2x2 blocks
    # in a 3x5x5 grid.
    
    # A 2x2x2 block at (i0,j0,k0) covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # For non-overlapping, we need |i0 - i0'| >= 2 or |j0 - j0'| >= 2 or |k0 - k0'| >= 2
    
    # Possible i0: 0,1 (since i0+1 <= 2)
    # Possible j0: 0,1,2,3 (since j0+1 <= 4)
    # Possible k0: 0,1,2,3 (since k0+1 <= 4)
    
    # Let's enumerate all sets of 4 non-overlapping blocks.
    
    import itertools
    
    all_positions = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    def blocks_overlap(b1, b2):
        i0_1, j0_1, k0_1 = b1
        i0_2, j0_2, k0_2 = b2
        # Blocks overlap if their ranges intersect in all dimensions
        # Range for block (i0,j0,k0) is [i0,i0+1] x [j0,j0+1] x [k0,k0+1]
        # They don't overlap if in any dimension, the ranges are disjoint
        # [i0_1, i0_1+1] and [i0_2, i0_2+1] are disjoint if i0_1+1 < i0_2 or i0_2+1 < i0_1
        if i0_1 + 1 < i0_2 or i0_2 + 1 < i0_1:
            return False
        if j0_1 + 1 < j0_2 or j0_2 + 1 < j0_1:
            return False
        if k0_1 + 1 < k0_2 or k0_2 + 1 < k0_1:
            return False
        return True
    
    # Find all sets of 4 mutually non-overlapping blocks
    candidate_sets = []
    for combo in itertools.combinations(all_positions, 4):
        # Check all pairs
        valid = True
        for idx in range(4):
            for jdx in range(idx+1, 4):
                if blocks_overlap(combo[idx], combo[jdx]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            candidate_sets.append(combo)
    
    # For each candidate set, compute the rank
    for combo in candidate_sets:
        if time.time() > deadline:
            break
        
        U2 = [[0.0] * (4*7 + 43) for _ in range(n * m)]
        V2 = [[0.0] * (4*7 + 43) for _ in range(m * p)]
        W2 = [[0.0] * (4*7 + 43) for _ in range(p * n)]
        
        col2 = 0
        for i0, j0, k0 in combo:
            add_strassen_block(i0, j0, k0, col2)
            col2 += 7
        
        covered_3d2 = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in combo:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered_3d2[i0+di][j0+dj][k0+dk] = True
        
        schoolbook_count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d2[i][j][k]:
                        schoolbook_count += 1
        
        # The rank is 28 + schoolbook_count
        rank2 = 28 + schoolbook_count
        if rank2 < best_rank:
            best_rank = rank2
            # Rebuild with correct size
            U2 = [[0.0] * rank2 for _ in range(n * m)]
            V2 = [[0.0] * rank2 for _ in range(m * p)]
            W2 = [[0.0] * rank2 for _ in range(p * n)]
            
            col2 = 0
            for i0, j0, k0 in combo:
                add_strassen_block(i0, j0, k0, col2)
                col2 += 7
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not covered_3d2[i][j][k]:
                            U2[i*m+j][col2] += 1
                            V2[j*p+k][col2] += 1
                            W2[k*n+i][col2] += 1
                            col2 += 1
            
            best_result = {"U": U2, "V": V2, "W": W2}
    
    return best_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify correctness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    print(f"Max error: {max_err}")
