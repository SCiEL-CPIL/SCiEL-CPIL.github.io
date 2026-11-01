"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_block():
    """Return (U, V, W) for a 2x2x2 Strassen decomposition, rank 7.
    U has shape (2*2, 7) = (4,7), indexed by [i*2+j, r]
    V has shape (2*2, 7) = (4,7), indexed by [j*2+k, r]
    W has shape (2*2, 7) = (4,7), indexed by [k*2+i, r]
    Convention: W[k*n+i] with n=2.
    """
    # Strassen products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P7
    U = [
        [1, 0, 0, 0, 1, 1, 0],  # A00
        [0, 0, 1, 0, 0, 0, 1],  # A01
        [1, 1, 0, 0, 0, 0, -1], # A10
        [1, 1, 0, 1, 0, 0, -1], # A11
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [0, 0, 1, 1, 0, 1, 0],  # B01
        [1, 0, 1, -1, 0, 0, 0], # B10
        [1, 0, -1, 0, 1, 0, 1], # B11
    ]
    W = [
        [1, 0, 0, -1, 1, 0, -1], # C00 (k=0,i=0)
        [0, 0, 1, 1, 0, 0, 0],   # C01 (k=1,i=0)
        [0, 1, 0, 0, 1, 0, 0],   # C10 (k=0,i=1)
        [1, -1, 1, 0, 0, 0, 1],  # C11 (k=1,i=1)
    ]
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140

    # Decomposition: j = {0,1} (Strassen block 1), {2,3} (Strassen block 2), {4} (remainder)
    # k = {0,1} (block 1), {2,3} (block 2), {4,5} (block 3)
    # Block 1: j∈{0,1}, k∈{0,1} → Strassen rank 7
    # Block 2: j∈{2,3}, k∈{2,3} → Strassen rank 7
    # Block 3: j∈{4}, k∈{4,5} → schoolbook rank 2
    # Total rank: 7 + 7 + 2 = 16? No wait...
    
    # Actually let me reconsider. The tensor equation is:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This means: C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We need: C[i,k] = sum_j A[i,j]*B[j,k]
    # So for each (i,j,k) triple, the sum over r must equal 1.
    
    # For a Strassen block covering j∈{j1,j2} and k∈{k1,k2}:
    # For i∈{0,1}, j∈{j1,j2}, k∈{k1,k2}:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is exactly the 2x2x2 multiplication tensor condition.
    
    # But wait - the Strassen decomposition satisfies:
    # For the 2x2x2 tensor: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k in {0,1}
    # So if I map local (i,j,k) to global, it works.
    
    # Let me set up:
    # Block 1: j_local = j - 0 (j=0,1), k_local = k - 0 (k=0,1)
    # Block 2: j_local = j - 2 (j=2,3), k_local = k - 2 (k=2,3)
    # Block 3: j_local = j - 4 (j=4), k_local = k - 4 (k=4,5) → this is 1x2, schoolbook rank 2
    
    # Total rank = 7 + 7 + 2 = 16
    
    # Wait, that seems too low. Let me re-examine.
    # The schoolbook rank is n*m*p = 2*5*6 = 60.
    # Each (i,j,k) triple needs the sum to be 1.
    # For a Strassen block with j∈{j1,j2}, k∈{k1,k2}: it handles 2*2*2 = 8 triples (for i=0,1)
    # with rank 7 instead of 8. Saves 1.
    # For block 3: j=4, k∈{4,5}: handles 2*1*2 = 4 triples with rank 2 (schoolbook). No saving.
    # Total: 8+8+4 = 20 triples handled by rank 7+7+2 = 16.
    # But we have 2*5*6 = 60 triples total!
    
    # I'm confusing myself. Let me restart.
    # The condition is: for ALL (i,j,k) with i∈[0,2), j∈[0,5), k∈[0,6):
    # sum_r U[i*5+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1
    # That's 60 equations.
    
    # A Strassen block for j∈{j1,j2}, k∈{k1,k2} satisfies these 8 equations (2 i × 2 j × 2 k)
    # using 7 rank terms. But those 7 rank terms also contribute to OTHER (i,j',k') equations
    # where j'∉{j1,j2} or k'∉{k1,k2}!
    
    # Wait no - U[i*m+j, r] is zero for j not in the block's j-range (if we set it up that way).
    # Similarly V[j*p+k, r] is zero for k not in the block's k-range.
    # So each block's rank terms only affect the (i,j,k) triples within that block.
    
    # So: Block 1 (j∈{0,1}, k∈{0,1}): 8 triples, rank 7
    # Block 2 (j∈{2,3}, k∈{2,3}): 8 triples, rank 7
    # Block 3 (j∈{4}, k∈{4,5}): 4 triples, rank 2 (schoolbook for 2x1x2)
    # Remaining: j∈{0,1}, k∈{2,3,4,5}: 2*4*2 = 16 triples, schoolbook rank 16
    #            j∈{2,3}, k∈{0,1,4,5}: 2*4*2 = 16 triples, schoolbook rank 16
    #            j∈{4}, k∈{0,1,2,3}: 2*4*1 = 8 triples, schoolbook rank 8
    # Total: 7+7+2+16+16+8 = 56
    
    # Hmm, that's 56. The finding said 54. Let me re-read:
    # "Applying Strassen's rank-7 decomposition to all six disjoint 2x2x2 sub-blocks
    # (j-pairs (0,1),(2,3) crossed with k-pairs (0,1),(2,3),(4,5)) reduces rank from 58 to 54"
    # So they used 6 Strassen blocks: j∈{0,1}×k∈{0,1}, j∈{0,1}×k∈{2,3}, j∈{0,1}×k∈{4,5},
    # j∈{2,3}×k∈{0,1}, j∈{2,3}×k∈{2,3}, j∈{2,3}×k∈{4,5}
    # That's 6 blocks × rank 7 = 42, plus j=4 remainder: 2*1*6 = 12 schoolbook = 12
    # Total: 42 + 12 = 54. Yes!
    
    # Let me build this:
    # 6 Strassen blocks for j∈{0,1,2,3} and k∈{0,1,2,3,4,5} in pairs
    # 1 schoolbook block for j=4, k∈{0,1,2,3,4,5}
    
    rank_total = 6 * 7 + 1 * (n * 1 * p)  # 42 + 12 = 54
    rank_total = 6 * 7 + n * 1 * p  # 42 + 12 = 54
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    sU, sV, sW = _strassen_block()
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for r in range(7):
                for i in range(n):
                    for j_loc in range(2):
                        j = j1 + j_loc
                        U[i * m + j][col] = sU[i * 2 + j_loc][r]
                    for k_loc in range(2):
                        k = k1 + k_loc
                        V[j1 * p + k][col] = sV[0 * 2 + k_loc][r]  # j_loc=0
                        V[j2 * p + k][col] = sV[1 * 2 + k_loc][r]  # j_loc=1
                    for k_loc in range(2):
                        k = k1 + k_loc
                        W[k * n + i][col] = sW[k_loc * 2 + i][r]
            col += 7
    
    # Schoolbook for j=4, k=0..5
    for k in range(p):
        for i in range(n):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Verify
    valid, err = _verify(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
