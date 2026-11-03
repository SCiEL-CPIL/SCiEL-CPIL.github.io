import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Target tensor entries: T[i*m+j, j*p+k, k*n+i] = 1
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.add((i * m + j, j * p + k, k * n + i))

    def verify_solution(sol):
        """Strict verification of tensor decomposition."""
        U = sol["U"]; V = sol["V"]; W = sol["W"]
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Check all target entries
        for (u, v, w) in target_entries:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False

        # Check non-target entries
        for u in range(n * m):
            Uu = U_np[u]
            for v in range(m * p):
                UV = Uu * V_np[v]
                if not np.any(UV):
                    continue
                for w in range(p * n):
                    if (u, v, w) in target_entries:
                        continue
                    val = float(np.dot(UV, W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    def build_solution_from_blocks(i_blocks, j_blocks, k_blocks, strassen_blocks):
        """
        Build a solution by partitioning dimensions into blocks.
        Strassen is applied to blocks listed in strassen_blocks (tuples of (i_idx, j_idx, k_idx) indices into the block lists).
        All other blocks use schoolbook (rank 1 per term).
        """
        # Determine total rank
        total_rank = 0
        for i_idx, i_range in enumerate(i_blocks):
            for j_idx, j_range in enumerate(j_blocks):
                for k_idx, k_range in enumerate(k_blocks):
                    if (i_idx, j_idx, k_idx) in strassen_blocks:
                        total_rank += 7
                    else:
                        total_rank += len(i_range) * len(j_range) * len(k_range)
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for i_idx, i_range in enumerate(i_blocks):
            for j_idx, j_range in enumerate(j_blocks):
                for k_idx, k_range in enumerate(k_blocks):
                    if (i_idx, j_idx, k_idx) in strassen_blocks:
                        # Apply Strassen to this 2x2x2 block
                        i0, i1 = i_range
                        j0, j1 = j_range
                        k0, k1 = k_range
                        
                        # P1 = (a+d)(e+h) -> W[k0,i0], W[k1,i1]
                        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
                        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
                        col += 1
                        
                        # P2 = (c+d)(e-f) -> W[k0,i1], W[k0,i0] (with signs)
                        # C = i1, A=i0, B=j1, D=j0? No, standard Strassen:
                        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
                        # P1 = (a+d)(e+h)
                        # P2 = (c+d)(e-f)
                        # P3 = (a-b)(g+h)
                        # P4 = (c-a)(f-h)
                        # P5 = (a+b)(g-h)
                        # P6 = (c+d)(f-h)
                        # P7 = (a-d)(e+g)
                        
                        # Let's re-verify the W contributions for 2x2x2 Strassen.
                        # Output C = [[c00, c01], [c10, c11]]
                        # c00 = P1 - P2 - P4 + P7
                        # c01 = P3 + P4
                        # c10 = P2 + P5
                        # c11 = P1 - P3 - P5 + P6
                        
                        # My previous code for P1 was: W[k0*n+i0] += 1; W[k1*n+i1] += 1. This is WRONG for standard Strassen.
                        # Standard Strassen terms are rank-1 tensors in the 2x2x2 space.
                        # Let's define the 7 rank-1 tensors for 2x2x2 exactly.
                        
                        # R1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] -> (a+d)(e+h) contributes to c00 and c11?
                        # Actually, the rank-1 decomposition of the 2x2x2 multiplication tensor is:
                        # T = sum_{r=1}^7 U_r V_r W_r
                        # Where U_r, V_r, W_r are vectors of length 4 (flattened 2x2).
                        
                        # Let's use the known explicit Strassen tensors.
                        # Indices: i in {0,1}, j in {0,1}, k in {0,1}.
                        # U vector has entries for (i,j). V for (j,k). W for (k,i).
                        
                        # Term 1: (a+d)(e+h)
                        # U: (0,0):1, (1,1):1
                        # V: (0,0):1, (1,1):1
                        # W: (0,0):1, (1,1):1  -> This gives c00 += (a+d)(e+h) and c11 += (a+d)(e+h).
                        # But we need c00 = P1 - P2 - P4 + P7 and c11 = P1 - P3 - P5 + P6.
                        # So W1 for P1 should be (0,0):1, (1,1):1.
                        
                        # Term 2: (c+d)(e-f)
                        # U: (1,0):1, (1,1):1
                        # V: (0,0):1, (0,1):-1
                        # W: (0,0):-1, (0,1):1 ? 
                        # Let's check: c00 gets -P2. c10 gets +P2.
                        # c00 is at (i=0, k=0). c10 is at (i=1, k=0).
                        # So W2 should have W[0,0] = -1, W[1,0] = 1.
                        
                        # Term 3: (a-b)(g+h)
                        # U: (0,0):1, (0,1):-1
                        # V: (1,0):1, (1,1):1
                        # W: (0,1):1, (1,1):-1 ?
                        # c01 gets +P3. c11 gets -P3.
                        # c01 is (i=0, k=1). c11 is (i=1, k=1).
                        # So W3 should have W[0,1] = 1, W[1,1] = -1.
                        
                        # Term 4: (c-a)(f-h)
                        # U: (1,0):1, (0,0):-1
                        # V: (0,1):1, (1,1):-1
                        # W: (0,0):-1, (0,1):1 ?
                        # c00 gets -P4. c01 gets +P4.
                        # c00 is (0,0). c01 is (0,1).
                        # So W4 should have W[0,0] = -1, W[0,1] = 1.
                        
                        # Term 5: (a+b)(g-h)
                        # U: (0,0):1, (0,1):1
                        # V: (1,0):1, (1,1):-1
                        # W: (1,0):1, (1,1):1 ?
                        # c10 gets +P5. c11 gets -P5.
                        # c10 is (1,0). c11 is (1,1).
                        # So W5 should have W[1,0] = 1, W[1,1] = -1.
                        
                        # Term 6: (c+d)(f-h)
                        # U: (1,0):1, (1,1):1
                        # V: (0,1):1, (1,1):-1
                        # W: (1,1):1, (1,0):-1 ?
                        # c11 gets +P6. c10 gets -P6.
                        # c11 is (1,1). c10 is (1,0).
                        # So W6 should have W[1,1] = 1, W[1,0] = -1.
                        
                        # Term 7: (a-d)(e+g)
                        # U: (0,0):1, (1,1):-1
                        # V: (0,0):1, (1,0):1
                        # W: (0,0):1, (0,1):-1 ?
                        # c00 gets +P7. c01 gets -P7.
                        # c00 is (0,0). c01 is (0,1).
                        # So W7 should have W[0,0] = 1, W[0,1] = -1.

                        # Let's implement these 7 terms explicitly.
                        # P1
                        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
                        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
                        col += 1
                        
                        # P2
                        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
                        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += -1
                        W[k0*n+i0, col] += -1; W[k0*n+i1, col] += 1
                        col += 1
                        
                        # P3
                        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += -1
                        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += -1
                        col += 1
                        
                        # P4
                        U[i1*m+j0, col] += 1; U[i0*m+j0, col] += -1
                        V[j0*p+k1, col] += 1; V[j1*p+k1, col] += -1
                        W[k0*n+i0, col] += -1; W[k1*n+i0, col] += 1
                        col += 1
                        
                        # P5
                        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
                        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += -1
                        W[k0*n+i1, col] += 1; W[k1*n+i1, col] += -1
                        col += 1
                        
                        # P6
                        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
                        V[j0*p+k1, col] += 1; V[j1*p+k1, col] += -1
                        W[k1*n+i1, col] += 1; W[k0*n+i1, col] += -1
                        col += 1
                        
                        # P7
                        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += -1
                        V[j0*p+k0, col] += 1; V[j1*p+k0, col] += 1
                        W[k0*n+i0, col] += 1; W[k1*n+i0, col] += -1
                        col += 1
                        
                    else:
                        # Schoolbook for this block
                        for i in i_range:
                            for j in j_range:
                                for k in k_range:
                                    U[i*m+j, col] = 1
                                    V[j*p+k, col] = 1
                                    W[k*n+i, col] = 1
                                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Candidate partitions
    # We partition the 4x4x7 space into 2x2x2 blocks where possible.
    # Since p=7, we can have three 2x2x2 blocks (covering k=0..5) and one 2x2x1 block (k=6).
    # Or mix them.
    
    # Strategy: Try all combinations of which 2x2x2 sub-blocks to apply Strassen to.
    # Possible 2x2x2 blocks are defined by:
    # i in {0,1} or {2,3}
    # j in {0,1} or {2,3}
    # k in {0,1}, {2,3}, {4,5}
    
    # Let's define the block grid.
    # i_blocks: [(0,1), (2,3)]
    # j_blocks: [(0,1), (2,3)]
    # k_blocks: [(0,1), (2,3), (4,5), (6,)]  # Last one is size 1
    
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5), (6,)]
    
    # We can only apply Strassen to blocks where all dimensions are size 2.
    # So valid Strassen blocks are (i_idx, j_idx, k_idx) where i_idx in {0,1}, j_idx in {0,1}, k_idx in {0,1,2}.
    # There are 2*2*3 = 12 potential Strassen blocks.
    # We want to select a subset of these to maximize reduction.
    # Each Strassen block replaces 8 schoolbook terms with 7, saving 1 rank.
    # However, we must ensure we don't double-count or miss terms.
    # The partition is disjoint, so we just decide for each potential 2x2x2 block: Strassen or Schoolbook.
    # The remaining terms (in the 2x2x1 block at k=6) must be schoolbook.
    
    # Actually, we can also mix. For example, use Strassen for k=0,1 and schoolbook for k=2,3,4,5,6.
    # Or Strassen for all 12 potential blocks?
    # If we use Strassen for all 12, we cover k=0..5 completely with Strassen.
    # The k=6 slice (2x2x1 for each i,j block) is schoolbook.
    
    # Let's try the "Max Strassen" approach: Apply Strassen to ALL possible 2x2x2 blocks.
    # i_blocks: 2 blocks of size 2.
    # j_blocks: 2 blocks of size 2.
    # k_blocks: 3 blocks of size 2, 1 block of size 1.
    
    # Potential Strassen blocks:
    # (0,0,0), (0,0,1), (0,0,2)
    # (0,1,0), (0,1,1), (0,1,2)
    # (1,0,0), (1,0,1), (1,0,2)
    # (1,1,0), (1,1,1), (1,1,2)
    
    # If we apply Strassen to all 12, rank = 12*7 + (terms in k=6 slice).
    # k=6 slice: i in {0,1,2,3}, j in {0,1,2,3}, k=6.
    # This is 4*4*1 = 16 terms.
    # Total rank = 84 + 16 = 100.
    
    # Let's verify this solution.
    strassen_blocks = set()
    for i_idx in range(2):
        for j_idx in range(2):
            for k_idx in range(3): # k=0,1,2 correspond to k_blocks 0,1,2
                strassen_blocks.add((i_idx, j_idx, k_idx))
                
    sol_max_strassen = build_solution_from_blocks(i_blocks, j_blocks, k_blocks, strassen_blocks)
    
    if verify_solution(sol_max_strassen):
        return sol_max_strassen
        
    # Fallback: Try a subset, e.g., only Strassen on first k-block.
    strassen_blocks_sub = set()
    for i_idx in range(2):
        for j_idx in range(2):
            strassen_blocks_sub.add((i_idx, j_idx, 0))
            
    sol_sub = build_solution_from_blocks(i_blocks, j_blocks, k_blocks, strassen_blocks_sub)
    if verify_solution(sol_sub):
        return sol_sub

    # Fallback: Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    sol_sb = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    if verify_solution(sol_sb):
        return sol_sb
    return sol_sb
# EVOLVE-BLOCK-END
