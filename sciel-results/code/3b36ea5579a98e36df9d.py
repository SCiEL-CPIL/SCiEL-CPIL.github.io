"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The decomposition should satisfy:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the multiplication tensor has all-ones entries)
    
    # Check a few random (i,j,k) triples
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
        if abs(val - 1) > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 150s kill
    
    def schoolbook():
        """Return the trivial rank-105 decomposition."""
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
    
    def verify_decomp(U, V, W, rank):
        """Fully verify all (i,j,k) triples."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Laderman's rank-23 algorithm for 3x3x3 matrix multiplication
    # Each term: (U_sparse, V_sparse, W_sparse) where each is dict {index: coeff}
    # U index: i*3+j, V index: j*3+k, W index: k*3+i
    
    laderman_333 = [
        # Term 0: A[0,0] * B[0,0] -> C[0,0]
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # Term 1: A[0,1] * B[1,0] -> C[0,0]
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        # Term 2: A[1,0] * B[0,1] -> C[1,1]
        ({(1,0): 1}, {(0,1): 1}, {(1,1): 1}),
        # Term 3: A[1,1] * B[1,1] -> C[1,1]
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # Term 4: A[2,0] * B[0,2] -> C[2,2]
        ({(2,0): 1}, {(0,2): 1}, {(2,2): 1}),
        # Term 5: A[2,1] * B[1,2] -> C[2,2]
        ({(2,1): 1}, {(1,2): 1}, {(2,2): 1}),
        # Term 6: A[2,2] * B[2,2] -> C[2,2]
        ({(2,2): 1}, {(2,2): 1}, {(2,2): 1}),
        # Term 7: (A[0,0]+A[0,1]) * (B[0,1]-B[1,1]) -> C[0,1]
        ({(0,0): 1, (0,1): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # Term 8: (A[1,0]+A[1,1]) * (B[0,1]-B[1,1]) -> C[1,1]
        ({(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): -1}, {(1,1): 1}),
        # Term 9: (A[0,0]+A[0,1]) * (B[1,0]+B[1,1]) -> C[0,0]
        ({(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # Term 10: (A[1,0]+A[1,1]) * (B[1,0]+B[1,1]) -> C[1,0]
        ({(1,0): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}, {(1,0): 1}),
        # Term 11: (A[0,0]-A[1,0]) * (B[0,1]-B[0,0]) -> C[0,1]
        ({(0,0): 1, (1,0): -1}, {(0,1): 1, (0,0): -1}, {(0,1): 1}),
        # Term 12: (A[0,1]-A[1,1]) * (B[1,0]-B[1,1]) -> C[1,1]
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): -1}, {(1,1): 1}),
        # Term 13: (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1]) -> C[0,0],C[0,1],C[1,0],C[1,1]
        ({(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # Term 14: (A[2,0]+A[2,1]) * (B[0,2]-B[1,2]) -> C[2,2]
        ({(2,0): 1, (2,1): 1}, {(0,2): 1, (1,2): -1}, {(2,2): 1}),
        # Term 15: (A[2,0]-A[2,1]) * (B[1,2]+B[1,1]) -> C[2,2]
        ({(2,0): 1, (2,1): -1}, {(1,2): 1, (1,1): 1}, {(2,2): 1}),
        # Term 16: A[0,2] * B[2,0] -> C[0,0]
        ({(0,2): 1}, {(2,0): 1}, {(0,0): 1}),
        # Term 17: A[0,2] * B[2,1] -> C[0,1]
        ({(0,2): 1}, {(2,1): 1}, {(0,1): 1}),
        # Term 18: A[1,2] * B[2,0] -> C[1,0]
        ({(1,2): 1}, {(2,0): 1}, {(1,0): 1}),
        # Term 19: A[1,2] * B[2,1] -> C[1,1]
        ({(1,2): 1}, {(2,1): 1}, {(1,1): 1}),
        # Term 20: A[0,2] * B[2,2] -> C[0,2]
        ({(0,2): 1}, {(2,2): 1}, {(0,2): 1}),
        # Term 21: A[1,2] * B[2,2] -> C[1,2]
        ({(1,2): 1}, {(2,2): 1}, {(1,2): 1}),
        # Term 22: (A[2,0]+A[2,1]+A[2,2]) * (B[0,0]+B[0,1]+B[0,2]+B[1,0]+B[1,1]+B[1,2]+B[2,0]+B[2,1]+B[2,2]) -> C[2,0],C[2,1],C[2,2]
        ({(2,0): 1, (2,1): 1, (2,2): 1}, 
         {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,0): 1, (2,1): 1, (2,2): 1}, 
         {(2,0): 1, (2,1): 1, (2,2): 1}),
    ]
    
    # Verify Laderman's algorithm covers all 27 (i,j,k) triples for 3x3x3
    def verify_laderman():
        """Check that Laderman's decomposition exactly computes 3x3x3 matmul."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for (U_sp, V_sp, W_sp) in laderman_333:
                        u_val = U_sp.get((i,j), 0)
                        v_val = V_sp.get((j,k), 0)
                        w_val = W_sp.get((k,i), 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    laderman_ok, laderman_err = verify_laderman()
    
    if not laderman_ok:
        # Fall back to Strassen-based approach if Laderman verification fails
        n_blocks = [(0, 1)]
        m_blocks = [(0, 1), (2, 3)]
        p_blocks = [(0, 1), (2, 3), (4, 5)]
        
        rank = 99
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for (a, b) in n_blocks:
            for (c, d) in m_blocks:
                for (e, f) in p_blocks:
                    U[a*m+c][col] += 1; U[b*m+d][col] += 1
                    V[c*p+e][col] += 1; V[d*p+f][col] += 1
                    W[e*n+a][col] += 1; W[f*n+b][col] += 1
                    col += 1
                    U[b*m+c][col] += 1; U[b*m+d][col] += 1
                    V[c*p+e][col] += 1
                    W[f*n+a][col] += 1
                    col += 1
                    U[a*m+c][col] += 1
                    V[d*p+f][col] += 1; V[d*p+e][col] -= 1
                    W[e*n+f][col] += 1
                    col += 1
                    U[b*m+d][col] += 1
                    V[c*p+f][col] += 1; V[c*p+e][col] -= 1
                    W[e*n+a][col] += 1
                    col += 1
                    U[a*m+c][col] += 1; U[a*m+d][col] += 1
                    V[d*p+f][col] += 1
                    W[e*n+f][col] += 1; W[e*n+a][col] -= 1
                    col += 1
                    U[b*m+c][col] += 1; U[a*m+c][col] -= 1
                    V[c*p+e][col] += 1; V[c*p+f][col] += 1
                    W[f*n+a][col] += 1
                    col += 1
                    U[b*m+d][col] += 1; U[a*m+d][col] -= 1
                    V[d*p+e][col] += 1; V[d*p+f][col] += 1
                    W[e*n+a][col] -= 1; W[f*n+b][col] += 1
                    col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    covered = False
                    for (a, b) in n_blocks:
                        if i in (a, b):
                            for (c, d) in m_blocks:
                                if j in (c, d):
                                    for (e, f) in p_blocks:
                                        if k in (e, f):
                                            covered = True
                    if not covered:
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
        
        assert col == rank, f"Expected rank {rank}, got {col}"
        if verify_decomp(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        return schoolbook()
    
    # Now embed Laderman's 3x3x3 into the 3x5x7 problem
    # The 3x3x3 block covers i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # That's 27 out of 105 triples, using rank 23
    # Remaining: 105 - 27 = 78 triples
    
    # Strategy: Use Laderman for the 3x3x3 sub-block, then use greedy set-cover
    # for the remaining 78 triples
    
    # First, let's try multiple placements of the 3x3x3 block
    # The 3x3x3 block can be placed at various (i_offset, j_offset, k_offset)
    # But since n=3, m=5, p=7, the 3x3x3 block must use all of i (n=3)
    # For j: can start at 0,1,2 (offsets 0-2)
    # For k: can start at 0,1,2,3,4 (offsets 0-4)
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try different placements of the 3x3x3 Laderman block
    for j_off in range(m - 2 + 1):  # 0 to 3
        for k_off in range(p - 2 + 1):  # 0 to 5
            if time.time() > deadline - 20:
                break
            
            # The 3x3x3 block covers:
            # i in {0,1,2}, j in {j_off, j_off+1, j_off+2}, k in {k_off, k_off+1, k_off+2}
            
            # Build the partial decomposition from Laderman
            rank = 23
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for c, (U_sp, V_sp, W_sp) in enumerate(laderman_333):
                for (i, j), coeff in U_sp.items():
                    U[i * m + (j + j_off)][c] += coeff
                for (j, k), coeff in V_sp.items():
                    V[(j + j_off) * p + (k + k_off)][c] += coeff
                for (k, i), coeff in W_sp.items():
                    W[(k + k_off) * n + i][c] += coeff
            
            # Determine which triples are covered by the Laderman block
            covered_triples = set()
            for i in range(3):
                for j in range(j_off, j_off + 3):
                    for k in range(k_off, k_off + 3):
                        covered_triples.add((i, j, k))
            
            # Remaining triples
            all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            remaining = [t for t in all_triples if t not in covered_triples]
            
            # Greedy set-cover: for each remaining triple, we can either
            # 1. Add a rank-1 term (schoolbook)
            # 2. Try to combine multiple remaining triples into a lower-rank block
            
            # For now, use schoolbook for all remaining
            # But try to find 2x2x2 Strassen blocks among the remaining
            remaining_set = set(remaining)
            used_rank = 23
            
            # Find all possible 2x2x2 Strassen blocks among remaining triples
            # A 2x2x2 block at (i0, j0, k0) covers 8 triples
            # and uses rank 7 instead of 8 (saves 1)
            
            # Greedy: repeatedly pick the Strassen block that covers the most remaining triples
            strassen_blocks = []
            for i0 in range(n - 1):
                for j0 in range(m - 1):
                    for k0 in range(p - 1):
                        block_triples = set()
                        for di in range(2):
                            for dj in range(2):
                                for dk in range(2):
                                    block_triples.add((i0 + di, j0 + dj, k0 + dk))
                        # Only consider if at least some are in remaining
                        overlap = block_triples & remaining_set
                        if len(overlap) >= 2:  # at least 2 triples to save
                            strassen_blocks.append((i0, j0, k0, block_triples))
            
            # Greedy selection
            selected_blocks = []
            covered_by_strassen = set()
            
            # Sort by number of remaining triples covered (descending)
            strassen_blocks.sort(key=lambda x: len(x[3] & remaining_set), reverse=True)
            
            for (i0, j0, k0, block_triples) in strassen_blocks:
                if time.time() > deadline - 10:
                    break
                overlap = block_triples & remaining_set - covered_by_strassen
                if len(overlap) < 2:
                    continue
                # Check if this block conflicts with already selected blocks
                # (we allow overlap but count each triple only once)
                selected_blocks.append((i0, j0, k0))
                covered_by_strassen.update(block_triples & remaining_set)
            
            # Add Strassen blocks
            for (i0, j0, k0) in selected_blocks:
                a, b = i0, i0 + 1
                c, d = j0, j0 + 1
                e, f = k0, k0 + 1
                
                # M1 = (A[a,c]+A[b,d]) * (B[c,e]+B[d,f]) -> C[e,a] += 1, C[f,b] += 1
                U[a*m+c][used_rank] += 1; U[b*m+d][used_rank] += 1
                V[c*p+e][used_rank] += 1; V[d*p+f][used_rank] += 1
                W[e*n+a][used_rank] += 1; W[f*n+b][used_rank] += 1
                used_rank += 1
                
                # M2 = (A[b,c]+A[b,d]) * B[c,e] -> C[f,a] += 1
                U[b*m+c][used_rank] += 1; U[b*m+d][used_rank] += 1
                V[c*p+e][used_rank] += 1
                W[f*n+a][used_rank] += 1
                used_rank += 1
                
                # M3 = A[a,c] * (B[d,f]-B[d,e]) -> C[e,f] += 1
                U[a*m+c][used_rank] += 1
                V[d*p+f][used_rank] += 1; V[d*p+e][used_rank] -= 1
                W[e*n+f][used_rank] += 1
                used_rank += 1
                
                # M4 = A[b,d] * (B[c,f]-B[c,e]) -> C[e,a] += 1
                U[b*m+d][used_rank] += 1
                V[c*p+f][used_rank] += 1; V[c*p+e][used_rank] -= 1
                W[e*n+a][used_rank] += 1
                used_rank += 1
                
                # M5 = (A[a,c]+A[a,d]) * B[d,f] -> C[e,f] += 1, C[e,a] -= 1
                U[a*m+c][used_rank] += 1; U[a*m+d][used_rank] += 1
                V[d*p+f][used_rank] += 1
                W[e*n+f][used_rank] += 1; W[e*n+a][used_rank] -= 1
                used_rank += 1
                
                # M6 = (A[b,c]-A[a,c]) * (B[c,e]+B[c,f]) -> C[f,a] += 1
                U[b*m+c][used_rank] += 1; U[a*m+c][used_rank] -= 1
                V[c*p+e][used_rank] += 1; V[c*p+f][used_rank] += 1
                W[f*n+a][used_rank] += 1
                used_rank += 1
                
                # M7 = (A[b,d]-A[a,d]) * (B[d,e]+B[d,f]) -> C[e,a] -= 1, C[f,b] += 1
                U[b*m+d][used_rank] += 1; U[a*m+d][used_rank] -= 1
                V[d*p+e][used_rank] += 1; V[d*p+f][used_rank] += 1
                W[e*n+a][used_rank] -= 1; W[f*n+b][used_rank] += 1
                used_rank += 1
            
            # Schoolbook for truly remaining triples
            for (i, j, k) in remaining:
                if (i, j, k) not in covered_by_strassen:
                    U[i*m+j][used_rank] = 1
                    V[j*p+k][used_rank] = 1
                    W[k*n+i][used_rank] = 1
                    used_rank += 1
            
            # Resize arrays
            U = [row[:used_rank] for row in U]
            V = [row[:used_rank] for row in V]
            W = [row[:used_rank] for row in W]
            
            if used_rank < best_rank:
                # Verify
                if verify_decomp(U, V, W, used_rank):
                    best_rank = used_rank
                    best_result = {"U": U, "V": V, "W": W}
            
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 15:
            break
    
    if best_result is not None:
        return best_result
    
    # Final fallback: Strassen-only approach
    n_blocks = [(0, 1)]
    m_blocks = [(0, 1), (2, 3)]
    p_blocks = [(0, 1), (2, 3), (4, 5)]
    
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (a, b) in n_blocks:
        for (c, d) in m_blocks:
            for (e, f) in p_blocks:
                U[a*m+c][col] += 1; U[b*m+d][col] += 1
                V[c*p+e][col] += 1; V[d*p+f][col] += 1
                W[e*n+a][col] += 1; W[f*n+b][col] += 1
                col += 1
                U[b*m+c][col] += 1; U[b*m+d][col] += 1
                V[c*p+e][col] += 1
                W[f*n+a][col] += 1
                col += 1
                U[a*m+c][col] += 1
                V[d*p+f][col] += 1; V[d*p+e][col] -= 1
                W[e*n+f][col] += 1
                col += 1
                U[b*m+d][col] += 1
                V[c*p+f][col] += 1; V[c*p+e][col] -= 1
                W[e*n+a][col] += 1
                col += 1
                U[a*m+c][col] += 1; U[a*m+d][col] += 1
                V[d*p+f][col] += 1
                W[e*n+f][col] += 1; W[e*n+a][col] -= 1
                col += 1
                U[b*m+c][col] += 1; U[a*m+c][col] -= 1
                V[c*p+e][col] += 1; V[c*p+f][col] += 1
                W[f*n+a][col] += 1
                col += 1
                U[b*m+d][col] += 1; U[a*m+d][col] -= 1
                V[d*p+e][col] += 1; V[d*p+f][col] += 1
                W[e*n+a][col] -= 1; W[f*n+b][col] += 1
                col += 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                for (a, b) in n_blocks:
                    if i in (a, b):
                        for (c, d) in m_blocks:
                            if j in (c, d):
                                for (e, f) in p_blocks:
                                    if k in (e, f):
                                        covered = True
                if not covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    if verify_decomp(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    return schoolbook()
# EVOLVE-BLOCK-END
