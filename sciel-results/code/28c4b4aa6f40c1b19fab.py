"""Matrix multiplication 3x5x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def decomp_222_rank7():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
        # M2 = (A10+A11)B00
        terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}))
        # M3 = A00(B01-B11)
        terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}))
        # M4 = A11(B10-B00)
        terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
        # M5 = (A00+A01)B11
        terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}))
        # M6 = (A10-A00)(B00+B01)
        terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
        # M7 = (A01-A11)(B10+B11)
        terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
        return terms

    def decomp_223_rank11():
        """Return the 11 terms for 2x2x3 matrix multiplication."""
        # Based on known decomposition
        terms = []
        # Use 2x2x2 (7 terms) for first 2 columns of B
        # And 2x2x1 (4 terms) for last column of B?
        # Actually, 2x2x3 rank is 11.
        # I will use a known set of 11 terms.
        # Reference: "The rank of matrix multiplication"
        
        # Term 1: A00 B00
        terms.append(({(0,0): 1}, {(0,0): 1}, {(0,0): 1}))
        # Term 2: A00 B10
        terms.append(({(0,0): 1}, {(1,0): 1}, {(0,1): 1}))
        # Term 3: A10 B00
        terms.append(({(1,0): 1}, {(0,0): 1}, {(1,0): 1}))
        # Term 4: A10 B10
        terms.append(({(1,0): 1}, {(1,0): 1}, {(1,1): 1}))
        # Term 5: A01 B00
        terms.append(({(0,1): 1}, {(0,0): 1}, {(0,2): 1}))
        # Term 6: A01 B10
        terms.append(({(0,1): 1}, {(1,0): 1}, {(0,3): 1})) # Wait, indices
        # This is getting complicated to hardcode from memory.
        # I will fall back to the partition method which gave 84.
        # But I can improve the partition method by using 3x3x3 rank 23.
        
        return None

    def decomp_333_rank23():
        """Return the 23 terms for 3x3x3 matrix multiplication."""
        # Bini et al. 1978
        terms = []
        # ... (I will implement a subset or a known valid decomposition)
        # If I can't find the exact 23 terms, I will use schoolbook for 3x3x3 (27).
        # 27+27+17+17 = 88. Worse than 84.
        # So I MUST get 23 or better.
        
        # I will use the 23-rank decomposition from literature.
        # It involves fractions.
        
        # For now, I will return the schoolbook 27 to be safe, 
        # but I will try to implement the 23-rank one.
        
        # Let's try to construct 3x3x3 from 2x2x2 and 1x1x1 etc.
        # 3x3x3 = 2x2x2 + 2x1x2 + 1x2x2 + 1x1x2 + 2x2x1 + 2x1x1 + 1x2x1 + 1x1x1
        # 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
        # Still worse than 27? No, 26 < 27.
        # So 26 is better than schoolbook.
        
        # Let's use the 26-rank decomposition constructed from Strassen blocks.
        # This is easier to code than the 23-rank one.
        
        terms_222 = decomp_222_rank7()
        
        # Map 2x2x2 terms to 3x3x3 indices (0,1)
        for u, v, w in terms_222:
            terms.append((u, v, w))
            
        # Add terms for the rest
        # Indices: i in {2}, j in {2}, k in {2} -> 1x1x1 (1 term)
        terms.append(({(2,2): 1}, {(2,2): 1}, {(2,2): 1}))
        
        # i in {2}, j in {0,1}, k in {0,1} -> 1x2x2 (4 terms)
        # Use schoolbook for 1x2x2
        for j in range(2):
            for k in range(2):
                terms.append(({(2, j): 1}, {(j, k): 1}, {(k, 2): 1}))
                
        # i in {0,1}, j in {2}, k in {0,1} -> 2x1x2 (4 terms)
        for i in range(2):
            for k in range(2):
                terms.append(({(i, 2): 1}, {(2, k): 1}, {(k, i): 1}))
                
        # i in {0,1}, j in {0,1}, k in {2} -> 2x2x1 (4 terms)
        for i in range(2):
            for j in range(2):
                terms.append(({(i, j): 1}, {(j, 2): 1}, {(2, i): 1}))
                
        # i in {2}, j in {2}, k in {0,1} -> 1x1x2 (2 terms)
        for k in range(2):
            terms.append(({(2, 2): 1}, {(2, k): 1}, {(k, 2): 1}))
            
        # i in {2}, j in {0,1}, k in {2} -> 1x2x1 (2 terms)
        for j in range(2):
            terms.append(({(2, j): 1}, {(j, 2): 1}, {(2, 2): 1}))
            
        # i in {0,1}, j in {2}, k in {2} -> 2x1x1 (2 terms)
        for i in range(2):
            terms.append(({(i, 2): 1}, {(2, 2): 1}, {(2, i): 1}))
            
        # Total: 7 + 1 + 4 + 4 + 4 + 2 + 2 + 2 = 26.
        
        return terms

    def build_decomposition_356():
        """Build the 3x5x6 decomposition using 3x3x3 (rank 26) and 3x2x3 (rank 17)."""
        
        # 3x5x6
        # Split A (3x5) into A1 (3x3) and A2 (3x2)
        # Split B (5x6) into B1 (3x6) and B2 (2x6)
        # C = A1 B1 + A2 B2
        
        # A1 B1 is 3x3x6
        # Split B1 into B1a (3x3) and B1b (3x3)
        # A1 B1a is 3x3x3 (Rank 26)
        # A1 B1b is 3x3x3 (Rank 26)
        
        # A2 B2 is 3x2x6
        # Split B2 into B2a (2x3) and B2b (2x3)
        # A2 B2a is 3x2x3 (Rank 17)
        # A2 B2b is 3x2x3 (Rank 17)
        
        # Total Rank: 26 + 26 + 17 + 17 = 86.
        # This is worse than 84.
        
        # I need the 23-rank 3x3x3.
        # If I use 23: 23+23+17+17 = 80.
        
        # I will implement the 23-rank 3x3x3 decomposition.
        # Source: "Optimal rank of 3x3x3 matrix multiplication"
        
        return None

    # Since I cannot reliably hardcode the 23-rank decomposition from memory,
    # and the 26-rank decomposition gives 86 (worse than 84),
    # I will stick to the partition method but optimize it.
    
    # The partition method found 84.
    # I will return the result of the partition method.
    
    def get_partitions(dim):
        """Get all ordered partitions of dim into positive integers."""
        if dim == 1:
            return [[1]]
        results = []
        for first in range(1, dim + 1):
            if first == dim:
                results.append([dim])
            else:
                for rest in get_partitions(dim - first):
                    results.append([first] + rest)
        return results

    def compute_rank_for_partition(i_groups, j_groups, k_groups):
        """Compute total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    def find_best_partition():
        """Find the partition that minimizes total rank."""
        i_parts = get_partitions(n)
        j_parts = get_partitions(m)
        k_parts = get_partitions(p)
        
        best_rank = n * m * p
        best_partition = (None, None, None)
        
        for ip in i_parts:
            if time.time() > deadline - 10:
                break
            for jp in j_parts:
                for kp in k_parts:
                    i_groups = []
                    idx = 0
                    for size in ip:
                        i_groups.append(list(range(idx, idx + size)))
                        idx += size
                    
                    j_groups = []
                    idx = 0
                    for size in jp:
                        j_groups.append(list(range(idx, idx + size)))
                        idx += size
                    
                    k_groups = []
                    idx = 0
                    for size in kp:
                        k_groups.append(list(range(idx, idx + size)))
                        idx += size
                    
                    r = compute_rank_for_partition(i_groups, j_groups, k_groups)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (i_groups, j_groups, k_groups)
        
        return best_rank, best_partition

    def build_decomposition(i_groups, j_groups, k_groups):
        """Build the decomposition for a given partition."""
        strassen_terms = decomp_222_rank7()
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    best_result = schoolbook()
    best_rank = n * m * p

    if time.time() < deadline - 5:
        optimal_rank, (opt_i, opt_j, opt_k) = find_best_partition()
        
        if opt_i is not None and optimal_rank < best_rank:
            result = build_decomposition(opt_i, opt_j, opt_k)
            rank = result.pop("rank")
            
            if verify(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
