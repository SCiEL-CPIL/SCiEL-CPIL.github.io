"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition using BCLR 3x3x3 and Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin

    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24

    # Schoolbook fallback (Rank 120)
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    # BCLR 3x3x3 rank-23 decomposition terms
    # Each term: (u_indices, v_indices, w_indices) where each is list of (local_idx, coeff)
    # Local indices: U is 3x3 (row,col), V is 3x3 (row,col), W is 3x3 (row,col)
    # W convention: W[k*n+i] so local W is (k,i) for 3x3x3
    BCLR_333 = [
        # Term 1: (0,0)
        ([((0,0),1)], [((0,0),1)], [((0,0),1)]),
        # Term 2: (1,0)
        ([((1,0),1)], [((0,0),1)], [((0,1),1)]),
        # Term 3: (2,0)
        ([((2,0),1)], [((0,0),1)], [((0,2),1)]),
        # Term 4: (0,1)
        ([((0,1),1)], [((1,0),1)], [((1,0),1)]),
        # Term 5: (1,1)
        ([((1,1),1)], [((1,0),1)], [((1,1),1)]),
        # Term 6: (2,1)
        ([((2,1),1)], [((1,0),1)], [((1,2),1)]),
        # Term 7: (0,2)
        ([((0,2),1)], [((2,0),1)], [((2,0),1)]),
        # Term 8: (1,2)
        ([((1,2),1)], [((2,0),1)], [((2,1),1)]),
        # Term 9: (2,2)
        ([((2,2),1)], [((2,0),1)], [((2,2),1)]),
        # Term 10: (0,0)+(1,1)
        ([((0,0),1),((1,1),1)], [((0,1),1),((1,1),1)], [((1,0),1),((2,1),1)]),
        # Term 11: (1,0)+(0,1)
        ([((1,0),1),((0,1),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,0),-1)]),
        # Term 12: (2,0)+(1,1)
        ([((2,0),1),((1,1),1)], [((0,2),1),((1,1),1)], [((2,0),1),((0,1),1)]),
        # Term 13: (0,1)+(2,0)
        ([((0,1),1),((2,0),1)], [((0,2),1),((1,1),-1)], [((0,2),1),((2,0),-1)]),
        # Term 14: (1,1)+(2,2)
        ([((1,1),1),((2,2),1)], [((1,2),1),((2,2),1)], [((1,2),1),((2,2),1)]),
        # Term 15: (2,1)+(1,2)
        ([((2,1),1),((1,2),1)], [((1,2),1),((2,2),-1)], [((2,1),1),((1,2),-1)]),
        # Term 16: (0,2)+(1,1)
        ([((0,2),1),((1,1),1)], [((1,2),1),((2,2),1)], [((1,1),1),((0,2),1)]),
        # Term 17: (1,2)+(0,2)
        ([((1,2),1),((0,2),1)], [((1,2),1),((2,2),-1)], [((1,1),1),((2,2),-1)]),
        # Term 18: (0,0)+(2,2)
        ([((0,0),1),((2,2),1)], [((2,1),1),((1,2),1)], [((2,1),1),((1,2),1)]),
        # Term 19: (1,0)+(2,1)
        ([((1,0),1),((2,1),1)], [((2,1),1),((1,2),-1)], [((2,0),1),((2,2),-1)]),
        # Term 20: (2,0)+(0,1)
        ([((2,0),1),((0,1),1)], [((2,1),1),((1,2),1)], [((2,0),1),((0,1),1)]),
        # Term 21: (0,0)+(1,2)
        ([((0,0),1),((1,2),1)], [((0,2),1),((2,1),1)], [((0,1),1),((2,1),1)]),
        # Term 22: (1,0)+(0,2)
        ([((1,0),1),((0,2),1)], [((0,2),1),((2,1),-1)], [((0,1),1),((1,0),-1)]),
        # Term 23: (2,0)+(1,1)+(0,2)
        ([((2,0),1),((1,1),1),((0,2),1)], [((0,2),1),((1,2),1),((2,1),1)], [((0,2),1),((1,1),1),((2,0),1)]),
    ]

    # Strassen 2x2x2 terms
    STRASSEN_222 = [
        # M1
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # M2
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((0,1),1),((1,1),-1)]),
        # M3
        ([((0,0),1)], [((1,0),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
        # M4
        ([((1,1),1)], [((0,1),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        # M5
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((1,0),1)]),
        # M6
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        # M7
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]

    # Rank-1 block for a x 1 x b (outer product)
    def make_rank1_block(a, b):
        terms = []
        u_list = [((i,0),1) for i in range(a)]
        v_list = [((0,k),1) for k in range(b)]
        w_list = [((k,i),1) for k in range(b) for i in range(a)]
        terms.append((u_list, v_list, w_list))
        return terms

    # Helper to get global indices
    def u_global(i, j): return i * m + j
    def v_global(j, k): return j * p + k
    def w_global(k, i): return k * n + i

    def add_terms(U, V, W, col_idx, terms, R, C, K):
        """Add decomposition terms with given row/col/offset mappings."""
        for u_terms, v_terms, w_terms in terms:
            for (li, lj), coeff in u_terms:
                U[u_global(R[li], C[lj]), col_idx] += coeff
            for (lj, lk), coeff in v_terms:
                V[v_global(C[lj], K[lk]), col_idx] += coeff
            for (lk, li), coeff in w_terms:
                W[w_global(K[lk], R[li]), col_idx] += coeff
            col_idx += 1
        return col_idx

    # Strategy: Try multiple partitionings and pick the best valid one
    def try_partition(R_parts, C_parts, K_parts, get_block_terms):
        """Build decomposition with given partitioning."""
        # Calculate total rank
        total_rank = 0
        for R in R_parts:
            for C in C_parts:
                for K in K_parts:
                    terms = get_block_terms(len(R), len(C), len(K))
                    total_rank += len(terms)
        
        U = np.zeros((num_U, total_rank), dtype=int)
        V = np.zeros((num_V, total_rank), dtype=int)
        W = np.zeros((num_W, total_rank), dtype=int)
        
        col_idx = 0
        for R in R_parts:
            for C in C_parts:
                for K in K_parts:
                    terms = get_block_terms(len(R), len(C), len(K))
                    col_idx = add_terms(U, V, W, col_idx, terms, R, C, K)
        
        return U, V, W, total_rank

    def get_block_terms(a, b, c):
        """Get terms for an a x b x c multiplication block."""
        if a == 2 and b == 2 and c == 2:
            return STRASSEN_222
        elif a == 3 and b == 3 and c == 3:
            return BCLR_333
        else:
            # Schoolbook fallback
            terms = []
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        terms.append(([((i,j),1)], [((j,k),1)], [((k,i),1)]))
            return terms

    # Partitioning 1: Current (2+2) x (2+2+1) x (2+2+2)
    R1 = [[0,1],[2,3]]
    C1 = [[0,1],[2,3],[4]]
    K1 = [[0,1],[2,3],[4,5]]
    
    # Partitioning 2: (3+1) x (3+2) x (3+3)
    R2 = [[0,1,2],[3]]
    C2 = [[0,1,2],[3,4]]
    K2 = [[0,1,2],[3,4,5]]
    
    # Partitioning 3: (2+2) x (3+2) x (3+3)
    R3 = [[0,1],[2,3]]
    C3 = [[0,1,2],[3,4]]
    K3 = [[0,1,2],[3,4,5]]
    
    # Partitioning 4: (3+1) x (2+2+1) x (2+2+2)
    R4 = [[0,1,2],[3]]
    C4 = [[0,1],[2,3],[4]]
    K4 = [[0,1],[2,3],[4,5]]
    
    # Partitioning 5: (2+2) x (2+3) x (3+3)
    R5 = [[0,1],[2,3]]
    C5 = [[0,1],[2,3,4]]
    K5 = [[0,1,2],[3,4,5]]
    
    # Partitioning 6: (4) x (5) x (6) - full schoolbook (fallback)
    R6 = [[0,1,2,3]]
    C6 = [[0,1,2,3,4]]
    K6 = [[0,1,2,3,4,5]]

    # Verify function
    def verify(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Check on-diagonal: T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check off-diagonal: T[i,j,k] = 0 when not (i,j,k) pattern
        # T[a,b,c] nonzero only if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # i.e., a%m == b//p and b%p == c//n and c%n == a//m
        for a in range(num_U):
            for b in range(num_V):
                for c in range(num_W):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
        return True

    # Try partitionings in order of expected rank
    partitions = [
        (R2, C2, K2),  # 3x3x3 + 3x2x3 + 1x3x3 + 1x2x3 = 23+18+9+6 = 56
        (R3, C3, K3),  # 2x3x3 + 2x2x3 + 2x3x3 + 2x2x3 = 18+18+18+18 = 72 (schoolbook for 2x3x3)
        (R4, C4, K4),  # 3x2x2 + 3x2x2 + 3x1x2 + 1x2x2 + 1x2x2 + 1x1x2 = 18+18+6+4+4+2 = 52
        (R5, C5, K5),  # 2x2x3 + 2x3x3 + 2x2x3 + 2x3x3 = 18+23+18+23 = 82
        (R1, C1, K1),  # Current: 90
        (R6, C6, K6),  # Schoolbook: 120
    ]

    best_result = None
    best_rank = rank_school

    for R_parts, C_parts, K_parts in partitions:
        if time.time() > deadline - 20:
            break
        U, V, W, rank = try_partition(R_parts, C_parts, K_parts, get_block_terms)
        if rank < best_rank and verify(U, V, W):
            best_result = (U, V, W, rank)
            best_rank = rank
            if rank <= 50:  # Good enough, stop
                break

    if best_result is not None:
        U, V, W, _ = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
