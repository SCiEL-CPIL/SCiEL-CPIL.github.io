import time
import numpy as np
from itertools import product

N, M, P = 3, 4, 8

def _all_partitions(n):
    """Generate all set partitions of range(n) as lists of sorted lists."""
    if n == 0:
        yield [[]]
        return
    def _partitions_helper(current, remaining):
        if not remaining:
            yield current
            return
        first = remaining[0]
        rest = remaining[1:]
        # Add first to each existing block
        for i in range(len(current)):
            new_current = [list(b) for b in current]
            new_current[i].append(first)
            yield from _partitions_helper(new_current, rest)
        # Add first as a new block
        new_current = current + [[first]]
        yield from _partitions_helper(new_current, rest)
    yield from _partitions_helper([], list(range(n)))

def _get_block_decomp(n_block, m_block, p_block):
    """
    Return (U_local, V_local, W_local) for a block of sizes (n_block, m_block, p_block).
    U_local: (n_block * m_block, rank)
    V_local: (m_block * p_block, rank)
    W_local: (p_block * n_block, rank)
    Uses known optimal decompositions for small blocks.
    """
    nb, mb, pb = n_block, m_block, p_block
    
    # Known optimal ranks for small rectangular matrix multiplication:
    # 1x1x1: rank 1
    # 1x2x2: rank 4 (schoolbook is 4, no known better)
    # 2x1x2: rank 4
    # 2x2x1: rank 4
    # 2x2x2: rank 7 (Strassen)
    # 2x2x4: ? Let's use schoolbook or try to find better. Schoolbook is 16.
    # 2x4x2: schoolbook 16
    # 4x2x2: schoolbook 16
    # 1x1x2: rank 2
    # 1x2x1: rank 2
    # 2x1x1: rank 2
    # 1x1x4: rank 4
    # 1x4x1: rank 4
    # 4x1x1: rank 4
    # 2x2x3: ? Schoolbook 12. Known rank? Maybe 11 or 10? Let's use schoolbook for safety unless known.
    # Actually, for 2x2x2 it is 7. For 2x2x3, we can think of it as 2x2x2 + 2x2x1.
    # But they are not independent. Let's just use schoolbook for anything not 2x2x2 or 1x1x1 or 1x1xk etc.
    # Wait, 2x2x2 is the only one with a famous better-than-schoolbook rank (7 vs 8).
    # Are there others? 3x3x3 is 23 (Laderman) vs 27. But we don't have 3x3x3 blocks likely.
    # For 2x2x4: Schoolbook 16. Can we do better? 
    # If we split p=4 into two 2s, we have two 2x2x2 blocks? No, the i and j indices are shared.
    # The tensor is T[i,j,k]. If we split k into k1, k2, the tensor is T[i,j,k1] + T[i,j,k2]? No, it's a direct sum in the k-dimension?
    # No, the multiplication tensor for m x n is a tensor in A (m x n) x B (n x p) x C (p x m).
    # If we partition the index sets, the tensor is the tensor product of the block tensors?
    # Yes! If we partition the indices, the total tensor is the Kronecker product (tensor product) of the block tensors.
    # So the rank of the whole is the product of the ranks of the blocks?
    # No, the rank of a tensor product T1 x T2 is rank(T1) * rank(T2) if the fields are algebraically closed and we are looking for tensor rank?
    # Actually, the rank of a tensor product is at most rank(T1)*rank(T2), and often equal.
    # But here we are summing over disjoint index sets.
    # If we partition the indices, the total tensor is the SUM of the block tensors, not the product.
    # Wait, if we partition the indices, the tensor T is the direct sum of the block tensors T_block.
    # The rank of a direct sum is the sum of the ranks.
    # So if we partition n, m, p, the total rank is sum over all blocks of rank(T_block).
    # And for each block, we can use the optimal rank for that block size.
    
    # So we need the optimal rank for each (nb, mb, pb).
    # For (2,2,2), optimal rank is 7.
    # For (1,1,1), optimal rank is 1.
    # For (1,1,k), optimal rank is k.
    # For (1,k,1), optimal rank is k.
    # For (k,1,1), optimal rank is k.
    # For (2,2,1), optimal rank is 4? (Schoolbook is 4. Is there a rank 3? No, 2x2x1 is 4x4 matrix multiplication? No.
    # 2x2 * 2x1 -> 2x1. Tensor is in (2x2) x (2x1) x (1x2).
    # This is just 4 scalar multiplications. Rank 4.
    # For (2,1,2), rank 4.
    # For (1,2,2), rank 4.
    # For (2,2,3), schoolbook 12. Is there a better rank?
    # 2x2x3 can be viewed as 2x2x2 + 2x2x1? No.
    # Let's just use schoolbook for all blocks except (2,2,2) and (1,1,1) and (1,1,k) etc.
    # Actually, (2,2,2) is the only one where Strassen helps (7 < 8).
    # Are there other small blocks with known sub-schoolbook rank?
    # 3x3x3 is 23 < 27. But we might not have 3x3x3 blocks.
    # 2x3x3? Unknown.
    # Let's assume only (2,2,2) is improved by Strassen.
    
    if nb == 2 and mb == 2 and pb == 2:
        rank = 7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Strassen terms for 2x2x2
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # M1 = (a+d)(e+h)
        U[:,0] = [1,0,0,1]
        V[:,0] = [1,0,0,1]
        W[:,0] = [1,0,0,1]
        # M2 = (c+d)e
        U[:,1] = [0,0,1,1]
        V[:,1] = [1,0,0,0]
        W[:,1] = [0,0,1,-1]
        # M3 = a(f-h)
        U[:,2] = [1,1,0,0]
        V[:,2] = [0,1,0,-1]
        W[:,2] = [0,1,0,1]
        # M4 = d(g-e)
        U[:,3] = [0,0,0,1]
        V[:,3] = [0,-1,0,1]
        W[:,3] = [1,0,-1,0]
        # M5 = (a+b)h
        U[:,4] = [1,0,1,0]
        V[:,4] = [0,0,0,1]
        W[:,4] = [0,-1,1,0]
        # M6 = (c-a)(e+f)
        U[:,5] = [-1,0,1,0]
        V[:,5] = [1,1,0,0]
        W[:,5] = [0,0,0,1]
        # M7 = (b-d)(g+h)
        U[:,6] = [0,1,0,-1]
        V[:,6] = [1,0,0,1]
        W[:,6] = [1,0,0,0]
        return U, V, W, rank
    else:
        # Schoolbook
        rank = nb * mb * pb
        U = np.zeros((nb * mb, rank))
        V = np.zeros((mb * pb, rank))
        W = np.zeros((pb * nb, rank))
        col = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    U[i * mb + j, col] = 1
                    V[j * pb + k, col] = 1
                    W[k * nb + i, col] = 1
                    col += 1
        return U, V, W, rank

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate all partitions
    part_n = list(_all_partitions(n))
    part_m = list(_all_partitions(m))
    part_p = list(_all_partitions(p))
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook fallback
    U_fb = np.zeros((n * m, best_rank))
    V_fb = np.zeros((m * p, best_rank))
    W_fb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_result = (U_fb, V_fb, W_fb)
    
    for pn in part_n:
        if time.time() > deadline - 10:
            break
        for pm in part_m:
            if time.time() > deadline - 10:
                break
            for pp in part_p:
                if time.time() > deadline - 10:
                    break
                
                # Check if this partition is valid (it always is, by definition)
                # Calculate total rank
                total_rank = 0
                blocks = []
                for bn in pn:
                    for bm in pm:
                        for bp in pp:
                            nb, mb, pb = len(bn), len(bm), len(bp)
                            U_l, V_l, W_l, r = _get_block_decomp(nb, mb, pb)
                            total_rank += r
                            blocks.append((bn, bm, bp, U_l, V_l, W_l, r))
                
                if total_rank < best_rank:
                    # Construct the full matrices
                    U = np.zeros((n * m, total_rank))
                    V = np.zeros((m * p, total_rank))
                    W = np.zeros((p * n, total_rank))
                    col = 0
                    for bn, bm, bp, U_l, V_l, W_l, r in blocks:
                        for i_idx, i in enumerate(bn):
                            for j_idx, j in enumerate(bm):
                                U[i * m + j, col:col+r] = U_l[i_idx * len(bm) + j_idx, :]
                        for j_idx, j in enumerate(bm):
                            for k_idx, k in enumerate(bp):
                                V[j * p + k, col:col+r] = V_l[j_idx * len(bp) + k_idx, :]
                        for k_idx, k in enumerate(bp):
                            for i_idx, i in enumerate(bn):
                                W[k * n + i, col:col+r] = W_l[k_idx * len(bn) + i_idx, :]
                        col += r
                    
                    if _verify(U, V, W, n, m, p):
                        best_rank = total_rank
                        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
