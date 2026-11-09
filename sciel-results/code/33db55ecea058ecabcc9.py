"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s budget, leave margin
    
    n, m, p = DIMENSIONS
    # U: shape (n*m, rank) = (8, rank)
    # V: shape (m*p, rank) = (24, rank)
    # W: shape (p*n, rank) = (12, rank)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if all entries match."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = 1 for matrix mult (identity-like tensor)
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-32 decomposition via structured search
    # Strategy: For each k, we have a 2x4 matrix product producing one scalar per (i).
    # The tensor T[i,j,k] = 1 means: for fixed k, the 2x4 matrix A * column k of B
    # gives C[:,k]. The rank structure per k-slice is rank 2 (since n=2).
    
    # Alternative: use the fact that T is the matrix multiplication tensor.
    # Known: rank(M_{2,4,6}) can be reduced below 48.
    # Try rank 32: use 4 "blocks" of 8, or 2 blocks of 16.
    
    # Let's try a direct construction for rank 32:
    # For each (i, j) pair, instead of having p separate k-factors,
    # we can potentially merge using the linear structure in k.
    # 
    # Actually, the standard trick: for M_{n,m,p}, if we can find a decomposition
    # where the k-dependence is linear, we can reduce.
    # 
    # For rank 32 = 4 * 8: group the 6 k-values into structures.
    # 
    # Let me try: for each i (2), j (4), we need to cover k=0..5.
    # If we use 2 factors per (i,j) that each cover 3 k-values, that's 2*4*2=16 per i,
    # times 2 for i = 32 total.
    
    # Construction for rank 32:
    # For i in {0,1}, j in {0,1,2,3}, r in {0,1}:
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1 if k in group_r, else 0
    #   W[k*n+i, c] = 1 if k in group_r, else 0
    # But this gives sum over c of [1 * 1_{k in group} * 1_{k in group}] = 1_{k in group} * 1_{k in group}
    # = 1 if k in group_r for some r, but we need exactly 1 for each k.
    # If groups partition {0..5} into two sets of 3, then for each k, exactly one r has it,
    # so sum = 1. This works!
    
    def try_rank32():
        # Groups: {0,1,2} and {3,4,5}
        groups = [[0, 1, 2], [3, 4, 5]]
        rank = n * m * len(groups)  # 2*4*2 = 16... wait that's 16, not 32
        # Hmm, 2*4*2 = 16. That's too low. Let me recount.
        # For each i (2), j (4), group (2): 2*4*2 = 16 factors.
        # But rank 16 < 24 lower bound, so this can't work.
        # The issue: W[k*n+i, c] depends on i, so different i can't share the same factor.
        # Let me be more careful.
        
        # Factor index: (i, j, g) where g is group index
        # U[i*m + j, (i,j,g)] = 1
        # V[j*p + k, (i,j,g)] = 1 if k in groups[g] else 0
        # W[k*n + i, (i,j,g)] = 1 if k in groups[g] else 0
        # 
        # Check: T[i,j,k] = sum_{i',j',g} U[i*m+j, (i',j',g)] * V[j*p+k, (i',j',g)] * W[k*n+i, (i',j',g)]
        # = sum_{i',j',g} [i*m+j == i'*m+j'] * [1_{k in groups[g]}] * [k*n+i == k*n+i' and 1_{k in groups[g]}]
        # = sum_{g} 1_{j==j'} * 1_{i==i'} * 1_{k in groups[g]}^2
        # = sum_{g} 1_{k in groups[g]}  (when i'=i, j'=j)
        # = 1 if k in any group = 1 (since groups partition 0..5)
        # 
        # Wait, this gives rank = n*m*num_groups = 2*4*2 = 16. But 16 < 24, contradiction!
        # The error: U[i*m+j, c] and W[k*n+i, c] both depend on i. Let me recheck.
        # 
        # U has shape (8, 16), indexed by row = i*m+j (0..7)
        # W has shape (12, 16), indexed by row = k*n+i (0..11)
        # 
        # For factor c = (i0, j0, g0):
        #   U[row, c] = 1 if row == i0*m + j0
        #   V[row, c] = 1 if row == j0*p + k for k in groups[g0]
        #   W[row, c] = 1 if row == k*n + i0 for k in groups[g0]
        # 
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Only c = (i, j, g) contributes (since U requires i0=i, j0=j; W requires i0=i)
        # = sum_g 1 * 1_{k in groups[g]} * 1_{k in groups[g]}
        # = sum_g 1_{k in groups[g]}
        # = 1 (since groups partition)
        # 
        # This gives rank 16! But 16 < 24, which violates the lower bound. 
        # There must be an error. Let me check the lower bound.
        # Lower bound for M_{n,m,p} is max(nm, mp, pn) = max(8, 24, 12) = 24.
        # So rank 16 is impossible. Where's the mistake?
        
        # Ah! The issue is that the decomposition formula is:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where U is n x m x rank, V is m x p x rank, W is p x n x rank.
        # But in the flattened form:
        # U_flat[i*m+j, c], V_flat[j*p+k, c], W_flat[k*n+i, c]
        # 
        # Let me re-examine: for c = (i0, j0, g0):
        # U_flat[i*m+j, c] = 1 iff i == i0 and j == j0
        # V_flat[j*p+k, c] = 1 iff j == j0 and k in groups[g0]
        # W_flat[k*n+i, c] = 1 iff k in groups[g0] and i == i0
        # 
        # So T[i,j,k] = sum_{i0,j0,g0} [i==i0][j==j0][j==j0][k in groups[g0]][k in groups[g0]][i==i0]
        # = sum_{g0} [k in groups[g0]]
        # = 1
        # 
        # This seems correct! But rank 16 < 24 is impossible.
        # The resolution: the lower bound of 24 is for the *tensor rank*, but maybe
        # my construction has a flaw. Let me verify with actual numbers.
        # n=2, m=4, p=6. groups = [[0,1,2],[3,4,5]]
        # rank = 2*4*2 = 16
        # 
        # Wait, I think the issue is that I'm double-counting or the groups need to be
        # disjoint in a specific way. Let me just test it numerically.
        
        rank = 16
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for g in range(2):
                    U[i * m + j][col] = 1
                    for k in groups[g]:
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                    col += 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try rank 32 with different grouping
    def try_rank32_v2():
        # Use 4 groups of 1.5? No, must be integers.
        # Try: groups = [[0,1],[2,3],[4,5]] -> 3 groups, rank = 2*4*3 = 24
        # Or: [[0],[1],[2],[3],[4],[5]] -> 6 groups, rank = 2*4*6 = 48 (schoolbook)
        # Try 2 groups but with different structure...
        
        # Actually, let me try the 16-rank construction and verify it carefully
        # The lower bound might be wrong, or my construction might have a subtle error.
        
        # Let's just test all small ranks via the group construction
        for num_groups in range(2, 7):
            # Create groups as evenly as possible
            group_sizes = [p // num_groups] * num_groups
            for i_extra in range(p % num_groups):
                group_sizes[i_extra] += 1
            
            groups = []
            k_start = 0
            for sz in group_sizes:
                groups.append(list(range(k_start, k_start + sz)))
                k_start += sz
            
            rank = n * m * num_groups
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for g in range(num_groups):
                        U[i * m + j][col] = 1
                        for k in groups[g]:
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                        col += 1
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Fallback to schoolbook immediately to ensure validity
    # The previous attempts at low rank were invalid for the matrix multiplication tensor
    return schoolbook()
# EVOLVE-BLOCK-END
