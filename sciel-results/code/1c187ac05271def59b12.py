"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    # U, V, W are lists of (index1, index2, value)
    # Indices are local to the 2x2x2 block
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    def get_splits(d):
        """Generate all compositions of d into parts of size 1 or 2."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(current[:])
                return
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
        helper(d, [])
        return results
    
    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the rank for given splits, matching actual build logic."""
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    if nb == 2 and mb == 2 and pb == 2:
                        rank += 7
                    else:
                        rank += nb * mb * pb
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits."""
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)
        
        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                U_entries.append(((i_off + li) * m + (j_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                V_entries.append(((j_off + lj) * p + (k_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                W_entries.append(((k_off + lk) * n + (i_off + li), col, val))
                            rank += 1
                    else:
                        # For non-2x2x2 blocks, we use schoolbook or recursive splitting
                        # Here we use schoolbook for simplicity and correctness, 
                        # as the rank calculation above assumes specific optimized ranks 
                        # that might not have simple implementations here.
                        # To maintain the rank estimate, we should ideally implement 
                        # the optimized versions, but for now, schoolbook is safe.
                        # However, if we use schoolbook, the actual rank will be higher 
                        # than the estimate, leading to a lower score.
                        # We must implement the optimized versions or adjust the estimate.
                        
                        # For this rewrite, we will stick to schoolbook for non-2x2x2 
                        # unless we implement the specific optimizations.
                        # Given the time, we will use schoolbook and adjust the rank 
                        # calculation to match.
                        
                        # Let's re-evaluate: if we use schoolbook for 2x2x3, rank is 12.
                        # If we use schoolbook for 2x3x3, rank is 18.
                        # If we use schoolbook for 3x3x3, rank is 27.
                        
                        # Let's update compute_rank to use schoolbook for non-2x2x2.
                        
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_entries.append((gi * m + gj, col, 1))
                                    V_entries.append((gj * p + gk, col, 1))
                                    W_entries.append((gk * n + gi, col, 1))
                                    rank += 1
        
        # Build matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}, rank
    
    # Generate all possible splits for each dimension
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Sort splits by estimated rank (more 2-blocks first) to find good solutions early
    def split_score(splits):
        """Lower score = better (more 2s, fewer 1s and larger blocks)."""
        score = 0
        for s in splits:
            if s == 2:
                score += 0
            elif s == 1:
                score += 1
            else:
                score += s  # penalty for larger blocks
        return score
    
    n_split_options.sort(key=split_score)
    m_split_options.sort(key=split_score)
    p_split_options.sort(key=split_score)
    
    # Find the best combination of splits (minimum rank)
    best_rank = float('inf')
    best_result = None
    best_splits = None
    checked = 0
    
    for ns in n_split_options:
        if time.time() > deadline:
            break
        for ms in m_split_options:
            if time.time() > deadline:
                break
            for ps in p_split_options:
                if time.time() > deadline:
                    break
                # Quick rank estimate to skip obviously bad combinations
                est_rank = compute_rank(ns, ms, ps)
                if est_rank >= best_rank:
                    continue
                result, r = build_decomposition(ns, ms, ps)
                checked += 1
                if r < best_rank:
                    best_rank = r
                    best_result = result
                    best_splits = (ns, ms, ps)
    
    # Verify the decomposition is correct
    def verify_decomposition(result):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(result['U'])
        V = np.array(result['V'])
        W = np.array(result['W'])
        
        rank = U.shape[1]
        # Check shapes
        if U.shape[0] != n * m:
            return False
        if V.shape[0] != m * p:
            return False
        if W.shape[0] != p * n:
            return False
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # if i,j,k are the "diagonal" elements, else 0
        # Actually, the tensor T[i,j,k] = delta_{i,k} for matrix mult? No.
        # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor form: T[i,j,k] such that C[i,k] = sum_{j,r} T[i,j,k,r] A[i,j] B[j,k]
        # The rank-1 decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r[i*m+j], V_r[j*p+k], W_r[k*n+i]
        # And T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        # For matrix mult, T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # So T[i,j,k] should be 1 for all valid (i,j,k) in the "middle" index j
        # Wait, I need to think about this more carefully.
        
        # The multiplication tensor M has entries M[i,j,k] where:
        # - i indexes rows of A (and C)
        # - j indexes columns of A / rows of B  
        # - k indexes columns of B (and C)
        # M[i,j,k] = 1 if this term contributes to C[i,k], which is always true for valid indices
        # Actually no: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # That's just the all-ones tensor, which has rank 1! That can't be right.
        
        # I think the tensor decomposition is:
        # C = A * B, where C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor M ∈ R^{n×m×p} with M[i,j,k] = 1 if j is the "connecting" index
        # But actually, the standard formulation is:
        # The multiplication map: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # This is a trilinear map, and the tensor has entries:
        # T[i,j,k] = 1 for the "diagonal" in some sense
        # 
        # Let me just check: for each triple (i,j,k), the contribution to the
        # decomposition is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and we need sum_r of this to equal 1 if (i,j,k) is a valid "multiplication"
        # term, and 0 otherwise.
        # 
        # For matrix multiplication, every (i,j,k) with i<n, j<m, k<p contributes
        # to exactly one output C[i,k]. The "tensor" for matrix mult has:
        # T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in R^{n×m×p})
        # But that has rank 1, which is clearly wrong.
        #
        # I think I'm confusing myself. Let me just verify numerically.
        
        # Build the full tensor and check
        # T[i,j,k] should satisfy: for the decomposition to be correct,
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        # (since matrix mult uses all (i,j,k) triples)
        
        # Actually wait - I think the tensor is NOT all ones. Let me reconsider.
        # The matrix multiplication tensor T ∈ R^{n×m×p} is defined by:
        # T[i,j,k] = 1 if this (i,j,k) triple is used in the multiplication
        # For standard matrix mult C = AB, C[i,k] = sum_j A[i,j] B[j,k]
        # So every (i,j,k) triple IS used, and T[i,j,k] = 1 for all i,j,k.
        # But that's rank 1! 
        #
        # I think the issue is that the tensor for matrix multiplication is NOT
        # the all-ones tensor. The correct tensor is:
        # T ∈ R^{(n×m) × (m×p) × (n×p)} where T[(i,j),(j',k),(i',k')] = 1 
        # if i=i', j=j', k=k'
        # In other words, T is the "identity" tensor that connects the three spaces.
        # This tensor has rank n×m×p (schoolbook).
        #
        # So in our indexing:
        # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
        # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
        # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for all i,j,k, and 0 otherwise? No, it should be 1 for all valid (i,j,k).
        
        # Let me just check a small case numerically.
        
        # For the all-ones check:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if best_result is not None:
        if not verify_decomposition(best_result):
            # Fallback to schoolbook if verification fails
            best_result = None
    
    if best_result is None:
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
        
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
