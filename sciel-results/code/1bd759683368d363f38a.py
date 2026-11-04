import numpy as np
import time
import random
from itertools import product

N, M, P = 3, 5, 7
TENSOR_DIM = N * M * P  # 105

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon = U[i * m + j] @ (V[j * p + k] * W[k * n + i])
                if abs(recon - 1.0) > 1e-7:
                    return False
    return True

def _schoolbook(n=N, m=M, p=P):
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

def _tensor_entries(n, m, p):
    """Return list of (i,j,k) triples."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def _basis_vectors(n, m, p):
    """
    Return the 105 basis tensors as vectors in R^(n*m * m*p * p*n).
    Tensor T[i,j,k] has a 1 at position ((i*m+j), (j*p+k), (k*n+i)).
    We represent each basis tensor as a vector of length n*m*p = 105,
    indexed by (i,j,k) -> i*m*p + j*p + k.
    """
    dim = n * m * p
    basis = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(dim)
                vec[i * m * p + j * p + k] = 1.0
                basis.append(vec)
    return basis

def _random_decomposition_trial(n, m, p, rank, rng, max_iter=50):
    """
    Try to find a rank-`rank` decomposition using the anchor method.
    
    Key insight: If T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r, then every basis tensor
    e_{ijk} = sum_r a_{ijk,r} (u_r ⊗ v_r ⊗ w_r) for some coefficients a_{ijk,r}.
    
    So the 105 basis tensors must all lie in the `rank`-dimensional subspace
    spanned by the rank-1 tensors u_r ⊗ v_r ⊗ w_r.
    
    Strategy: Pick `rank` random linear combinations of basis tensors, check they
    are linearly independent, then check if all 105 basis tensors lie in their span.
    If yes, extract the decomposition.
    
    This is a necessary condition check. If it passes, we can extract coefficients.
    But extracting the actual u_r, v_r, w_r from the span is the hard part.
    
    Alternative: Use the fact that for a valid decomposition, the tensor's
    "matricizations" have rank <= `rank`. We can use this to guide the search.
    
    For now, use a simpler approach: random search with structure.
    """
    dim = n * m * p
    basis = _basis_vectors(n, m, p)
    
    for trial in range(max_iter):
        # Generate random rank-1 tensors
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Check if the span of {u_r ⊗ v_r ⊗ w_r} contains all basis tensors
        # Build the 105 x rank matrix of inner products
        # For each basis tensor e_ijk, its inner product with u_r ⊗ v_r ⊗ w_r is
        # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        
        # Build matrix B where B[ijk, r] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        B = np.zeros((dim, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    for r in range(rank):
                        B[idx, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        # For the decomposition to be valid, we need:
        # For each (i,j,k): sum_r B[idx, r] * c_r = 1 for all (i,j,k)
        # where c_r are fixed coefficients... no, that's not right.
        
        # Actually, the decomposition says:
        # T_{ijk} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For the multiplication tensor, T_{ijk} = 1 for all (i,j,k).
        # So we need: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of 105 equations in the entries of U, V, W.
        # It's bilinear, so hard to solve directly.
        
        # Check if our random guess works:
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j] @ (V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication, rank 7."""
    n = m = p = 2
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # m1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C[0,0] and C[1,1]
    
    # m2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C[1,0]
    
    # m3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1  # C[0,1] and C[1,1]
    
    # m4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = 1  # C[1,0] and C[0,0]
    
    # m5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1  # C[0,1] and C[0,0]
    
    # m6 = (A10-A11)*(B01+B11)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[3][5] = 1  # C[1,1]
    
    # m7 = (A01-A10)*(B00+B10)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1  # C[0,0]
    
    return {"U": U, "V": V, "W": W}

def _build_block_decomp(n, m, p, n_parts, m_parts, p_parts):
    """Build a block decomposition using Strassen for 2x2x2 blocks."""
    assert sum(n_parts) == n
    assert sum(m_parts) == m
    assert sum(p_parts) == p
    assert len(n_parts) == len(m_parts) == len(p_parts)
    
    total_rank = 0
    for bi in range(len(n_parts)):
        for bj in range(len(m_parts)):
            for bk in range(len(p_parts)):
                ni, mj, pk = n_parts[bi], m_parts[bj], p_parts[bk]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for bi in range(len(n_parts)):
        ni = n_parts[bi]
        i0 = sum(n_parts[:bi])
        for bj in range(len(m_parts)):
            mj = m_parts[bj]
            j0 = sum(m_parts[:bj])
            for bk in range(len(p_parts)):
                pk = p_parts[bk]
                k0 = sum(p_parts[:bk])
                
                if ni == 2 and mj == 2 and pk == 2:
                    sub = _strassen_2x2x2()
                else:
                    sub = _schoolbook(ni, mj, pk)
                
                sub_rank = len(sub["U"][0])
                sub_U = sub["U"]
                sub_V = sub["V"]
                sub_W = sub["W"]
                
                for r in range(sub_rank):
                    for i in range(ni):
                        for j in range(mj):
                            U[(i0 + i) * m + (j0 + j)][col] = sub_U[i * mj + j][r]
                    for j in range(mj):
                        for k in range(pk):
                            V[(j0 + j) * p + (k0 + k)][col] = sub_V[j * pk + k][r]
                    for k in range(pk):
                        for i in range(ni):
                            W[(k0 + k) * n + (i0 + i)][col] = sub_W[k * ni + i][r]
                    col += 1
    
    # Trim trailing zero columns
    while col > 0:
        if all(U[i][col-1] == 0 for i in range(n*m)) and \
           all(V[i][col-1] == 0 for i in range(m*p)) and \
           all(W[i][col-1] == 0 for i in range(p*n)):
            col -= 1
        else:
            break
    
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _get_all_splits(dim, max_part=3):
    """Get all compositions of dim into parts of size 1..max_part."""
    results = []
    def helper(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            helper(remaining - part, current)
            current.pop()
    helper(dim, [])
    return results

def _greedy_block_search(n, m, p, time_budget=60):
    """Search over block partitions to find minimum rank."""
    start_time = time.time()
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    # Try all combinations
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if time.time() - start_time > time_budget:
                    return best_result, best_rank
                if len(ns) != len(ms) or len(ns) != len(ps):
                    continue
                # Compute rank
                rank = 0
                for idx in range(len(ns)):
                    ni, mj, pk = ns[idx], ms[idx], ps[idx]
                    if ni == 2 and mj == 2 and pk == 2:
                        rank += 7
                    else:
                        rank += ni * mj * pk
                
                if rank < best_rank:
                    try:
                        result = _build_block_decomp(n, m, p, list(ns), list(ms), list(ps))
                        actual_rank = len(result["U"][0])
                        if actual_rank == rank and _verify(result["U"], result["V"], result["W"], n, m, p):
                            best_rank = rank
                            best_result = result
                    except:
                        pass
    
    return best_result, best_rank

def _anisotropic_strassen_3x3x3():
    """
    Attempt to use a known rank-23 decomposition for 3x3x3 matrix multiplication.
    This is based on the work of Bini, Capovani, Lotti, Romani (1978) who showed
    that 3x3 matrix multiplication can be done with 23 multiplications.
    
    The 23 bilinear forms for 3x3x3 matrix multiplication:
    A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    This is complex to hardcode correctly, so let's use a verified construction.
    """
    n = m = p = 3
    # For now, return schoolbook as fallback since hardcoding 23 terms is error-prone
    return _schoolbook(n, m, p)

def construct():
    n, m, p = N, M, P
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Phase 1: Exhaustive block partition search (fast)
    block_result, block_rank = _greedy_block_search(n, m, p, time_budget=30)
    if block_rank < best_rank:
        best_rank = block_rank
        best = block_result
    
    # Phase 2: Try to find even better via targeted random search
    # Focus on ranks just below the block result
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        rng = np.random.RandomState(42)
        
        # Try ranks from best_rank-1 down to max(20, best_rank-5)
        for target_rank in range(best_rank - 1, max(20, best_rank - 5) - 1, -1):
            if time.time() > deadline - 15:
                break
            
            # Multiple random trials
            for trial in range(20):
                if time.time() > deadline - 15:
                    break
                
                result = _random_decomposition_trial(n, m, p, target_rank, rng, max_iter=1)
                if result is not None:
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best_rank = target_rank
                        best = result
                        break
    
    # Final verification
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best
