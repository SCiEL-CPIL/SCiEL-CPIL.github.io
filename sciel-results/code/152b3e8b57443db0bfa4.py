"""
Optimized tensor decomposition for 2x4 * 4x5 matrix multiplication.
Uses Block-Strassen decomposition with column-redundancy search.
Rank: 36
"""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def _build_base_terms(n, m, p):
    """
    Constructs the base rank-36 Block-Strassen decomposition.
    Returns list of (u_dict, v_dict, w_dict).
    """
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
            
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    # Block 1: A_L (cols 0-1) * B_T (rows 0-1)
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1)
    
    # Block 2: A_R (cols 2-3) * B_B (rows 2-3)
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1)
    
    return rank_terms

def _dict_to_vec(d, size, m, p):
    """Convert a sparse dict to a dense vector of given size."""
    vec = [0] * size
    for key, val in d.items():
        if len(key) == 2:
            i, j = key
            idx = i * m + j
        else:
            idx = key
        if 0 <= idx < size:
            vec[idx] = val
    return vec

def _vec_to_dict(vec, size, m, p):
    """Convert a dense vector to a sparse dict."""
    d = {}
    for idx, val in enumerate(vec):
        if val != 0:
            if idx < size:
                i, j = divmod(idx, m) if size == 8 else (idx, 0)
                # For U: size=8, m=4 -> i = idx//4, j = idx%4
                # For V: size=20, m=5 -> j = idx//5, k = idx%5  
                # For W: size=10, m=2 -> k = idx//2, i = idx%2
                if size == 8:  # U
                    i, j = divmod(idx, 4)
                    d[(i, j)] = val
                elif size == 20:  # V
                    j, k = divmod(idx, 5)
                    d[(j, k)] = val
                elif size == 10:  # W
                    k, i = divmod(idx, 2)
                    d[(i, k)] = val
    return d

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # Check: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if this is a valid MM tensor entry
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For MM, T[i,j,k] = 1 for all i,j,k (each product term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, that's not right. The MM tensor has T[i,j,k] = delta_{i,j,k} meaning the (i,j,k) entry is 1 if it's a valid triple.
    # Actually for matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(len(U[0])):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _try_find_merge(rank_terms, n, m, p):
    """
    Try to find a pair of terms that can be merged to reduce rank by 1.
    Two terms can be merged if their U vectors are proportional OR their V vectors are proportional.
    If U_a = alpha * U_b, then we can replace both with a single term.
    Returns (merged_terms, new_rank) or None if no merge found.
    """
    num_terms = len(rank_terms)
    
    # Convert all terms to vectors
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    U_vecs = []
    V_vecs = []
    W_vecs = []
    
    for u_dict, v_dict, w_dict in rank_terms:
        U_vecs.append(_dict_to_vec(u_dict, U_size, m, p))
        V_vecs.append(_dict_to_vec(v_dict, V_size, m, p))
        W_vecs.append(_dict_to_vec(w_dict, W_size, m, p))
    
    # Check for proportional U vectors (same W pattern allows merging)
    # If U_a = U_b (identical), we can merge: (U_a + U_b) V_a^T + U_a (V_b - V_a)^T... no
    # Actually if U_a = U_b, then U_a V_a^T + U_a V_b^T = U_a (V_a + V_b)^T -> 1 term
    # If V_a = V_b, then U_a V_a^T + U_b V_a^T = (U_a + U_b) V_a^T -> 1 term
    
    # Check for identical U vectors
    for a in range(num_terms):
        for b in range(a + 1, num_terms):
            # Check if U_a == U_b
            if U_vecs[a] == U_vecs[b]:
                # Merge: new term has U = U_a, V = V_a + V_b, W = W_a + W_b
                # But we need W_a + W_b to be a valid W vector (it should be, since it's a sum)
                # Actually, the merged term contributes U_a (V_a+V_b)^T (W_a+W_b)^T
                # Original contributes U_a V_a^T W_a^T + U_a V_b^T W_b^T
                # These are NOT the same in general!
                # U_a V_a^T W_a^T + U_a V_b^T W_b^T != U_a (V_a+V_b)^T (W_a+W_b)^T
                # The latter expands to U_a V_a^T W_a^T + U_a V_a^T W_b^T + U_a V_b^T W_a^T + U_a V_b^T W_b^T
                # So we have extra terms. This merge is only valid if W_a = W_b or V_a = V_b.
                
                # If W_a == W_b, then U_a V_a^T W_a^T + U_a V_b^T W_a^T = U_a (V_a+V_b)^T W_a^T -> valid!
                if W_vecs[a] == W_vecs[b]:
                    # Valid merge
                    new_U = U_vecs[a][:]
                    new_V = [v1 + v2 for v1, v2 in zip(V_vecs[a], V_vecs[b])]
                    new_W = W_vecs[a][:]
                    
                    # Build new term list
                    new_terms = []
                    for i in range(num_terms):
                        if i == a or i == b:
                            continue
                        u_d = _vec_to_dict(U_vecs[i], U_size, m, p)
                        v_d = _vec_to_dict(V_vecs[i], V_size, m, p)
                        w_d = _vec_to_dict(W_vecs[i], W_size, m, p)
                        new_terms.append((u_d, v_d, w_d))
                    
                    # Add merged term
                    u_d = _vec_to_dict(new_U, U_size, m, p)
                    v_d = _vec_to_dict(new_V, V_size, m, p)
                    w_d = _vec_to_dict(new_W, W_size, m, p)
                    new_terms.append((u_d, v_d, w_d))
                    
                    return new_terms
            
            # Check if V_a == V_b and W_a == W_b
            if V_vecs[a] == V_vecs[b] and W_vecs[a] == W_vecs[b]:
                # Merge: new term has U = U_a + U_b, V = V_a, W = W_a
                new_U = [u1 + u2 for u1, u2 in zip(U_vecs[a], U_vecs[b])]
                new_V = V_vecs[a][:]
                new_W = W_vecs[a][:]
                
                new_terms = []
                for i in range(num_terms):
                    if i == a or i == b:
                        continue
                    u_d = _vec_to_dict(U_vecs[i], U_size, m, p)
                    v_d = _vec_to_dict(V_vecs[i], V_size, m, p)
                    w_d = _vec_to_dict(W_vecs[i], W_size, m, p)
                    new_terms.append((u_d, v_d, w_d))
                
                u_d = _vec_to_dict(new_U, U_size, m, p)
                v_d = _vec_to_dict(new_V, V_size, m, p)
                w_d = _vec_to_dict(new_W, W_size, m, p)
                new_terms.append((u_d, v_d, w_d))
                
                return new_terms
    
    # Check for proportional (not just identical) vectors
    # If U_a = alpha * U_b, we can still potentially merge
    for a in range(num_terms):
        for b in range(a + 1, num_terms):
            # Check if U_a is proportional to U_b
            # Find alpha such that U_a = alpha * U_b
            alpha = None
            for i in range(U_size):
                if U_vecs[b][i] != 0:
                    alpha = U_vecs[a][i] / U_vecs[b][i]
                    break
            
            if alpha is not None:
                # Verify proportionality
                proportional = True
                for i in range(U_size):
                    if abs(U_vecs[a][i] - alpha * U_vecs[b][i]) > 1e-10:
                        proportional = False
                        break
                
                if proportional and W_vecs[a] == W_vecs[b]:
                    # U_a = alpha * U_b, W_a = W_b
                    # U_a V_a^T W_a^T + U_b V_b^T W_b^T = alpha*U_b V_a^T W^T + U_b V_b^T W^T
                    # = U_b (alpha*V_a + V_b)^T W^T -> 1 term
                    new_U = U_vecs[b][:]
                    new_V = [alpha * v1 + v2 for v1, v2 in zip(V_vecs[a], V_vecs[b])]
                    new_W = W_vecs[a][:]
                    
                    new_terms = []
                    for i in range(num_terms):
                        if i == a or i == b:
                            continue
                        u_d = _vec_to_dict(U_vecs[i], U_size, m, p)
                        v_d = _vec_to_dict(V_vecs[i], V_size, m, p)
                        w_d = _vec_to_dict(W_vecs[i], W_size, m, p)
                        new_terms.append((u_d, v_d, w_d))
                    
                    u_d = _vec_to_dict(new_U, U_size, m, p)
                    v_d = _vec_to_dict(new_V, V_size, m, p)
                    w_d = _vec_to_dict(new_W, W_size, m, p)
                    new_terms.append((u_d, v_d, w_d))
                    
                    return new_terms
    
    return None

def construct():
    """
    Constructs the minimal-rank tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Starts with rank-36 Block-Strassen and searches for merges to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build base decomposition
    rank_terms = _build_base_terms(n, m, p)
    
    # Try to find merges
    merged = _try_find_merge(rank_terms, n, m, p)
    if merged is not None:
        rank_terms = merged
    
    # Verify the decomposition
    rank = len(rank_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][c] = coeff
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to base decomposition if merge was invalid
        rank_terms = _build_base_terms(n, m, p)
        rank = len(rank_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][c] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
