"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _decompose_strassen(n, m, p):
    """Build the known rank-17 decomposition (Strassen + direct)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Strassen for A1(2x2) x B1_left(2x2)
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Direct for A1(2x2) x B1_right(2x1)
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Direct for A2(2x1) x B2(1x3)
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _get_targets():
    """The 10 target positions needing coverage beyond Strassen."""
    return [
        (0, 2, 4),  # A[0,0]*B[0,2]->C[0,2]
        (1, 5, 4),  # A[0,1]*B[1,2]->C[0,2]
        (3, 2, 5),  # A[1,0]*B[0,2]->C[1,2]
        (4, 5, 5),  # A[1,1]*B[1,2]->C[1,2]
        (2, 6, 0),  # A[0,2]*B[2,0]->C[0,0]
        (2, 7, 2),  # A[0,2]*B[2,1]->C[0,1]
        (2, 8, 4),  # A[0,2]*B[2,2]->C[0,2]
        (5, 6, 1),  # A[1,2]*B[2,0]->C[1,0]
        (5, 7, 3),  # A[1,2]*B[2,1]->C[1,1]
        (5, 8, 5),  # A[1,2]*B[2,2]->C[1,2]
    ]

def _build_rank16_candidate(pair_indices, strassen_U, strassen_V, strassen_W, targets, T):
    """
    Build a rank-16 candidate with one pair term and 8 direct terms.
    pair_indices: tuple (i, j) of target indices covered by the pair term.
    """
    n_a, n_b, n_c = 6, 9, 6
    i, j = pair_indices
    a1, b1, c1 = targets[i]
    a2, b2, c2 = targets[j]
    
    U = np.zeros((n_a, 16))
    V = np.zeros((n_b, 16))
    W = np.zeros((n_c, 16))
    
    U[:, :7] = strassen_U[:, :7]
    V[:, :7] = strassen_V[:, :7]
    W[:, :7] = strassen_W[:, :7]
    
    # Pair term at column 7
    U[a1, 7] = 1
    U[a2, 7] = 1
    V[b1, 7] = 1
    V[b2, 7] = 1
    W[c1, 7] = 1
    W[c2, 7] = 1
    
    # Direct terms for remaining 8 targets
    remaining = [t for idx, t in enumerate(targets) if idx != i and idx != j]
    for k, (a, b, c) in enumerate(remaining):
        col = 8 + k
        U[a, col] = 1
        V[b, col] = 1
        W[c, col] = 1
    
    if _verify_decomposition(U, V, W, T):
        return U, V, W
    return None

def _try_rank16_pair_terms(deadline):
    """
    Search for rank-16 with one pair term.
    Enumerate all valid pairs where the cross-product only hits those 2 targets.
    """
    targets = _get_targets()
    target_set = set(targets)
    T = _build_tensor(2, 3, 3)
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    
    # Pre-compute valid pairs
    valid_pairs = []
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            a1, b1, c1 = targets[i]
            a2, b2, c2 = targets[j]
            
            u_set = {a1, a2}
            v_set = {b1, b2}
            w_set = {c1, c2}
            
            produced = set()
            for a in u_set:
                for b in v_set:
                    for c in w_set:
                        produced.add((a, b, c))
            
            if produced & target_set == {(a1,b1,c1), (a2,b2,c2)}:
                valid_pairs.append((i, j))
    
    # Try each valid pair
    for pair in valid_pairs:
        if time.time() > deadline:
            return None
        result = _build_rank16_candidate(pair, strassen_U, strassen_V, strassen_W, targets, T)
        if result is not None:
            return result
    
    return None

def _try_rank16_two_pairs(deadline):
    """
    Search for rank-16 with two pair terms covering 4 targets total,
    and 6 direct terms for the remaining 6 targets.
    """
    targets = _get_targets()
    target_set = set(targets)
    T = _build_tensor(2, 3, 3)
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    
    # Find all valid pairs
    valid_pairs = []
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            a1, b1, c1 = targets[i]
            a2, b2, c2 = targets[j]
            u_set = {a1, a2}
            v_set = {b1, b2}
            w_set = {c1, c2}
            produced = set()
            for a in u_set:
                for b in v_set:
                    for c in w_set:
                        produced.add((a, b, c))
            if produced & target_set == {(a1,b1,c1), (a2,b2,c2)}:
                valid_pairs.append((i, j))
    
    # Try all pairs of valid pairs that are disjoint
    for pi in range(len(valid_pairs)):
        if time.time() > deadline:
            return None
        for pj in range(pi+1, len(valid_pairs)):
            p1 = valid_pairs[pi]
            p2 = valid_pairs[pj]
            
            # Check disjointness
            if len(set(p1) & set(p2)) > 0:
                continue
            
            i1, j1 = p1
            i2, j2 = p2
            
            a1, b1, c1 = targets[i1]
            a2, b2, c2 = targets[j1]
            a3, b3, c3 = targets[i2]
            a4, b4, c4 = targets[j2]
            
            n_a, n_b, n_c = 6, 9, 6
            U = np.zeros((n_a, 16))
            V = np.zeros((n_b, 16))
            W = np.zeros((n_c, 16))
            
            U[:, :7] = strassen_U[:, :7]
            V[:, :7] = strassen_V[:, :7]
            W[:, :7] = strassen_W[:, :7]
            
            # Pair term 1 at column 7
            U[a1, 7] = 1
            U[a2, 7] = 1
            V[b1, 7] = 1
            V[b2, 7] = 1
            W[c1, 7] = 1
            W[c2, 7] = 1
            
            # Pair term 2 at column 8
            U[a3, 8] = 1
            U[a4, 8] = 1
            V[b3, 8] = 1
            V[b4, 8] = 1
            W[c3, 8] = 1
            W[c4, 8] = 1
            
            # Direct terms for remaining 6 targets
            used = {i1, j1, i2, j2}
            remaining = [t for idx, t in enumerate(targets) if idx not in used]
            for k, (a, b, c) in enumerate(remaining):
                col = 9 + k
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
            
            if _verify_decomposition(U, V, W, T):
                return U, V, W
    
    return None

def _try_rank16_triple_term(deadline):
    """
    Search for rank-16 with one triple term covering 3 targets,
    and 7 direct terms for the remaining 7 targets.
    """
    targets = _get_targets()
    target_set = set(targets)
    T = _build_tensor(2, 3, 3)
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    
    n_a, n_b, n_c = 6, 9, 6
    
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            for k in range(j+1, len(targets)):
                if time.time() > deadline:
                    return None
                
                a1, b1, c1 = targets[i]
                a2, b2, c2 = targets[j]
                a3, b3, c3 = targets[k]
                
                u_set = {a1, a2, a3}
                v_set = {b1, b2, b3}
                w_set = {c1, c2, c3}
                
                produced = set()
                for a in u_set:
                    for b in v_set:
                        for c in w_set:
                            produced.add((a, b, c))
                
                if produced & target_set == {(a1,b1,c1), (a2,b2,c2), (a3,b3,c3)}:
                    U = np.zeros((n_a, 16))
                    V = np.zeros((n_b, 16))
                    W = np.zeros((n_c, 16))
                    
                    U[:, :7] = strassen_U[:, :7]
                    V[:, :7] = strassen_V[:, :7]
                    W[:, :7] = strassen_W[:, :7]
                    
                    # Triple term at column 7
                    U[a1, 7] = 1
                    U[a2, 7] = 1
                    U[a3, 7] = 1
                    V[b1, 7] = 1
                    V[b2, 7] = 1
                    V[b3, 7] = 1
                    W[c1, 7] = 1
                    W[c2, 7] = 1
                    W[c3, 7] = 1
                    
                    # Direct terms for remaining 7
                    used = {i, j, k}
                    remaining = [t for idx, t in enumerate(targets) if idx not in used]
                    for idx, (a, b, c) in enumerate(remaining):
                        col = 8 + idx
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                    
                    if _verify_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def _try_rank16_half_coefficients(deadline):
    """
    Try rank-16 with half-integer coefficients.
    Explore pairs where using 0.5 coefficients on both targets
    allows the cross-product to cancel unwanted entries.
    """
    targets = _get_targets()
    target_set = set(targets)
    T = _build_tensor(2, 3, 3)
    strassen_U, strassen_V, strassen_W = _decompose_strassen(2, 3, 3)
    
    n_a, n_b, n_c = 6, 9, 6
    
    # For each pair, try if half-coefficients work
    # U[a1]=0.5, U[a2]=0.5, V[b1]=1, V[b2]=1, W[c1]=2, W[c2]=2
    # Then at (a1,b1,c1): 0.5*1*2 = 1
    # At (a2,b2,c2): 0.5*1*2 = 1
    # But cross terms like (a1,b2,c1): 0.5*1*2 = 1 (unwanted)
    # So this only works if no cross terms hit targets.
    
    # Alternative: U[a1]=1, U[a2]=1, V[b1]=0.5, V[b2]=0.5, W[c1]=2, W[c2]=2
    # Same issue.
    
    # Try: U[a1]=1, U[a2]=1, V[b1]=1, V[b2]=1, W[c1]=0.5, W[c2]=0.5
    # At (a1,b1,c1): 1*1*0.5 = 0.5 (not 1)
    # Need to scale: multiply all by 2. But then other entries scale too.
    
    # Actually, for a pair term to give exactly 1 at both targets with halves:
    # su1*sv1*sc1 = 1 and su2*sv2*sc2 = 1
    # With halves: e.g., su1=1, sv1=0.5, sc1=2 -> product=1
    # su2=1, sv2=0.5, sc2=2 -> product=1
    # Cross: (a1,b2,c1): 1*0.5*2=1 (bad if in target_set)
    # (a1,b1,c2): 1*0.5*2=1 (bad if in target_set)
    
    # So half coefficients don't help unless the cross terms don't hit targets.
    # Let's just verify this is equivalent to the integer case for valid pairs.
    
    # Try a different approach: two terms that partially overlap
    # Term A covers target i with coefficient 1, and also hits target j with coefficient 1
    # Term B covers target j with coefficient 1, and also hits target i with coefficient -1
    # Net: target i gets 1-1=0 (bad), target j gets 1+1=2 (bad)
    
    # Or: Term A gives 1 at target i, 1 at target j
    # Term B gives 0 at target i, -1 at some other target k
    # This doesn't help reduce rank.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank-16 with one pair term
    result = _try_rank16_pair_terms(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-16 with two pair terms
    result = _try_rank16_two_pairs(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-16 with a triple term
    result = _try_rank16_triple_term(deadline)
    if result is not None:
        U, V, W = result
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-17
    U, V, W = _decompose_strassen(n, m, p)
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Schoolbook fallback
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
