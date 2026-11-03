"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy (vectorized)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,l,k] = sum_r U[i*m+l,r] * V[l*p+k,r] * W[k*n+i,r]
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        T += np.einsum('il,lk,ki->ilk', u_2d, v_2d, w_2d)
    
    return np.all(np.abs(T - 1.0) < 1e-9)

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs, k_singles=None):
    """Build a Strassen-based decomposition with given pairings.
    
    k_pairs: list of (k0,k1) pairs for Strassen 2x2 blocks
    k_singles: list of single k indices (default: all k not in k_pairs)
    """
    if k_singles is None:
        used_k = set()
        for k0, k1 in k_pairs:
            used_k.add(k0)
            used_k.add(k1)
        k_singles = [k for k in range(p) if k not in used_k]
    
    # Rank calculation:
    # For each j_pair and k_pair: 7 terms (Strassen 2x2)
    # For each j_pair and k_single: 4 terms (standard 2x1)
    # For each j_single and k: 2 terms
    rank = 7 * len(j_pairs) * len(k_pairs) + 4 * len(j_pairs) * len(k_singles) + 2 * (m - 2*len(j_pairs)) * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
        
        # For each k_single with this j_pair: 4 standard products
        for ks in k_singles:
            U[0*m + j0][col] += 1
            V[j0*p + ks][col] += 1
            W[ks*n + 0][col] += 1
            col += 1
            
            U[0*m + j1][col] += 1
            V[j1*p + ks][col] += 1
            W[ks*n + 0][col] += 1
            col += 1
            
            U[1*m + j0][col] += 1
            V[j0*p + ks][col] += 1
            W[ks*n + 1][col] += 1
            col += 1
            
            U[1*m + j1][col] += 1
            V[j1*p + ks][col] += 1
            W[ks*n + 1][col] += 1
            col += 1
    
    # Remaining j indices: standard products
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    
    for j in range(m):
        if j in used_j:
            continue
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val - round(val)) < tol:
                new_row.append(float(round(val)))
            elif abs(val - (round(val*2)/2.0)) < tol:
                new_row.append(round(val*2)/2.0)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def _eliminate_redundant_terms(U, V, W, n, m, p, rank, time_fn):
    """Try to eliminate redundant terms. Returns (U, V, W, new_rank)."""
    current_U, current_V, current_W, current_rank = U, V, W, rank
    
    # Convert to numpy for faster operations
    U_arr = np.array(current_U, dtype=float)
    V_arr = np.array(current_V, dtype=float)
    W_arr = np.array(current_W, dtype=float)
    
    # Precompute full tensor T
    T_full = np.zeros((n, m, p), dtype=float)
    for r in range(current_rank):
        u_2d = U_arr[:, r].reshape(n, m)
        v_2d = V_arr[:, r].reshape(m, p)
        w_2d = W_arr[:, r].reshape(p, n)
        T_full += np.einsum('il,lk,ki->ilk', u_2d, v_2d, w_2d)
    
    # Compute individual term contributions
    T_terms = np.zeros((current_rank, n, m, p), dtype=float)
    for r in range(current_rank):
        u_2d = U_arr[:, r].reshape(n, m)
        v_2d = V_arr[:, r].reshape(m, p)
        w_2d = W_arr[:, r].reshape(p, n)
        T_terms[r] = np.einsum('il,lk,ki->ilk', u_2d, v_2d, w_2d)
    
    active = [True] * current_rank
    count = current_rank
    
    # Greedy elimination: try to remove any term whose removal keeps T == 1
    for iteration in range(current_rank):
        if time_fn() < 3:
            break
        
        # Find removable terms: T_full - T_terms[r] == 1 for all (i,l,k)
        removed_this_round = False
        for r in range(current_rank):
            if not active[r]:
                continue
            T_without = T_full - T_terms[r]
            if np.all(np.abs(T_without - 1.0) < 1e-9):
                # Remove term r
                active[r] = False
                count -= 1
                T_full = T_without  # Update T_full
                removed_this_round = True
                break  # Restart from beginning after removal
        
        if not removed_this_round:
            break
    
    if count == current_rank:
        return current_U, current_V, current_W, current_rank
    
    # Build output arrays
    keep_indices = [r for r in range(current_rank) if active[r]]
    U_out = U_arr[:, keep_indices].tolist()
    V_out = V_arr[:, keep_indices].tolist()
    W_out = W_arr[:, keep_indices].tolist()
    
    return U_out, V_out, W_out, count

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin before 150s limit
    
    def _time_left():
        return _deadline - time.time()
    
    best_rank = float('inf')
    best_result = None
    
    # Generate all possible k-pair configurations for p=6
    # We can have 0, 1, 2, or 3 k-pairs
    # For each number of pairs, generate all matchings
    
    def _gen_k_configs(p):
        """Generate all ways to partition k indices into pairs and singles."""
        configs = []
        
        # 3 pairs (all paired)
        k_list = list(range(p))
        for perm in set(combinations(k_list, 2)):
            remaining = [x for x in k_list if x not in perm]
            for perm2 in set(combinations(remaining, 2)):
                remaining2 = [x for x in remaining if x not in perm2]
                if len(remaining2) == 2:
                    perm3 = tuple(sorted(remaining2))
                    k_pairs = [tuple(sorted(perm)), tuple(sorted(perm2)), perm3]
                    k_pairs = tuple(sorted(k_pairs))
                    configs.append((list(k_pairs), []))
        
        # 2 pairs + 2 singles
        for pair1 in combinations(k_list, 2):
            remaining1 = [x for x in k_list if x not in pair1]
            for pair2 in combinations(remaining1, 2):
                remaining2 = [x for x in remaining1 if x not in pair2]
                k_pairs = [tuple(sorted(pair1)), tuple(sorted(pair2))]
                k_pairs = tuple(sorted(k_pairs))
                k_singles = sorted(remaining2)
                configs.append((list(k_pairs), k_singles))
        
        # 1 pair + 4 singles
        for pair in combinations(k_list, 2):
            remaining = [x for x in k_list if x not in pair]
            k_pairs = [tuple(sorted(pair))]
            k_singles = sorted(remaining)
            configs.append((k_pairs, k_singles))
        
        # 0 pairs + 6 singles
        configs.append(([], list(range(p))))
        
        # Deduplicate
        seen = set()
        unique_configs = []
        for kp, ks in configs:
            key = (tuple(sorted(kp)), tuple(ks))
            if key not in seen:
                seen.add(key)
                unique_configs.append((kp, ks))
        
        return unique_configs
    
    k_configs = _gen_k_configs(p)
    
    # Generate j-pair configurations for m=5
    j_configs = []
    # 2 pairs + 1 single
    for jp1 in combinations(range(m), 2):
        remaining1 = [x for x in range(m) if x not in jp1]
        for jp2 in combinations(remaining1, 2):
            remaining2 = [x for x in remaining1 if x not in jp2]
            j_pairs = [tuple(sorted(jp1)), tuple(sorted(jp2))]
            j_pairs = tuple(sorted(j_pairs))
            j_singles = sorted(remaining2)
            j_configs.append((list(j_pairs), j_singles))
    
    # 1 pair + 3 singles
    for jp in combinations(range(m), 2):
        remaining = [x for x in range(m) if x not in jp]
        j_pairs = [tuple(sorted(jp))]
        j_singles = sorted(remaining)
        j_configs.append((j_pairs, j_singles))
    
    # 0 pairs + 5 singles
    j_configs.append(([], list(range(m))))
    
    # Deduplicate j_configs
    seen_j = set()
    unique_j_configs = []
    for jp, js in j_configs:
        key = (tuple(sorted(jp)), tuple(js))
        if key not in seen_j:
            seen_j.add(key)
            unique_j_configs.append((jp, js))
    j_configs = unique_j_configs
    
    # Try all combinations
    total_configs = len(j_configs) * len(k_configs)
    configs_tried = 0
    
    for j_pairs, j_singles in j_configs:
        if _time_left() < 10:
            break
        for k_pairs, k_singles in k_configs:
            if _time_left() < 5:
                break
            
            configs_tried += 1
            
            U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs, k_singles)
            rank = len(U[0])
            
            if rank >= best_rank:
                continue
            
            if not _verify_decomp(U, V, W, n, m, p, rank):
                continue
            
            # Try to eliminate redundant terms
            U2, V2, W2, rank2 = _eliminate_redundant_terms(U, V, W, n, m, p, rank, _time_left)
            
            if rank2 < best_rank:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
                # Early exit if we hit theoretical lower bound
                # Lower bound: n*m*p / max(n,m,p) is not quite right
                # For 2x5x6: we need at least ceil(2*5*6 / 6) = 10? No.
                # The tensor has 60 entries. Each term covers at most n*m*p = 60 positions.
                # But each term is a rank-1 tensor. The tensor rank is what we're minimizing.
                # Known lower bound: max(n*m, m*p, p*n) / max(n,m,p) doesn't apply directly.
                # For now, just keep searching unless we find rank <= 30 (heuristic).
    
    # Fallback: standard configuration
    if best_result is None:
        U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
        best_result = {"U": U, "V": V, "W": W}
        best_rank = len(U[0])
    
    # Snap to half-integers for cleaner output
    best_result["U"] = _snap_to_half_integer(best_result["U"])
    best_result["V"] = _snap_to_half_integer(best_result["V"])
    best_result["W"] = _snap_to_half_integer(best_result["W"])
    
    # Final verification
    if not _verify_decomp(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
        # Fallback to standard
        U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
        best_result = {"U": U, "V": V, "W": W}
        best_rank = len(U[0])
    
    return best_result
# EVOLVE-BLOCK-END
