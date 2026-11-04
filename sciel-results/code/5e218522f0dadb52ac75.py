"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
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
    return U, V, W


def _build_index_maps(n, m, p):
    """Precompute which (i,j,k) triples each row/col in U,V,W participates in."""
    # For U[i*m+j], it participates in triples where that i,j and any k
    u_map = {}  # u_idx -> list of (i,j,k)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            u_map[u_idx] = [(i, j, k) for k in range(p)]
    
    # For V[j*p+k], it participates in triples where that j,k and any i
    v_map = {}  # v_idx -> list of (i,j,k)
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            v_map[v_idx] = [(i, j, k) for i in range(n)]
    
    # For W[k*n+i], it participates in triples where that k,i and any j
    w_map = {}  # w_idx -> list of (i,j,k)
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            w_map[w_idx] = [(i, j, k) for j in range(m)]
    
    return u_map, v_map, w_map


def _search_rank(r, n, m, p, time_budget, seed):
    """Local search for an exact rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(seed)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Precompute index maps
    u_map, v_map, w_map = _build_index_maps(n, m, p)
    
    # Precompute triple-to-row mappings
    # triple (i,j,k) -> (u_idx, v_idx, w_idx)
    triple_to_rows = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_rows[(i,j,k)] = (i*m+j, j*p+k, k*n+i)
    
    # Sparse random init with better density
    density = 0.4
    for row in range(n * m):
        for col in range(r):
            if rng.random() < density:
                U[row][col] = rng.choice([1, -1])
    for row in range(m * p):
        for col in range(r):
            if rng.random() < density:
                V[row][col] = rng.choice([1, -1])
    for row in range(p * n):
        for col in range(r):
            if rng.random() < density:
                W[row][col] = rng.choice([1, -1])

    # Precompute current sums for each triple
    # sums[(i,j,k)] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    sums = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                u_row = U[u_idx]
                v_row = V[v_idx]
                w_row = W[w_idx]
                for c in range(r):
                    s += u_row[c] * v_row[c] * w_row[c]
                sums[(i,j,k)] = s

    def compute_error():
        e = 0
        for (i,j,k), s in sums.items():
            e += (s - 1) * (s - 1)
        return e

    cur = compute_error()
    best = cur
    best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
    no_improve = 0
    steps = 0
    max_steps = 8000

    # Fast delta computation: when we change U[u_idx][c] by delta,
    # the change in sum for triple (i,j,k) is delta * V[v_idx][c] * W[w_idx][c]
    def get_affected_triples(factor, idx, col):
        """Get list of (triple_key, other_factor_idx, other_col_value) for affected triples."""
        if factor == 'U':
            triples = u_map[idx]
            return [(t, triple_to_rows[t][1], V[triple_to_rows[t][1]][col]) for t in triples]
        elif factor == 'V':
            triples = v_map[idx]
            return [(t, triple_to_rows[t][2], W[triple_to_rows[t][2]][col]) for t in triples]
        else:  # 'W'
            triples = w_map[idx]
            return [(t, triple_to_rows[t][0], U[triple_to_rows[t][0]][col]) for t in triples]

    factors = [('U', U, n * m), ('V', V, m * p), ('W', W, p * n)]
    
    while no_improve < 300 and steps < max_steps:
        if time.time() - start > time_budget:
            break
        steps += 1
        
        # Choose factor, row, col
        factor_name, M, nrows = factors[rng.randrange(3)]
        row = rng.randrange(nrows)
        col = rng.randrange(r)
        old = M[row][col]
        
        # Try a new value
        candidates = [v for v in [0, 1, -1] if v != old]
        new = rng.choice(candidates)
        
        # Compute delta error using precomputed sums
        delta_val = new - old
        affected = get_affected_triples(factor_name, row, col)
        
        # Compute new error
        new_err = cur
        for (t, other_idx, other_val) in affected:
            old_sum = sums[t]
            new_sum = old_sum + delta_val * other_val
            new_err -= (old_sum - 1) * (old_sum - 1)
            new_err += (new_sum - 1) * (new_sum - 1)
        
        if new_err < cur:
            # Accept: update sums
            for (t, other_idx, other_val) in affected:
                sums[t] += delta_val * other_val
            M[row][col] = new
            cur = new_err
            no_improve = 0
            if cur < best:
                best = cur
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            # Reject with some probability (simulated annealing-like)
            if rng.random() > 0.15:
                pass  # keep old value
            else:
                # Accept anyway to escape local minima
                for (t, other_idx, other_val) in affected:
                    sums[t] += delta_val * other_val
                M[row][col] = new
                cur = new_err
            no_improve += 1

    Ub, Vb, Wb = best_state
    if _verify(Ub, Vb, Wb, n, m, p):
        return Ub, Vb, Wb
    
    # Try a final polish: greedy improvement on the best state
    U2, V2, W2 = [row[:] for row in Ub], [row[:] for row in Vb], [row[:] for row in Wb]
    # Recompute sums for polish
    sums2 = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                for c in range(r):
                    s += U2[u_idx][c] * V2[v_idx][c] * W2[w_idx][c]
                sums2[(i,j,k)] = s
    
    cur2 = sum((s-1)**2 for s in sums2.values())
    improved = True
    polish_steps = 0
    while improved and polish_steps < 500:
        improved = False
        polish_steps += 1
        for factor_name, M2, nrows in [('U', U2, n*m), ('V', V2, m*p), ('W', W2, p*n)]:
            for row in range(nrows):
                for col in range(r):
                    old = M2[row][col]
                    for new in [v for v in [0,1,-1] if v != old]:
                        delta_val = new - old
                        affected = get_affected_triples(factor_name, row, col)
                        new_err2 = cur2
                        for (t, other_idx, other_val) in affected:
                            # Use original V/W/U values from best_state, not M2
                            if factor_name == 'U':
                                ov = Vb[other_idx][col]
                            elif factor_name == 'V':
                                ov = Wb[other_idx][col]
                            else:
                                ov = Ub[other_idx][col]
                            old_sum = sums2[t]
                            new_sum = old_sum + delta_val * ov
                            new_err2 -= (old_sum - 1) * (old_sum - 1)
                            new_err2 += (new_sum - 1) * (new_sum - 1)
                        if new_err2 < cur2:
                            for (t, other_idx, other_val) in affected:
                                if factor_name == 'U':
                                    sums2[t] += delta_val * Vb[other_idx][col]
                                elif factor_name == 'V':
                                    sums2[t] += delta_val * Wb[other_idx][col]
                                else:
                                    sums2[t] += delta_val * Ub[other_idx][col]
                            M2[row][col] = new
                            cur2 = new_err2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    if _verify(U2, V2, W2, n, m, p):
        return U2, V2, W2
    return None


def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    # Guaranteed valid baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # Try progressively lower ranks; stop once time runs low.
    # Known optimal rank for 3x3x3 matrix multiplication is 23 (Laderman is 23 for 3x3? Actually it's 23 for general, but let's try down to 19)
    # Actually the known minimal rank for 3x3 matrix multiplication is 23 (Laderman's algorithm)
    # Wait, Laderman is for 3x3 and has rank 23. Let me verify... Actually Laderman's algorithm achieves rank 23 for 3x3.
    # But we should try to find better if possible.
    for target_rank in range(23, 18, -1):
        if time.time() - start_time > timeout - 8:
            break
        for seed in range(6):
            if time.time() - start_time > timeout - 8:
                break
            res = _search_rank(target_rank, n, m, p, time_budget=12.0, seed=seed)
            if res is not None:
                U, V, W = res
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # found valid at this rank, try next (lower) rank

    # Final safety: ensure we return something valid
    if time.time() - start_time > timeout - 1:
        return best

    return best
# EVOLVE-BLOCK-END
