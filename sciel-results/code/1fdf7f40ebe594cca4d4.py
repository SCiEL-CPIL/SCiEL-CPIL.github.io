"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 138  # Safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        if rank == 0:
            return False
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def solve_for_W(U, V, rank, tol=1e-8):
        """
        Given U and V, solve for W such that sum_r U_ij_r V_jk_r W_ki_r = 1.
        Returns W as a list of lists or None if no integer solution found.
        Uses rational arithmetic for exactness when close to integers.
        """
        num_unknowns = pn * rank
        num_eqs = n * m * p

        A = np.zeros((num_eqs, num_unknowns), dtype=np.float64)
        b = np.ones(num_eqs, dtype=np.float64)

        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        u_val = U[i * m + j][r]
                        v_val = V[j * p + k][r]
                        if u_val != 0 and v_val != 0:
                            w_idx = (k * n + i) * rank + r
                            A[eq_idx, w_idx] = float(u_val * v_val)
                    eq_idx += 1

        try:
            w_real, residuals, rank_solver, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        # Check if solution is close to integers
        rounded = np.round(w_real)
        max_diff = np.max(np.abs(w_real - rounded))
        
        if max_diff < tol:
            w_int = rounded.astype(np.int64)
            W_flat = w_int.reshape(pn, rank)
        elif max_diff < 0.5:
            # Near-integer: try exact rational solve on a subset
            # Identify which equations are most constrained
            w_int = rounded.astype(np.int64)
            W_flat = w_int.reshape(pn, rank)
            # Verify; if fails, try small perturbations
            U_np = np.array(U, dtype=np.int64)
            V_np = np.array(V, dtype=np.int64)
            W_np = W_flat
            U3 = U_np.reshape(n, m, rank)
            V3 = V_np.reshape(m, p, rank)
            W3 = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            if np.all(T == 1):
                return W_np.tolist()
            return None
        else:
            return None

        # Verify exactly
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = W_flat
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

        if np.all(T == 1):
            return W_np.tolist()
        else:
            return None

    def try_rank_fix_uv(target_rank, max_attempts, value_range):
        """Fix U and V randomly, solve for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            U = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(mp)]

            W = solve_for_W(U, V, target_rank)
            if W is not None:
                return {"U": U, "V": V, "W": W}

        return None

    def try_rank_structured(target_rank):
        """
        Use structured patterns exploiting the rectangular asymmetry.
        For n=2, we can use patterns based on the 2x3 block structure.
        Key insight: U[i*m+j, r] can factor as a_i * b_j * c_r where a_i in {1,-1}
        This reduces the effective dimensionality and increases integrality chance.
        """
        if time.time() > deadline - 5:
            return None

        attempts = 40
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                return None

            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]

            # Structured U: U[i*m+j, r] = a_i[r] * b_j[r]
            # a_i: 2 vectors of length target_rank, entries in {-1, 0, 1}
            # b_j: 3 vectors of length target_rank, entries in {-1, 0, 1}
            a_vals = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n)]
            b_vals = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m)]

            for i in range(n):
                for j in range(m):
                    for r in range(target_rank):
                        U[i * m + j][r] = a_vals[i][r] * b_vals[j][r]

            # Structured V: V[j*p+k, r] = d_j[r] * e_k[r]
            d_vals = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m)]
            e_vals = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p)]

            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        V[j * p + k][r] = d_vals[j][r] * e_vals[k][r]

            W = solve_for_W(U, V, target_rank)
            if W is not None:
                return {"U": U, "V": V, "W": W}

        return None

    def try_kronecker_block(target_rank):
        """
        Exploit the 2x3x7 structure via block decomposition.
        Treat the 7 columns as 2 blocks of 3 + 1, or use the fact that
        rank(2x3x7) can leverage rank(2x3x3) = 5 structure.
        """
        if time.time() > deadline - 5:
            return None

        attempts = 30
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                return None

            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]

            # Pattern: U depends on i via a sign flip (n=2 symmetry)
            # U[0*m+j, r] = u_j[r], U[1*m+j, r] = s_j[r] * u_j[r]
            # where s_j[r] in {-1, 1}
            u_vals = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m)]
            s_vals = [[random.choice([-1, 1]) for _ in range(target_rank)] for _ in range(m)]

            for j in range(m):
                for r in range(target_rank):
                    U[0 * m + j][r] = u_vals[j][r]
                    U[1 * m + j][r] = s_vals[j][r] * u_vals[j][r]

            # V: V[j*p+k, r] = v_jk[r] with some structure
            # Group k into {0,1,2} and {3,4,5} and {6}
            # Use different patterns for different k blocks
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        if k < 3:
                            V[j * p + k][r] = random.choice([-1, 0, 1])
                        elif k < 6:
                            V[j * p + k][r] = random.choice([-1, 0, 1])
                        else:
                            V[j * p + k][r] = random.choice([-1, 0, 1])

            W = solve_for_W(U, V, target_rank)
            if W is not None:
                return {"U": U, "V": V, "W": W}

        return None

    def try_gf2_sparse(target_rank, max_iter=50):
        """
        Sparse search: generate U, V, W with mostly zeros and verify directly.
        """
        if time.time() > deadline - 5:
            return None

        for attempt in range(max_iter):
            if time.time() > deadline - 5:
                return None

            density = random.choice([0.05, 0.1, 0.15, 0.2])
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]

            for r in range(target_rank):
                # Each rank component is sparse
                for idx in range(nm):
                    if random.random() < density:
                        U[idx][r] = random.choice([-1, 1])
                for idx in range(mp):
                    if random.random() < density:
                        V[idx][r] = random.choice([-1, 1])
                for idx in range(pn):
                    if random.random() < density:
                        W[idx][r] = random.choice([-1, 1])

            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}

        return None

    def try_polished_search(target_rank, max_attempts=30):
        """
        Generate random U, V, solve for W, then polish near-integer solutions
        by trying small integer perturbations to W.
        """
        if time.time() > deadline - 5:
            return None

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]

            num_unknowns = pn * target_rank
            num_eqs = n * m * p

            A = np.zeros((num_eqs, num_unknowns), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)

            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            u_val = U[i * m + j][r]
                            v_val = V[j * p + k][r]
                            if u_val != 0 and v_val != 0:
                                w_idx = (k * n + i) * target_rank + r
                                A[eq_idx, w_idx] = float(u_val * v_val)
                        eq_idx += 1

            try:
                w_real, residuals, rank_solver, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue

            rounded = np.round(w_real)
            max_diff = np.max(np.abs(w_real - rounded))

            if max_diff < 0.3:
                # Try the rounded solution and small perturbations
                candidates = [rounded]
                # Try perturbing entries that are closest to half-integers
                frac_parts = np.abs(w_real - rounded)
                # Get indices of the largest fractional parts
                flat_indices = np.argsort(frac_parts)[::-1]
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    perturbed = rounded.copy()
                    # Pick a random subset of the worst-fitting entries
                    num_perturb = random.randint(1, 3)
                    indices = random.sample(list(flat_indices[:20]), min(num_perturb, 20))
                    for idx in indices:
                        perturbed[idx] += random.choice([-1, 1])
                    candidates.append(perturbed)

                for cand in candidates:
                    W_flat = cand.astype(np.int64).reshape(pn, target_rank)
                    U_np = np.array(U, dtype=np.int64)
                    V_np = np.array(V, dtype=np.int64)
                    W_np = W_flat
                    U3 = U_np.reshape(n, m, target_rank)
                    V3 = V_np.reshape(m, p, target_rank)
                    W3 = W_np.reshape(p, n, target_rank)
                    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    if np.all(T == 1):
                        return {"U": U, "V": V, "W": W_np.tolist()}

        return None

    # Main search: try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = 21
    lower_bound = 21

    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break

        # Multiple strategies for each rank, ordered by expected effectiveness
        strategies = [
            ("structured", lambda: try_rank_structured(target_rank)),
            ("kronecker", lambda: try_kronecker_block(target_rank)),
            ("fix_uv_±1", lambda: try_rank_fix_uv(target_rank, 80, (-1, 1))),
            ("polished", lambda: try_polished_search(target_rank, 25)),
            ("fix_uv_±2", lambda: try_rank_fix_uv(target_rank, 40, (-2, 2))),
            ("fix_uv_01", lambda: try_rank_fix_uv(target_rank, 30, (0, 1))),
            ("gf2_sparse", lambda: try_gf2_sparse(target_rank, 40)),
        ]

        # Shuffle strategies for variety
        random.shuffle(strategies)

        found_at_rank = False
        for name, strategy in strategies:
            if time.time() > deadline - 5:
                break

            result = strategy()
            if result is not None:
                best = result
                best_rank = target_rank
                found_at_rank = True
                break

        if not found_at_rank:
            # If we didn't find anything at this rank, continue to next
            continue

    return best
# EVOLVE-BLOCK-END
