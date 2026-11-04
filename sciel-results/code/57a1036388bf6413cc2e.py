"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # Safety margin before 150s kill

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

    def solve_for_W(U, V, rank):
        """
        Given U and V, solve for W such that sum_r U_ij_r V_jk_r W_ki_r = 1.
        Returns W as a list of lists or None if no integer solution found.
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
        tol = 1e-8
        if not np.all(np.abs(w_real - np.round(w_real)) < tol):
            return None

        w_int = np.round(w_real).astype(np.int64)
        W_flat = w_int.reshape(pn, rank)

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
        """
        if time.time() > deadline - 5:
            return None

        # Strategy: Use the known rank-5 decomposition for 2x3x3 as a template
        # and extend to 2x3x7 by combining with 3x3x7 structure

        # Known rank-5 for 2x3x3 (Strassen-like for rectangular):
        # We'll try patterns where U has structure based on i (2 values)
        # and V/W have structure based on j,k

        # Pattern 1: Block-diagonal in the p dimension
        # Split p=7 into blocks and use structured combinations
        attempts = 30
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                return None

            # Generate U with structure: U[i*m+j, r] depends on i in a patterned way
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]

            # Assign random "types" to each rank component
            types = [random.randint(0, 3) for _ in range(target_rank)]

            for r in range(target_rank):
                t = types[r]
                # U pattern: for each (i,j), assign based on type
                for i in range(n):
                    for j in range(m):
                        # Use structured assignment
                        if t == 0:
                            U[i * m + j][r] = 1
                        elif t == 1:
                            U[i * m + j][r] = 1 if i == 0 else -1
                        elif t == 2:
                            U[i * m + j][r] = 1 if j == 0 else 0
                        else:
                            U[i * m + j][r] = random.choice([-1, 1])

                # V pattern
                for j in range(m):
                    for k in range(p):
                        if t == 0:
                            V[j * p + k][r] = 1
                        elif t == 1:
                            V[j * p + k][r] = 1 if k < 3 else 0
                        elif t == 2:
                            V[j * p + k][r] = 1 if j == 0 else -1
                        else:
                            V[j * p + k][r] = random.choice([-1, 0, 1])

            W = solve_for_W(U, V, target_rank)
            if W is not None:
                return {"U": U, "V": V, "W": W}

        return None

    def try_gf2_peel(target_rank, max_iter=100):
        """
        GF(2) peeling: find sparse rank-1 components over GF(2)
        and lift to integers.
        """
        if time.time() > deadline - 5:
            return None

        # Work over GF(2) first to find sparse patterns
        # T[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor)
        # We want to find U, V, W with entries in {0,1} over GF(2)
        # such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 mod 2

        # This is a different problem (XOR), but sparse solutions over GF(2)
        # can sometimes be lifted to integer solutions

        # Instead, use a direct sparse search
        for attempt in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Generate sparse U, V, W with mostly zeros
            density = random.choice([0.1, 0.2, 0.3])
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]

            for r in range(target_rank):
                # Each rank component is sparse
                u_sparsity = random.random() < density
                v_sparsity = random.random() < density
                w_sparsity = random.random() < density

                if u_sparsity:
                    for idx in range(random.randint(1, nm)):
                        U[idx][r] = random.choice([-1, 1])
                if v_sparsity:
                    for idx in range(random.randint(1, mp)):
                        V[idx][r] = random.choice([-1, 1])
                if w_sparsity:
                    for idx in range(random.randint(1, pn)):
                        W[idx][r] = random.choice([-1, 1])

            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}

        return None

    def try_kronecker_block(target_rank):
        """
        Exploit rectangular asymmetry: decompose 2x3x7 as
        (2x3x3) composed with (3x3x7) structure.
        Use block Kronecker patterns.
        """
        if time.time() > deadline - 5:
            return None

        # The 2x3x3 multiplication has rank 5 (known result)
        # For 2x3x7, we can think of it as:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Group k into blocks: k = 3*q + r, q in {0,1}, r in {0,1,2}
        # But 7 = 3*2 + 1, so not perfectly divisible

        # Alternative: Use the fact that n=2 is very small
        # Try patterns where U depends only weakly on i
        for attempt in range(50):
            if time.time() > deadline - 5:
                return None

            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]

            # Pattern: U[i*m+j, r] = a_i * b_j * c_r where a_i in {1,-1}, b_j in {1,0,-1}
            a_vals = [random.choice([1, -1]) for _ in range(target_rank)]
            b_vals = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m)]

            for i in range(n):
                for j in range(m):
                    for r in range(target_rank):
                        U[i * m + j][r] = a_vals[r] * b_vals[j][r]

            # V: V[j*p+k, r] = d_j * e_k * f_r
            d_vals = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m)]
            e_vals = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p)]

            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        V[j * p + k][r] = d_vals[j][r] * e_vals[k][r]

            W = solve_for_W(U, V, target_rank)
            if W is not None:
                return {"U": U, "V": V, "W": W}

        return None

    # Main search: try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = 21
    # But actual lower bound for 2x3x7 might be higher

    lower_bound = 21

    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 5:
            break

        # Multiple strategies for each rank
        strategies = [
            ("fix_uv_±1", lambda: try_rank_fix_uv(target_rank, 100, (-1, 1))),
            ("fix_uv_01", lambda: try_rank_fix_uv(target_rank, 50, (0, 1))),
            ("fix_uv_±2", lambda: try_rank_fix_uv(target_rank, 50, (-2, 2))),
            ("structured", lambda: try_rank_structured(target_rank)),
            ("kronecker", lambda: try_kronecker_block(target_rank)),
            ("gf2_sparse", lambda: try_gf2_peel(target_rank, 50)),
        ]

        # Shuffle strategies for variety
        random.shuffle(strategies)

        for name, strategy in strategies:
            if time.time() > deadline - 5:
                break

            result = strategy()
            if result is not None:
                best = result
                best_rank = target_rank
                break

        if best_rank == target_rank:
            # Found a decomposition at this rank, continue to try lower
            continue

    return best
# EVOLVE-BLOCK-END
