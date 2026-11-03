"""Matrix multiplication 4x4 * 4x5 tensor decomposition.
Searches for low-rank exact decomposition using structured search."""

import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best = make_schoolbook()
    best_rank = n * m * p

    # Try structured ±1 search for small ranks
    # Key insight: for ±1 factors, each term in the sum is ±1.
    # We need sum of r terms (each ±1) = 1, so r must be odd.
    # For even r, we need mixed parity.
    
    # Strategy 1: Pure ±1 search for small odd ranks
    # Strategy 2: Allow {0, ±1, ±1/2} entries
    
    def random_search(rank, seed, max_iters=200, allow_zero=False):
        rng = random.Random(seed)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            vals = [-1, 0, 1] if allow_zero else [-1, 1]
            if rank % 2 == 0 and not allow_zero:
                vals = [-1, 0, 1]  # need 0 to get odd sum from even count
            else:
                vals = [-1, 1]
            
            U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(rank, U, V, W):
                return {"U": U, "V": V, "W": W}
        return None
    
    # Also try a smarter structured approach:
    # Set U[i*m+j, c] to depend only on (i,j) pattern, similarly for V, W
    # Use the fact that the all-ones tensor can be decomposed using Kronecker products
    
    # For 4x4x4, the known optimal is rank 19 (Strassen-based for 2x2 extended).
    # For 4x4x5, let's just do exhaustive ±1 search for ranks 17-24.
    
    # Exhaustive search is infeasible for large ranks, so use random + local search
    
    def local_search(rank, U, V, W, seed, max_steps=50):
        """Try flipping individual entries to fix mismatches."""
        rng = random.Random(seed)
        current_U = [row[:] for row in U]
        current_V = [row[:] for row in V]
        current_W = [row[:] for row in W]
        
        def get_errors(U_, V_, W_):
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U_[i*m+j][c] * V_[j*p+k][c] * W_[k*n+i][c]
                        if abs(s - 1) > 1e-9:
                            errors.append((i, j, k, s))
            return errors
        
        errors = get_errors(current_U, current_V, current_W)
        if not errors:
            return {"U": current_U, "V": current_V, "W": current_W}
        
        best_err_count = len(errors)
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not errors:
                break
            
            # Try flipping a random entry in a random factor
            choice = rng.choice(['U', 'V', 'W'])
            if choice == 'U':
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                if current_U[idx][c] == 0:
                    continue
                old = current_U[idx][c]
                current_U[idx][c] = -old
            elif choice == 'V':
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                if current_V[idx][c] == 0:
                    continue
                old = current_V[idx][c]
                current_V[idx][c] = -old
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                if current_W[idx][c] == 0:
                    continue
                old = current_W[idx][c]
                current_W[idx][c] = -old
            
            new_errors = get_errors(current_U, current_V, current_W)
            if len(new_errors) < best_err_count:
                best_err_count = len(new_errors)
                errors = new_errors
            else:
                # revert
                if choice == 'U':
                    current_U[idx][c] = old
                elif choice == 'V':
                    current_V[idx][c] = old
                else:
                    current_W[idx][c] = old
        
        errors = get_errors(current_U, current_V, current_W)
        if not errors:
            return {"U": current_U, "V": current_V, "W": current_W}
        return None
    
    # Search from lowest feasible rank upward
    for rank in range(17, min(n*m*p, 28)):  # Try up to rank 27
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        # For even rank, must allow zeros
        allow_zero = (rank % 2 == 0)
        
        # Multiple random restarts
        num_restarts = 10 if rank <= 20 else 5
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            result = random_search(rank, seed, max_iters=50, allow_zero=allow_zero)
            if result:
                found = result
                break
            else:
                # Try local search from a random init
                vals = [-1, 0, 1] if allow_zero else [-1, 1]
                rng = random.Random(seed + 9999)
                U0 = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
                V0 = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
                W0 = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
                result = local_search(rank, U0, V0, W0, seed + 7777, max_steps=100)
                if result:
                    found = result
                    break
        
        if found:
            best = found
            best_rank = rank
            break  # Found the minimum rank
    
    return best

# EVOLVE-BLOCK-END
