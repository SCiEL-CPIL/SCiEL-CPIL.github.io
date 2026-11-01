import itertools
import time
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, milp, Bounds
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def _generate_words(q, n):
    return [list(word) for word in itertools.product(range(q), repeat=n)]

def _hamming_distance(w1, w2):
    return sum(1 for a, b in zip(w1, w2) if a != b)

def _seed_solution(q, n, radius):
    # Direct sum of ternary tetracode
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: simple greedy
        words = _generate_words(q, n)
        covered = [False] * len(words)
        code = []
        # Map word to index
        word_to_idx = {tuple(w): i for i, w in enumerate(words)}
        
        # Precompute coverage
        cover_sets = []
        for c in words:
            cov = []
            for u in words:
                if _hamming_distance(c, u) <= radius:
                    cov.append(word_to_idx[tuple(u)])
            cover_sets.append(cov)
            
        # Greedy
        while any(not c for c in covered):
            best_idx = -1
            best_gain = -1
            for i, c in enumerate(words):
                gain = sum(1 for j in cover_sets[i] if not covered[j])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0: break # Should not happen if radius >= 0
            code.append(words[best_idx])
            for j in cover_sets[best_idx]:
                covered[j] = True
        return code

    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    if q == 3 and n == 6 and radius == 1:
        try:
            words = _generate_words(q, n)
            N = len(words)
            word_to_idx = {tuple(w): i for i, w in enumerate(words)}
            
            # Build sparse incidence matrix A (N x N)
            # A[u, c] = 1 if u is covered by c (dist(u,c) <= 1)
            rows = []
            cols = []
            data = []
            
            for c_idx, c in enumerate(words):
                # c covers itself and words at distance 1
                # Self
                rows.append(c_idx)
                cols.append(c_idx)
                data.append(1)
                
                # Distance 1
                for pos in range(n):
                    original_val = c[pos]
                    for new_val in range(q):
                        if new_val == original_val: continue
                        c_new = c.copy()
                        c_new[pos] = new_val
                        u_idx = word_to_idx[tuple(c_new)]
                        rows.append(u_idx)
                        cols.append(c_idx)
                        data.append(1)
                        
            # Note: A is N_rows (constraints) x N_cols (variables)
            # Constraint: sum_{c covers u} x_c >= 1 for each u
            # So A[u, c] = 1 if u is covered by c.
            # In my loop above, I added (u_idx, c_idx).
            # So A is N x N.
            A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
            
            # Objective: minimize sum x_c
            c_obj = np.ones(N)
            
            # Constraints: A @ x >= 1
            # LinearConstraint(A, lb=np.ones(N), ub=np.inf)
            cons = LinearConstraint(A, lb=np.ones(N), ub=np.inf)
            
            # Bounds: x_c in [0, 1]
            bounds = Bounds(lb=0, ub=1)
            
            # Variables must be integer
            integrality = np.ones(N)
            
            def solve_milp():
                res = milp(c=c_obj, constraints=cons, integrality=integrality, bounds=bounds)
                return res
                
            # Run in thread to allow timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_milp)
                try:
                    res = future.result(timeout=deadline - time.time())
                    if res.status == 0:
                        # res.x contains the solution
                        x = res.x
                        code = [words[i] for i in range(N) if x[i] > 0.5]
                        # Verify
                        # Check if all words are covered
                        # This is a sanity check
                        covered = [False] * N
                        for c_word in code:
                            c_idx = word_to_idx[tuple(c_word)]
                            # Update covered
                            # This is O(N * |code| * N) which is slow
                            # Better to use the matrix
                            pass 
                        
                        # Just trust the solver for now, but we should verify.
                        # Verification:
                        # For each u, check if any c in code is within dist 1.
                        # Precompute distances? No, just check.
                        is_valid = True
                        for u in words:
                            u_t = tuple(u)
                            found = False
                            for c_word in code:
                                if _hamming_distance(u, c_word) <= radius:
                                    found = True
                                    break
                            if not found:
                                is_valid = False
                                break
                        if is_valid:
                            return code
                        else:
                            # Solver returned invalid? Fallback
                            return _seed_solution(q, n, radius)
                    else:
                        return _seed_solution(q, n, radius)
                except FuturesTimeoutError:
                    return _seed_solution(q, n, radius)
        except Exception:
            return _seed_solution(q, n, radius)
    else:
        return _seed_solution(q, n, radius)
