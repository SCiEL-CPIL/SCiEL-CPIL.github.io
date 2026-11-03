"""Matrix multiplication 2x3x10 rank-minimal exact tensor decomposition search."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
DEADLINE = 120.0  # seconds, with safety margin below

def _elapsed():
    return time.time() - START_TIME

def _time_left(margin=10.0):
    return DEADLINE - _elapsed() - margin

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback (always valid, rank = n*m*p)
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Lower bound on rank
    lb = max(n*m, m*p, n*p)  # = max(6, 30, 20) = 30

    # We search for exact rational decompositions using the elimination method.
    # The multiplication tensor T[i,j,k] = 1 for all (i,j,k).
    # We need: for all (i,k): sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    #
    # Elimination: fix rank-1 term r=0 to isolate A[0,0].
    # Set U[0*m+0, 0] = 1, V[0*p+k, 0] = 1 for all k, W[k*n+0, 0] = 1 for all k,
    # and U[i*m+j, 0] = 0 for (i,j) != (0,0).
    # Then term r=0 contributes: for output (i,k), sum_j U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0]
    #   = U[0,0]*V[0,k]*W[k*n+0] (only j=0,i=0 nonzero) = 1*1*1 = 1 for all k.
    # So C[0,k] gets exactly 1 from term r=0, and we need sum over r>=1 to be 0 for (i=0,k).
    # For i != 0, term r=0 contributes 0 (since U[i*m+j,0]=0 for i!=0).
    # So we need sum_{r>=1} ... = 1 for i!=0, and = 0 for i=0.

    # This is a bilinear system. Let's set it up for R total terms.
    # Variables: U[i*m+j,r], V[j*p+k,r], W[k*n+i,r] for r in 1..R-1.
    # Constraints:
    #   For (i=0, k): sum_j sum_{r=1}^{R-1} U[j, r]*V[j*p+k, r]*W[k, r] == 0   (W[k*n+0,r])
    #   For (i>=1, k): sum_j sum_{r=1}^{R-1} U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1

    # This is still bilinear. To make it tractable, we can fix one factor and solve
    # linearly for the others. But that's a large search.

    # Given the time budget, let's do a randomized search with exact rational arithmetic
    # for small ranks, and accept the schoolbook if we can't beat it quickly.

    # Actually, for 2x3x10, known results suggest rank might be around 30-40.
    # Let's try a structured approach: use the fact that n=2 is small.

    # For n=2, the output has 2 rows. We can think of this as:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    #
    # A known technique: for 2 x m x p, the rank is at most m*p + 2*(m-1)*(p-1) or similar.
    # But let's just do a bounded numerical search.

    # Try ranks from lb to best_rank-1, using a simple randomized bilinear solver.
    # For each R, fix W randomly (rational), then the system becomes linear in U,V?
    # No, it's still bilinear in U and V.

    # Alternative: fix U and V, solve for W linearly. W has p*n*R = 10*2*R = 20R unknowns.
    # The constraints are n*p = 20 equations (one per output entry).
    # If we fix U and V, W is determined linearly (if consistent).
    # So: random U, V -> solve linear system for W -> check if W gives exact 1.

    # But we also need W to be rational and the whole thing exact.

    # Let me implement: for a given R, randomly generate U, V with rational entries,
    # set up the linear system M * W_flat = b where M is 20 x (20R), and solve.
    # If a rational solution exists, we have a valid decomposition.

    # M[i,k, (j, r)] = U[i*m+j, r] * V[j*p+k, r]
    # b[i,k] = 1  (for all i,k)
    # Unknowns: W[k*n+i, r] for r in 0..R-1.

    # This is an underdetermined system (20 equations, 20R unknowns for R>=1).
    # For R=1: 20 equations, 20 unknowns -> square system.
    # For R>1: underdetermined -> many solutions.

    # The catch: we need W to be rational (or half-integer), and the solution must exist.
    # For R=1, we need det(M) != 0 and the solution to be rational.

    # Let me try R from lb upward, with randomized U,V, solving for W.

    # For efficiency, use numpy for the linear solve, then check rationality.

    # Actually, the requirement is "integer/half-integer entries". So entries must be
    # in {k/2 : k in Z}. Let me search for such solutions.

    # Strategy: for each R, try many random rational U,V (with small denominators),
    # solve M W = b, check if W entries are half-integers (within tolerance), and
    # verify the full decomposition exactly.

    # Given time constraints, let's be pragmatic.

    max_rank_to_try = best_rank - 1
    # We'll try a limited number of ranks and random seeds.

    rng = np.random.default_rng(42)

    # Precompute the structure
    # M has shape (n*p, m*R) if we group W by (i,k) and unknowns by (j,r)... 
    # Wait, let me re-index.
    # Constraint for output (i,k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Unknowns are W[k*n+i, r] for r in 0..R-1. There are p*n*R = 10*2*R = 20R unknowns.
    # Equations: n*p = 20.
    # For equation (i,k), the coefficient of unknown W[k*n+i', r] is:
    #   nonzero only when i' == i (since W is indexed by k*n+i, and the equation is for (i,k))
    #   coefficient = U[i*m+j, r] * V[j*p+k, r] summed over j
    # So for equation (i,k), only unknowns W[k*n+i, r] for r in 0..R-1 appear, with
    #   coeff_r = sum_j U[i*m+j, r] * V[j*p+k, r]
    # So the system is block-diagonal! Each (i,k) equation only involves W[k*n+i, :].
    # That means: for each (i,k), we need sum_r [sum_j U[i*m+j,r]*V[j*p+k,r]] * W[k*n+i,r] = 1.
    # This is R unknowns (W[k*n+i,0..R-1]) and 1 equation per (i,k).
    # So for each (i,k), we have 1 equation in R unknowns. Always solvable for R>=1!
    # We just need W[k*n+i, r] to be half-integers.

    # So the problem reduces to: find U, V (rational, half-integer) such that for each
    # (i,k), the vector c_{i,k} = [sum_j U[i*m+j,r]*V[j*p+k,r] for r in 0..R-1] is
    # nonzero (so we can find a half-integer W satisfying the dot product = 1).
    # Actually we need: there exist half-integers W_{i,k,0..R-1} with c_{i,k} · W_{i,k} = 1.
    # This requires that gcd of the entries of c_{i,k} (when written as rationals) divides 1,
    # i.e., the entries of c_{i,k} are not all multiples of 2 (in half-integer terms).

    # Simpler: if any entry of c_{i,k} is 1 (or -1), set W to 1 there and 0 elsewhere.
    # So we need: for each (i,k), at least one r has sum_j U[i*m+j,r]*V[j*p+k,r] = ±1.

    # Even simpler: if we can make c_{i,k} have a 1 in it, we're done.

    # Let's try: for R=1, we need for each (i,k): sum_j U[i*m+j,0]*V[j*p+k,0] = 1 (up to sign,
    # and W=1). So we need a matrix product structure where U (n*m x 1) and V (m*p x 1)
    # give U_col[i*m+j] * V_col[j*p+k] summed over j = 1 for all (i,k).
    # That's: for all i,k, sum_j u_{i,j} v_{j,k} = 1. This means the matrix (u_i as row, v_k as col)
    # product is all-ones. For n=2, m=3, p=10: u is 2x3, v is 3x10, and u@v = ones(2,10).
    # This is possible! e.g., u = [[1,0,0],[0,1,0]], v = [[1,1,...],[1,1,...],[0,0,...]] but then
    # u@v row 0 = [1,1,...], row 1 = [1,1,...]. Yes! 
    # So R=1 works?? But rank 1 for matrix multiplication is impossible for nontrivial cases.
    # Wait, I think I made an error in the indexing.

    # Let me re-examine. The decomposition is:
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]  summed over j AND r.
    # 
    # For R=1 (single term r=0):
    # C[i,k] = sum_j U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # Let a_{i,j} = U[i*m+j,0], b_{j,k} = V[j*p+k,0], w_{i,k} = W[k*n+i,0].
    # Then C[i,k] = w_{i,k} * sum_j a_{i,j} b_{j,k} = w_{i,k} * (A*B)[i,k] where A is 2x3, B is 3x10.
    # We need C[i,k] = (A*B)[i,k] for ALL A,B. So w_{i,k} * (A*B)[i,k] = (A*B)[i,k] for all A,B.
    # This requires w_{i,k} = 1 for all (i,k) (since (A*B)[i,k] can be any value).
    # So with w_{i,k}=1, we get C[i,k] = (A*B)[i,k]. 
    # But wait, that means a single rank-1 term computes the full product? That's impossible
    # unless a_{i,j} b_{j,k} captures the full bilinear form, which it doesn't for m>1.
    # 
    # The issue: a_{i,j} b_{j,k} is a rank-1 matrix (as a function of (i,k) for fixed A,B entries),
    # but (A*B)[i,k] = sum_j A[i,j] B[j,k] is a sum of m rank-1 terms. So a single term a_{i,j}b_{j,k}
    # cannot equal sum_j A[i,j]B[j,k] for all A,B.
    # 
    # I see my error: in the decomposition, U, V, W are FIXED (they define the algorithm),
    # and A, B are the INPUTS. The entries of U, V, W are constants, not functions of A, B.
    # The formula is: C[i,k] = sum_{j,r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]?
    # 
    # No wait. Let me re-read the interface. The factor matrices U, V, W are the algorithm.
    # The standard Strassen-like decomposition: 
    #   M_r = U_r * A * V_r  (some bilinear form)
    #   C = sum_r W_r * M_r
    # 
    # But the interface here is different. Let me re-read:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    # 
    # The multiplication tensor T is a 3-tensor T[i,j,k] such that (A*B)[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all (i,j,k).
    # 
    # A rank-R decomposition of T means: T[i,j,k] = sum_{r=1}^R T1[i,j,r] * T2[j,k,r] * T3[i,k,r]
    # where T1 is (n*m x R), T2 is (m*p x R), T3 is (p*n x R).
    # 
    # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k).
    # 
    # THIS is the constraint! Not the one I had before. The sum is over r only, and it must equal
    # 1 for every (i,j,k) triple. There are n*m*p = 60 constraints, and 3*R variables (well,
    # n*m*R + m*p*R + p*n*R = R*(6+30+20) = 56R variables).
    # 
    # So for R=1: 60 equations, 56 unknowns -> overdetermined. Unlikely to have a solution.
    # For R>=2: 60 equations, 112 unknowns -> underdetermined.
    # 
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    # 
    # This is a system of 60 bilinear (in U,V,W) equations.
    # 
    # Elimination method: fix U[0,0,0] = 1 (i=0,j=0,r=0), and set U[i*m+j,0]=0 for (i,j)!=(0,0).
    # Then for (i=0,j=0,k): sum_r U[0,0,r] V[k,r] W[k,r] = V[0,k,0]*W[0,k,0] + sum_{r>=1} 0 = 1
    #   so V[0,k,0]*W[0,k,0] = 1 for all k. Set V[0,k,0]=1, W[0,k,0]=1.
    # For (i=0,j=0,k) is done.
    # For (i=0, j!=0, k): sum_r U[0*m+j, r] V[j*p+k, r] W[k*n+0, r] = 1
    #   U[0*m+j,0]=0 for j!=0, so sum_{r>=1} U[j, r] V[j*p+k, r] W[k, r] = 1  (W[k*n+0,r])
    # For (i!=0, j=0, k): sum_r U[i*m, r] V[k, r] W[k*n+i, r] = 1
    # For (i!=0, j!=0, k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # 
    # This is still complex. Let me just do a numerical search.

    # Given the complexity and time budget, let me implement a randomized search for small R
    # using the exact constraint, and fall back to schoolbook.

    # For each R, we have 60 equations: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    # 
    # Approach: fix W to random rational values, then the system becomes linear in (U*V) products.
    # Actually, fix U and W, then linear in V. Or fix two factors, solve for the third.
    # 
    # Fix U and W: then for each (i,j,k), V[j*p+k, r] appears with coefficient U[i*m+j,r]*W[k*n+i,r].
    # For fixed (j,k), V[j*p+k, r] for r=0..R-1 appear in equations for all (i) with:
    #   sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1  for all i.
    # That's n equations in R unknowns (V[j*p+k, 0..R-1]) for each (j,k).
    # There are m*p = 30 such groups, each with n=2 equations and R unknowns.
    # For R>=2, each group is underdetermined -> solvable.
    # 
    # So: fix U and W (random rational), solve for V. Check if V is rational/half-integer.
    # 
    # Let me implement this.

    # Variables: U has n*m*R = 6R entries, W has p*n*R = 20R entries, V has m*p*R = 30R entries.
    # Fix U and W randomly, solve linear system for V.
    # 
    # For each (j,k): unknowns v_r = V[j*p+k, r], r=0..R-1.
    # Equations: for i=0..n-1: sum_r v_r * U[i*m+j, r] * W[k*n+i, r] = 1.
    # This is a 2 x R linear system. Solve it.
    # 
    # For R=2: 2x2 system per (j,k). 30 systems. Solve each.
    # 
    # Let me code this up.

    def try_rank(R, num_trials=200):
        """Try to find a valid rank-R decomposition by fixing U,W and solving for V."""
        nonlocal best, best_rank
        if R >= best_rank:
            return False
        if _time_left(5.0) < 0:
            return False
        for trial in range(num_trials):
            if _time_left(5.0) < 0:
                break
            # Random U, W with small integer entries
            U_rand = rng.integers(-2, 3, size=(n*m, R))
            W_rand = rng.integers(-2, 3, size=(p*n, R))
            # Solve for V
            V_sol = np.zeros((m*p, R), dtype=float)
            valid = True
            for j in range(m):
                for k in range(p):
                    # 2 x R system: A_sys @ v = b
                    A_sys = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A_sys[i, r] = U_rand[i*m+j, r] * W_rand[k*n+i, r]
                    b = np.ones(n)
                    # Solve least squares / exact
                    if R == 1:
                        if np.any(np.abs(A_sys[:,0]) < 1e-12):
                            valid = False
                            break
                        v0 = b[0] / A_sys[0,0]
                        if abs(A_sys[1,0]*v0 - b[1]) > 1e-9:
                            valid = False
                            break
                        V_sol[j*p+k, 0] = v0
                    else:
                        # Overdetermined or square; use lstsq and check residual
                        v, res, rank_, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                        if np.linalg.norm(A_sys @ v - b) > 1e-9:
                            valid = False
                            break
                        V_sol[j*p+k, :] = v
                if not valid:
                    break
            if not valid:
                continue
            # Check if V_sol entries are close to half-integers
            V_half = np.round(V_sol * 2) / 2
            if np.max(np.abs(V_sol - V_half)) > 1e-6:
                continue
            # Now verify the full decomposition exactly with U_rand, V_half, W_rand
            U_int = U_rand.astype(int)
            W_int = W_rand.astype(int)
            V_int = V_half  # half-integers
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                # Found a valid decomposition!
                U_out = [[int(x) for x in U_int[row]] for row in range(n*m)]
                V_out = [[float(x) for x in V_int[row]] for row in range(m*p)]
                W_out = [[int(x) for x in W_int[row]] for row in range(p*n)]
                # Convert half-integers to int or float appropriately
                for row in range(m*p):
                    for c in range(R):
                        if V_out[row][c] == int(V_out[row][c]):
                            V_out[row][c] = int(V_out[row][c])
                best = {"U": U_out, "V": V_out, "W": W_out}
                best_rank = R
                return True
        return False

    # Try ranks from lower bound upward
    for R in range(lb, best_rank):
        if _time_left(10.0) < 0:
            break
        if try_rank(R, num_trials=100):
            # Found a valid decomposition at rank R; try to go lower
            # (but we're going upward, so this is the first found)
            # Continue trying lower ranks? We started at lb, so R is the first found.
            break

    return best
