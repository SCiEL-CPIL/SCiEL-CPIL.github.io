"""Matrix multiplication 2x3x5 rank minimization."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n  # 6, 15, 10

    deadline = 140  # seconds, safety margin before 150s kill

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is valid."""
        rank = len(U[0])
        for i0 in range(n):
            for j0 in range(m):
                for j1 in range(m):
                    for k1 in range(p):
                        for i in range(n):
                            for k in range(p):
                                expected = (1 if i0 == i and j0 == j1 and k1 == k else 0)
                                actual = sum(U[i0 * m + j0][c] * V[j1 * p + k1][c] * W[k * n + i][c]
                                            for c in range(rank))
                                if abs(actual - expected) > tol:
                                    return False
        return True

    def verify_np(U, V, W, n, m, p, tol=1e-9):
        """Vectorized verification using numpy."""
        rank = U.shape[1]
        # Build constraint tensor: for each (i0,j0,j1,k1,i,k), check sum over c
        # U[i0*m+j0, c] * V[j1*p+k1, c] * W[k*n+i, c]
        for i0 in range(n):
            for j0 in range(m):
                for j1 in range(m):
                    for k1 in range(p):
                        # U_slice: (r,) for fixed (i0,j0)
                        u_vec = U[i0 * m + j0, :]  # (r,)
                        # V_slice: (r,) for fixed (j1,k1)
                        v_vec = V[j1 * p + k1, :]  # (r,)
                        # W_slice: (p*n, r) -> for each (k,i), W[k*n+i, :]
                        # product: u_vec[c] * v_vec[c] * W[k*n+i, c]
                        uv = u_vec * v_vec  # (r,)
                        # Now for each (k,i): sum_c uv[c] * W[k*n+i, c]
                        # W is (p*n, r), so W @ uv gives (p*n,) vector
                        w_result = W @ uv  # (p*n,)
                        for i in range(n):
                            for k in range(p):
                                expected = (1 if i0 == i and j0 == j1 and k1 == k else 0)
                                actual = w_result[k * n + i]
                                if abs(actual - expected) > tol:
                                    return False
        return True

    # Schoolbook solution (rank 30) as fallback
    rank_school = n * m * p  # 30
    U_school = np.zeros((N, rank_school), dtype=float)
    V_school = np.zeros((M, rank_school), dtype=float)
    W_school = np.zeros((P, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Try to find lower rank solutions
    # Strategy: for each target rank, try structured + random approaches
    # Key insight: the constraints are multilinear. Fix one matrix, solve for others.

    def try_solve_with_fixed_W(W_fixed, r, n, m, p, N, M, P, max_iter=100, tol=1e-10):
        """
        Given W_fixed (P x r), try to find U (N x r) and V (M x r) satisfying constraints.
        The constraints are bilinear in U and V (for fixed W).
        Use iterative approach: alternate solving for U (given V) and V (given U).
        """
        # Initialize
        np.random.seed(42)
        U_cur = np.random.randn(N, r) * 0.1
        V_cur = np.random.randn(M, r) * 0.1

        for iteration in range(max_iter):
            # Solve for U given V: constraints are linear in U
            # For each (i0, j0, j1, k1, i, k):
            # sum_c U[i0*m+j0, c] * V[j1*p+k1, c] * W[k*n+i, c] = delta
            # This is linear in U entries.
            # Build system: unknowns are U[i0*m+j0, c] for all i0,j0,c
            # Each constraint gives one equation.
            # Number of unknowns: N * r
            # Number of constraints: n * m * m * p * n * p = n^2 * m^2 * p^2 = 4*9*25 = 900
            # But many are trivially 0=0 when delta=0 and the equation is automatically satisfied.
            
            # Let's build the full linear system for U (given V and W)
            # Unknowns: U[i*m+j, c] for i in [0,n), j in [0,m), c in [0,r) -> N*r unknowns
            num_unknowns = N * r
            
            # Build A matrix and b vector
            # Each constraint (i0, j0, j1, k1, i, k) gives:
            # sum_c U[i0*m+j0, c] * (V[j1*p+k1, c] * W[k*n+i, c]) = delta(i0,i)*delta(j0,j1)*delta(k1,k)
            # Coefficient of U[i0*m+j0, c] in this equation: V[j1*p+k1, c] * W[k*n+i, c]
            # But U[i0*m+j0, c] only appears in constraints where the first two indices match.
            
            # Actually, U[i0*m+j0, c] appears in ALL constraints (regardless of j1, k1, i, k)
            # with coefficient V[j1*p+k1, c] * W[k*n+i, c]
            
            # This is a large system. Let's use a smarter approach.
            # Group constraints by (i0, j0): for fixed (i0, j0), the unknowns are U[i0*m+j0, :] (r values)
            # Constraints involving U[i0*m+j0, c]: all (j1, k1, i, k) combinations
            # For fixed (i0, j0): m*p*n*p = m*n*p^2 constraints, each involving all r unknowns U[i0*m+j0, :]
            
            # Actually, let me just build the full system and solve with lstsq.
            # Number of constraints: n*m*m*p*n*p = 900
            # Number of unknowns: N*r = 6*r
            
            A_mat = np.zeros((n * m * m * p * n * p, num_unknowns))
            b_vec = np.zeros(n * m * m * p * n * p)
            row = 0
            for i0 in range(n):
                for j0 in range(m):
                    for j1 in range(m):
                        for k1 in range(p):
                            for i in range(n):
                                for k in range(p):
                                    expected = (1 if i0 == i and j0 == j1 and k1 == k else 0)
                                    b_vec[row] = expected
                                    for c in range(r):
                                        A_mat[row, i0 * m + j0]  # WRONG - need to index by (i0*m+j0)*r + c
                                    row += 1
            # This approach is getting complicated. Let me use a different strategy.
            return None  # Skip this approach for now

    # Better approach: direct construction for specific ranks
    # Let me try rank 6 with a structured approach
    
    # For rank r, the key observation: we need the "routing" to work.
    # Let me try a parameterized family.
    
    # Approach: try all ranks from 5 to 29, use random search with numpy for speed
    np.random.seed(12345)
    
    for target_rank in range(5, 30):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        num_trials = 50 if target_rank <= 10 else 20
        found = False
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U_r = np.random.randint(-2, 3, size=(N, target_rank)).astype(float)
            V_r = np.random.randint(-2, 3, size=(M, target_rank)).astype(float)
            W_r = np.random.randint(-2, 3, size=(P, target_rank)).astype(float)
            
            if verify_np(U_r, V_r, W_r, n, m, p, tol=1e-9):
                best_rank = target_rank
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                found = True
                break
        
        if found:
            break  # Found the minimum rank (we're going in order)
    
    # Also try a structured approach for specific ranks
    if best_rank == 30:
        # Try structured: rank = n*p = 10
        # For each (i,k) output, dedicate columns to compute the dot product
        # This won't work directly (as shown in analysis), but let's try variants
        
        # Try: for each j, have a "j-channel" that routes A[·,j] and B[j,·]
        # Rank = m * n * p would be schoolbook. 
        # Try rank = m * max(n,p) = 3*5 = 15
        target_rank = 15
        for trial in range(100):
            if time.time() > deadline:
                break
            # Structured: U depends on (i,j), V depends on (j,k), W depends on (k,i)
            # Try: U[i*m+j, c] where c encodes (j, something)
            # For c = j*p + k: U[i*m+j, j*p+k] = 1, V[j*p+k, j*p+k] = 1, W[k*n+i, j*p+k] = delta(i,i)*delta(k,k)
            # But W[k*n+i, j*p+k] can only depend on (k,i), not on j. So this doesn't work.
            
            # Alternative: c = i*p + k (n*p = 10 columns)
            # U[i*m+j, i'*p+k'] = delta(i,i')  (independent of j!)
            # V[j*p+k, i'*p+k'] = delta(k,k')  (independent of j!)  
            # W[k*n+i, i'*p+k'] = delta(i,i')*delta(k,k')
            # Check: sum_c U[i0*m+j0,c]*V[j1*p+k1,c]*W[k*n+i,c]
            # = sum_{i',k'} delta(i0,i')*delta(k1,k')*delta(i,i')*delta(k,k')
            # = delta(i0,i)*delta(k1,k)  -- missing delta(j0,j1)!
            # So this gives wrong answer (no j-matching).
            
            # We NEED the j-matching. The only way is to have U depend on j and V depend on j
            # in a correlated way.
            
            # Try: c = j * n * p + i * p + k  (m*n*p = 30, schoolbook)
            # Or: c = j * p + k with U[i*m+j, j'*p+k'] = delta(j,j')*f(i) 
            #     V[j*p+k, j'*p+k'] = delta(j,j')*g(k)
            #     W[k*n+i, j'*p+k'] = h(i,k,j')
            # Check: sum_{j',k'} delta(j0,j')*f(i0) * delta(j1,j')*g(k1) * h(i,k,j')
            # = f(i0)*g(k1)*h(i,k,j0)  (since j'=j0=j1 required)
            # Need this = delta(i0,i)*delta(k1,k)
            # So f(i0)*g(k1)*h(i,k,j0) = delta(i0,i)*delta(k1,k)
            # This requires h(i,k,j0) to depend on j0 to cancel... but h doesn't have j0 as independent var.
            # If f(i)=1, g(k)=1, h(i,k,j)=delta(i,i)*delta(k,k) then it works!
            # But h(i,k,j) = delta(i,i)*delta(k,k) is independent of j, so:
            # U[i*m+j, j'*p+k'] = delta(j,j')  (independent of i!)
            # V[j*p+k, j'*p+k'] = delta(j,j')  (independent of k!)
            # W[k*n+i, j'*p+k'] = delta(i,i)*delta(k,k)  (independent of j'!)
            # But W[k*n+i, c] where c = j'*p+k' must be independent of j' for this to work.
            # W[k*n+i, j'*p+k'] = delta(i,i)*delta(k,k) for ALL j'. 
            # This means W[k*n+i, :] has the same value delta(i,i)*delta(k,k) for all entries where k'=k.
            # Wait, c = j'*p+k', so for fixed (k,i), W[k*n+i, j'*p+k'] for j'=0,1,2 all equal delta(i,i)*delta(k,k).
            # That's fine! W is (10, 15) and W[k*n+i, j*p+k] = delta(i,i)*delta(k,k) for all j.
            
            # Let's verify this rank-15 solution:
            # U[i*m+j, j'*p+k'] = delta(j, j')
            # V[j*p+k, j'*p+k'] = delta(j, j')
            # W[k*n+i, j'*p+k'] = delta(i, i)*delta(k, k)  [independent of j']
            
            # Coefficient of A[i0,j0]*B[j1,k1] in C[i,k]:
            # sum_{j',k'} U[i0*m+j0, j'*p+k'] * V[j1*p+k1, j'*p+k'] * W[k*n+i, j'*p+k']
            # = sum_{j',k'} delta(j0,j') * delta(j1,j') * delta(i,i)*delta(k,k)
            # = delta(j0,j1) * delta(i,i)*delta(k,k) * (number of k' values = p)
            # = p * delta(j0,j1) * delta(i,i)*delta(k,k)
            # = 5 * delta(j0,j1) * delta(i0,i)*delta(k1,k)  [since i0=i required for delta(i,i)]
            
            # This gives 5 instead of 1! So we need to scale: divide U by sqrt(5) or V by sqrt(5) or W by 5.
            # Use W[k*n+i, j'*p+k'] = delta(i,i)*delta(k,k)/5
            
            # But we need integer/half-integer entries! 1/5 is not a half-integer.
            # Alternative: scale U by 1/sqrt(5) -- not rational.
            # Or: use U[i*m+j, j'*p+k'] = delta(j,j'), V[j*p+k, j'*p+k'] = delta(j,j'), W[k*n+i, j'*p+k'] = delta(i,i)*delta(k,k)/5
            # 1/5 is not a half-integer. Problem.
            
            # What if we use a different split? 
            # U[i*m+j, j'*p+k'] = delta(j,j')/sqrt(p) -- not rational
            # 
            # Alternative: make the "overlap" be 1 instead of p.
            # If U[i*m+j, c] and V[j*p+k, c] are 1 for exactly one c each (when j matches),
            # and W[k*n+i, c] = 1, then we get delta(j0,j1)*delta(i0,i)*delta(k1,k).
            # But that requires c to encode (j, i, k) uniquely -> rank = m*n*p = 30.
            
            # What if we use c = j*p + k and have U depend on i?
            # U[i*m+j, j'*p+k'] = delta(j,j') * a(i, k')
            # V[j*p+k, j'*p+k'] = delta(j,j') * b(k, k')  [wait, V[j*p+k, c] where c=j'*p+k']
            # Hmm, this is getting complicated.
            
            pass  # Skip structured for now, rely on random search

    # Final: try a more targeted search for rank 6-15 with better initialization
    if best_rank == 30 and time.time() < deadline - 10:
        # Try rank 6 with structured initialization
        # Idea: use the fact that n=2, so we have 2 "rows" of A
        # For each column c, U has a 2x3 pattern (2 rows of A, 3 cols of A)
        # V has a 3x5 pattern (3 rows of B, 5 cols of B)
        # W has a 5x2 pattern (5 cols of C, 2 rows of C)
        
        for target_rank in [6, 7, 8, 9, 10, 12, 15]:
            if time.time() > deadline:
                break
            for trial in range(200):
                if time.time() > deadline:
                    break
                # Use a mix of structured and random
                U_r = np.zeros((N, target_rank), dtype=float)
                V_r = np.zeros((M, target_rank), dtype=float)
                W_r = np.zeros((P, target_rank), dtype=float)
                
                # Initialize with random small integers
                for c in range(target_rank):
                    U_r[:, c] = np.random.randint(-2, 3, size=N)
                    V_r[:, c] = np.random.randint(-2, 3, size=M)
                    W_r[:, c] = np.random.randint(-2, 3, size=P)
                
                if verify_np(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best_rank = target_rank
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                    break
            if best_rank < 30:
                break

    # Convert to lists of integers (or half-integers)
    # The task says "integer/half-integer entries"
    # Check if entries are integers or half-integers
    def to_half_int_list(mat):
        """Convert numpy matrix to list of lists, ensuring entries are int or half-int."""
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-8:
                    row_list.append(rounded if rounded != int(rounded) else int(rounded))
                else:
                    # Shouldn't happen for valid solutions
                    row_list.append(rounded)
            result.append(row_list)
        return result

    U_out = to_half_int_list(best_U)
    V_out = to_half_int_list(best_V)
    W_out = to_half_int_list(best_W)

    result = {"U": U_out, "V": V_out, "W": W_out}
    return result
