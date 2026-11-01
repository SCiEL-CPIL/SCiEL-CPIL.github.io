"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])

        if np.max(np.abs(R - target)) > tol:
            return False

        # Check integer/half-integer vectorized
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < tol):
                return False
        return True

    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def gf3_search(rank):
        """
        Search for a sparse support pattern over GF(3) that satisfies
        the matrix multiplication tensor identity modulo 3.
        Returns factor matrices over GF(3) or None.
        """
        # The tensor T[a,b,c] = 1 if (a,b,c) corresponds to (i,j,k) with
        # a=i*m+j, b=j*p+k, c=k*n+i
        # We need U,V,W with entries in {0,1,2} such that
        # sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] (mod 3)
        
        # For each rank component r, the support is a set of (a,b,c) triples
        # where U[a,r], V[b,r], W[c,r] are all nonzero.
        # The contribution to T[a,b,c] from component r is U[a,r]*V[b,r]*W[c,r].
        
        # Strategy: enumerate small support patterns and use backtracking
        # over GF(3) to find assignments.
        
        # First, identify the "essential" positions where T=1
        positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    positions.append((a, b, c))
        
        # For a given rank, try to assign each position to a component
        # and solve the resulting system over GF(3)
        
        # This is complex; instead, use a structured approach:
        # For each component, pick a small set of positions and solve
        # for U,V,W entries that make the sum work out.
        
        # Simpler approach: use random initialization over GF(3) and
        # check if the tensor is reconstructed correctly.
        
        for attempt in range(200):
            if time.time() > deadline - 20:
                return None
            
            U = np.zeros((nm, rank), dtype=np.int64)
            V = np.zeros((mp, rank), dtype=np.int64)
            W = np.zeros((pn, rank), dtype=np.int64)
            
            # Random sparse initialization over GF(3)
            for r in range(rank):
                # Pick a few nonzero entries
                nz_u = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(mp, size=np.random.randint(1, 5), replace=False)
                nz_w = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
                U[nz_u, r] = np.random.randint(1, 3, size=len(nz_u))
                V[nz_v, r] = np.random.randint(1, 3, size=len(nz_v))
                W[nz_w, r] = np.random.randint(1, 3, size=len(nz_w))
            
            # Check if the tensor is reconstructed correctly mod 3
            R = np.zeros((nm, mp, pn), dtype=np.int64)
            for r in range(rank):
                R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            R %= 3
            
            # Compare with target mod 3
            if np.array_equal(R, target.astype(np.int64)):
                # Found a valid GF(3) decomposition
                # Lift to rationals: entries are 0, 1, 2
                return U.astype(float), V.astype(float), W.astype(float)
        
        return None

    def gf3_structured_search(rank):
        """
        Structured GF(3) search using the block structure of the problem.
        The matrix multiplication has a natural block structure:
        A is 2x3, B is 3x7, C is 2x7.
        Each entry C[i,k] = sum_j A[i,j]*B[j,k]
        
        We can exploit this by creating components that correspond to
        specific (i,j,k) patterns.
        """
        # For each (i,k) pair in the output, we need sum_j A[i,j]*B[j,k]
        # This suggests components that focus on specific (i,k) pairs
        # and distribute the j-sum across components.
        
        # Number of output entries: n*p = 2*7 = 14
        # For each output entry, we need at least 1 component (for the sum over j)
        # But we can share components if they work for multiple (i,k) pairs.
        
        # Try a structured approach: create components indexed by (i,k) pairs
        # and for each, solve for the j-dependence.
        
        # Actually, let's try a different approach: use the fact that
        # the problem is small enough to do an exhaustive search over
        # small support patterns.
        
        # For rank r, each component has at most r nonzeros in each of U,V,W
        # The total number of "active" positions is at most r*(nm+mp+pn)
        
        # Let's try a greedy approach: build up the decomposition one
        # component at a time, choosing the component that best matches
        # the remaining tensor.
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        remaining = target.copy()
        
        for r in range(rank):
            if time.time() > deadline - 20:
                return None
            
            # Find the best rank-1 component to subtract
            # Try all possible small support patterns
            best_component = None
            best_error = float('inf')
            
            # For each possible support size (small), try random patterns
            for _ in range(50):
                # Pick a random small support
                sz_u = np.random.randint(1, 4)
                sz_v = np.random.randint(1, 5)
                sz_w = np.random.randint(1, 4)
                
                nz_u = np.random.choice(nm, size=sz_u, replace=False)
                nz_v = np.random.choice(mp, size=sz_v, replace=False)
                nz_w = np.random.choice(pn, size=sz_w, replace=False)
                
                # Try all combinations of values in {1, 2} (GF(3) nonzero)
                # This is too many; instead, use random values
                u_vals = np.random.randint(1, 3, size=sz_u)
                v_vals = np.random.randint(1, 3, size=sz_v)
                w_vals = np.random.randint(1, 3, size=sz_w)
                
                u_vec = np.zeros(nm, dtype=np.int64)
                v_vec = np.zeros(mp, dtype=np.int64)
                w_vec = np.zeros(pn, dtype=np.int64)
                u_vec[nz_u] = u_vals
                v_vec[nz_v] = v_vals
                w_vec[nz_w] = w_vals
                
                # Compute the contribution
                contrib = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec) % 3
                
                # Error: how well does this match the remaining tensor?
                # We want to minimize the mismatch
                error = np.sum((remaining - contrib) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_component = (u_vec, v_vec, w_vec)
            
            if best_component is None:
                return None
            
            u_vec, v_vec, w_vec = best_component
            U[:, r] = u_vec
            V[:, r] = v_vec
            W[:, r] = w_vec
            
            # Update remaining
            contrib = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec) % 3
            remaining = (remaining - contrib) % 3
        
        # Final check
        R = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        R %= 3
        
        if np.array_equal(R, target.astype(np.int64)):
            return U.astype(float), V.astype(float), W.astype(float)
        
        return None

    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R

    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)

    def als_optimize(U, V, W, max_iters=100):
        """
        Optimize U, V, W using Alternating Least Squares with time guard.
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        prev_error = tensor_error(U, V, W)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update U
            M_U = np.zeros((mp * pn, r))
            for c in range(r):
                M_U[:, c] = np.outer(V[:, c], W[:, c]).flatten()

            for a in range(nm):
                rhs = target[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    U[a, :] = 0.0

            # Update V
            M_V = np.zeros((nm * pn, r))
            for c in range(r):
                M_V[:, c] = np.outer(U[:, c], W[:, c]).flatten()

            for b in range(mp):
                rhs = target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    V[b, :] = 0.0

            # Update W
            M_W = np.zeros((nm * mp, r))
            for c in range(r):
                M_W[:, c] = np.outer(U[:, c], V[:, c]).flatten()

            for c in range(pn):
                rhs = target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    W[c, :] = 0.0

            current_error = tensor_error(U, V, W)
            if abs(current_error - prev_error) < 1e-12:
                break
            prev_error = current_error

        return U, V, W, prev_error

    def polish_pass(U, V, W, max_iters=300):
        """
        High-precision polish: alternate between rounding to half-integers
        and doing a tight local solve. This snaps the solution onto the lattice.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Round to half-integers
            U_r = round_to_half_integer(U)
            V_r = round_to_half_integer(V)
            W_r = round_to_half_integer(W)

            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r

            # Use rounded values as starting point for a tight local solve
            M_U = np.zeros((mp * pn, r))
            for c in range(r):
                M_U[:, c] = np.outer(V_r[:, c], W_r[:, c]).flatten()

            for a in range(nm):
                rhs = target[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                except:
                    pass

            M_V = np.zeros((nm * pn, r))
            for c in range(r):
                M_V[:, c] = np.outer(U_r[:, c], W_r[:, c]).flatten()

            for b in range(mp):
                rhs = target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                except:
                    pass

            M_W = np.zeros((nm * mp, r))
            for c in range(r):
                M_W[:, c] = np.outer(U_r[:, c], V_r[:, c]).flatten()

            for c in range(pn):
                rhs = target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                except:
                    pass

        return U, V, W, tensor_error(U, V, W)

    def structured_init(rank):
        """
        Create a structured initialization based on the tensor's support.
        """
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    positions.append((i * m + j, j * p + k, k * n + i))

        for idx, (a, b, c) in enumerate(positions):
            comp = idx % rank
            U[a, comp] += 1.0 + 0.1 * np.random.randn()
            V[b, comp] += 1.0 + 0.1 * np.random.randn()
            W[c, comp] += 1.0 + 0.1 * np.random.randn()

        return U, V, W

    def random_sparse_init(rank):
        """Random sparse initialization with structure-aware sparsity."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        for c in range(rank):
            nz_u = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
            nz_v = np.random.choice(mp, size=np.random.randint(1, 5), replace=False)
            nz_w = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
            U[nz_u, c] = np.random.choice([-1, 1], size=len(nz_u)) * 0.5
            V[nz_v, c] = np.random.choice([-1, 1], size=len(nz_v)) * 0.5
            W[nz_w, c] = np.random.choice([-1, 1], size=len(nz_w)) * 0.5

        return U, V, W

    def try_rank(rank, num_attempts=10):
        """Try to find a valid decomposition for a given rank."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None

        # First, try GF(3) search
        gf3_result = gf3_search(rank)
        if gf3_result is not None:
            U, V, W = gf3_result
            if verify(U, V, W):
                return U, V, W
            # Try polishing the GF(3) result
            U_pol, V_pol, W_pol = polish_pass(U, V, W, max_iters=50)
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol

        gf3_structured = gf3_structured_search(rank)
        if gf3_structured is not None:
            U, V, W = gf3_structured
            if verify(U, V, W):
                return U, V, W
            U_pol, V_pol, W_pol = polish_pass(U, V, W, max_iters=50)
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol

        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break

            if attempt % 4 == 0:
                U, V, W = structured_init(rank)
            elif attempt % 4 == 1:
                U, V, W = random_sparse_init(rank)
            else:
                U = np.random.randn(nm, rank) * 0.3
                V = np.random.randn(mp, rank) * 0.3
                W = np.random.randn(pn, rank) * 0.3

            U_opt, V_opt, W_opt, final_error = als_optimize(U, V, W, max_iters=80)

            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()

            if final_error < 1.0:
                U_pol, V_pol, W_pol = polish_pass(U_opt, V_opt, W_opt, max_iters=100)
                if verify(U_pol, V_pol, W_pol):
                    return U_pol, V_pol, W_pol

            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)

            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded

        if best_U is not None and best_error < 5.0:
            U_pol, V_pol, W_pol = polish_pass(best_U, best_V, best_W, max_iters=150)
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol

        return None, None, None

    best_result = None
    best_rank = 42

    for rank in range(24, 42):
        if time.time() > deadline - 35:
            break

        num_attempts = 10 if rank <= 30 else 6
        U, V, W = try_rank(rank, num_attempts=num_attempts)

        if U is not None:
            best_result = (U, V, W)
            best_rank = rank
            break

    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
