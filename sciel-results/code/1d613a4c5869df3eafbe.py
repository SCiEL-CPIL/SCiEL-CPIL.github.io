"""Matrix multiplication 3x3 * 3x5 rank minimization via exact rank peeling."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin for 150s limit


def construct():
    n, m, p = 3, 3, 5
    total = time.time()
    
    def deadline_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    # Build the multiplication tensor as dict of (i,j,k) -> 1
    # U has row i*m+j, V has row j*p+k, W has row k*n+i (trace convention)
    # Rank-r decomposition: sum_{c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta(i,j,k)
    # where delta=1 if (i,j,k) is a valid product term, 0 otherwise
    # Valid terms: for each (i,j,k), the product A[i,j]*B[j,k] appears in C[i,k]
    # So tensor T[i,j,k] = 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # That's just the all-ones tensor! The product A*B has C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear map is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the trilinear tensor for the multiplication is T[i,j,k] where we pair A[i,j] with B[j,k] to get C[i,k]
    
    # Wait - the matrix multiplication tensor T_{ijk} is defined as:
    # (T * (a,b,c)) = sum_{i,j,k} T_{ijk} a_i b_j c_k = sum_k C_k * x_k
    # where C_k = sum_i a_i * B_{ik} ... 
    
    # Let me reconsider. For multiplying A (n x m) by B (m x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear tensor has T[i,j,k] = 1 (A's row i, B's column k, summation over j)
    # Actually the tensor entries are T[i,k,j] or similar depending on convention.
    # 
    # Given the factor convention:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The reconstruction is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This must equal delta(i,j,k) where delta=1 means "A[i,j]*B[j,k] contributes to C[i,k]"
    # Which is 1 for all valid (i,j,k) since the formula is C[i,k] = sum_j A[i,j]*B[j,k]
    
    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # This is the rank-1 tensor of all ones! Its tensor rank is 1??
    
    # Wait, that can't be right. The multiplication tensor rank is not 1.
    # Let me re-examine. The bilinear form for C[i,k] is:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # So the multiplication is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}
    # The structure constants: T is a 3-tensor with T_{(i,j),(j,k),(i,k)} ...
    
    # Actually I think the correct interpretation is:
    # We want to decompose the bilinear map. The tensor T has:
    # T[i,j,k] in the "right" ordering such that the bilinear form gives C.
    
    # With U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For this to compute C correctly, we need:
    # For each output C[i,k]: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor should satisfy: the "product" reconstructed from U,V,W matches A*B
    
    # I think the decomposition is:
    # For each c in 0..rank-1, we have vectors u_c (length n*m), v_c (length m*p), w_c (length n*p)
    # And: sum_c outer product appropriately... 
    
    # Actually the standard way: the multiplication tensor T in E_{nm} ⊗ E_{mp} ⊗ E_{np}^*
    # T((i,j),(j',k),(i',k')) = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}... no
    
    # Let me think differently. The bilinear map f: R^{nm} × R^{mp} -> R^{np}
    # f(a,b)_{i,k} = sum_j a_{i,j} * b_{j,k}
    # 
    # A rank-r decomposition writes f as:
    # f(a,b)_{i,k} = sum_{c=1}^{r} u_c(i*m to ...) ... 
    
    # I think the correct reading of the contract is:
    # We need U (nm × r), V (mp × r), W (np × r) such that
    # for all i,k: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And this should equal sum_j A[i,j]*B[j,k]
    # 
    # Hmm, but that has a W depending on both k and i, which is the output index.
    # 
    # Actually I think the convention is:
    # U, V, W factor the tensor T ∈ R^{nm} ⊗ R^{mp} ⊗ R^{np} (with appropriate identification)
    # where T_{(i,m+j), (j,p+k), (k,n+i)} = 1 if and only if this represents the multiplication
    #
    # The multiplication tensor for A_{n×m} × B_{m×p} → C_{n×p} is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_{i,j,k} where e_{i,j,k} corresponds to the (i,k)-th entry
    
    # With the given indexing, I believe the target is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{something}
    # 
    # For standard matmul, the "tensor" is: for each j, the product A[i,j]*B[j,k] feeds into C[i,k]
    # So T_{(i*m+j), (j*p+k), (k*n+i)} should be 1 for all valid i,j,k (since each triple (i,j,k)
    # uniquely determines a contribution to C[i,k] from the j-th term).
    
    # Wait, that IS just the all-ones tensor again. But the matmul tensor has rank much more than 1.
    
    # I think I'm confusing the tensor. Let me look at it from the perspective of:
    # The matmul as a 3-tensor T_{αβγ} where α indexes A, β indexes B, γ indexes C.
    # T_{(i,j),(j',k),(i',k')} = 1 if i=i', j=j', k=k' (the standard Kronecker structure)
    # 
    # With the given factor shapes:
    # U[α, c], V[β, c], W[γ, c] where α=i*m+j, β=j*p+k, γ=k'*n+i'
    #
    # The constraint is: sum_c U[α,c]*V[β,c]*W[γ,c] = T_{α,β,γ}
    # where T_{α,β,γ} = 1 if α corresponds to (i,j), β to (j',k), γ to (i',k') 
    # AND i=i', j=j', k=k'... but that's always true since each index is unique.
    
    # Hmm, I think the issue is that the matmul tensor is NOT simply all ones in this indexing.
    # Let me reconsider: the tensor T has T_{α,β,γ} = 1 when the indices are consistent.
    # α = i*m+j means row i, column j of A
    # β = j'*p+k means row j', column k of B  
    # γ = k'*n+i' means row i', column k' of C
    #
    # The multiplication rule: C[i',k'] = sum_j' A[i',j'] * B[j',k']
    # So T_{α,β,γ} = 1 iff i=i' AND j=j' AND k=k'
    # But since α determines (i,j), β determines (j',k), γ determines (i',k'),
    # we need i=i', j=j', k=k' simultaneously.
    
    # This means T is not all ones! It's 1 only when the three indices are "consistent".
    # Specifically, T[α,β,γ]=1 when:
    #   (α//m) == (γ%m)    [i from A == i' from C]
    #   (α%m) == (β//p)    [j from A == j' from B]  
    #   (β%p) == (γ//n)    [k from B == k' from C]
    
    n, m, p = 3, 3, 5
    N_A = n * m  # 9
    N_B = m * p  # 15
    N_C = p * n  # 15
    
    # Build target tensor T as dict
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[(alpha, beta, gamma)] = 1
    
    # Schoolbook: rank 45, each rank-1 term is a single (i,j,k)
    # U[alpha,c]=1, V[beta,c]=1, W[gamma,c]=1 for the corresponding c
    
    # Rank peeling: start from schoolbook, try to merge columns
    # Random initialization: use schoolbook
    rank = n * m * p
    # U, V, W as lists of lists (for exact arithmetic)
    # Initialize with schoolbook
    U = [[Fraction(0)] * rank for _ in range(N_A)]
    V = [[Fraction(0)] * rank for _ in range(N_B)]
    W = [[Fraction(0)] * rank for _ in range(N_C)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    def compute_residual(U, V, W, rank):
        """Compute T - sum_c U[:,c]*V[:,c]*W[:,c] over supported entries"""
        R = {}
        for c in range(rank):
            for alpha in range(N_A):
                ua = U[alpha][c]
                if ua == 0:
                    continue
                for beta in range(N_B):
                    uv = ua * V[beta][c]
                    if uv == 0:
                        continue
                    for gamma in range(N_C):
                        uvg = uv * W[gamma][c]
                        if uvg == 0:
                            continue
                        key = (alpha, beta, gamma)
                        R[key] = R.get(key, Fraction(0)) + uvg
        return R
    
    def try_remove_column(U, V, W, rank, col_idx):
        """Try to remove column col_idx by expressing it as combination of others.
        Return True if successful, modifying U, V, W in place."""
        if rank <= 1:
            return False
        
        # We want to set U[:,col_idx]=0, V[:,col_idx]=0, W[:,col_idx]=0
        # which means absorbing it into other columns
        # Simple approach: try to find a column c' such that we can "merge"
        # For now, just try brute force: can we zero out this column?
        # This requires the rank-1 tensor u⊗v⊗w to be in the span of others.
        
        # Simpler: try pairwise merging of two columns into one
        # (a,b) -> c where the sum can be represented
        return False
    
    def check_exactness(U, V, W, rank):
        """Check if the decomposition exactly reproduces T"""
        R = compute_residual(U, V, W, rank)
        for (alpha, beta, gamma) in T:
            expected = T[(alpha, beta, gamma)]
            actual = R.get((alpha, beta, gamma), Fraction(0))
            if actual != expected:
                return False
        # Check no spurious entries (entries not in T should be 0)
        for key, val in R.items():
            if val != 0 and key not in T:
                return False
        return True
    
    # Strategy: random rank peeling with small random perturbations
    # Try to reduce rank by finding linearly dependent columns
    
    best_U, best_V, best_W, best_rank = U[:], V[:], W[:], rank
    
    # Try various random merge strategies
    if not deadline_ok():
        return {"U": [[float(x) for x in row] for row in best_U],
                "V": [[float(x) for x in row] for row in best_V],
                "W": [[float(x) for x in row] for row in best_W]}
    
    # The matmul tensor T_{α,β,γ} = 1[consistent] is actually the "permutation-like" tensor
    # Its rank is known to be at most n*m*p and the question is if we can do better.
    
    # For 3x3x5: n*m*p = 45. Known lower bound considerations suggest rank >= n*m*p/(max) ... 
    # Actually for rectangular matmul, the Strassen-like algorithms show rank can be less than n*m*p
    # for large enough matrices, but for 3x3x3 the rank is 23 (Strassen-like) not 27.
    
    # For 3x3 x 3x5, the tensor is 9 x 15 x 15. 
    # The rank is at most 9*15 = 135 (naive in one dimension) but schoolbook is 45.
    
    # Let me try a different approach: use the known structure.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can think of this as: for each j, compute A[:,j] * B[j,:] (outer product) and sum.
    # That gives rank m=3 if we could do outer products, but outer product of (n×1)(1×p) needs rank n*p=15
    # So total rank = m * n * p = 45 (schoolbook).
    
    # To beat 45, we'd need some kind of Strassen-like trick exploiting linearity.
    # For 3x3x3, rank is 23 < 27. For 3x3x5, let's see if we can do < 45.
    
    # Try a random search: start from schoolbook, randomly perturb and project back
    # to satisfy the constraints, reducing effective rank.
    
    # Alternative: use the fact that T can be viewed as a matrix and find low-rank approximations
    # Flatten T as a matrix of size (N_A*N_B) x N_C or similar, then use SVD-like approach.
    
    # Let's use a "column merging" approach with exact arithmetic
    # Start with rank 45, try to merge pairs of columns
    
    def merge_columns(U, V, W, rank, i, j):
        """Try to merge columns i and j into a single column.
        We replace two rank-1 tensors with one, hoping the result is exact."""
        if i >= rank or j >= rank or i == j:
            return None
        # After merging, the combined column c should satisfy:
        # For all (a,b,g) in T:
        #   U[a,i]*V[b,i]*W[g,i] + U[a,j]*V[b,j]*W[g,j] = U[a,c]*V[b,c]*W[g,c]
        # This is generally impossible for arbitrary columns.
        # Instead, we can try: set column c = column i, and add column j's contribution
        # by modifying other columns. This is the "peeling" approach.
        pass
        return None
    
    # Different approach: use bilinear factorization with random init and optimize
    # Since we need exact integer/half-integer results, let's try a structured approach.
    
    # Key insight: The tensor T[i,j,k] (with consistency constraint) has a special structure.
    # For the (n,m,p) = (3,3,5) case, let's see if a rank < 45 exists.
    
    # Lower bound: The tensor has "slice rank" considerations.
    # For a fixed j, the matrix T_{i,k;j} is the identity-like matrix (only T[i,j,k,j]=1... 
    # wait no. Let me re-examine.
    
    # T[α,β,γ] = 1 iff α=(i,j), β=(j,k), γ=(k,i) for some i,j,k
    # Note: γ = k*n+i, so γ encodes (output_row=k, output_col=i)? 
    # Wait: γ = k*n + i means row k of C, column i of C.
    # And C has dimensions n×p, so row index i ∈ [0,n), column index k ∈ [0,p)
    # But γ = k*n + i puts column index first, row index second. That's transposed indexing.
    
    # Let me re-read the contract: W[k*n+i, r]
    # So W's row index is k*n+i. If C is n×p, then C[i,k] (row i, col k) would be indexed as...
    # k*n+i means: for p columns and n rows, index = col*n + row.
    # So W indexes C in column-major order: W[col*n + row] = C[row, col] = C[i, k] where col=k, row=i.
    
    # OK so the indexing is consistent. The tensor T has T=1 at exactly 45 positions
    # (i,j,k) mapped to (i*m+j, j*p+k, k*n+i).
    
    # For rank minimization, let's try the approach of finding a low-rank decomposition
    # by solving a system of equations.
    
    # Start with a candidate rank r and try to find U, V, W with that rank.
    # This is an NP-hard problem in general, but for small cases we can try.
    
    # Let's try rank from 27 upward (since for 3x3x3 we know 23 works)
    # For 3x3x5, a reasonable target might be around 30-40.
    
    # Use a randomized approach: fix U and V randomly, solve for W, check if residual is zero.
    
    def attempt_rank(r, iterations=50):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        if not deadline_ok():
            return None
        
        for it in range(iterations):
            if not deadline_ok():
                return None
            
            # Random integer initializations for U and V
            U_rand = [[random.randint(-2, 2) for _ in range(r)] for _ in range(N_A)]
            V_rand = [[random.randint(-2, 2) for _ in range(r)] for _ in range(N_B)]
            
            # For each (α,β,γ) in T, we need:
            # sum_c U[α,c]*V[β,c]*W[γ,c] = 1
            # For entries not in T: sum_c U[α,c]*V[β,c]*W[γ,c] = 0
            
            # This gives us equations. Let's collect them.
            # Unknowns: W[γ,c] for γ in 0..N_C-1, c in 0..r-1. Total N_C*r unknowns.
            # Equations: one per (α,β,γ) pair.
            
            # For fixed (α,β), the equation for each γ is:
            # sum_c (U[α,c]*V[β,c]) * W[γ,c] = T[α,β,γ]
            # 
            # Let M_{α,β} be the vector (U[α,c]*V[β,c])_c
            # Then for each γ: sum_c M_{α,β}[c] * W[γ,c] = T[α,β,γ]
            
            # Collect all equations
            # Unknowns: W[0][0..r-1], W[1][0..r-1], ..., W[N_C-1][0..r-1]
            # Each equation involves at most one W[γ,*] row
            
            # Actually each equation is linear in the r components of a single W[γ,*]
            # No wait - each equation couples U, V, W for specific indices.
            
            # Let me restructure: fix U, V. Then for each (α,β,γ):
            #   sum_c M_{α,β,c} * W_{γ,c} = T_{α,β,γ}
            # where M_{α,β,c} = U_{α,c} * V_{β,c}
            
            # Group by (α,β): for each pair (α,β), we have equations for all γ:
            #   M_{α,β} · W_{γ} = T_{α,β,γ}
            # This means W_{γ} must satisfy M_{α,β} · W_{γ} = T_{α,β,γ} for ALL (α,β)
            # that are paired with this γ in the tensor.
            
            # Actually, for a given γ, which (α,β) pairs have T_{α,β,γ} ≠ 0?
            # T[α,β,γ]=1 iff there exists (i,j,k) with α=i*m+j, β=j*p+k, γ=k*n+i
            # Given γ=k*n+i, we have k=γ//n, i=γ%n
            # Then α=i*m+j (j free, 0..m-1), β=j*p+k (same j)
            # So for given γ, the valid (α,β) pairs are: for j=0..m-1:
            #   α = i*m + j, β = j*p + k
            
            # So for each γ, we have m equations (one per j):
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[γ, c] = 1
            
            # And for entries NOT in T, the sum must be 0.
            # T[α,β,γ]=0 otherwise.
            
            # Let me just set up the full linear system for W given U, V.
            
            # Equations: for each (α,β,γ) in the full index space [N_A]×[N_B]×[N_C]:
            #   sum_c (U_{α,c} * V_{β,c}) * W_{γ,c} = T_{α,β,γ}
            
            # This is a large system. Let's use the structure.
            # For efficiency, only process non-zero T entries and a subset of zeros.
            
            # Let me use a simpler approach: since we need W_{γ,c} for all γ,c,
            # and the equations are:
            # For γ = k*n+i, and for each j:
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  ... (must be 1)
            # For other (α,β,γ) combinations: sum = 0
            
            # The non-zero equations give us N_A equations (one per (α,β) in the support pattern)
            # Actually for each γ there are m equations, total n*p*m = n*m*p = 45 equations.
            
            # But we also need the zero constraints. Without them, W might have spurious values.
            
            # Let me try: only enforce the non-zero constraints and check the rest.
            
            # Build the system: unknowns are W[γ][c] for all γ, c. Total N_C * r.
            # Equations from non-zero T: for each (i,j,k): 
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is 45 equations. Unknowns: 15*r.
            # For r < 15, we have more unknowns than equations, so solutions exist generically.
            # But we also need the zero constraints.
            
            # Let me try: solve the underdetermined system for the minimum-norm solution,
            # then check if zero constraints are satisfied.
            
            from fractions import Fraction as Fr
            
            # Collect equations
            # Each equation: sum_c coeff[c] * W[gamma][c] = 1
            # where coeff[c] = U[i*m+j, c] * V[j*p+k, c]
            
            equations = []  # list of (gamma, coeffs, rhs)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        gamma = k * n + i
                        coeffs = [Fr(U_rand[i*m+j][c] * V_rand[j*p+k][c]) for c in range(r)]
                        equations.append((gamma, coeffs, Fr(1)))
            
            # Group by gamma
            from collections import defaultdict
            gamma_eqs = defaultdict(list)
            for gamma, coeffs, rhs in equations:
                gamma_eqs[gamma].append((coeffs, rhs))
            
            # For each gamma, solve the subsystem:
            # For gamma with e equations: sum over c of coeffs[t][c] * W[gamma][c] = rhs[t]
            # This is a system of e equations in r unknowns (W[gamma][0..r-1])
            
            W_sol = [[Fr(0) for _ in range(r)] for _ in range(N_C)]
            solvable = True
            
            for gamma in range(N_C):
                if gamma not in gamma_eqs:
                    continue
                eqs = gamma_eqs[gamma]
                e = len(eqs)
                # Solve e equations in r unknowns (W[gamma][c])
                # Use simple approach: if e <= r, find any solution
                # Represent as: A * x = b where A is e×r, b is e-vector
                A = [list(coeffs) for coeffs, rhs in eqs]
                b = [rhs for coeffs, rhs in eqs]
                
                # Solve using Gaussian elimination (exact)
                x = solve_system(A, b, r)
                if x is None:
                    solvable = False
                    break
                W_sol[gamma] = x
            
            if not solvable:
                continue
            
            # Check zero constraints (sample a few)
            valid = True
            check_count = 0
            random.seed(hash(it))
            for _ in range(200):
                alpha = random.randint(0, N_A - 1)
                beta = random.randint(0, N_B - 1)
                gamma = random.randint(0, N_C - 1)
                target = T.get((alpha, beta, gamma), Fr(0))
                actual = Fr(0)
                for c in range(r):
                    actual += Fr(U_rand[alpha][c]) * Fr(V_rand[beta][c]) * W_sol[gamma][c]
                if actual != target:
                    valid = False
                    break
                check_count += 1
            
            if valid:
                # Full verification
                for alpha in range(N_A):
                    if not deadline_ok():
                        return None
                    for beta in range(N_B):
                        for gamma in range(N_C):
                            target = T.get((alpha, beta, gamma), Fr(0))
                            actual = Fr(0)
                            for c in range(r):
                                actual += Fr(U_rand[alpha][c]) * Fr(V_rand[beta][c]) * W_sol[gamma][c]
                            if actual != target:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U_rand, V_rand, W_sol
            
            if not deadline_ok():
                return None
    
        return None
    
    def solve_system(A, b, num_vars):
        """Solve A*x=b where A is list of lists (equations), b is list, num_vars total.
        Returns solution x or None if inconsistent."""
        if not A:
            return [Fraction(0) for _ in range(num_vars)]
        
        m_eq = len(A)
        # Augmented matrix
        aug = [[x for x in eq] + [b[i]] for i, eq in enumerate(A)]
        
        # Gaussian elimination
        rank_r = 0
        cols = [[aug[eq][col] for eq in range(m_eq)] for col in range(num_vars)]
        
        for col in range(num_vars):
            if rank_r >= m_eq:
                break
            # Find pivot
            pivot = -1
            for row in range(rank_r, m_eq):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap
            if pivot != rank_r:
                aug[pivot], aug[rank_r] = aug[rank_r], aug[pivot]
            # Normalize
            pv = aug[rank_r][col]
            aug[rank_r] = [x / pv for x in aug[rank_r]]
            # Eliminate
            for row in range(m_eq):
                if row != rank_r and aug[row][col] != 0:
                    factor = aug[row][col]
                    aug[row] = [a - factor * b_val for a, b_val in zip(aug[row], aug[rank_r])]
            rank_r += 1
        
        # Check consistency
        for row in range(rank_r, m_eq):
            if any(x != 0 for x in aug[row][:-1]):
                return None
        
        # Solution: free variables set to 0
        x = [Fraction(0) for _ in range(num_vars)]
        for i in range(rank_r):
            # Find pivot column
            for col in range(num_vars):
                if aug[i][col] != 0:
                    x[col] = aug[i][-1]
                    break
        return x
    
    # Try ranks from 27 to 44
    random.seed(42)
    best_result = (best_U, best_V, best_W, best_rank)
    
    for r in range(27, 45):
        if not deadline_ok():
            break
        result = attempt_rank(r, iterations=10)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r, r)
            best_U, best_V, best_W, best_rank = U_r, V_r, W_r, r
            break  # Found a valid decomposition with rank r
    
    # Also try the known Strassen-style approach for the 3x3x3 sub-block
    # Convert best result to float lists
    final_U = [[float(x) for x in row] for row in best_U]
    final_V = [[float(x) for x in row] for row in best_V]
    final_W = [[float(x) for x in row] for row in best_W]
    
    return {"U": final_U, "V": final_V, "W": final_W}
