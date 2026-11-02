"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy broadcasting.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    r = U.shape[1]
    U_arr = U.reshape(n, m, r)
    V_arr = V.reshape(m, p, r)
    W_arr = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    Ue = U_arr[:, :, None, :]  # (n, m, 1, r)
    Ve = V_arr[None, :, :, :]  # (1, m, p, r)
    # W[k,i,r] -> need as function of (i, j, k, r)
    # W_arr is (p, n, r), so W_arr.T is (n, p, r), W_arr.T[i, k, r] = W[k, i, r]
    We = W_arr.T[:, None, :, :]  # (n, 1, p, r)
    
    T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification with rational arithmetic (entries should be half-integers)."""
    r = U.shape[1]
    # Multiply all by 2 to work with integers
    U2 = (2 * U).round()
    V2 = (2 * V).round()
    W2 = (2 * W).round()
    
    # Check they're close to integers
    if not (np.allclose(U2, (2*U), atol=1e-6) and 
            np.allclose(V2, (2*V), atol=1e-6) and 
            np.allclose(W2, (2*W), atol=1e-6)):
        return False
    
    U2 = U2.astype(np.int64)
    V2 = V2.astype(np.int64)
    W2 = W2.astype(np.int64)
    
    U_arr = U2.reshape(n, m, r)
    V_arr = V2.reshape(m, p, r)
    W_arr = W2.reshape(p, n, r)
    
    Ue = U_arr[:, :, None, :]
    Ve = V_arr[None, :, :, :]
    We = W_arr.T[:, None, :, :]
    
    T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
    # T should be 8 (= 2^3) for all entries since we scaled by 2
    return np.all(T == 8)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W

def _try_grouped_by_j(n, m, p):
    """
    Try decomposition grouped by j (the middle dimension).
    For each j, we need to handle the n*p combinations.
    This gives m * n * p = same as schoolbook, but let's try to merge.
    """
    # For each j, the contribution is A[:,j] * B[j,:]
    # This is an outer product of an n-vector and p-vector
    # Can we represent multiple j's with shared structure?
    
    # Rank m * n * p is schoolbook. Let's try rank m * max(n,p) with structure.
    # Actually, for each j: C += A[:,j] @ B[j,:]
    # This is rank 1 in the (i,k) space for each j
    # Total rank = m (if we could share), but we can't because A[:,j] and B[j,:] vary
    
    # Try: for each j, use n*p rank-1 tensors -> total m*n*p (schoolbook)
    # Or: try to use fewer by exploiting linearity
    
    # Let's try a different grouping: by i
    # For each i: C[i,:] = sum_j A[i,j] * B[j,:]
    # This is a linear combination of m vectors in R^p
    # Each A[i,j] is a scalar, so we need m rank-1 contributions per i
    # Total: n * m * p (same)
    
    # Try by k: C[:,k] = sum_j A[:,j] * B[j,k]
    # Same thing.
    
    return None  # No improvement found with simple grouping

def _try_straight_line_program(n, m, p):
    """
    Try to find a straight-line program (SLP) for matrix multiplication.
    This is equivalent to finding a bilinear algorithm.
    
    For 3x4 * 4x4, we're looking for fewer than 48 multiplications.
    
    Known results:
    - 2x2: rank 7 (Strassen)
    - 3x3: rank 23 (best known)
    - 4x4: rank 49 (best known)
    
    For rectangular, results are less well-known.
    Let's try a parameterized family.
    """
    # Try rank 16 (the lower bound)
    # This would require each rank-1 term to contribute to exactly 3 entries (i,j,k)
    # Since we have 48 entries and 16 terms, each term covers ~3 entries on average
    
    # For rank 16, we need:
    # sum_{r=0}^{15} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of 48 polynomial equations in 16*(12+16+12) = 576 variables
    # Very underdetermined, but the equations are highly structured
    
    # Try a structured ansatz:
    # U[i*m+j, r] = a_{i,j,r}
    # V[j*p+k, r] = b_{j,k,r}  
    # W[k*n+i, r] = c_{k,i,r}
    # with small integer/half-integer values
    
    # Strategy: use symmetry. The problem has some symmetry between dimensions.
    # n=3, m=4, p=4
    
    # Try: for each r, the rank-1 tensor U_r ⊗ V_r ⊗ W_r should have
    # support on a small set of (i,j,k) triples
    
    # Let's try to construct rank 16 by hand using the structure:
    # The tensor T[i,j,k] = 1 is the "all-ones" tensor of size 3x4x4
    
    # A rank-1 tensor has the form u_i * v_j * w_k
    # We need sum_r u_r(i) * v_r(j) * w_r(k) = 1 for all i,j,k
    
    # This is a tensor decomposition of the all-ones tensor!
    # The all-ones tensor of size n x m x p has rank 1 (it's u⊗v⊗w where u,v,w are all-ones)
    # But we need the DECOMPOSITION to be in the specific format where
    # U has shape (n*m, r), not (n, r) x (m, r)
    
    # Wait, I'm confusing myself. Let me re-read the problem.
    
    # The matrix multiplication tensor for M(n,m) * N(m,p) -> P(n,p) is:
    # T[i,j,k] = 1 for all valid i,j,k
    # And we want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So U is indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs
    # This is NOT the standard tensor decomposition of an n x m x p tensor
    # It's a different structure!
    
    # Let me think about this differently.
    # For each r, define:
    # u_r: R^{n*m} (indexed by i*m+j)
    # v_r: R^{m*p} (indexed by j*p+k)
    # w_r: R^{p*n} (indexed by k*n+i)
    
    # The constraint is: for all (i,j,k):
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # This is equivalent to: the tensor T of size (n*m) x (m*p) x (p*n) 
    # with T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j', k=k', i=i' else 0
    # Wait no, that's not right either.
    
    # Let me be very careful. The matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # The bilinear form is: for each i,k, C[i,k] = sum_j A[i,j] B[j,k]
    # This involves n*p outputs, each being an inner product of length m
    
    # The tensor for this computation, in the standard format, is:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    # So T is a (n*m) x (m*p) x (p*n) tensor where T[a, b, c] = 1 
    # if the j-index of a equals the j-index of b, 
    # the k-index of b equals the k-index of c, 
    # and the i-index of c equals the i-index of a, else 0.
    
    # Let's verify: a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 iff j=j' AND k=k' AND i=i'
    
    # So the tensor has 1s at positions where the "cycle" i->j->k->i is consistent.
    # Number of 1s: n*m*p = 48 (for each valid (i,j,k) triple, there's exactly one 1)
    
    # Now we want to decompose this tensor as:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a standard 3-tensor decomposition!
    # The tensor has size (n*m) x (m*p) x (p*n) = 12 x 16 x 12
    # with 48 non-zero entries (all equal to 1)
    
    # The rank of this tensor is what we're looking for.
    
    # Key insight: this tensor has a lot of structure. Each 1 is at position
    # (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p)
    
    # Let's try to find a low-rank decomposition by exploiting this structure.
    
    # For rank r, we need r vectors u_r in R^{12}, v_r in R^{16}, w_r in R^{12}
    # such that for all (i,j,k): u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] sums to 1
    # and for all other (a,b,c): the sum is 0.
    
    # This is a system of (12*16*12 - 48) = 2256 equations (the zeros) plus 48 equations (the ones)
    # in 12r + 16r + 12r = 40r variables.
    
    # For r=16: 640 variables, 2304 equations. Overdetermined but structured.
    
    # Let's try a constructive approach.
    
    # Idea: use the fact that the 1s form a "permutation-like" structure.
    # For each (i,j,k), the 1 is at (i*m+j, j*p+k, k*n+i)
    
    # Let's try to use rank-1 tensors that each cover exactly one 1 and zero out everything else.
    # That would give rank 48 (schoolbook).
    
    # To do better, we need rank-1 tensors that cover multiple 1s.
    # A rank-1 tensor u ⊗ v ⊗ w has u[a] * v[b] * w[c] non-zero wherever 
    # u[a], v[b], w[c] are all non-zero.
    
    # For it to cover multiple 1s at positions (i1*m+j1, j1*p+k1, k1*n+i1), 
    # (i2*m+j2, j2*p+k2, k2*n+i2), etc., we need:
    # u[i1*m+j1] * v[j1*p+k1] * w[k1*n+i1] = alpha1
    # u[i2*m+j2] * v[j2*p+k2] * w[k2*n+i2] = alpha2
    # and all other products = 0
    
    # This is hard to satisfy exactly.
    
    # Alternative: use linear combinations. If we have two rank-1 tensors that 
    # each cover one 1, their sum covers both. But that's still rank 2 for 2 ones.
    
    # The key is to find rank-1 tensors that are "sparse" in a useful way.
    
    # Let's try a different approach: use the structure to build a block decomposition.
    
    # Group the 1s by i: for each i, we have m*p = 16 ones at positions
    # (i*m+j, j*p+k, k*n+i) for j in [0,m), k in [0,p)
    
    # For fixed i, the positions are:
    # a = i*m + j (j ranges over 0..3)
    # b = j*p + k (j ranges over 0..3, k ranges over 0..3)
    # c = k*n + i (k ranges over 0..3, i is fixed)
    
    # So for fixed i, the a-values are {i*4, i*4+1, i*4+2, i*4+3}
    # The c-values are {i, i+3, i+6, i+9}
    # The b-values depend on j and k
    
    # This suggests a block structure where each "block" corresponds to a fixed i.
    
    # For rank 16, we could try: 4 blocks of rank 4, one for each i.
    # But that gives rank 16 only if the blocks don't interfere.
    
    # Actually, let's just try to numerically solve for small ranks.
    
    return None  # Placeholder, will implement actual search below

def _construct_rank_r(n, m, p, r, seed=42):
    """
    Try to construct a rank-r decomposition using numerical optimization.
    Uses the structure of the matrix multiplication tensor.
    """
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    # The tensor T has 1s at positions (i*m+j, j*p+k, k*n+i)
    # and 0s elsewhere.
    
    # We want to minimize: sum_{all (a,b,c)} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
    
    # But this is 12*16*12 = 2304 terms, most of which are 0.
    
    # Let's use gradient descent on the loss function.
    
    # Precompute the positions of 1s
    ones_positions = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_positions.append((a, b, c))
    
    # Loss = sum_{(a,b,c) in ones} (S[a,b,c] - 1)^2 + sum_{(a,b,c) not in ones} S[a,b,c]^2
    # where S[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # For efficiency, we can compute S as a 3D tensor using broadcasting.
    
    def compute_S(U, V, W):
        """Compute S[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        # S = U @ V^T @ W^T ... no, that's not right.
        # S[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-tensor contraction.
        
        # Use einsum: S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # = np.einsum('ar,br,cr->abc', U, V, W)
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def loss_and_grad(U, V, W):
        S = compute_S(U, V, W)
        
        # Target tensor
        T = np.zeros((n*m, m*p, p*n))
        for a, b, c in ones_positions:
            T[a, b, c] = 1.0
        
        # Loss
        L = np.sum((S - T)**2)
        
        # Gradient: dL/dU[a,r] = 2 * sum_{b,c} (S[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # = 2 * (S - T) contracted with V and W over b and c
        
        # dL/dU[a,r] = 2 * sum_{b,c} (S[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # = 2 * einsum('abc,br,cr->ar', S-T, V, W)
        
        R = S - T
        dU = 2 * np.einsum('abc,br,cr->ar', R, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', R, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', R, U, V)
        
        return L, dU, dV, dW
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(5000):
        L, dU, dV, dW = loss_and_grad(U, V, W)
        
        if L < best_loss:
            best_loss = L
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 0:
            lr = min(0.1, max(0.001, 0.01 / (grad_norm + 1e-6)))
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid solution
    S = compute_S(best_U, best_V, best_W)
    T = np.zeros((n*m, m*p, p*n))
    for a, b, c in ones_positions:
        T[a, b, c] = 1.0
    
    if np.allclose(S, T, atol=1e-6):
        return best_U, best_V, best_W, True
    
    return None, None, None, False

def _try_structured_decomposition(n, m, p):
    """
    Try a structured decomposition based on the algebraic structure.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can view this as: for each j, compute the outer product A[:,j] * B[j,:]
    and sum them up.
    
    The outer product A[:,j] * B[j,:] is a rank-1 n x p matrix.
    So C is a sum of m rank-1 matrices, giving a "rank" of m in the 
    matrix sense, but in the tensor sense we need to account for the 
    bilinear structure.
    
    In the tensor decomposition sense, each term A[i,j] * B[j,k] * 1 
    contributes to C[i,k]. The "1" here is the identity in the output space.
    
    So the natural decomposition is:
    For each (i,j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    where r = i*m*p + j*p + k
    
    This is the schoolbook decomposition of rank n*m*p.
    
    To do better, we need to find linear dependencies among these terms.
    
    Idea: use the fact that the output C[i,k] only depends on the sum over j,
    not on individual j values. This suggests we might be able to combine
    terms with different j.
    
    Let's try: for each (i,k), we need sum_j A[i,j] B[j,k].
    This is an inner product of the i-th row of A and the k-th column of B.
    
    If we could represent this inner product using fewer than m multiplications,
    we'd have a better algorithm. But for a single inner product of length m,
    the rank is m (you need m multiplications).
    
    However, we're computing n*p inner products simultaneously, and they share
    the same A and B matrices. This sharing is what allows for potential savings.
    
    For square matrices, Strassen's algorithm exploits this sharing.
    For rectangular matrices, similar techniques may apply.
    
    Let's try to adapt Strassen's idea to 3x4 * 4x4.
    """
    
    # Strassen's 2x2 algorithm uses 7 multiplications instead of 8.
    # The key is to compute 7 bilinear forms:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 3x4 * 4x4, we can try to partition the matrices into blocks
    # and apply Strassen-like ideas.
    
    # Partition A (3x4) into blocks:
    # A = [A1 | A2] where A1 is 3x2 and A2 is 3x2
    # Or: A = [A11 A12; A21 A22] with A11, A21 being 1x2 or 2x2
    
    # This is getting complex. Let's try a simpler approach:
    # use the known lower bound of 16 and try to construct a rank-16 solution.
    
    # For rank 16, we need 16 rank-1 tensors that sum to the target tensor.
    # Each rank-1 tensor u ⊗ v ⊗ w has u in R^12, v in R^16, w in R^12.
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms.
    
    # Schoolbook has 48 terms. We want to reduce to 16.
    # Each merge reduces the rank by 1 if we can express 2 terms as 1 term.
    # But that's not how it works. We need to find a new set of 16 terms
    # that sum to the same tensor.
    
    # Let's try to use symmetry. The tensor has some symmetry:
    # swapping (i,j,k) with (i',j',k') where the cycle is preserved.
    
    # Actually, let's just try the numerical optimization for rank 16, 20, 24, etc.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find better decompositions
    # Try ranks from low to high
    for target_rank in [16, 18, 20, 24, 28, 32, 36, 40, 44]:
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try multiple seeds
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            try:
                U_r, V_r, W_r, success = _construct_rank_r(n, m, p, target_rank, seed=42+seed)
                if success:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
            except Exception:
                continue
        
        if found:
            continue  # Found a solution at this rank, try lower
    
    # Convert to list format
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
