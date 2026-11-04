import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = 140.0

    def to_lists(mat):
        return mat.tolist()

    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def decompose_to_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def verify(U, V, W, atol=1e-9):
        T = decompose_to_tensor(U, V, W)
        return np.allclose(T, T_target, atol=atol)

    # Schoolbook rank-18 fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # --- Construct explicit rank-16 decomposition ---
    # Strategy: use structured factorizations.
    # The 2x3x3 multiplication tensor has rank 16.
    # We build it by combining a rank-2 decomposition of 2x1x1 with
    # rank-3 decompositions for the 3x3 part, exploiting Kronecker structure.
    #
    # Known approach: decompose M2*M3*M3 as sum of 16 rank-1 terms.
    # We use a systematic construction based on the Strassen-like recursion.

    def build_rank16():
        """
        Construct a rank-16 decomposition for 2x3 * 3x3 -> 2x3.
        
        Indexing:
          U: (6, 16)  rows = (i,j) with i in 0..1, j in 0..2
          V: (9, 16)  rows = (j,k) with j in 0..2, k in 0..2
          W: (6, 16)  rows = (k,i) with k in 0..2, i in 0..1
        """
        r = 16
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        W = np.zeros((pn, r), dtype=float)

        # We use a construction based on splitting the 3x3 multiplication.
        # For each of the 2 rows of the left matrix, we need to compute
        # a 1x3 * 3x3 = 1x3 product, which has rank 9 schoolbook,
        # but we can share structure.
        #
        # Alternative: use the known explicit rank-16 decomposition.
        # Let's use a systematic approach with half-integer entries.
        
        # Construction via Kronecker-like product:
        # Think of 2x3x3 as (2x1x1) ⊗ (1x3x3) but that gives rank 2*9=18.
        # We need to save 2 ranks by exploiting symmetries.
        
        # Explicit rank-16 construction:
        # We'll use a pattern based on the following idea:
        # Split the 3 columns into pairs and use Strassen-like tricks.
        
        # Let me try a direct construction with specific patterns.
        # Based on known results, the rank-16 decomposition for 2x3x3
        # can be built as follows:
        
        # Use the fact that rank(2x3x3) = rank(2x1x1) * rank(1x3x3) - savings
        # Actually, let's use a concrete known decomposition.
        
        # I'll construct it by solving the system directly with a smart initialization.
        
        # Use a structured random initialization guided by the tensor structure.
        np.random.seed(42)
        
        # Initialize with a structured pattern
        for t in range(r):
            # U[t] should be nonzero in rows corresponding to (i,j)
            # V[t] nonzero in rows (j,k)
            # W[t] nonzero in rows (k,i)
            # For the term to contribute to T[i*m+j, j*p+k, k*n+i],
            # U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] must match.
            
            # Structured initialization:
            # Pick random i, j, k patterns
            i0 = t % n
            j0 = t % m
            k0 = t % p
            
            # Set some entries
            U[i0*m + j0, t] = 1.0
            V[j0*p + k0, t] = 1.0
            W[k0*n + i0, t] = 1.0
            
            # Add some more entries to allow for the rank-16 structure
            if t < 8:
                # First 8 terms: cover the "diagonal" structure
                pass
            else:
                # Last 8 terms: cover the "off-diagonal" structure
                i1 = (i0 + 1) % n
                j1 = (j0 + 1) % m
                k1 = (k0 + 1) % p
                U[i1*m + j1, t] = 1.0
                V[j1*p + k1, t] = 1.0
                W[k1*n + i1, t] = 1.0

        return U, V, W

    def polish_decomposition(U, V, W, max_iter=5000):
        """
        Polish a decomposition to exact values using gradient descent
        with careful convergence criteria.
        """
        r = U.shape[1]
        lr = 0.01
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            T_approx = decompose_to_tensor(U, V, W)
            diff = T_approx - T_target
            
            # Gradient w.r.t. U: grad_U[a,t] = sum_{b,c} diff[a,b,c] * V[b,t] * W[c,t]
            grad_U = np.einsum('abc,bt,ct->at', diff, V, W)
            grad_V = np.einsum('abc,at,ct->bt', diff, U, W)
            grad_W = np.einsum('abc,at,bt->ct', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            err = np.sum(diff**2)
            if err < 1e-10:
                break
        
        return U, V, W

    def round_and_verify(U, V, W):
        """Round to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
        return None

    # Try to find rank-16 decomposition
    best_result = None
    
    # Attempt 1: Use structured initialization + polish
    if time.time() - start_time < deadline - 10:
        U0, V0, W0 = build_rank16()
        U1, V1, W1 = polish_decomposition(U0.copy(), V0.copy(), W0.copy(), max_iter=3000)
        result = round_and_verify(U1, V1, W1)
        if result is not None:
            best_result = result
    
    # Attempt 2: Multiple random structured inits
    if best_result is None and time.time() - start_time < deadline - 10:
        for seed in range(20):
            if time.time() - start_time > deadline - 10:
                break
            np.random.seed(seed * 1000 + 7)
            
            U0 = np.zeros((nm, 16))
            V0 = np.zeros((mp, 16))
            W0 = np.zeros((pn, 16))
            
            for t in range(16):
                # Random structured init
                for i in range(n):
                    for j in range(m):
                        if np.random.rand() < 0.5:
                            U0[i*m+j, t] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                for j in range(m):
                    for k in range(p):
                        if np.random.rand() < 0.4:
                            V0[j*p+k, t] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                for k in range(p):
                    for i in range(n):
                        if np.random.rand() < 0.5:
                            W0[k*n+i, t] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            U1, V1, W1 = polish_decomposition(U0.copy(), V0.copy(), W0.copy(), max_iter=2000)
            result = round_and_verify(U1, V1, W1)
            if result is not None:
                best_result = result
                break
    
    # Attempt 3: Direct systematic search for rank-16 with sparse patterns
    if best_result is None and time.time() - start_time < deadline - 10:
        # Try a more systematic construction
        # Based on the algebraic structure, try specific patterns
        
        # Pattern: use the fact that we can decompose as
        # sum over t of u_t ⊗ v_t ⊗ w_t
        # where the supports are chosen to cover all (i,j,k) triples
        
        # Explicit known rank-16 for 2x3x3:
        # We'll try a construction based on splitting into 2x2 and 2x1 blocks
        
        # Let's try a concrete explicit construction:
        U = np.zeros((6, 16))
        V = np.zeros((9, 16))
        W = np.zeros((6, 16))
        
        # Terms 0-7: for i=0
        # Terms 8-15: for i=1
        
        # For i=0, j=0,1,2 and k=0,1,2:
        # We need 9 terms but can share with i=1 terms
        
        # Actually, let me try a different explicit construction
        # using the known minimal decomposition pattern.
        
        # Use a Kronecker-like construction:
        # rank(2x3x3) <= rank(2x2x2) + rank(2x1x1)*rank(1x3x3) - ...
        
        # Let me just try harder with optimization from many random starts
        for seed in range(50):
            if time.time() - start_time > deadline - 10:
                break
            np.random.seed(seed * 31337 + 11)
            
            U0 = np.random.randn(6, 16) * 0.5
            V0 = np.random.randn(9, 16) * 0.5
            W0 = np.random.randn(6, 16) * 0.5
            
            U1, V1, W1 = polish_decomposition(U0.copy(), V0.copy(), W0.copy(), max_iter=1500)
            result = round_and_verify(U1, V1, W1)
            if result is not None:
                best_result = result
                break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
    
    # Fallback: schoolbook rank 18
    U, V, W = schoolbook()
    return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
# EVOLVE-BLOCK-END
