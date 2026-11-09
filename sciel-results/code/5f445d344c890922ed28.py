"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations, product
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The multiplication tensor T has entries:
    # T[i*m + j, j*p + k, k*n + i] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # All other entries are 0.
    
    # We want to find rank-1 decompositions: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where each term contributes to exactly one (i,j,k) triple.
    # 
    # Key insight: For matrix multiplication, the tensor is "sparse" in a structured way.
    # Each rank-1 term u ⊗ v ⊗ w satisfies:
    #   u[a] * v[b] * w[c] = 1 if (a,b,c) corresponds to some (i,j,k)
    #   u[a] * v[b] * w[c] = 0 otherwise
    #
    # For the standard basis approach, each rank-1 term can be associated with a subset
    # of the (i,j,k) triples. We need to cover all n*m*p = 96 triples with rank-1 terms.
    #
    # A rank-1 term u ⊗ v ⊗ w where u has support on rows {i*m+j : ...}, v on columns, w on outputs.
    # 
    # Let's think about this differently. The tensor T can be viewed as a 3-way array.
    # We can use the fact that for matrix multiplication, there exist known low-rank decompositions.
    # 
    # For 3x4 x 4x8, the schoolbook rank is 3*4*8 = 96.
    # We want to find if we can do better.
    #
    # Approach: Use the fact that each rank-1 component must have a specific structure.
    # Let's enumerate possible "patterns" for rank-1 components.
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly reconstruct T."""
        r = U.shape[1]
        recon = np.tensordot(U, V, axes=([1], [1]))  # (nm, mp, r)
        recon = np.tensordot(recon, W, axes=([2], [1]))  # (nm, mp, pn)
        
        # Build expected tensor
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        diff = np.abs(recon - T)
        return np.max(diff) < tol
    
    def to_half_integer_matrix(A, tol=1e-10):
        """Convert to half-integers if possible, else return as-is."""
        B = np.round(A * 2) / 2
        if np.max(np.abs(B - A)) < tol:
            return B
        B = np.round(A)
        if np.max(np.abs(B - A)) < tol:
            return B
        return A
    
    # Build the list of all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For each triple (i,j,k), the corresponding tensor entry is at:
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # Strategy: Try to find a decomposition where each rank-1 term covers multiple triples.
    # A rank-1 term u ⊗ v ⊗ w covers triple (i,j,k) if u[i*m+j] * v[j*p+k] * w[k*n+i] != 0.
    # For it to be exact, we need the sum over all terms to give exactly 1 at each (i,j,k) and 0 elsewhere.
    
    # Let's use a greedy/ILP-like approach. First, let's try small ranks.
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # But for matrix multiplication, the lower bound is tighter.
    
    # Let's try to construct a decomposition using the "blocking" technique.
    # Group indices to reduce rank.
    
    # For now, let's use a systematic search over possible rank-1 patterns.
    # Each rank-1 pattern is defined by which (i,j,k) triples it "covers".
    
    # A simpler approach: use known results. For 3x4x8 matrix multiplication,
    # the rank is known to be at most 96 (schoolbook). Let's try to find improvements.
    
    # Let's use ALS with better initialization and more iterations.
    def als_decompose(T, rank, max_iter=500, tol=1e-12, seed=42):
        """Alternating least squares for tensor decomposition."""
        np.random.seed(seed)
        nm, mp, pn = T.shape
        
        # Initialize with structured random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        for it in range(max_iter):
            # Fix V, W; solve for U
            M = np.outer(V, W).reshape(rank, mp * pn)
            T_flat = T.reshape(nm, mp * pn)
            U_new = T_flat @ np.linalg.pinv(M)
            
            # Fix U, W; solve for V
            M2 = np.outer(U, W).reshape(rank, nm * pn)
            T_flat2 = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
            V_new = T_flat2 @ np.linalg.pinv(M2)
            
            # Fix U, V; solve for W
            M3 = np.outer(U, V).reshape(rank, nm * mp)
            T_flat3 = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
            W_new = T_flat3 @ np.linalg.pinv(M3)
            
            # Check convergence
            if (np.max(np.abs(U_new - U)) < tol and
                np.max(np.abs(V_new - V)) < tol and
                np.max(np.abs(W_new - W)) < tol):
                U, V, W = U_new, V_new, W_new
                break
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    # Build T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_rank = nm * mp  # 96
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 95 down to 32
    for rank in range(95, 31, -1):
        found = False
        for seed in range(20):
            try:
                U, V, W = als_decompose(T, rank, max_iter=300, seed=seed)
                if check_decomposition(U, V, W):
                    # Try to convert to half-integers
                    U_r = to_half_integer_matrix(U)
                    V_r = to_half_integer_matrix(V)
                    W_r = to_half_integer_matrix(W)
                    if check_decomposition(U_r, V_r, W_r):
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                continue
        if found:
            break
    
    if best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_school = np.zeros((nm, nm * mp))
        V_school = np.zeros((mp, nm * mp))
        W_school = np.zeros((pn, nm * mp))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
