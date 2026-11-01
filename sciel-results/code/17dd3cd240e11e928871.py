"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
_DEADLINE_SECONDS = 135.0  # safety margin under 150s kill

def _strassen_rank7():
    """Known valid rank-7 Strassen decomposition for 2x2 matrix multiply."""
    n, m, p = 2, 2, 2
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    # W[k*n+i, c] with n=2: row index = k*2 + i
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0 -> C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # k=0,i=1 -> C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0 -> C[0,1]
        [1, -1, 1, 0, 0, 1, 0],  # k=1,i=1 -> C[1,1]
    ]
    return (np.array(U_data, dtype=float),
            np.array(V_data, dtype=float),
            np.array(W_data, dtype=float))

def _verify_decomposition(U, V, W, tol=1e-9):
    """Check that sum_c U[:,c] outer W[:,c] with V reproduces the 2x2 matmul tensor.
    Tensor T[i,j,k] = sum_{a,b} A[i,a] B[a,k] = (i,j,k) entry.
    Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  -- NO.
    Correct: C[i,k] = sum_j A[i,j] B[j,k].
    Tensor form: T[i,j,k] (where T is the bilinear map A x B -> C) has entries
    T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    In rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    is NOT right either. The correct tensor network:
    C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] ... no.
    
    Actually the standard tensor decomposition for matrix multiplication:
    The matmul tensor M has M[i,j,k,l] = delta_{i,k} * delta_{j,l}? No.
    
    Let me be precise. We want: for all A (n x m) and B (m x p),
    C = A @ B, so C[i,k] = sum_j A[i,j] B[j,k].
    
    Rank-r decomposition means there exist U (n*m x r), V (m*p x r), W (p*n x r)
    such that for all i,j,k: the coefficient of A[i,j]*B[j,k] in C[i,k] equals
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... 
    
    No. The correct formulation:
    C[i,k] = sum_j A[i,j] B[j,k]
    = sum_j sum_c (U[i*m+j, c] * something) ... 
    
    The standard: define tensor T with T[i,j,k] = 1 if (i,j,k) is a valid triple
    for matrix multiplication, i.e., the term A[i,j]*B[j,k] appears in C[i,k].
    So T[i,j,k] = delta_{i,k} is wrong. T[i,j,k] should be 1 for all valid (i,j,k)
    triples where A[i,j]*B[j,k] contributes to C[i,k]. Since every (i,j,k) triple
    has A[i,j]*B[j,k] contributing to C[i,k], T[i,j,k] = 1 for all i,j,k? No!
    
    Wait: C[i,k] = sum_j A[i,j]*B[j,k]. The tensor of the bilinear map has
    entries indexed by (i, j, k) where i indexes rows of A/C, j indexes cols of A
    and rows of B, k indexes cols of B/C. The entry T[i,j,k] = 1 means A[i,j]*B[j,k]
    contributes to C[i,k]. Since it always does, T[i,j,k] = 1 for all i,j,k in range.
    
    But that's a 2x2x2 all-ones tensor, which has rank 1! That can't be right.
    
    I'm confusing myself. Let me use the standard definition from the literature:
    The matrix multiplication tensor M_{n,m,p} is in (F^n)^* ⊗ (F^m)^* ⊗ (F^p)^*
    ⊗ (F^n ⊗ F^p)^* ... no, it's simpler.
    
    Standard: M ∈ (F^{n×m})^* ⊗ (F^{m×p})^* ⊗ (F^{n×p})^*
    M(A,B) = A@B.
    
    In coordinates with A flattened to (nm,) and B to (mp,), C to (np,):
    M[α, β, γ] where α = i*m+j, β = j*p+k, γ = k*n+i
    M[α,β,γ] = 1 if the triple (i,j,k) is consistent (same j for A and B, same i,k for C).
    
    So M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    
    A rank-r decomposition: M = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    where u_c ∈ F^{nm}, v_c ∈ F^{mp}, w_c ∈ F^{np}.
    
    So: M[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And for invalid triples, M = 0 = sum_c ... (must also hold).
    
    So verification: for all i,j,k in {0,1}, check
    sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] == 1
    And for all other index combinations (where the triple is inconsistent), the sum should be 0.
    But "other index combinations" in the flattened indices: α = i'*2+j', β = j''*2+k'', γ = k''*2+i''
    where j' != j'' or i' != i'' or k' != k''. For those, sum_c U[α,c]*V[β,c]*W[γ,c] should be 0.
    """
    n, m, p = 2, 2, 2
    r = U.shape[1]
    # Check all 2x2x2 = 8 valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    # Check invalid triples: total (nm)*(mp)*(np) = 4*4*4 = 64 entries
    # Valid ones are 8, invalid ones are 56
    for a in range(n*m):
        for b in range(m*p):
            for g in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(g, n)
                if j == j2 and i == i2 and k == k2:
                    continue  # valid triple, already checked
                s = 0.0
                for c in range(r):
                    s += U[a, c] * V[b, c] * W[g, c]
                if abs(s) > tol:
                    return False
    return True

def _try_heisenberg_rank6(t_start, deadline):
    """Attempt to construct a rank-6 decomposition using Heisenberg Lie algebra structure constants.
    The 3D Heisenberg algebra has basis {X, Y, Z} with [X,Y]=Z, [X,Z]=0, [Y,Z]=0.
    Structure constants: f_{XY}^Z = 1, f_{YX}^Z = -1, all others 0.
    
    We parameterize a rank-6 candidate by exploiting the antisymmetric structure constants
    and check if it yields a valid matmul decomposition. This is a heuristic search within
    the Heisenberg-inspired family.
    """
    # Heisenberg structure constants (antisymmetric in lower indices)
    # f_{ab}^c for a,b,c in {0,1,2}
    # f_{0,1}^2 = 1, f_{1,0}^2 = -1, rest 0
    # 
    # For rank-6 matmul, we need 6 rank-1 tensors. The Heisenberg algebra gives us
    # a natural 3-dimensional structure. We attempt to map the 3 structure constants
    # to 6 bilinear forms (3 positive, 3 negative by antisymmetry).
    #
    # This is a highly constrained search. We'll try a few parameterizations.
    
    best_result = None
    
    # Parameterization 1: Use the Heisenberg commutator structure directly
    # Map 3D algebra to 2x2 matrices via Pauli matrices
    # X -> sigma_x, Y -> sigma_y, Z -> sigma_z (or identity)
    # The commutator [sigma_i, sigma_j] = 2i epsilon_{ijk} sigma_k
    
    # Try constructing U, V, W from Pauli matrix entries
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Use real parts and construct rank-6 candidate
    # This is a heuristic; we check validity
    candidates = []
    
    # Candidate: 6 products based on Heisenberg structure
    # M1 = A[0,0]*B[0,0] + A[1,1]*B[1,1]  (trace-like)
    # M2 = A[0,0]*B[1,1] + A[1,1]*B[0,0]
    # M3 = A[0,1]*B[0,0] + A[1,0]*B[1,0]
    # M4 = A[0,1]*B[1,0] + A[1,0]*B[0,1]
    # M5 = A[0,0]*B[0,1] + A[1,1]*B[1,0]
    # M6 = A[0,1]*B[0,1] + A[1,0]*B[1,1]
    
    U_cand = np.zeros((4, 6))
    V_cand = np.zeros((4, 6))
    W_cand = np.zeros((4, 6))
    
    # Product 1: A[0,0]*B[0,0] + A[1,1]*B[1,1]
    U_cand[0, 0] = 1; U_cand[3, 0] = 1  # A[0,0], A[1,1]
    V_cand[0, 0] = 1; V_cand[3, 0] = 1  # B[0,0], B[1,1]
    W_cand[0, 0] = 1; W_cand[3, 0] = 1  # C[0,0], C[1,1]
    
    # Product 2: A[0,0]*B[1,1] + A[1,1]*B[0,0]
    U_cand[0, 1] = 1; U_cand[3, 1] = 1
    V_cand[3, 1] = 1; V_cand[0, 1] = 1
    W_cand[0, 1] = 1; W_cand[3, 1] = 1
    
    # Product 3: A[0,1]*B[0,0] + A[1,0]*B[1,0]
    U_cand[1, 2] = 1; U_cand[2, 2] = 1
    V_cand[0, 2] = 1; V_cand[2, 2] = 1
    W_cand[0, 2] = 1; W_cand[3, 2] = 1
    
    # Product 4: A[0,1]*B[1,0] + A[1,0]*B[0,1]
    U_cand[1, 3] = 1; U_cand[2, 3] = 1
    V_cand[2, 3] = 1; V_cand[1, 3] = 1
    W_cand[0, 3] = 1; W_cand[3, 3] = 1
    
    # Product 5: A[0,0]*B[0,1] + A[1,1]*B[1,0]
    U_cand[0, 4] = 1; U_cand[3, 4] = 1
    V_cand[1, 4] = 1; V_cand[2, 4] = 1
    W_cand[0, 4] = 1; W_cand[3, 4] = 1
    
    # Product 6: A[0,1]*B[0,1] + A[1,0]*B[1,1]
    U_cand[1, 5] = 1; U_cand[2, 5] = 1
    V_cand[1, 5] = 1; V_cand[3, 5] = 1
    W_cand[0, 5] = 1; W_cand[3, 5] = 1
    
    if _verify_decomposition(U_cand, V_cand, W_cand):
        best_result = (U_cand, V_cand, W_cand)
    
    # Check deadline
    if time.time() - t_start > deadline:
        return None
    
    # Try more parameterizations with signs
    for sign_pattern in range(4):
        if time.time() - t_start > deadline:
            break
        U_s = U_cand.copy()
        V_s = V_cand.copy()
        W_s = W_cand.copy()
        # Flip signs in various ways
        if sign_pattern & 1:
            U_s[:, 1] *= -1
            V_s[:, 1] *= -1
        if sign_pattern & 2:
            U_s[:, 3] *= -1
            W_s[:, 3] *= -1
        if _verify_decomposition(U_s, V_s, W_s):
            best_result = (U_s, V_s, W_s)
            break
    
    return best_result

def construct():
    """
    Attempts rank-6 via Heisenberg Lie algebra parameterization;
    falls back to rank-7 Strassen if rank-6 fails.
    """
    t_start = time.time()
    deadline = t_start + _DEADLINE_SECONDS
    
    # Try rank-6 Heisenberg-based construction
    result6 = _try_heisenberg_rank6(t_start, deadline)
    if result6 is not None:
        U, V, W = result6
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-7 Strassen
    U, V, W = _strassen_rank7()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
