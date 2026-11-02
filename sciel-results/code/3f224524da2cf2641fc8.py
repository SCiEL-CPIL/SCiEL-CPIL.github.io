import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import product
from collections import defaultdict
from fractions import Fraction
from sympy import Matrix, Rational, symbols, eye, zeros

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Uses block-Strassen decomposition as starting point, then searches for linear
    dependencies among rank-1 terms to reduce rank.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    T_target = np.ones((n, m, p))
    
    def get_block_strassen_36():
        terms = []
        def add_strassen_2x2x2(i0, j0, k0):
            terms.append((
                {(i0, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(i0, k0): 1, (i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1},
                {(i0+1, k0): 1, (i0+1, k0+1): -1}
            ))
            terms.append((
                {(i0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(i0, k0+1): 1, (i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(i0, k0): 1, (i0+1, k0): 1}
            ))
            terms.append((
                {(i0, j0): 1, (i0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(i0, k0): -1, (i0, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0): 1, (i0, j0): -1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0, j0+1): 1, (i0+1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(i0, k0): 1}
            ))
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            terms.append(( {(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1} ))
            terms.append(( {(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1} ))
            terms.append(( {(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1} ))
            terms.append(( {(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1} ))

        # Block decomposition for 2x4 * 4x5
        # Split m=4 into 2+2, p=5 into 4+1
        add_strassen_2x2x2(0, 0, 0)
        add_strassen_2x2x2(0, 2, 0)
        add_strassen_2x2x2(0, 0, 2)
        add_strassen_2x2x2(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return terms

    def terms_to_matrices(terms):
        R = len(terms)
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for r, (u_d, v_d, w_d) in enumerate(terms):
            for (i, j), val in u_d.items():
                U[i*m + j, r] += val
            for (j, k), val in v_d.items():
                V[j*p + k, r] += val
            for (i, k), val in w_d.items():
                W[k*n + i, r] += val
        return U, V, W

    def compute_tensor(U, V, W):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return T

    def verify_decomposition(U, V, W, tol=1e-9):
        T_comp = compute_tensor(U, V, W)
        return np.allclose(T_comp, T_target, atol=tol)

    def try_reduce_rank_by_dependency(terms, max_remove=5):
        """
        Try to find linear dependencies among the rank-1 terms.
        A dependency means sum_r alpha_r * term_r = 0 for some alpha_r not all zero.
        If we find such a dependency, we can potentially eliminate one term.
        """
        R = len(terms)
        if R <= 1:
            return None
        
        # Build the tensor representation as a 3D array of shape (n*m, m*p, p*n)
        # Each term is a rank-1 tensor: u_r ⊗ v_r ⊗ w_r
        # We want to check if these R vectors are linearly dependent in R^(n*m*m*p*p*n)
        
        # For efficiency, use the fact that a rank-1 tensor u⊗v⊗w can be represented
        # by its flattening. Check linear dependence via rank of the matrix of flattened terms.
        
        flattened_terms = []
        for u_d, v_d, w_d in terms:
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            for (i, j), val in u_d.items():
                u_vec[i*m + j] = val
            for (j, k), val in v_d.items():
                v_vec[j*p + k] = val
            for (i, k), val in w_d.items():
                w_vec[k*n + i] = val
            # Kronecker product of the three vectors
            flat = np.kron(np.kron(u_vec, v_vec), w_vec)
            flattened_terms.append(flat)
        
        # Stack into a matrix: each row is a flattened term
        M = np.array(flattened_terms)  # shape (R, n*m*m*p*p*n)
        
        # Check rank using SVD
        try:
            U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
            rank = np.sum(S > 1e-10)
            
            if rank < R:
                # There are linear dependencies
                # The null space vectors give the dependencies
                # Vt has shape (R, R), last (R-rank) rows span the null space
                null_dim = R - rank
                null_vectors = Vt[rank:]  # shape (null_dim, R)
                
                # For each null vector, try to eliminate the term with largest coefficient
                best_result = None
                for nv in null_vectors:
                    # Find the index with largest absolute coefficient
                    idx_max = np.argmax(np.abs(nv))
                    if abs(nv[idx_max]) < 1e-10:
                        continue
                    
                    # Try removing term idx_max
                    remaining = [t for i, t in enumerate(terms) if i != idx_max]
                    
                    # Verify the remaining terms still give the correct tensor
                    U_rem, V_rem, W_rem = terms_to_matrices(remaining)
                    if verify_decomposition(U_rem, V_rem, W_rem):
                        if best_result is None or len(remaining) < len(best_result):
                            best_result = remaining
                
                return best_result
        except Exception:
            pass
        
        return None

    def try_direct_rank_reduction(terms, target_rank):
        """
        Try to find a subset of 'target_rank' terms that still decompose the tensor.
        This is a combinatorial search, so we use heuristics.
        """
        R = len(terms)
        if target_rank >= R:
            return terms if verify_decomposition(*terms_to_matrices(terms)) else None
        
        if time.time() > deadline - 5:
            return None
        
        # Try removing terms one at a time, greedily
        current_terms = list(terms)
        
        # First, try to find if any single term can be removed
        for i in range(len(current_terms)):
            candidate = current_terms[:i] + current_terms[i+1:]
            U_c, V_c, W_c = terms_to_matrices(candidate)
            if verify_decomposition(U_c, V_c, W_c):
                current_terms = candidate
                break
        
        # Continue trying to remove more terms
        improved = True
        while improved and len(current_terms) > target_rank:
            improved = False
            for i in range(len(current_terms)):
                candidate = current_terms[:i] + current_terms[i+1:]
                U_c, V_c, W_c = terms_to_matrices(candidate)
                if verify_decomposition(U_c, V_c, W_c):
                    current_terms = candidate
                    improved = True
                    break
        
        if len(current_terms) <= target_rank:
            return current_terms
        return None

    # Generate initial 36 terms
    init_terms = get_block_strassen_36()
    U0, V0, W0 = terms_to_matrices(init_terms)
    
    # Verify the initial decomposition
    if not verify_decomposition(U0, V0, W0):
        # Fallback: return the unverified decomposition (should not happen)
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    best_terms = init_terms
    best_rank = 36
    
    # Step 1: Try dependency-based reduction
    if time.time() < deadline - 30:
        reduced = try_reduce_rank_by_dependency(init_terms, max_remove=5)
        if reduced is not None and len(reduced) < best_rank:
            best_terms = reduced
            best_rank = len(reduced)
    
    # Step 2: Try greedy direct reduction for various target ranks
    for target_rank in range(best_rank - 1, 25, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_direct_rank_reduction(best_terms, target_rank)
        if result is not None:
            best_terms = result
            best_rank = len(result)
    
    # Step 3: Try a few random subsets as additional heuristic
    if time.time() < deadline - 20:
        np.random.seed(42)
        for target_rank in range(best_rank - 1, 28, -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Random subset of size target_rank
                idx = np.random.choice(len(best_terms), size=target_rank, replace=False)
                candidate = [best_terms[i] for i in sorted(idx)]
                
                U_c, V_c, W_c = terms_to_matrices(candidate)
                if verify_decomposition(U_c, V_c, W_c):
                    best_terms = candidate
                    best_rank = target_rank
                    found = True
                    break
    
    U_final, V_final, W_final = terms_to_matrices(best_terms)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
