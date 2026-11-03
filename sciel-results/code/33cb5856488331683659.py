import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 115  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    def build_68_rank():
        """Build the deterministic rank-68 decomposition using 4 Strassen blocks."""
        rank = 68
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        U_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]
        
        for g in range(4):
            k0 = 2 * g
            k1 = 2 * g + 1
            
            for l in range(7):
                for li, coeff in U_coeffs[l].items():
                    i_local = li // 2
                    j_local = li % 2
                    U[i_local*m + j_local, col] += coeff
                for lj, coeff in V_coeffs[l].items():
                    j_local = lj // 2
                    k_local = lj % 2
                    V[j_local*p + (k0 + k_local), col] += coeff
                for lc, coeff in W_coeffs[l].items():
                    i_local = lc // 2
                    k_local = lc % 2
                    W[(k0 + k_local)*n + i_local, col] += coeff
                col += 1
            
            for i in range(2):
                for k in [k0, k1]:
                    U[i*m + 2, col] = 1.0
                    V[2*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
            for j in range(2):
                for k in [k0, k1]:
                    U[2*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + 2, col] = 1.0
                    col += 1
            for k in [k0, k1]:
                U[2*m + 2, col] = 1.0
                V[2*p + k, col] = 1.0
                W[k*n + 2, col] = 1.0
                col += 1
        
        return U, V, W
    
    def try_reduce_rank(U0, V0, W0, max_reduction=4):
        """
        Try to reduce rank by finding linear dependencies among terms.
        Uses the pencil method: fix k, look at 3x3 matrices, find span structure.
        
        Strategy: For each pair of terms, check if one can be expressed as a 
        linear combination of others (in the tensor sense). If so, eliminate it.
        """
        rank = U0.shape[1]
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank
        
        # Compute tensor contribution of each term
        term_tensors = np.zeros((rank, n, m, p))
        for c in range(rank):
            u_mat = U0[:, c].reshape(n, m)
            v_mat = V0[:, c].reshape(m, p)
            w_mat = W0[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        term_tensors[c, i, j, k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        term_flat = term_tensors.reshape(rank, -1)
        
        # Try to eliminate terms by finding linear dependencies
        # A term c can be eliminated if term_flat[c] is in the span of others
        # AND the coefficients are "nice" (small integers/half-integers)
        
        for c in range(rank):
            if time_left() < 3:
                break
            
            others = [x for x in range(rank) if x != c]
            A = term_flat[others].T  # (n*m*p, rank-1)
            b = term_flat[c]
            
            try:
                x, residuals, r, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.max(np.abs(A @ x - b)) < 1e-10:
                    # Check if x has nice values
                    x_rounded = np.round(x * 2) / 2
                    if np.max(np.abs(A @ x_rounded - b)) < 1e-10:
                        # Found a dependency! Term c = sum x_i * term_i for i in others
                        # But this doesn't directly give us a lower-rank decomposition
                        # We need to check if the resulting U,V,W entries are still nice
                        
                        # Actually, for matrix multiplication tensor, if term c is a 
                        # linear combination of others in tensor space, we can't simply
                        # remove it because the decomposition requires specific U,V,W structure.
                        # 
                        # Instead, let's try a different approach: look for terms that
                        # have sparse structure allowing merging.
                        pass
            except Exception:
                pass
        
        # Alternative: try to find a 3x3x2 block with rank < 17
        # This is the key insight from the pencil method
        
        return best_U, best_V, best_W, best_rank
    
    def find_rank_16_block():
        """
        Use the pencil method to find a rank-16 decomposition for 3x3x2.
        
        The pencil method: fix k=0 and k=1, get two 3x3 matrices M_0 and M_1.
        The tensor for 3x3x2 has rank r if and only if the combined structure
        allows r terms.
        
        For 3x3x2, we know rank 17 works. Let's try to find rank 16 by 
        exploiting the rectangular geometry.
        
        Key idea: The schoolbook terms for i=2 or j=2 rows might be mergeable
        with Strassen terms if we allow half-integer coefficients.
        """
        rank = 16
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Start with Strassen (7 terms)
        U_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]
        
        col = 0
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Now we need 9 more terms for the schoolbook parts (i=2 or j=2)
        # Instead of 10 schoolbook terms, try to merge one with a Strassen term
        
        # The schoolbook terms needed:
        # i∈{0,1}, j=2, k∈{0,1}: 2 terms
        # i=2, j∈{0,1}, k∈{0,1}: 2 terms  
        # i=2, j=2, k∈{0,1}: 2 terms
        # Total: 6 terms... wait, let me recount.
        
        # For 3x3x2:
        # All (i,j,k) pairs: 3*3*2 = 18
        # Strassen covers: i∈{0,1}, j∈{0,1}, k∈{0,1} = 4*2 = 8 entries (via 7 terms)
        # Remaining: 
        #   i∈{0,1}, j=2, k∈{0,1}: 2*2 = 4 entries
        #   i=2, j∈{0,1}, k∈{0,1}: 2*2 = 4 entries
        #   i=2, j=2, k∈{0,1}: 1*2 = 2 entries
        # Total remaining: 10 entries, need at least 10 terms if schoolbook
        
        # So we need 7 + 10 = 17 terms minimum with this structure.
        # To get 16, we need to merge one schoolbook term with a Strassen term.
        
        # Try: modify one Strassen term to also cover one schoolbook entry
        # This requires the modified term to have non-zero entries in both
        # the Strassen region and the schoolbook region.
        
        # Let's try a brute-force search over small modifications
        # This is expensive, so let's be smart about it.
        
        # Actually, for this specific problem, rank 16 for 3x3x2 is not known
        # to be achievable. Let's just return the rank-17 block.
        return None
    
    # Build the base 68-rank solution
    U, V, W = build_68_rank()
    
    if not validate(U, V, W):
        return schoolbook()
    
    # Try to improve using pencil method insights
    # For now, the 68-rank solution is our best deterministic construction
    # The pencil method analysis suggests that cross-block optimization is
    # limited by the structure of the 3x3 matrix multiplication.
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result
