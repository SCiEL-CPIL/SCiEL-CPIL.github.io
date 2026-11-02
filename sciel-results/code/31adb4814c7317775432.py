"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
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
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def try_rank_with_support(U, V, rank, time_limit=5.0):
        """
        Given fixed U and V matrices (nm x rank, mp x rank), solve for W.
        The constraint is: sum_c U[a,c] * V[b,c] * W[c_idx, c] = target[a,b,c_idx]
        for all valid (a,b,c_idx) combinations.
        
        This is a linear system in W. We set up the system and solve it.
        """
        if time.time() > start_time + time_limit:
            return None
        
        # Build the linear system: for each (a, b) pair that corresponds to valid (i,j,k),
        # we have: sum_c U[a,c] * V[b,c] * W[c_idx, c] = 1
        # where c_idx = k*n + i
        
        # Collect all equations
        equations = []  # list of (row_vector, rhs)
        # row_vector is a vector of length pn*rank where element (c_idx, c) maps to W[c_idx, c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    
                    # Equation: sum_c U[a,c] * V[b,c] * W[c_idx, c] = 1
                    row = np.zeros(pn * rank)
                    for c in range(rank):
                        coeff = U[a, c] * V[b, c]
                        row[c_idx * rank + c] = coeff
                    equations.append((row, 1.0))
        
        # Also need to ensure that for invalid combinations, the result is 0
        # But actually, the tensor only has non-zero entries at valid (i,j,k) positions,
        # and the decomposition automatically gives 0 elsewhere if the system is consistent.
        # Wait, no - we need to check ALL (a, b, c_idx) combinations.
        # For a given (a, b, c_idx), the value is sum_c U[a,c] * V[b,c] * W[c_idx, c]
        # This should be 1 if (a,b,c_idx) corresponds to a valid (i,j,k), else 0.
        
        # Let's add zero constraints for all other (a, b, c_idx) combinations
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    valid_set.add((a, b, c_idx))
        
        for a in range(nm):
            for b in range(mp):
                for c_idx in range(pn):
                    if (a, b, c_idx) in valid_set:
                        continue
                    row = np.zeros(pn * rank)
                    for c in range(rank):
                        coeff = U[a, c] * V[b, c]
                        row[c_idx * rank + c] = coeff
                    equations.append((row, 0.0))
        
        # Build the full system
        num_eqs = len(equations)
        A = np.zeros((num_eqs, pn * rank))
        rhs = np.zeros(num_eqs)
        for idx, (row, val) in enumerate(equations):
            A[idx] = row
            rhs[idx] = val
        
        # Solve the linear system using least squares
        if time.time() > start_time + time_limit:
            return None
        
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, rhs, rcond=None)
        except Exception:
            return None
        
        if time.time() > start_time + time_limit:
            return None
        
        W = W_flat.reshape(pn, rank)
        
        # Verify the solution
        if verify(U, V, W, tol=1e-6):
            return W
        
        # Try rounding to half-integers
        W_rounded = round_to_half_integer(W)
        if verify(U, V, W_rounded, tol=1e-6):
            return W_rounded
        
        return None
    
    def generate_sparse_U(rank, sparsity_level=0.3):
        """Generate a sparse U matrix with random entries."""
        U = np.zeros((nm, rank))
        for c in range(rank):
            # Each column of U has a few non-zero entries
            num_nz = max(1, int(nm * sparsity_level))
            nz_indices = np.random.choice(nm, size=num_nz, replace=False)
            for idx in nz_indices:
                U[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0, 1.5, 2.0])
        return U
    
    def generate_sparse_V(rank, sparsity_level=0.2):
        """Generate a sparse V matrix with random entries."""
        V = np.zeros((mp, rank))
        for c in range(rank):
            # Each column of V has a few non-zero entries
            num_nz = max(1, int(mp * sparsity_level))
            nz_indices = np.random.choice(mp, size=num_nz, replace=False)
            for idx in nz_indices:
                V[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0, 1.5, 2.0])
        return V
    
    # Try to find a low-rank decomposition
    best_result = None
    best_rank = 42  # default to schoolbook
    
    # Try ranks from 30 up to 41
    for rank in range(30, 42):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations of U and V
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            U = generate_sparse_U(rank, sparsity_level=0.3)
            V = generate_sparse_V(rank, sparsity_level=0.2)
            
            W = try_rank_with_support(U, V, rank, time_limit=3.0)
            
            if W is not None:
                best_result = (U, V, W)
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    # Convert to lists as required
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
