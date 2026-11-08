"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random
from fractions import Fraction

def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_28(deadline):
    """Try to find a rank-28 decomposition for 2x3x5 matrix multiplication.
    
    Strategy: Start from schoolbook rank-30, then try to merge pairs of terms.
    For each pair of terms (i1,j1,k1) and (i2,j2,k2) that we want to merge,
    we modify the corresponding columns to create a 2-term replacement that
    still satisfies all 30 equations.
    
    Key insight: If we have terms at positions (i,j,k) and (i',j',k'), we can
    try to replace them with a 2-dimensional subspace that covers both.
    """
    n, m, p = DIMENSIONS
    r = 28
    
    # Initialize with schoolbook pattern
    # We'll use a sparse structure: most columns are one-hot, 
    # but we'll try to create dependencies.
    
    # Approach: For each pair of "mergeable" terms, try to construct a valid decomposition.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if we can find
    # two vectors that reproduce both contributions.
    
    # Let's try a parameterized approach:
    # For a pair of terms to merge, we set:
    #   U[i1,j1,:] = [1, a], U[i2,j2,:] = [0, b]  (and others zero for these cols)
    #   V[j1,k1,:] = [1, c], V[j2,k2,:] = [0, d]
    #   W[k1,i1,:] = [1, e], W[k2,i2,:] = [0, f]
    # Then we need: 1*1*1 + a*c*e = 1 and 0 + b*d*f = 1
    # And all other (i,j,k) triples must still sum to 1.
    
    # Actually, a cleaner approach: use the fact that for rank r, we have r degrees of freedom.
    # Let's try random initialization with structure and use iterative refinement.
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    num_attempts = 50
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Create a structured random initialization
        # Most entries are 0, but we'll have some non-zero pattern
        
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        
        # Start with a base pattern: assign each of the 30 terms to a column
        # but allow some columns to serve multiple terms
        col_assign = list(range(30))
        random.shuffle(col_assign)
        
        # Map (i,j,k) -> column
        term_to_col = {}
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term_to_col[(i, j, k)] = col_assign[idx]
                    idx += 1
        
        # Initialize: each term gets its assigned column with value 1
        for (i, j, k), c in term_to_col.items():
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
        # Now try to reduce rank by creating dependencies
        # Pick 2 columns to merge: we'll try to make them linearly dependent in a way
        # that still satisfies all equations.
        
        # For now, try a simple perturbation approach
        # Add small random perturbations to some entries and see if we can maintain validity
        
        # Actually, let's try a different approach: 
        # Use the fact that we have 30 constraints and 31*28 = 868 unknowns.
        # Fix 28*31 - 30 = 838 variables arbitrarily (e.g., to 0 or 1),
        # and solve for the remaining 30.
        
        # Simpler: try specific structured patterns
        # Pattern: for each output entry (i,k), the sum over j should be 1.
        # We can group terms by (i,k) and try to create shared structure.
        
        # Let's try: for a fixed (i,k), the j=0 and j=1 terms can share a column
        # if we set up the right values.
        
        # Try merging terms that share the same (i,k) but different j
        merged = 0
        for i in range(n):
            for k in range(p):
                if time.time() > deadline:
                    break
                # Terms (i,0,k) and (i,1,k) share output position (i,k)
                # Try to merge them
                c1 = term_to_col[(i, 0, k)]
                c2 = term_to_col[(i, 1, k)]
                
                if c1 == c2:
                    continue  # Already in same column
                
                # Try to merge: set up so that column c1 handles both
                # U[i,0,c1] and U[i,1,c1] both non-zero
                # V[0,k,c1] and V[1,k,c1] both non-zero
                # W[k,i,c1] shared
                
                # This is getting complex. Let's use a numerical solver approach.
                pass
        
        # Alternative: use scipy or manual Gauss-Jordan for a subset
        # For now, return None if we can't find it quickly
        pass
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank 30)
    r_school = n * m * p
    U_school = np.zeros((n * m, r_school))
    V_school = np.zeros((m * p, r_school))
    W_school = np.zeros((p * n, r_school))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
                c += 1
    
    # Try to find rank 28
    # Use a systematic search: try all pairs of terms to merge
    found_rank_28 = False
    
    # Strategy: For rank 28, we need to eliminate 2 independent terms.
    # We'll try to find a 2-dimensional correction to the schoolbook decomposition.
    
    # Let U = U_school[:, :28] + delta_U, etc.
    # But this doesn't directly work because we need to remove 2 columns.
    
    # Better approach: construct rank-28 directly.
    # For each of the 28 columns, assign it to cover one or more terms.
    
    # Let's try a constructive approach based on the structure of 2x3x5.
    # The output is 2x5, so there are 10 output entries.
    # Each output entry (i,k) depends on 3 input pairs (i,j,k) for j=0,1,2.
    
    # Key idea: For output entry (i,k), we need:
    #   sum_j A[i,j] * B[j,k] = C[i,k]
    # In tensor form: sum_j U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for each (i,j,k)
    
    # For rank 28, we can afford to have 2 "shared" columns that each handle 2 terms.
    
    # Let's try: pick two pairs of terms to share columns.
    # Term pairs that can share: terms with same (i,k) but different j, or same (j,k) but different i, etc.
    
    # Try all combinations of 2 pairs to merge
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    term_idx = {t: idx for idx, t in enumerate(terms)}
    
    # Find all valid pairs to merge (terms that can share a column)
    # Two terms (i1,j1,k1) and (i2,j2,k2) can share a column if:
    # We can assign values such that both are satisfied.
    
    # For simplicity, try merging terms that differ in exactly one index.
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = r_school
    
    # Try to construct rank 28 by merging 2 pairs
    # For each pair of terms to merge, solve the linear system
    
    # Let's try a specific approach: 
    # Merge (0,0,0) with (0,1,0) and (1,0,1) with (1,1,1) as an example
    # Then verify and adjust
    
    # Actually, let's use a more general numerical approach:
    # Set up the system and use least squares with sparsity constraints
    
    # For rank r=28, we have 30 equations:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of 30 cubic equations in 31*28=868 unknowns.
    # We can fix most variables and solve for the rest.
    
    # Strategy: 
    # 1. Assign 26 columns to be "standard" (one term each)
    # 2. Use 2 columns to handle 2 pairs of terms
    # 3. Solve for the values in those 2 special columns
    
    # Let's try all ways to pick 2 pairs of terms to merge
    # A pair is valid if the two terms can be handled by one column
    
    # For terms (i1,j1,k1) and (i2,j2,k2) to share column c:
    # We need U[i1*m+j1,c]*V[j1*p+k1,c]*W[k1*n+i1,c] = 1
    # and U[i2*m+j2,c]*V[j2*p+k2,c]*W[k2*n+i2,c] = 1
    # And for all other (i,j,k) not in {these two}, the contribution from col c should be 0.
    
    # This means: for any (i,j,k) not equal to either term, at least one of
    # U[i*m+j,c], V[j*p+k,c], W[k*n+i,c] must be 0.
    
    # Let's try to find such columns.
    
    # For a column to handle terms (i1,j1,k1) and (i2,j2,k2):
    # If (i1,j1) != (i2,j2), then U has two non-zero entries.
    # If (j1,k1) != (j2,k2), then V has two non-zero entries.
    # If (k1,i1) != (k2,i2), then W has two non-zero entries.
    
    # The constraint is that no other (i,j,k) triple can have all three non-zero.
    
    # Let's enumerate possible merges:
    # For simplicity, try merging terms that differ in exactly one coordinate.
    
    merge_candidates = []
    for idx1 in range(30):
        for idx2 in range(idx1+1, 30):
            t1 = terms[idx1]
            t2 = terms[idx2]
            # Count how many coordinates differ
            diff = sum(1 for a, b in zip(t1, t2) if a != b)
            if diff == 1:
                merge_candidates.append((idx1, idx2))
    
    # Try each merge candidate
    for (idx1, idx2) in merge_candidates:
        if time.time() > deadline:
            break
        
        t1 = terms[idx1]
        t2 = terms[idx2]
        
        # Build a rank-28 decomposition with these two terms sharing a column
        # Columns 0-25: standard terms (the 26 terms not in {idx1, idx2})
        # Column 26: handles t1 and t2
        # Column 27: we need one more... wait, we have 30 terms, merging 2 pairs gives 28 terms.
        # So we need to merge 2 pairs to get from 30 to 28.
        
        # Let's pick a second merge pair
        # Find another valid merge candidate that doesn't conflict
        for (idx3, idx4) in merge_candidates:
            if time.time() > deadline:
                break
            if idx3 in (idx1, idx2) or idx4 in (idx1, idx2):
                continue
            
            t3 = terms[idx3]
            t4 = terms[idx4]
            
            # Check if the two merges are compatible (no overlapping terms)
            # They are compatible since we checked indices
            
            # Now construct the decomposition
            U = np.zeros((n * m, 28), dtype=float)
            V = np.zeros((m * p, 28), dtype=float)
            W = np.zeros((p * n, 28), dtype=float)
            
            # Assign standard terms to columns 0-25
            standard_terms = [t for idx, t in enumerate(terms) 
                           if idx not in (idx1, idx2, idx3, idx4)]
            
            for col, (i, j, k) in enumerate(standard_terms):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
            
            # Column 26: handle t1 and t2
            i1, j1, k1 = t1
            i2, j2, k2 = t2
            U[i1 * m + j1, 26] = 1.0
            U[i2 * m + j2, 26] = 1.0
            V[j1 * p + k1, 26] = 1.0
            V[j2 * p + k2, 26] = 1.0
            W[k1 * n + i1, 26] = 1.0
            W[k2 * n + i2, 26] = 1.0
            
            # But this gives value 1 for both terms, and also 1 for any (i,j,k) where
            # all three are non-zero. We need to check if this creates spurious contributions.
            
            # For term (i1,j1,k1): U[i1*m+j1,26]*V[j1*p+k1,26]*W[k1*n+i1,26] = 1*1*1 = 1 ✓
            # For term (i2,j2,k2): U[i2*m+j2,26]*V[j2*p+k2,26]*W[k2*n+i2,26] = 1*1*1 = 1 ✓
            # For other terms: need to check if any (i,j,k) not in {t1,t2} has all three non-zero.
            
            # Check for spurious contributions
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) == t1 or (i, j, k) == t2:
                            continue
                        u_val = U[i * m + j, 26]
                        v_val = V[j * p + k, 26]
                        w_val = W[k * n + i, 26]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Column 27: handle t3 and t4
            i3, j3, k3 = t3
            i4, j4, k4 = t4
            U[i3 * m + j3, 27] = 1.0
            U[i4 * m + j4, 27] = 1.0
            V[j3 * p + k3, 27] = 1.0
            V[j4 * p + k4, 27] = 1.0
            W[k3 * n + i3, 27] = 1.0
            W[k4 * n + i4, 27] = 1.0
            
            # Check for spurious contributions from column 27
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) == t3 or (i, j, k) == t4:
                            continue
                        u_val = U[i * m + j, 27]
                        v_val = V[j * p + k, 27]
                        w_val = W[k * n + i, 27]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Also check cross-interactions between columns 26 and 27
            # For any (i,j,k), the total contribution should be 1.
            # Standard columns contribute 1 if (i,j,k) is a standard term.
            # Column 26 contributes 1 if (i,j,k) is t1 or t2 (and no spurious).
            # Column 27 contributes 1 if (i,j,k) is t3 or t4 (and no spurious).
            
            # Verify the full decomposition
            if _verify(U, V, W):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = 28
                found_rank_28 = True
                break
        
        if found_rank_28:
            break
    
    if found_rank_28:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback: schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
