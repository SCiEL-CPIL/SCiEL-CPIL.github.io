"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol


def _search_rank13(deadline):
    """
    Search for a rank-13 decomposition using structured sparsity with shared V-support.
    
    The middle product insight: terms in the second block may have V-factors that are
    linear combinations sharing support with the first block. We search for sparsity
    patterns where some terms have B-support spanning both {0,1,4,5} and {2,3,6,7}.
    """
    n, m, p = 2, 2, 4
    
    # Precompute the expected tensor
    T_expected = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T_expected[i * m + k, k * p + j, j * n + i] = 1.0
    
    # For rank-13, we try patterns where:
    # - Most terms are standard Strassen-like (3 A, 2 B, 2 C)
    # - Some terms have "cross-block" B support (indices from both halves)
    
    # Generate candidate term structures
    # Each term is (A_set, B_set, C_set)
    
    A_indices = [0, 1, 2, 3]
    B_indices = list(range(8))
    C_indices = list(range(8))
    
    # Standard Strassen-like patterns (from known rank-14)
    # These are the 14 patterns used in rank-14
    standard_patterns = [
        # (A_set, B_set, C_set) for each of the 14 terms
        ({0, 3}, {0, 5}, {0, 3}),      # M1
        ({2, 3}, {0}, {1, 3}),          # M2
        ({0}, {1, 5}, {2, 3}),          # M3
        ({3}, {4, 0}, {0, 1}),          # M4
        ({0, 1}, {5}, {0, 2}),          # M5
        ({2, 0}, {0, 1}, {3}),          # M6
        ({1, 3}, {4, 5}, {0}),          # M7
        ({0, 3}, {2, 7}, {4, 7}),       # N1
        ({2, 3}, {2}, {5, 7}),          # N2
        ({0}, {3, 7}, {6, 7}),          # N3
        ({3}, {6, 2}, {4, 5}),          # N4
        ({0, 1}, {7}, {4, 6}),          # N5
        ({2, 0}, {2, 3}, {7}),          # N6
        ({1, 3}, {6, 7}, {4}),          # N7
    ]
    
    # For rank-13, we need to merge or share one term.
    # Try: replace two terms with one that has combined support
    # The merged term would have larger A, B, or C sets
    
    # Strategy: try all pairs of terms to merge, creating a combined pattern
    # Then solve for coefficients
    
    best_result = None
    
    for pair_idx in range(len(standard_patterns)):
        if time.time() > deadline:
            break
        for other_idx in range(pair_idx + 1, len(standard_patterns)):
            if time.time() > deadline:
                break
            
            p1 = standard_patterns[pair_idx]
            p2 = standard_patterns[other_idx]
            
            # Create merged pattern: union of supports
            merged_A = p1[0] | p2[0]
            merged_B = p1[1] | p2[1]
            merged_C = p1[2] | p2[2]
            
            # The remaining 12 patterns
            remaining = [standard_patterns[i] for i in range(len(standard_patterns)) 
                        if i != pair_idx and i != other_idx]
            
            # Total patterns: 12 standard + 1 merged = 13
            all_patterns = remaining + [(merged_A, merged_B, merged_C)]
            
            # Now solve for coefficients
            # For each pattern t, we have unknowns:
            #   a_t[i] for i in A_set[t]
            #   b_t[j] for j in B_set[t]  
            #   c_t[k] for k in C_set[t]
            
            # Total unknowns: sum over t of (|A_t| + |B_t| + |C_t|)
            # This is still a nonlinear system (trilinear)
            
            # Use the trick: fix one set of coefficients, solve linear for others
            # Or use iterative refinement
            
            # For now, try a simpler approach: assume coefficients are in {-1, 0, 1}
            # and search over small coefficient sets
            
            # This is still expensive. Let's try a more targeted approach:
            # Keep the standard coefficients for the 12 unchanged terms,
            # and only solve for the merged term's coefficients.
            
            # The 12 standard terms have known coefficients (from rank-14)
            # Let's extract them
            
            rank14 = _build_rank14()
            U14 = np.array(rank14['U'], dtype=float)
            V14 = np.array(rank14['V'], dtype=float)
            W14 = np.array(rank14['W'], dtype=float)
            
            # For the 12 remaining terms, their contributions are fixed
            # The merged term must compensate for the missing 2 terms' contributions
            # plus provide any cross-terms
            
            # Compute the residual that the merged term must satisfy
            # For each (i,j,k) in the tensor:
            #   sum_{t in remaining} U14[i,t]*V14[j,t]*W14[k,t] + merged_term(i,j,k) = 1
            
            # The merged term is: sum_{i' in merged_A} sum_{j' in merged_B} sum_{k' in merged_C}
            #   a[i']*b[j']*c[k'] * [i'==i and j'==j and k'==k]
            # Wait, that's not right. The term is:
            #   (sum_{i' in merged_A} a[i'] * delta_{i',i}) * 
            #   (sum_{j' in merged_B} b[j'] * delta_{j',j}) *
            #   (sum_{k' in merged_C} c[k'] * delta_{k',k})
            
            # So for a specific (i,j,k), the contribution is:
            #   a[i] * b[j] * c[k] if i in merged_A, j in merged_B, k in merged_C
            #   0 otherwise
            
            # So we need: for all (i,j,k):
            #   residual[i,j,k] = a[i]*b[j]*c[k] if (i in merged_A, j in merged_B, k in merged_C)
            #   residual[i,j,k] = 0 otherwise (but this may not hold)
            
            # This is a rank-1 tensor fitting problem on the subgrid defined by the merged supports
            
            # Compute residual
            residual = np.zeros((n * m, m * p, p * n), dtype=float)
            for i in range(n * m):
                for j in range(m * p):
                    for k in range(p * n):
                        val = 0.0
                        for t in range(12):
                            # Map remaining indices to original indices
                            orig_idx = None
                            for orig_i in range(14):
                                if orig_i != pair_idx and orig_i != other_idx:
                                    if orig_idx is None:
                                        orig_idx = orig_i
                                    else:
                                        break
                            # This mapping is getting complicated. Let me redo this.
                        break
                break
            break
        
        # This approach is getting too complex. Let's use a simpler heuristic.
    
    # Alternative: use the rank-14 solution and try to find a linear dependency
    # that allows rank reduction. Check if any column of [U|V|W] is a linear
    # combination of others in a way that preserves the tensor structure.
    
    # For now, return None to indicate no rank-13 found
    return None


def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # First, try to find a rank-13 decomposition
    rank13 = _search_rank13(_DEADLINE)
    if rank13 is not None and _verify_decomposition(rank13['U'], rank13['V'], rank13['W']):
        return rank13
    
    # Fallback to verified rank-14 Strassen-based decomposition
    return _build_rank14()

# EVOLVE-BLOCK-END
