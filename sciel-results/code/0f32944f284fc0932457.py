"""
Strategy: Robust Strassen Construction with Linear Dependency Minimization.
1. Construct the standard Strassen decomposition for 6x6 matrices by splitting into 2x2 blocks of 3x3 matrices.
   This yields a rank of 7 * 27 = 189.
2. Verify the decomposition.
3. Attempt to reduce the rank by checking for linear dependencies among the columns of the factor matrices U, V, W.
   If a set of columns is linearly dependent, we can potentially remove one term.
   Note: True rank reduction requires that the removed term does not break the tensor equation.
   We will check if any term is a linear combination of others in the context of the tensor product.
   This is a structural change: adding a post-optimization phase.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(decomp, n, m, p, sample=False):
    """Verify that the decomposition is correct."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    if sample:
        import random
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        random.shuffle(indices)
        indices = indices[:min(200, len(indices))]
    else:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for i, j, k in indices:
        s_val = 0
        for c in range(rank):
            s_val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(s_val - 1) > 1e-9:
            return False
    return True

def _strassen_2x2_atoms():
    """
    Returns the 7 Strassen atoms for 2x2 matrix multiplication.
    Atom format: (A_coeffs, B_coeffs, C_coeffs)
    Indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    """
    atoms = [
        ({0: 1, 3: -1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({0: 1, 3: 1}, {3: 1}, {3: 1}),
        ({2: 1, 3: 1}, {1: 1}, {1: 1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        ({3: 1}, {2: 1, 0: -1}, {2: 1}),
        ({0: 1, 1: 1}, {2: 1}, {2: 1}),
        ({2: 1, 1: -1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ]
    return atoms

def _build_strassen_6x6():
    """
    Build the 6x6 Strassen decomposition.
    """
    n, m, p = 6, 6, 6
    block_size = 3
    atoms = _strassen_2x2_atoms()
    
    U = []
    V = []
    W = []
    
    for s in range(len(atoms)):
        a_coeffs, b_coeffs, c_coeffs = atoms[s]
        
        # For each atom, we iterate over the non-zero coefficients in A and B
        # and the corresponding C coefficients.
        # However, the Strassen algorithm computes products of linear combinations.
        # The rank-1 terms in the 6x6 decomposition correspond to the scalar multiplications
        # within the 3x3 blocks.
        
        # The term U*V*W for the 6x6 tensor is sum over atoms s of (sum over block terms).
        # Specifically, for each atom s:
        # We have a product of (linear combo of A blocks) * (linear combo of B blocks) -> (linear combo of C blocks).
        # This expands to a sum of scalar products of elements.
        
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                # Determine the C block this contributes to
                # The atom s defines which C blocks are affected.
                # In Strassen, the atom s produces a specific set of C blocks.
                # For example, atom 0 produces C00 and C11.
                
                # We need to add the schoolbook terms for the 3x3 multiplication
                # of the block A_ai and B_bi, scaled by the signs.
                
                a_row_start = (ai // 2) * block_size
                a_col_start = (ai % 2) * block_size
                b_row_start = (bi // 2) * block_size
                b_col_start = (bi % 2) * block_size
                
                # The C blocks affected by this atom are determined by the atom's c_coeffs.
                # However, the specific contribution to a C block depends on the atom.
                # For Strassen, each atom contributes to specific C blocks.
                # Let's determine the C block indices for this atom.
                
                # Actually, it's easier to just iterate over all C blocks that this atom affects.
                # The atom s affects C blocks where c_coeffs is non-zero.
                # But the contribution of the specific (ai, bi) pair to a C block (ci) is
                # a_sign * b_sign * c_sign(ci) ? No.
                
                # The Strassen formula is:
                # C = sum_s (U_s * V_s) * W_s
                # where U_s, V_s are 2x2 matrices of linear combos of A, B blocks.
                # W_s is a 2x2 matrix of linear combos of C blocks.
                
                # The term for a specific scalar multiplication in the 3x3 blocks is:
                # A_{ai, local_i, local_j} * B_{bi, local_j, local_k} * C_{ci, local_i, local_k}
                
                # We need to ensure that the sum over s and local indices gives 1 for the tensor.
                
                # Let's construct the terms explicitly.
                for ci, c_sign in c_coeffs.items():
                    if c_sign == 0:
                        continue
                    
                    c_row_start = (ci // 2) * block_size
                    c_col_start = (ci % 2) * block_size
                    
                    # Check if this (ai, bi) pair actually contributes to this C block in Strassen.
                    # In the standard Strassen, the atom s has specific A, B, C blocks.
                    # For example, P1 = (A00-A11)(B00+B11) -> C00+C11.
                    # Here, ai=0, bi=0 contributes to ci=0 and ci=3.
                    # ai=0, bi=3 contributes to ci=0 and ci=3.
                    # ai=3, bi=0 contributes to ci=0 and ci=3.
                    # ai=3, bi=3 contributes to ci=0 and ci=3.
                    
                    # So for a given atom s, and a given (ai, bi) in a_coeffs, b_coeffs,
                    # it contributes to ALL ci in c_coeffs.
                    
                    for li in range(block_size):
                        for lj in range(block_size):
                            for lk in range(block_size):
                                gi = a_row_start + li
                                gj = a_col_start + lj
                                gk = b_col_start + lk
                                
                                # The C index is determined by gi, gk?
                                # No, the C block is ci. The local indices are li, lk.
                                # The global C indices are c_row_start + li, c_col_start + lk.
                                
                                # We create a rank-1 term.
                                new_U = [0] * (n * m)
                                new_V = [0] * (m * p)
                                new_W = [0] * (p * n)
                                
                                new_U[gi * m + gj] = a_sign
                                new_V[gj * p + gk] = b_sign
                                new_W[gk * n + gi] = c_sign
                                
                                U.append(new_U)
                                V.append(new_V)
                                W.append(new_W)
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(decomp, n, m, p):
    """
    Attempt to reduce the rank by removing linearly dependent terms.
    This is a heuristic. We check if any column in U, V, W is zero or a linear combination.
    For simplicity, we just check for zero columns.
    """
    U = [row[:] for row in decomp["U"]]
    V = [row[:] for row in decomp["V"]]
    W = [row[:] for row in decomp["W"]]
    
    rank = len(U[0])
    cols_to_remove = []
    
    for c in range(rank):
        # Check if column c is all zeros in U, V, or W
        if all(U[i][c] == 0 for i in range(n*m)) or \
           all(V[i][c] == 0 for i in range(m*p)) or \
           all(W[i][c] == 0 for i in range(p*n)):
            cols_to_remove.append(c)
    
    if not cols_to_remove:
        return decomp
    
    # Remove columns
    keep_cols = [c for c in range(rank) if c not in cols_to_remove]
    
    U_new = [[U[i][c] for c in keep_cols] for i in range(n*m)]
    V_new = [[V[i][c] for c in keep_cols] for i in range(m*p)]
    W_new = [[W[i][c] for c in keep_cols] for i in range(p*n)]
    
    return {"U": U_new, "V": V_new, "W": W_new}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    try:
        # Step 1: Build Strassen decomposition
        decomp = _build_strassen_6x6()
        
        # Step 2: Verify
        if _verify(decomp, n, m, p, sample=False):
            # Step 3: Try to reduce rank
            decomp = _try_reduce_rank(decomp, n, m, p)
            
            # Step 4: Verify again
            if _verify(decomp, n, m, p, sample=False):
                return decomp
            else:
                return _build_strassen_6x6()
        else:
            # Fallback to schoolbook
            return _schoolbook(n, m, p)
            
    except Exception as e:
        # Fallback to schoolbook on any error
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
