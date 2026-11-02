import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    TIME_LIMIT = 135.0  # Leave 15s margin before 150s kill

    def F(x):
        if isinstance(x, Fraction):
            return x
        if isinstance(x, (int, np.integer)):
            return Fraction(int(x))
        if isinstance(x, float):
            fr = Fraction(x).limit_denominator(10**6)
            if abs(fr - x) < 1e-12:
                return fr
            return Fraction(x)
        return Fraction(x)

    def build_initial_decomposition():
        """Build a decomposition of 2x5x5 using block Strassen + schoolbook."""
        terms = []

        def add_strassen(i_start, j_start, k_start):
            terms.append((
                {(i_start, j_start): 1, (i_start+1, j_start+1): 1},
                {(j_start, k_start): 1, (j_start+1, k_start+1): 1},
                {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            ))
            terms.append((
                {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1},
                {(j_start, k_start): 1},
                {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            ))
            terms.append((
                {(i_start, j_start): 1},
                {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1},
                {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            ))
            terms.append((
                {(i_start+1, j_start+1): 1},
                {(j_start+1, k_start): 1, (j_start, k_start): -1},
                {(k_start, i_start): 1, (k_start, i_start+1): 1}
            ))
            terms.append((
                {(i_start, j_start): 1, (i_start, j_start+1): 1},
                {(j_start+1, k_start+1): 1},
                {(k_start, i_start): -1, (k_start+1, i_start): 1}
            ))
            terms.append((
                {(i_start+1, j_start): 1, (i_start, j_start): -1},
                {(j_start, k_start): 1, (j_start, k_start+1): 1},
                {(k_start+1, i_start+1): 1}
            ))
            terms.append((
                {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1},
                {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1},
                {(k_start, i_start): 1}
            ))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        add_schoolbook(0, 2, 4, 5, 0, 5)

        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = F(val)
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = F(val)
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = F(val)
        return U, V, W

    def compute_tensor(terms):
        """Compute the full tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] as Fractions."""
        rank = len(terms)
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        u_val = terms[c][0].get((i, j), 0)
                        v_val = terms[c][1].get((j, k), 0)
                        w_val = terms[c][2].get((k, i), 0)
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            total += F(u_val) * F(v_val) * F(w_val)
                    T[(i,j,k)] = total
        return T

    def is_valid(terms):
        T = compute_tensor(terms)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[(i,j,k)] != 1:
                        return False
        return True

    def term_contribution(terms, c):
        """Return dict of (i,j,k) -> contribution of term c to the tensor."""
        u_dict, v_dict, w_dict = terms[c]
        contrib = {}
        for (i, j), u_val in u_dict.items():
            if u_val == 0:
                continue
            for (j2, k), v_val in v_dict.items():
                if j2 != j or v_val == 0:
                    continue
                for (k2, i2), w_val in w_dict.items():
                    if k2 != k or i2 != i or w_val == 0:
                        continue
                    contrib[(i, j, k)] = F(u_val) * F(v_val) * F(w_val)
        return contrib

    def remove_redundant_terms_fast(terms):
        """Remove terms whose contribution is zero or linearly dependent."""
        current = list(terms)
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            rank = len(current)
            # Precompute contributions for all terms
            contributions = [term_contribution(current, c) for c in range(rank)]
            
            for c_remove in range(rank):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                # Check if removing term c_remove still gives valid tensor
                # The residual after removal is -contribution[c_remove]
                # We need: for all (i,j,k), sum_{c != c_remove} contrib_c[i,j,k] == 1
                # i.e., total[i,j,k] - contrib_c_remove[i,j,k] == 1
                # i.e., contrib_c_remove[i,j,k] == total[i,j,k] - 1 = 0 for all (i,j,k)
                # Wait, that's only if total == 1 already.
                # More generally: if current tensor is valid (all 1s), then removing term c
                # is valid iff contribution of term c is zero everywhere.
                # But we can also do: check if the sum of remaining terms equals 1.
                
                # Faster: compute total tensor once, then check if contrib[c_remove] is zero
                total_tensor = compute_tensor(current)
                # If total is valid, check if removing c_remove keeps it valid
                removable = True
                for (i,j,k), val in total_tensor.items():
                    c_val = contributions[c_remove].get((i,j,k), 0)
                    if val - c_val != 1:
                        removable = False
                        break
                if removable:
                    current = [t for idx, t in enumerate(current) if idx != c_remove]
                    improved = True
                    break
        return current

    def linear_dependency_removal(terms):
        """Find and remove terms that are linearly dependent in the tensor space.
        Build a matrix where each row is a tensor component and each column is a term's
        contribution vector. Find null space vectors."""
        current = list(terms)
        rank = len(current)
        
        # Build the contribution matrix: rows = (i,j,k) components, cols = terms
        # M[i,j,k,c] = contribution of term c to component (i,j,k)
        ncomp = n * m * p
        comp_index = {}
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    comp_index[(i,j,k)] = idx
                    idx += 1
        
        # Build matrix as list of lists of Fractions
        M = [[Fraction(0)] * rank for _ in range(ncomp)]
        for c in range(rank):
            contrib = term_contribution(current, c)
            for (i,j,k), val in contrib.items():
                M[comp_index[(i,j,k)]][c] = val
        
        # Find null space: solve M * x = 0
        # Use Gaussian elimination with Fractions
        # We want to find if any column is a linear combination of others
        
        # Simpler approach: for each column, check if it's in the span of others
        # by checking if removing it and re-solving still works
        # But that's what we already do.
        
        # Better: use rank of M. If rank(M) < rank, there are dependencies.
        # Find which columns are redundant.
        
        # Perform Gaussian elimination to find pivot columns
        A = [row[:] for row in M]  # copy
        row_count = ncomp
        col_count = rank
        
        pivot_cols = []
        row_idx = 0
        for col in range(col_count):
            if row_idx >= row_count:
                break
            # Find pivot in column col, from row_idx down
            pivot = -1
            for r in range(row_idx, row_count):
                if A[r][col] != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap rows
            A[row_idx], A[pivot] = A[pivot], A[row_idx]
            # Eliminate
            pivot_val = A[row_idx][col]
            for r in range(row_count):
                if r != row_idx and A[r][col] != 0:
                    factor = A[r][col] / pivot_val
                    for c in range(col, col_count):
                        A[r][c] -= factor * A[row_idx][c]
            pivot_cols.append(col)
            row_idx += 1
        
        # Non-pivot columns are linearly dependent
        pivot_set = set(pivot_cols)
        redundant_cols = [c for c in range(rank) if c not in pivot_set]
        
        # Remove redundant columns (terms)
        if redundant_cols:
            current = [t for idx, t in enumerate(current) if idx not in redundant_cols]
        
        return current

    # Main logic
    terms = build_initial_decomposition()
    
    if not is_valid(terms):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))

    # Remove zero-contribution terms
    terms = remove_redundant_terms_fast(terms)
    
    # Remove linearly dependent terms
    if time.time() - start_time < TIME_LIMIT - 20:
        terms = linear_dependency_removal(terms)
    
    # Final validation
    if not is_valid(terms):
        terms = build_initial_decomposition()

    U, V, W = terms_to_matrices(terms)

    def to_plain(val):
        if val.denominator == 1:
            return int(val.numerator)
        elif val.denominator == 2:
            return val.numerator / 2
        else:
            return float(val)

    U_out = [[to_plain(val) for val in row] for row in U]
    V_out = [[to_plain(val) for val in row] for row in V]
    W_out = [[to_plain(val) for val in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
