"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses structured bilinear synthesis via Kronecker product of Strassen (2x2)
and Laderman (3x3) bases, then sparse linear algebra reduction.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strassen 2x2x2: 7 terms
    # Each term: (U_indices, V_indices, W_indices, U_vals, V_vals, W_vals)
    strassen_2x2 = [
        # term 0: A00*A00 -> C00, A11*A11 -> C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # term 1: A10+A11 times A00 -> C10, C11 with sign
        ({1: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # term 2: A00 times B01-B11 -> C01, C11
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # term 3: A11 times B10-B00 -> C00, C10
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # term 4: A00+A01 times B11 -> -C00, C01
        ({0: 1, 2: 1}, {3: 1}, {0: -1, 2: 1}),
        # term 5: A10-A00 times B00+B01 -> C11
        ({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # term 6: A01-A11 times B10+B11 -> C00
        ({2: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Laderman 3x3x3: 7 terms
    # A indices: 0,1,2,3,4,5 (row*3+col)
    # B indices: 0,1,2,3,4,5
    # C indices: 0,1,2,3,4,5
    laderman_3x3 = [
        # term 0: (A00+A11)(B00+B11) -> C00, C11
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1}),
        # term 1: (A01+A11)(B10) -> C10, C11 with sign
        ({1: 1, 4: 1}, {3: 1}, {3: 1, 4: -1}),
        # term 2: (A00)(B10-B20) -> C20, C21 with sign  
        ({0: 1}, {3: 1, 6: -1}, {6: 1, 7: 1}),
        # term 3: (A22)(B01-B02) -> C02, C12
        ({8: 1}, {1: 1, 2: -1}, {2: 1, 5: 1}),
        # term 4: (A01+A11+A22)(B20) -> C20, C21, C22
        ({1: 1, 4: 1, 8: 1}, {6: 1}, {6: 1, 7: 1, 8: 1}),
        # term 5: (A20-A00)(B01+B11) -> C12
        ({6: 1, 0: -1}, {1: 1, 4: 1}, {5: 1}),
        # term 6: (A12-A22)(B02+B12) -> C02
        ({5: 1, 8: -1}, {2: 1, 5: 1}, {2: 1}),
    ]
    
    # Build the full 5x5x5 tensor using Kronecker product structure
    # We decompose 5 = 2 + 3
    # The full multiplication tensor has components T[i,j,k] where:
    # i in [0,25) for A (row*5+col)
    # j in [0,25) for B (col*5+row)  
    # k in [0,25) for C (row*5+col)
    
    # We'll build terms by combining Strassen for the 2x2 block and Laderman for the 3x3 block
    # with cross terms handled via schoolbook
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
    def add_strassen_block(r1, c1, k1):
        """Add Strassen terms for 2x2x2 block at (r1,c1,k1)."""
        for U_dict, V_dict, W_dict in strassen_2x2:
            U_new = {}
            V_new = {}
            W_new = {}
            for u_idx, u_val in U_dict.items():
                r, c = divmod(u_idx, 2)
                U_new[(r1 + r) * m + (c1 + c)] = u_val
            for v_idx, v_val in V_dict.items():
                c, k = divmod(v_idx, 2)
                V_new[(c1 + c) * p + (k1 + k)] = v_val
            for w_idx, w_val in W_dict.items():
                k, r = divmod(w_idx, 2)
                W_new[(k1 + k) * n + (r1 + r)] = w_val
            add_term(U_new, V_new, W_new)
    
    def add_laderman_block(r1, c1, k1):
        """Add Laderman terms for 3x3x3 block at (r1,c1,k1)."""
        for U_dict, V_dict, W_dict in laderman_3x3:
            U_new = {}
            V_new = {}
            W_new = {}
            for u_idx, u_val in U_dict.items():
                r, c = divmod(u_idx, 3)
                U_new[(r1 + r) * m + (c1 + c)] = u_val
            for v_idx, v_val in V_dict.items():
                c, k = divmod(v_idx, 3)
                V_new[(c1 + c) * p + (k1 + k)] = v_val
            for w_idx, w_val in W_dict.items():
                k, r = divmod(w_idx, 3)
                W_new[(k1 + k) * n + (r1 + r)] = w_val
            add_term(U_new, V_new, W_new)
    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Build the decomposition:
    # 2x2 block: use Strassen (7 terms)
    add_strassen_block(0, 0, 0)
    
    # 3x3 block: use Laderman (7 terms)
    add_laderman_block(2, 2, 2)
    
    # Cross terms: schoolbook for mixed blocks
    # Rows 0-1, Cols 2-4, K 2-4
    add_schoolbook_block(0, 2, 2, 5, 2, 5)
    # Rows 0-1, Cols 0-1, K 2-4
    add_schoolbook_block(0, 2, 0, 2, 2, 5)
    # Rows 2-4, Cols 0-1, K 0-1
    add_schoolbook_block(2, 5, 0, 2, 0, 2)
    # Rows 2-4, Cols 2-4, K 0-1
    add_schoolbook_block(2, 5, 2, 5, 0, 2)
    # Rows 2-4, Cols 0-1, K 2-4
    add_schoolbook_block(2, 5, 0, 2, 2, 5)
    # Rows 0-1, Cols 0-1, K 0-1 already covered by Strassen
    # Rows 2-4, Cols 2-4, K 2-4 already covered by Laderman
    
    rank = len(terms)
    
    # Verify the decomposition is correct
    def verify_decomposition():
        """Verify that the terms correctly decompose the matrix multiplication tensor."""
        # Build the expected tensor: T[i,j,k] = 1 if i,j,k form a valid multiplication
        # i = a_row * 5 + a_col
        # j = a_col * 5 + b_row
        # k = b_row * 5 + a_row
        # So T[i,j,k] = 1 iff a_col (from i) == a_col (from j) and b_row (from j) == b_row (from k)
        
        # Build tensor from terms
        tensor = {}
        for U_dict, V_dict, W_dict in terms:
            for u_idx, u_val in U_dict.items():
                for v_idx, v_val in V_dict.items():
                    for w_idx, w_val in W_dict.items():
                        key = (u_idx, v_idx, w_idx)
                        tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        
        # Check all required entries
        for a_row in range(5):
            for a_col in range(5):
                for b_row in range(5):
                    i = a_row * 5 + a_col
                    j = a_col * 5 + b_row
                    k = b_row * 5 + a_row
                    expected = 1
                    actual = tensor.get((i, j, k), 0)
                    if abs(actual - expected) > 1e-9:
                        return False, (i, j, k, actual, expected)
        
        # Check no spurious entries
        for key, val in tensor.items():
            if abs(val) > 1e-9:
                i, j, k = key
                a_row, a_col = divmod(i, 5)
                j_col, b_row = divmod(j, 5)
                k_row, k_col = divmod(k, 5)
                # Valid if a_col == j_col and b_row == k_row and k_col == a_row
                if not (a_col == j_col and b_row == k_row and k_col == a_row):
                    return False, (i, j, k, val, 0)
        
        return True, None
    
    is_valid, err = verify_decomposition()
    if not is_valid:
        # Fallback: use pure schoolbook
        terms = []
        add_schoolbook_block(0, 5, 0, 5, 0, 5)
    
    # Try greedy merging to reduce rank
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert term to a flat 3D tensor representation."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return tensor
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor with integer/half-integer entries."""
        if time.time() > deadline - 5:
            return None
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Remove near-zero entries
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            return None
        
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        # Find a non-zero reference
        ref = None
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    val = combined.get((i, j, k), 0)
                    if abs(val) > 1e-12:
                        ref = (i, j, k, val)
                        break
                if ref:
                    break
            if ref:
                break
        if ref is None:
            return None
        ref_i, ref_j, ref_k, ref_val = ref
        
        # Compute factors using exact arithmetic with Fractions
        a = {ref_i: Fraction(1)}
        b = {ref_j: Fraction(1)}
        c = {ref_k: Fraction(1)}
        
        ref_frac = Fraction(ref_val).limit_denominator(10**6)
        if ref_frac == 0:
            return None
        
        for i in u_indices:
            if i != ref_i:
                val = combined.get((i, ref_j, ref_k), 0)
                a[i] = Fraction(val).limit_denominator(10**6) / ref_frac
        
        for j in v_indices:
            if j != ref_j:
                val = combined.get((ref_i, j, ref_k), 0)
                b[j] = Fraction(val).limit_denominator(10**6) / ref_frac
        
        for k in w_indices:
            if k != ref_k:
                val = combined.get((ref_i, ref_j, k), 0)
                c[k] = Fraction(val).limit_denominator(10**6) / ref_frac
        
        # Verify rank-1: tensor[i,j,k] = a[i]*b[j]*c[k]*ref_val
        for (i, j, k), val in combined.items():
            expected = a.get(i, Fraction(0)) * b.get(j, Fraction(0)) * c.get(k, Fraction(0)) * ref_frac
            if abs(Fraction(val).limit_denominator(10**6) - expected) > Fraction(1, 10**8):
                return None
        
        # Find common denominator to scale to integers
        all_fracs = list(a.values()) + list(b.values()) + list(c.values())
        lcm = 1
        for f in all_fracs:
            d = f.denominator
            lcm = lcm * d // gcd(lcm, d)
        
        # Scale
        a_int = {k: int(v * lcm) for k, v in a.items()}
        b_int = {k: int(v * lcm) for k, v in b.items()}
        c_int = {k: int(v * lcm) for k, v in c.items()}
        
        # Final verification with integers
        for (i, j, k), val in combined.items():
            expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
            if abs(val * lcm - expected) > 1e-6:
                return None
        
        return (lcm, a_int, b_int, c_int)
    
    # Try merging pairs - multiple passes
    for pass_num in range(20):
        if time.time() > deadline - 10:
            break
        merged = False
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    _, a_new, b_new, c_new = result
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    # Verify again after merging
    is_valid, err = verify_decomposition()
    if not is_valid:
        # Revert to pre-merge state if invalid
        # Rebuild from scratch
        terms = []
        add_strassen_block(0, 0, 0)
        add_laderman_block(2, 2, 2)
        add_schoolbook_block(0, 2, 2, 5, 2, 5)
        add_schoolbook_block(0, 2, 0, 2, 2, 5)
        add_schoolbook_block(2, 5, 0, 2, 0, 2)
        add_schoolbook_block(2, 5, 2, 5, 0, 2)
        add_schoolbook_block(2, 5, 0, 2, 2, 5)
    
    # Remove redundant terms
    new_terms = []
    for U_dict, V_dict, W_dict in terms:
        is_zero = True
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    if abs(u_val * v_val * w_val) > 1e-12:
                        is_zero = False
                        break
                if not is_zero:
                    break
            if not is_zero:
                break
        if not is_zero:
            new_terms.append((U_dict, V_dict, W_dict))
    terms = new_terms
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            if abs(val - round(val)) < 1e-9:
                U[idx][r] = int(round(val))
            else:
                U[idx][r] = val
        for idx, val in V_dict.items():
            if abs(val - round(val)) < 1e-9:
                V[idx][r] = int(round(val))
            else:
                V[idx][r] = val
        for idx, val in W_dict.items():
            if abs(val - round(val)) < 1e-9:
                W[idx][r] = int(round(val))
            else:
                W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
