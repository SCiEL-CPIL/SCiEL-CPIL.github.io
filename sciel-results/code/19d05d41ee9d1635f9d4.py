"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Build the target MM tensor: T[u,v,w] = 1 iff u=3i+j, v=3j+k, w=3k+i
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = i * 3 + j
                v = j * 3 + k
                w = k * 3 + i
                T[u, v, w] = 1.0
    
    # Start with schoolbook rank-27 decomposition
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    # Verify schoolbook
    T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
    assert np.allclose(T_rec, T), "Schoolbook decomposition failed verification"
    
    # Attempt to reduce rank by merging columns
    # Two columns c1, c2 can be merged if there exists a scalar a such that:
    # U[:,c1] = a*U[:,c2], V[:,c1] = a*V[:,c2], W[:,c1] = a*W[:,c2]
    # This is unlikely for schoolbook, so we try a different approach.
    
    # Better approach: Try to find a rank-23 decomposition using known constructions
    # The key insight: use Strassen-like combinations for 2x2 sub-blocks
    
    # Let's try to construct a rank-23 decomposition systematically
    # We'll use the following strategy:
    # 1. For the 2x2 sub-blocks, use Strassen (7 multiplications each)
    # 2. For cross terms involving index 2, use direct multiplication
    
    # Actually, let me try a more systematic search:
    # We'll try to find a valid decomposition with rank < 27
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 27
    
    # Try to find a rank-23 decomposition
    # Known construction: partition into blocks and use optimized algorithms
    
    # Let's define a helper to verify a decomposition
    def verify_decomp(U_mat, V_mat, W_mat, target):
        T_rec = np.einsum('uc,vc,wc->uvw', U_mat, V_mat, W_mat)
        return np.allclose(T_rec, target, atol=1e-10)
    
    # Try various rank-23 constructions
    # Construction 1: Based on block decomposition
    
    # For a 3x3 matrix multiplication, we can think of it as:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    
    # One known rank-23 decomposition uses the following 23 terms:
    # I'll define them explicitly based on literature
    
    def make_vec(indices_values):
        v = np.zeros(9)
        for idx, val in indices_values:
            v[idx] = val
        return v
    
    # Define 23 terms for rank-23 decomposition
    # Each term is (U_support, V_support, W_support)
    
    # Term 0: (A00+A11)(B00+B11) contributes to C00, C01, C10, C11
    terms = []
    
    # Strassen-like terms for the top-left 2x2 block
    # M1 = (A00+A11)(B00+B11)
    terms.append((
        make_vec([(0,1),(4,1)]),
        make_vec([(0,1),(4,1)]),
        make_vec([(0,1),(1,1),(3,1),(4,1)])
    ))
    
    # M2 = (A10+A11)B00
    terms.append((
        make_vec([(3,1),(4,1)]),
        make_vec([(0,1)]),
        make_vec([(3,1),(4,1)])
    ))
    
    # M3 = A00(B10-B11)
    terms.append((
        make_vec([(0,1)]),
        make_vec([(3,1),(4,-1)]),
        make_vec([(1,1),(4,-1)])
    ))
    
    # M4 = A11(B01-B00)
    terms.append((
        make_vec([(4,1)]),
        make_vec([(1,1),(0,-1)]),
        make_vec([(3,1),(4,-1)])
    ))
    
    # M5 = (A00+A01)B11
    terms.append((
        make_vec([(0,1),(1,1)]),
        make_vec([(4,1)]),
        make_vec([(0,1),(1,1)])
    ))
    
    # M6 = (A10-A00)(B00+B01)
    terms.append((
        make_vec([(3,1),(0,-1)]),
        make_vec([(0,1),(1,1)]),
        make_vec([(3,1),(4,1)])
    ))
    
    # M7 = (A01-A11)(B10+B11)
    terms.append((
        make_vec([(1,1),(4,-1)]),
        make_vec([(3,1),(4,1)]),
        make_vec([(1,1),(4,1)])
    ))
    
    # Now we need 16 more terms to handle all entries involving index 2
    # Entries: C02, C12, C20, C21, C22 and their contributions
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # Let's add terms for these
    # Term 8: A02*B22 -> C02
    terms.append((
        make_vec([(2,1)]),
        make_vec([(8,1)]),
        make_vec([(2,1)])
    ))
    
    # Term 9: A00*B02 -> C02
    terms.append((
        make_vec([(0,1)]),
        make_vec([(2,1)]),
        make_vec([(2,1)])
    ))
    
    # Term 10: A01*B12 -> C02
    terms.append((
        make_vec([(1,1)]),
        make_vec([(5,1)]),
        make_vec([(2,1)])
    ))
    
    # Term 11: A12*B22 -> C12
    terms.append((
        make_vec([(5,1)]),
        make_vec([(8,1)]),
        make_vec([(5,1)])
    ))
    
    # Term 12: A10*B02 -> C12
    terms.append((
        make_vec([(3,1)]),
        make_vec([(2,1)]),
        make_vec([(5,1)])
    ))
    
    # Term 13: A11*B12 -> C12
    terms.append((
        make_vec([(4,1)]),
        make_vec([(5,1)]),
        make_vec([(5,1)])
    ))
    
    # Term 14: A20*B00 -> C20
    terms.append((
        make_vec([(6,1)]),
        make_vec([(0,1)]),
        make_vec([(6,1)])
    ))
    
    # Term 15: A21*B10 -> C20
    terms.append((
        make_vec([(7,1)]),
        make_vec([(3,1)]),
        make_vec([(6,1)])
    ))
    
    # Term 16: A22*B20 -> C20
    terms.append((
        make_vec([(8,1)]),
        make_vec([(6,1)]),
        make_vec([(6,1)])
    ))
    
    # Term 17: A20*B01 -> C21
    terms.append((
        make_vec([(6,1)]),
        make_vec([(1,1)]),
        make_vec([(7,1)])
    ))
    
    # Term 18: A21*B11 -> C21
    terms.append((
        make_vec([(7,1)]),
        make_vec([(4,1)]),
        make_vec([(7,1)])
    ))
    
    # Term 19: A22*B21 -> C21
    terms.append((
        make_vec([(8,1)]),
        make_vec([(7,1)]),
        make_vec([(7,1)])
    ))
    
    # Term 20: A20*B02 -> C22
    terms.append((
        make_vec([(6,1)]),
        make_vec([(2,1)]),
        make_vec([(8,1)])
    ))
    
    # Term 21: A21*B12 -> C22
    terms.append((
        make_vec([(7,1)]),
        make_vec([(5,1)]),
        make_vec([(8,1)])
    ))
    
    # Term 22: A22*B22 -> C22
    terms.append((
        make_vec([(8,1)]),
        make_vec([(8,1)]),
        make_vec([(8,1)])
    ))
    
    # We have 23 terms now. Let's verify
    if len(terms) == 23:
        U_test = np.column_stack([t[0] for t in terms])
        V_test = np.column_stack([t[1] for t in terms])
        W_test = np.column_stack([t[2] for t in terms])
        
        if verify_decomp(U_test, V_test, W_test, T):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = 23
            # Convert to lists
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    
    # If rank-23 didn't work, try to improve schoolbook by merging
    # For now, return the best we have
    if best_rank < 27:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Return schoolbook
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
