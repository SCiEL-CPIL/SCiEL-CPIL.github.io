import numpy as np
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 2, 4, 6
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Strassen 2x2x2 decomposition
    U_s = np.array([
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1]   # M7: A12 - A22
    ])
    
    V_s = np.array([
        [1, 0, 0, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B21 - B22
        [-1, 0, 1, 0],  # M4: B12 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1]    # M7: B21 + B22
    ])
    
    W_s = np.array([
        [1, 0, 0, 1],   # M1: C11 += M1, C22 += M1
        [0, 0, 1, -1],  # M2: C21 += M2, C22 -= M2
        [0, 1, 0, 1],   # M3: C12 += M3, C22 += M3
        [1, 0, 1, 0],   # M4: C11 += M4, C12 += M4
        [-1, 1, 0, 0],  # M5: C11 -= M5, C12 += M5
        [0, 0, 0, 1],   # M6: C22 += M6
        [1, 0, 0, 0]    # M7: C11 += M7
    ])
    
    # Build the 42-term decomposition
    rank = 42
    U = np.zeros((dim_A, rank))
    V = np.zeros((dim_B, rank))
    W = np.zeros((dim_C, rank))
    
    col_offset = 0
    for cb in range(3):  # 3 column blocks of B
        # Block: A_left (2x2) * B_top_block (2x2)
        for t in range(7):
            for lr in range(2):
                for lc in range(2):
                    val = U_s[t][lr*2+lc]
                    if val != 0:
                        U[lr*m + lc, col_offset] += val
            for lr in range(2):
                for lc in range(2):
                    val = V_s[t][lr*2+lc]
                    if val != 0:
                        V[lr*p + (cb*2 + lc), col_offset] += val
            for lr in range(2):
                for lc in range(2):
                    val = W_s[t][lr*2+lc]
                    if val != 0:
                        W[(cb*2 + lc)*n + lr, col_offset] += val
            col_offset += 1
        
        # Block: A_right (2x2) * B_bot_block (2x2)
        for t in range(7):
            for lr in range(2):
                for lc in range(2):
                    val = U_s[t][lr*2+lc]
                    if val != 0:
                        U[lr*m + (2 + lc), col_offset] += val
            for lr in range(2):
                for lc in range(2):
                    val = V_s[t][lr*2+lc]
                    if val != 0:
                        V[(2 + lr)*p + (cb*2 + lc), col_offset] += val
            for lr in range(2):
                for lc in range(2):
                    val = W_s[t][lr*2+lc]
                    if val != 0:
                        W[(cb*2 + lc)*n + lr, col_offset] += val
            col_offset += 1
    
    # Verify the decomposition
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.max(np.abs(T - T_reconstructed))
    
    if error > 1e-10:
        # Fallback to schoolbook rank 48
        rank = 48
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    U[a_idx, r] = 1
                    V[b_idx, r] = 1
                    W[c_idx, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Search for linear dependencies to reduce rank
    # Each term r has vectors u_r, v_r, w_r
    # A dependency is a set of coefficients c_r such that sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0
    # This means for all (a,b,c): sum_r c_r * U[a,r] * V[b,r] * W[c,r] = 0
    
    # Strategy: Look for pairs or small sets of terms that can be combined
    # For two terms r1, r2 with coefficients c1, c2:
    # c1*U[:,r1]⊗V[:,r1]⊗W[:,r1] + c2*U[:,r2]⊗V[:,r2]⊗W[:,r2] = 0
    # This requires U[:,r1] proportional to U[:,r2], etc., which is unlikely.
    
    # Better: Look for 3-term or 4-term dependencies
    # We can use the fact that Strassen has known identities
    
    # Let's try a more systematic approach: 
    # The 42 terms are organized as (cb, row_block, strassen_term)
    # where cb in {0,1,2}, row_block in {0,1}, strassen_term in {0..6}
    
    # Term index: cb*14 + row_block*7 + strassen_term
    
    # Search for dependencies among terms with the same (cb, strassen_term) but different row_block
    # These share the same B and C structure but different A structure
    
    # Actually, let's try a different approach: use linear algebra to find the null space
    # of the tensor product map
    
    # Flatten the tensor space: dim_A * dim_B * dim_C = 8 * 24 * 12 = 2304
    # Each term r maps to a vector in this space: vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r])
    # We want to find the rank of the 42 x 2304 matrix
    
    # But 2304 is large. Let's use a smarter approach.
    
    # For a fixed (b,c), the coefficient of T[a,b,c] is sum_r U[a,r] * V[b,r] * W[c,r]
    # For a dependency sum_r c_r * U[:,r] ⊗ V[:,r] ⊗ W[:,r] = 0, we need:
    # For all (a,b,c): sum_r c_r * U[a,r] * V[b,r] * W[c,r] = 0
    
    # This is equivalent to: for all (b,c), the vector (sum_r c_r * U[:,r] * V[b,r] * W[c,r])_a = 0
    
    # Let's build a system: for each (b,c) pair, we get dim_A equations
    # Total: dim_B * dim_C * dim_A = 24 * 12 * 8 = 2304 equations in 42 unknowns
    
    # This is over-determined, so dependencies are unlikely unless there's structure.
    
    # Let's check if there are any exact integer dependencies by looking at small subsets
    
    # Try all pairs first (42 choose 2 = 861 pairs)
    found_dependency = False
    best_rank = 42
    
    # For efficiency, let's only check dependencies of size 2 and 3
    # Size 2: c1*T1 + c2*T2 = 0 means T1 and T2 are proportional
    # This requires U[:,r1] ∝ U[:,r2], V[:,r1] ∝ V[:,r2], W[:,r1] ∝ W[:,r2]
    
    # Check for proportional terms
    def are_proportional(u1, u2, tol=1e-10):
        # Check if u1 = k*u2 for some scalar k
        nonzero_u1 = u1[np.abs(u1) > tol]
        nonzero_u2 = u2[np.abs(u2) > tol]
        if len(nonzero_u1) == 0 or len(nonzero_u2) == 0:
            return True if np.all(np.abs(u1) < tol) and np.all(np.abs(u2) < tol) else False
        # Try to find k from the first non-zero pair
        for i in range(len(u1)):
            for j in range(len(u2)):
                if np.abs(u1[i]) > tol and np.abs(u2[j]) > tol:
                    k = u1[i] / u2[j]
                    # Check if u1 = k*u2
                    if np.all(np.abs(u1 - k*u2) < tol):
                        return True
        return False
    
    # Check all pairs for proportionality
    for r1 in range(42):
        for r2 in range(r1+1, 42):
            if are_proportional(U[:,r1], U[:,r2]) and \
               are_proportional(V[:,r1], V[:,r2]) and \
               are_proportional(W[:,r1], W[:,r2]):
                # These two terms are proportional, so we can eliminate one
                found_dependency = True
                best_rank = 41
                break
        if found_dependency:
            break
    
    if not found_dependency:
        # Try 3-term dependencies
        # This is more expensive. Let's use a smarter search.
        # For each pair (r1, r2), check if there exists r3 and coefficients such that
        # c1*T1 + c2*T2 + c3*T3 = 0
        
        # This is equivalent to: T3 is in the span of T1 and T2 in the tensor space
        # But checking this directly is expensive.
        
        # Alternative: use the structure of the problem
        # The 42 terms come from 3 column blocks × 2 row blocks × 7 Strassen terms
        # Terms with the same (cb, strassen_term) but different row_block have:
        # - Different U vectors (different A rows)
        # - Same V vectors (same B columns)
        # - Same W vectors (same C columns)
        
        # So for fixed (cb, t), terms r_left and r_right have:
        # U_left, V, W and U_right, V, W
        # A dependency c_left*T_left + c_right*T_right = 0 requires:
        # c_left*U_left ⊗ V ⊗ W + c_right*U_right ⊗ V ⊗ W = 0
        # = (c_left*U_left + c_right*U_right) ⊗ V ⊗ W = 0
        # This requires c_left*U_left + c_right*U_right = 0 (since V,W are nonzero)
        # Which means U_left and U_right are proportional, which they're not.
        
        # What about terms with same (row_block, t) but different cb?
        # They have different V and W but same U.
        # c1*T1 + c2*T2 = 0 requires c1*U⊗V1⊗W1 + c2*U⊗V2⊗W2 = 0
        # = U ⊗ (c1*V1⊗W1 + c2*V2⊗W2) = 0
        # Requires c1*V1⊗W1 + c2*V2⊗W2 = 0, i.e., V1⊗W1 proportional to V2⊗W2
        
        # Let's check if any V1⊗W1 are proportional to V2⊗W2
        # V1⊗W1 is a 24×12 matrix, V2⊗W2 is also 24×12
        # They're proportional if V1⊗W1 = k*(V2⊗W2)
        
        # This is getting complex. Let's just return the rank-42 solution.
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
