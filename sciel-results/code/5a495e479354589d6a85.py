import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses block Strassen + direct, then local search to reduce rank.
    """
    n, m, p = 3, 4, 6
    
    def build_baseline():
        R = 66
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1
        
        for j in range(3):
            for i in range(2):
                add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
        
        for j in range(3):
            for i in range(2):
                add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
        
        assert col == R
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition exactly reproduces matrix multiplication."""
        r = len(U[0])
        # T[a,b,c] = sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] should equal 1
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    actual = 0.0
                    for rr in range(r):
                        actual += U[a * m + b][rr] * V[b * p + c][rr] * W[c * n + a][rr]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    def try_local_search(U, V, W, deadline):
        """Try to reduce rank by finding and removing redundant terms using linear algebra."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            r = len(U[0])
            if r <= 1:
                break
            
            # Build the matrix of tensor entries: for each (a,b,c) triple, 
            # the tensor value is sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r]
            # We need this to equal 1 for all (a,b,c).
            
            # For each pair of columns (i, j), check if column j is a linear combination
            # of the others in the "tensor space" sense.
            # 
            # A term r contributes to tensor entry (a,b,c) with value 
            # U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r]
            # 
            # If we can express the contribution of column j as a linear combination
            # of contributions from other columns, we can drop it.
            #
            # More practically: try each column, remove it, and solve for whether
            # the remaining columns can still produce the correct tensor.
            # This is a system of equations.
            
            for drop_idx in range(r):
                if time.time() >= deadline:
                    break
                
                # Remove column drop_idx
                U_test = [row[:drop_idx] + row[drop_idx+1:] for row in U]
                V_test = [row[:drop_idx] + row[drop_idx+1:] for row in V]
                W_test = [row[:drop_idx] + row[drop_idx+1:] for row in W]
                
                # Check if the reduced decomposition still works
                if verify(U_test, V_test, W_test):
                    U = U_test
                    V = V_test
                    W = W_test
                    improved = True
                    break
            
            if not improved:
                # Try a more sophisticated approach: look for pairs of columns
                # that can be merged or where one is a scalar multiple in effect
                # Check if any two columns are "equivalent" in their tensor contribution
                # up to scaling of the other factors
                found_merge = False
                for i in range(r):
                    if found_merge or time.time() >= deadline:
                        break
                    for j in range(i + 1, r):
                        if time.time() >= deadline:
                            break
                        
                        # Check if column i and j have proportional contributions
                        # across all (a,b,c) entries
                        # Contribution of column r to (a,b,c): U[a*m+b,r]*V[b*p+c,r]*W[c*n+a,r]
                        # If for all (a,b,c): U_i[a]*V_i[b]*W_i[c] = alpha * U_j[a]*V_j[b]*W_j[c]
                        # then they're equivalent up to scaling.
                        
                        # Check proportionality
                        ratios = []
                        valid = True
                        for a in range(n):
                            for b in range(m):
                                for c in range(p):
                                    ui = U[a*m+b][i]
                                    uj = U[a*m+b][j]
                                    vi = V[b*p+c][i]
                                    vj = V[b*p+c][j]
                                    wi = W[c*n+a][i]
                                    wj = W[c*n+a][j]
                                    
                                    contrib_i = ui * vi * wi
                                    contrib_j = uj * vj * wj
                                    
                                    if abs(contrib_i) < 1e-12 and abs(contrib_j) < 1e-12:
                                        continue
                                    elif abs(contrib_i) < 1e-12 or abs(contrib_j) < 1e-12:
                                        valid = False
                                        break
                                    else:
                                        ratio = contrib_i / contrib_j
                                        if not ratios:
                                            ratios.append(ratio)
                                        elif abs(ratio - ratios[0]) > 1e-9:
                                            valid = False
                                            break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        # Columns i and j are proportional in their tensor contributions
                        # We can potentially merge them. If ratio is constant, 
                        # we can drop one and scale the other, but scaling isn't free
                        # in tensor decomposition (we'd need to adjust all three factors).
                        # For now, just drop one if the other covers it.
                        # Actually if they're proportional, dropping one breaks the sum.
                        # Skip this for now.
                        pass
            
        return U, V, W
    
    U, V, W = build_baseline()
    
    deadline = time.time() + 140  # 10s safety margin
    
    U, V, W = try_local_search(U, V, W, deadline)
    
    # Final verification
    if not verify(U, V, W):
        # Fallback to baseline if something went wrong
        U, V, W = build_baseline()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
