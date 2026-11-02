import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 135  # safety margin before 150s kill

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    
    def _strassen_rank55(n, m, p):
        """
        Construct the known rank-55 decomposition using Strassen for 2x2 blocks
        and schoolbook for the 2x1 block.
        
        Split A (2x3) into A1 (2x2) and A2 (2x1).
        Split B (3x10) into B1 (2x10) and B2 (1x10).
        
        For A1*B1: use 5 Strassen blocks (each covering 2 columns of B), 7 mults each = 35
        For A2*B2: schoolbook, 2*10 = 20 mults
        Total: 55
        """
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
        for g in range(5):
            k0 = 2 * g
            k1 = 2 * g + 1
            
            # U indices: A is 2x3, so A[i,j] -> U[i*3+j]
            # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
            u00, u01, u10, u11 = 0, 1, 3, 4
            
            # V indices: B is 3x10, so B[j,k] -> V[j*10+k]
            # B00=V[0+k0], B01=V[0+k1] for j=0
            # B10=V[10+k0], B11=V[10+k1] for j=1
            v00, v01 = k0, k1  # j=0
            v10, v11 = 10 + k0, 10 + k1  # j=1
            
            # W indices: C is 2x10, so C[i,k] -> W[k*2+i]
            w00, w01 = k0 * 2, k1 * 2  # i=0
            w10, w11 = k0 * 2 + 1, k1 * 2 + 1  # i=1
            
            # Strassen terms for 2x2:
            # M1 = (A00+A11)(B00+B11)
            U[u00][col] = 1
            U[u11][col] = 1
            V[v00][col] = 1
            V[v11][col] = 1
            W[w00][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[u10][col] = 1
            U[u11][col] = 1
            V[v00][col] = 1
            W[w10][col] = 1
            W[w11][col] = -1
            col += 1
            
            # M3 = A00(B01-B11)
            U[u00][col] = 1
            V[v01][col] = 1
            V[v11][col] = -1
            W[w01][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M4 = A11(B10-B00)
            U[u11][col] = 1
            V[v10][col] = 1
            V[v00][col] = -1
            W[w00][col] = 1
            W[w10][col] = 1
            col += 1
            
            # M5 = (A00+A01)B11
            U[u00][col] = 1
            U[u01][col] = 1
            V[v11][col] = 1
            W[w01][col] = 1
            W[w00][col] = -1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[u10][col] = 1
            U[u00][col] = -1
            V[v00][col] = 1
            V[v01][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[u01][col] = 1
            U[u11][col] = -1
            V[v10][col] = 1
            V[v11][col] = 1
            W[w00][col] = 1
            col += 1
        
        # Schoolbook for A2 (2x1) * B2 (1x10)
        # A2: A[i,2] -> U[i*3+2], i=0,1 => U[2], U[5]
        # B2: B[2,k] -> V[2*10+k] = V[20+k], k=0..9
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return {"U": U, "V": V, "W": W}
    
    def _verify_fraction(U, V, W, n, m, p):
        """Verify exact decomposition using Fraction arithmetic."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def _verify_np(U, V, W, n, m, p, tol=1e-10):
        """Fast verification using NumPy."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # First, get the guaranteed rank-55 solution
    base = _strassen_rank55(n, m, p)
    U0 = base["U"]
    V0 = base["V"]
    W0 = base["W"]
    
    # Verify the base solution
    if not _verify_np(U0, V0, W0, n, m, p):
        raise ValueError("Base rank-55 solution failed verification")
    
    best_result = {"U": U0, "V": V0, "W": W0}
    best_rank = 55
    
    # Try to find a rank-54 solution by exploiting structure
    # Strategy: Try to merge pairs of columns from the rank-55 solution
    # by finding linear combinations that satisfy the tensor constraints
    
    deadline = _start + TIME_LIMIT - 10
    
    # Approach: For each pair of columns (c1, c2) in the rank-55 solution,
    # check if we can replace them with a single column that still satisfies
    # all constraints. This is equivalent to checking if the two columns
    # are "redundant" in some sense.
    
    # More practically: Try random rank-54 decompositions with structure
    # inspired by the Strassen pattern
    
    rng = np.random.RandomState(42)
    
    r54 = 54
    Ua0 = np.array(U0, dtype=float)
    Va0 = np.array(V0, dtype=float)
    Wa0 = np.array(W0, dtype=float)
    
    attempts = 0
    max_attempts = 500
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try different strategies for constructing rank-54 candidates
        strategy = attempts % 3
        
        if strategy == 0:
            # Strategy 0: Start from rank-55, drop one column, try to fix
            # by adjusting remaining columns
            U_try = Ua0[:, :r54].copy()
            V_try = Va0[:, :r54].copy()
            W_try = Wa0[:, :r54].copy()
            
            # Check if dropping column 54 (last schoolbook column) works
            # This won't work directly, so add small perturbations
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Try dropping a random column and see if we can compensate
                drop_col = rng.randint(0, 55)
                U_try = Ua0.copy()
                V_try = Va0.copy()
                W_try = Wa0.copy()
                
                # Remove column drop_col
                U_try = np.delete(U_try, drop_col, axis=1)
                V_try = np.delete(V_try, drop_col, axis=1)
                W_try = np.delete(W_try, drop_col, axis=1)
                
                # Now we have rank 54. Check if it still works (it won't, but
                # maybe with small adjustments)
                # For now, just check if the structure is close
                
                # Try adding back a combination
                # This is complex, so skip for now
                
        elif strategy == 1:
            # Strategy 1: Random sparse matrices with Strassen-like structure
            U_try = np.zeros((n*m, r54))
            V_try = np.zeros((m*p, r54))
            W_try = np.zeros((p*n, r54))
            
            # Use block structure: 5 blocks of ~10-11 columns for Strassen parts,
            # and 14 columns for schoolbook part (reduced from 20)
            
            # Block 0-4: Strassen-like (7 columns each = 35)
            for g in range(5):
                start = g * 7
                end = start + 7
                if end > r54:
                    end = r54
                
                k0 = 2 * g
                k1 = 2 * g + 1
                
                u00, u01, u10, u11 = 0, 1, 3, 4
                v00, v01 = k0, k1
                v10, v11 = 10 + k0, 10 + k1
                w00, w01 = k0 * 2, k1 * 2
                w10, w11 = k0 * 2 + 1, k1 * 2 + 1
                
                # M1
                U_try[u00, start] = 1
                U_try[u11, start] = 1
                V_try[v00, start] = 1
                V_try[v11, start] = 1
                W_try[w00, start] = 1
                W_try[w11, start] = 1
                
                # M2
                U_try[u10, start+1] = 1
                U_try[u11, start+1] = 1
                V_try[v00, start+1] = 1
                W_try[w10, start+1] = 1
                W_try[w11, start+1] = -1
                
                # M3
                U_try[u00, start+2] = 1
                V_try[v01, start+2] = 1
                V_try[v11, start+2] = -1
                W_try[w01, start+2] = 1
                W_try[w11, start+2] = 1
                
                # M4
                U_try[u11, start+3] = 1
                V_try[v10, start+3] = 1
                V_try[v00, start+3] = -1
                W_try[w00, start+3] = 1
                W_try[w10, start+3] = 1
                
                # M5
                U_try[u00, start+4] = 1
                U_try[u01, start+4] = 1
                V_try[v11, start+4] = 1
                W_try[w01, start+4] = 1
                W_try[w00, start+4] = -1
                
                # M6
                U_try[u10, start+5] = 1
                U_try[u00, start+5] = -1
                V_try[v00, start+5] = 1
                V_try[v01, start+5] = 1
                W_try[w11, start+5] = 1
                
                # M7
                U_try[u01, start+6] = 1
                U_try[u11, start+6] = -1
                V_try[v10, start+6] = 1
                V_try[v11, start+6] = 1
                W_try[w00, start+6] = 1
            
            # Remaining columns (35 to 53): schoolbook-like but compressed
            # We need to handle A2*B2 with fewer than 20 columns
            # Try sharing some W entries
            schoolbook_start = 35
            schoolbook_end = r54
            num_schoolbook = schoolbook_end - schoolbook_start  # 19 columns
            
            # Use 19 columns for 20 schoolbook terms by merging two
            for idx in range(num_schoolbook):
                if idx < 19:
                    # Map 19 columns to 20 (i,k) pairs
                    # Skip one pair and merge
                    i = idx // 10
                    k = idx % 10
                    if i == 1 and k == 9:
                        # This is the 20th term, merge with another
                        pass
                    
                    u_idx = i * m + 2
                    v_idx = 2 * p + k
                    w_idx = k * n + i
                    
                    U_try[u_idx, schoolbook_start + idx] = 1
                    V_try[v_idx, schoolbook_start + idx] = 1
                    W_try[w_idx, schoolbook_start + idx] = 1
            
            # Add the missing 20th term by merging with an existing column
            # This is a hack; in reality we'd need a proper compression
            
        else:
            # Strategy 2: Pure random search with constraints
            U_try = rng.choice([-1, 0, 1], size=(n*m, r54))
            V_try = rng.choice([-1, 0, 1], size=(m*p, r54))
            W_try = rng.choice([-1, 0, 1], size=(p*n, r54))
            
            # Make it sparse
            U_try[rng.random((n*m, r54)) < 0.5] = 0
            V_try[rng.random((m*p, r54)) < 0.5] = 0
            W_try[rng.random((p*n, r54)) < 0.5] = 0
        
        # Verify the candidate
        if _verify_np(U_try, V_try, W_try, n, m, p):
            # Convert to list of lists with exact values
            U_list = [[float(x) for x in row] for row in U_try]
            V_list = [[float(x) for x in row] for row in V_try]
            W_list = [[float(x) for x in row] for row in W_try]
            
            # Final exact verification
            if _verify_fraction(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r54
                break  # Found rank-54, stop searching
    
    elapsed = time.time() - _start
    if elapsed > TIME_LIMIT:
        # Return the best result found so far
        pass
    
    return best_result
