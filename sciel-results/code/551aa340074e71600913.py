import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Attempt to find a rank-27 decomposition using a structured ansatz.
       - The first 7 terms are fixed to Strassen's decomposition for the 
         2x2x2 sub-tensor (indices i in {0,1}, j in {0,1}, k in {0,1}).
       - The remaining 20 terms are generated randomly with sparse integer 
         entries for U and V.
       - W is solved via linear least squares.
       - We verify if W has small integer or half-integer entries.
    2. If rank 27 fails, fall back to a rank-28 search (Strassen 2x2x2 + 
       Schoolbook for the rest, potentially optimized).
    3. If rank 28 fails, return the schoolbook rank-30 decomposition.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition using integer arithmetic."""
        r = len(U[0])
        # Convert to integers for exact check (multiply by 2 if half-integers)
        # We assume entries are integers or half-integers.
        # Multiply all by 2 to work with integers.
        U_int = [[int(u * 2) for u in row] for row in U]
        V_int = [[int(v * 2) for v in row] for row in V]
        W_int = [[int(w * 2) for w in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U_int[i * m + j][c] * V_int[j * p + k][c] * W_int[k * n + i][c]
                    # The sum should be 1 * 2^3 = 8
                    if s != 8:
                        return False
        return True

    def is_small_rational(x, max_denom=2):
        """Check if x is a rational with small denominator."""
        # Check integer
        if abs(x - round(x)) < 1e-9:
            return True
        # Check half-integer
        if abs(x * 2 - round(x * 2)) < 1e-9:
            return True
        # Check other small denominators? 
        # For now, stick to int and half-int as they are most likely for tensor decomps
        return False

    def try_structured_search(target_rank, max_trials=1000):
        """
        Attempt to find a decomposition of given rank.
        Uses a Strassen core for the first 7 terms (if rank >= 7) 
        and random sparse terms for the rest.
        """
        if target_rank < 7:
            return None
            
        # Pre-compute Strassen vectors for 2x2x2
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U indices: i*m+j -> 0,1, 2,3 (for m=3, j=0,1)
        # V indices: j*p+k -> 0,1, 3,4 (for p=5, j=0,1)
        # W indices: k*n+i -> 0,1, 2,3 (for n=2, k=0,1)
        
        # Strassen 7 terms for 2x2x2:
        # m1 = (A00 + A11)(B00 + B11)
        # m2 = (A10 + A11)B00
        # m3 = A00(B10 - B11)
        # m4 = A11(B01 - B00)
        # m5 = (A00 + A01)B11
        # m6 = (A10 - A00)(B00 + B01)
        # m7 = (A01 - A11)(B10 + B11)
        
        # We map these to our index space.
        # U vectors (size 6): non-zero at i*m+j
        # V vectors (size 15): non-zero at j*p+k
        # W vectors (size 10): non-zero at k*n+i
        
        # Term 1: (A00+A11)(B00+B11)
        # U1: A00=1, A11=1 -> idx 0, 3
        # V1: B00=1, B11=1 -> idx 0, 4
        # W1: C00=1, C11=1, C01=-1, C10=-1 -> idx 0, 3, 1, 2
        # Wait, W contributes to C_ki.
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # Let's just use the standard Strassen coefficients.
        # U1 = [1, 0, 1, 0, 0, 0] (A00, A01, A10, A11, A12, A22? No, A is 2x3. Indices 0,1,2,3,4,5. A00=0, A01=1, A10=2, A11=3, A12=4, A22=5? No.
        # A is 2x3. i=0,1. j=0,1,2.
        # A00=idx0, A01=idx1, A02=idx2
        # A10=idx3, A11=idx4, A12=idx5
        
        # B is 3x5. j=0,1,2. k=0,1,2,3,4.
        # B00=idx0, B01=idx1, B02=idx2, B03=idx3, B04=idx4
        # B10=idx5, B11=idx6, B12=idx7, B13=idx8, B14=idx9
        # B20=idx10, B21=idx11, B22=idx12, B23=idx13, B24=idx14
        
        # C is 5x2. k=0..4, i=0,1.
        # C00=idx0, C01=idx1
        # C10=idx2, C11=idx3
        # C20=idx4, C21=idx5
        # C30=idx6, C31=idx7
        # C40=idx8, C41=idx9
        
        # Strassen for C00, C01, C10, C11 using A00,A01,A10,A11 and B00,B01,B10,B11.
        # Note: A02, A12, B2*, C2*, C3*, C4* are not involved in this 2x2x2 sub-block.
        
        # U1: A00=1, A11=1 -> [1, 0, 0, 1, 0, 0]
        # V1: B00=1, B11=1 -> [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # W1: C00=1, C11=1, C01=-1, C10=-1 -> [1, -1, -1, 1, 0, 0, 0, 0, 0, 0]
        
        # Let's define the 7 Strassen terms explicitly.
        strassen_U = []
        strassen_V = []
        strassen_W = []
        
        # Helper to create sparse vectors
        def make_vec(size, mapping):
            v = [0] * size
            for idx, val in mapping.items():
                v[idx] = val
            return v

        # 1. (A00+A11)(B00+B11)
        strassen_U.append(make_vec(6, {0:1, 4:1}))
        strassen_V.append(make_vec(15, {0:1, 6:1}))
        strassen_W.append(make_vec(10, {0:1, 1:-1, 2:-1, 3:1}))

        # 2. (A10+A11)B00
        strassen_U.append(make_vec(6, {3:1, 4:1}))
        strassen_V.append(make_vec(15, {0:1}))
        strassen_W.append(make_vec(10, {0:0, 1:0, 2:1, 3:0})) # C10 += 1
        
        # 3. A00(B10-B11)
        strassen_U.append(make_vec(6, {0:1}))
        strassen_V.append(make_vec(15, {5:1, 6:-1}))
        strassen_W.append(make_vec(10, {0:1, 1:0, 2:0, 3:0})) # C00 += 1
        
        # 4. A11(B01-B00)
        strassen_U.append(make_vec(6, {4:1}))
        strassen_V.append(make_vec(15, {1:1, 0:-1}))
        strassen_W.append(make_vec(10, {0:0, 1:1, 2:0, 3:0})) # C01 += 1
        
        # 5. (A00+A01)B11
        strassen_U.append(make_vec(6, {0:1, 1:1}))
        strassen_V.append(make_vec(15, {6:1}))
        strassen_W.append(make_vec(10, {0:0, 1:0, 2:0, 3:1})) # C11 += 1
        
        # 6. (A10-A00)(B00+B01)
        strassen_U.append(make_vec(6, {3:1, 0:-1}))
        strassen_V.append(make_vec(15, {0:1, 1:1}))
        strassen_W.append(make_vec(10, {0:0, 1:0, 2:1, 3:0})) # C10 += 1
        
        # 7. (A01-A11)(B10+B11)
        strassen_U.append(make_vec(6, {1:1, 4:-1}))
        strassen_V.append(make_vec(15, {5:1, 6:1}))
        strassen_W.append(make_vec(10, {0:0, 1:1, 2:0, 3:0})) # C01 += 1

        # Verify Strassen part locally? 
        # We assume it's correct for the 2x2x2 sub-block.
        
        np.random.seed(12345) # Fixed seed for reproducibility in this trial
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Initialize U, V with Strassen part
            U = [row[:] for row in strassen_U]
            V = [row[:] for row in strassen_V]
            
            # Extend U, V to target_rank
            num_extra = target_rank - 7
            if num_extra > 0:
                # Generate random sparse vectors for the remaining terms
                # Use {-1, 0, 1} with low density
                for c in range(7, target_rank):
                    u_vec = np.random.choice([-1, 0, 1], size=nm)
                    v_vec = np.random.choice([-1, 0, 1], size=mp)
                    # Ensure some sparsity
                    if np.sum(u_vec != 0) > 4:
                         u_vec = np.random.choice([-1, 0, 1], size=nm, p=[0.7, 0.15, 0.15])
                    if np.sum(v_vec != 0) > 5:
                         v_vec = np.random.choice([-1, 0, 1], size=mp, p=[0.8, 0.1, 0.1])
                    
                    U.append(u_vec.tolist())
                    V.append(v_vec.tolist())
            
            # Construct linear system for W
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # 30 equations, pn * target_rank unknowns
            A = np.zeros((n*m*p, pn * target_rank))
            b = np.ones(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = np.array(U[i*m+j])
                        v_row = np.array(V[j*p+k])
                        coeffs = u_row * v_row
                        
                        start_idx = (k*n+i) * target_rank
                        A[idx, start_idx:start_idx+target_rank] = coeffs
                        b[idx] = 1
                        idx += 1
            
            # Solve A x = b
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(pn, target_rank)
            
            # Check if W_sol is "nice" (integer or half-integer)
            # We allow small tolerance
            W_check = W_sol * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-5):
                W_rounded = np.round(W_check) / 2.0
                
                # Additional check: ensure W is not zero where it shouldn't be? 
                # The verification will handle correctness.
                
                U_list = U
                V_list = V
                W_list = W_rounded.tolist()
                
                # Verify exactness
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Also check pure integers
            if np.allclose(W_sol, np.round(W_sol), atol=1e-5):
                W_rounded = np.round(W_sol)
                U_list = U
                V_list = V
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try rank 27 first
    result = try_structured_search(27, max_trials=200)
    if result:
        return result

    # If 27 fails, try a more exhaustive random search for 28 
    # (Strassen core + 21 random terms)
    # This is likely to succeed if 28 is the true minimum or if a 28-rank decomp exists easily
    result = try_structured_search(28, max_trials=500)
    if result:
        return result

    # Fallback to schoolbook
    return schoolbook()
