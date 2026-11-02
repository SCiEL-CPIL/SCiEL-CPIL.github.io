import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    # We will build a decomposition for M_{n,m,p} using recursive block splitting.
    # The decomposition is represented as a list of terms: (U_row, V_col, W_row, coeff)
    # where U_row is an index into U (0..n*m-1), V_col into V (0..m*p-1), W_row into W (0..p*n-1)
    # Wait, the interface requires U to be (n*m, rank).
    # So each term corresponds to a column in U, V, W.
    # U[i*m+j, r] = coeff * delta_{U_row, i*m+j}
    # V[j*p+k, r] = coeff * delta_{V_col, j*p+k}  -- Wait, V index is j*p+k?
    # Let's check the interface again:
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    
    # The multiplication tensor T_{i,j,k} = 1.
    # Decomposition: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # This implies that for a term r to contribute to T_{i,j,k}, we need:
    # U_{i*m+j, r} != 0, V_{j*p+k, r} != 0, W_{k*n+i, r} != 0.
    # If we set U_{row_u, r}=1, V_{row_v, r}=1, W_{row_w, r}=1, it contributes 1 to T_{i,j,k} 
    # ONLY IF row_u == i*m+j AND row_v == j*p+k AND row_w == k*n+i.
    # This is the "sparse" or "schoolbook-like" representation where each term is a rank-1 tensor 
    # supported on a single (i,j,k) triple.
    # BUT, Strassen's algorithm uses terms that are sums of products.
    # Strassen for 2x2:
    # C11 = A11(B11-B22) + A21 B22 + A11 B11 ? No.
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mx is a product of two linear combinations of A and B blocks.
    # M1 = (A11+A22)(B11+B22)
    # This expands to A11B11 + A11B22 + A22B11 + A22B22.
    # In the tensor decomposition, a term like A11B11 corresponds to:
    # U[A11_idx, r] = 1, V[B11_idx, r] = 1, W[C11_idx, r] = 1
    # But Strassen combines these.
    # Actually, the rank of the tensor is the number of multiplications.
    # If we use Strassen, we have 7 multiplications for 2x2x2.
    # Each multiplication Mx involves:
    # 1. Computing linear combo of A: P = sum c_a A_a
    # 2. Computing linear combo of B: Q = sum c_b B_b
    # 3. Multiplying P * Q
    # 4. Adding to C: C_c += d_c * (P*Q)
    
    # In the tensor factorization U, V, W:
    # For each Strassen step (multiplication), we create ONE column r in U, V, W.
    # U[i*m+j, r] = coefficient of A[i,j] in P
    # V[j'*p+k, r] = coefficient of B[j',k] in Q  (Note: V index is j*p+k, so row j, col k)
    # W[k'*n+i', r] = coefficient for C[i',k'] in the addition step.
    
    # Wait, the V matrix in the interface has shape (m*p, rank).
    # The index is j*p+k.
    # In the multiplication A (n x m) * B (m x p):
    # A indices: i (0..n-1), j (0..m-1). U index: i*m+j.
    # B indices: j (0..m-1), k (0..p-1). V index: j*p+k.
    # C indices: i (0..n-1), k (0..p-1). W index: k*n+i.
    
    # Let's implement a recursive function that returns a list of "terms".
    # A term is a tuple: (u_idx, v_idx, w_idx, coeff)
    # But Strassen terms are not single indices. They are linear combinations.
    # So for one Strassen multiplication M = P*Q:
    # P = sum_{(i,j) in A_block} c_a_{i,j} A[i,j]
    # Q = sum_{(j',k) in B_block} c_b_{j',k} B[j',k]
    # Contribution to C: sum_{(i',k') in C_block} c_c_{i',k'} C[i',k'] += M
    # So for this single multiplication, we add ONE column to U, V, W.
    # U[row, r] = c_a_{row}
    # V[row, r] = c_b_{row}
    # W[row, r] = c_c_{row}
    # Where row in U is i*m+j, row in V is j*p+k, row in W is k*n+i.
    
    # This works perfectly! The rank is the total number of multiplications.
    
    def strassen_2x2x2(terms, a_offset, b_offset, c_offset, scale=1.0):
        """
        Adds Strassen terms for a 2x2x2 block.
        a_offset: (i0, j0) top-left of A block (2x2)
        b_offset: (j0, k0) top-left of B block (2x2)
        c_offset: (i0, k0) top-left of C block (2x2)
        """
        i0, j0 = a_offset
        j0_b, k0 = b_offset
        i0_c, k0_c = c_offset
        
        # A blocks:
        # A11: (i0, j0)
        # A12: (i0, j0+1)
        # A21: (i0+1, j0)
        # A22: (i0+1, j0+1)
        
        # B blocks:
        # B11: (j0, k0)
        # B12: (j0, k0+1)
        # B21: (j0+1, k0)
        # B22: (j0+1, k0+1)
        
        # C blocks:
        # C11: (i0, k0)
        # C12: (i0, k0+1)
        # C21: (i0+1, k0)
        # C22: (i0+1, k0+1)
        
        # We define 7 multiplications.
        # For each, we define the linear combos.
        
        def add_term(a_coeffs, b_coeffs, c_coeffs):
            """
            a_coeffs: dict {(i,j): coeff}
            b_coeffs: dict {(j,k): coeff}
            c_coeffs: dict {(i,k): coeff}
            """
            # Create one column
            # We need to map local indices to global indices.
            # This function is called within the context of the block, so offsets are applied by caller?
            # No, better to pass global indices.
            pass

        # Let's just directly generate the 7 terms with global indices.
        
        # M1 = (A11+A22)(B11+B22)
        # A: A11 at (i0, j0), A22 at (i0+1, j0+1)
        # B: B11 at (j0, k0), B22 at (j0+1, k0+1)
        # C: C11 at (i0, k0), C22 at (i0+1, k0+1)
        # C11 += M1, C22 += M1
        
        # Term 1
        a1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        b1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        c1 = {(i0, k0): 1, (i0+1, k0+1): 1}
        
        # M2 = (A21+A22)B11
        # A: A21 at (i0+1, j0), A22 at (i0+1, j0+1)
        # B: B11 at (j0, k0)
        # C: C21 += M2, C22 -= M2
        
        a2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        b2 = {(j0, k0): 1}
        c2 = {(i0+1, k0): 1, (i0+1, k0+1): -1}
        
        # M3 = A11(B12-B22)
        # A: A11 at (i0, j0)
        # B: B12 at (j0, k0+1), B22 at (j0+1, k0+1)
        # C: C12 += M3, C22 += M3
        
        a3 = {(i0, j0): 1}
        b3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        c3 = {(i0, k0+1): 1, (i0+1, k0+1): 1}
        
        # M4 = A22(B21-B11)
        # A: A22 at (i0+1, j0+1)
        # B: B21 at (j0+1, k0), B11 at (j0, k0)
        # C: C11 += M4, C21 += M4
        
        a4 = {(i0+1, j0+1): 1}
        b4 = {(j0+1, k0): 1, (j0, k0): -1}
        c4 = {(i0, k0): 1, (i0+1, k0): 1}
        
        # M5 = (A11+A12)B22
        # A: A11 at (i0, j0), A12 at (i0, j0+1)
        # B: B22 at (j0+1, k0+1)
        # C: C11 -= M5, C12 += M5
        
        a5 = {(i0, j0): 1, (i0, j0+1): 1}
        b5 = {(j0+1, k0+1): 1}
        c5 = {(i0, k0): -1, (i0, k0+1): 1}
        
        # M6 = (A21-A11)(B11+B12)
        # A: A21 at (i0+1, j0), A11 at (i0, j0)
        # B: B11 at (j0, k0), B12 at (j0, k0+1)
        # C: C22 += M6
        
        a6 = {(i0+1, j0): 1, (i0, j0): -1}
        b6 = {(j0, k0): 1, (j0, k0+1): 1}
        c6 = {(i0+1, k0+1): 1}
        
        # M7 = (A12-A22)(B21+B22)
        # A: A12 at (i0, j0+1), A22 at (i0+1, j0+1)
        # B: B21 at (j0+1, k0), B22 at (j0+1, k0+1)
        # C: C11 += M7
        
        a7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        b7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        c7 = {(i0, k0): 1}
        
        terms_list = [
            (a1, b1, c1),
            (a2, b2, c2),
            (a3, b3, c3),
            (a4, b4, c4),
            (a5, b5, c5),
            (a6, b6, c6),
            (a7, b7, c7)
        ]
        
        for a, b, c in terms_list:
            # Convert to global indices and add to terms
            # We store terms as (u_idx, v_idx, w_idx, coeff)
            # But here we have multiple indices per term.
            # Wait, my previous logic: "For this single multiplication, we add ONE column to U, V, W."
            # U[row, r] = c_a_{row}. If a has multiple keys, U has multiple non-zeros in column r.
            # This is correct.
            
            # We need to accumulate into the final matrices.
            # Since we are building a list of terms first, let's store them as dicts.
            # Or better, just append to a list of (u_dict, v_dict, w_dict).
            pass

    # Let's restructure to collect all coefficients for each rank column.
    # rank_columns = list of (U_dict, V_dict, W_dict)
    
    def decompose(n, m, p):
        """
        Returns a list of (U_dict, V_dict, W_dict)
        U_dict: {i*m+j: coeff}
        V_dict: {j*p+k: coeff}
        W_dict: {k*n+i: coeff}
        """
        cols = []
        
        def recurse(n_curr, m_curr, p_curr, i0, j0, k0):
            # i0, j0, k0 are the top-left offsets in the ORIGINAL matrix
            # n_curr, m_curr, p_curr are the current block sizes
            
            if time.time() > deadline:
                return
            
            if n_curr == 1 and m_curr == 1 and p_curr == 1:
                # Schoolbook for 1x1x1
                u_idx = i0 * m + j0  # Wait, m is original m? No, m_curr is 1.
                # The index in U is based on the ORIGINAL dimensions n, m, p.
                # U index: i * m + j. Here i is global i, j is global j.
                # So u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # Try to split
            # We want to use Strassen where possible (2x2x2).
            # If any dimension is 2, we can potentially use Strassen if the others allow.
            # But Strassen is specifically for 2x2x2.
            # If we have 2x2xN, we can split N into 1s and 2s?
            # If we have 2x2x3, split into 2x2x2 + 2x2x1.
            # If we have 2x2x1, split into 2x1x1 + 2x1x1? Or 1x2x1 + 1x2x1?
            
            # Strategy:
            # If n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
            #   Check if we can do a 2x2x2 block.
            #   If yes, take a 2x2x2 block at the top-left.
            #   Apply Strassen.
            #   Recurse on the remainder.
            
            # Remainder logic:
            # If we take 2x2x2 from (i0, j0, k0) of size (n_curr, m_curr, p_curr):
            # Remainder is L-shaped.
            # 1. (n_curr-2) x m_curr x p_curr at (i0+2, j0, k0)
            # 2. 2 x (m_curr-2) x p_curr at (i0, j0+2, k0)
            # 3. 2 x 2 x (p_curr-2) at (i0, j0, k0+2)
            
            # This works if n_curr, m_curr, p_curr are all >= 2.
            
            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                # Apply Strassen to 2x2x2 block at (i0, j0, k0)
                # We need to generate the 7 terms for this block.
                
                # Define the 7 terms for Strassen 2x2x2
                # Local indices 0,1. Global offsets i0, j0, k0.
                
                # M1
                a1 = {(i0, j0): 1, (i0+1, j0+1): 1}
                b1 = {(j0, k0): 1, (j0+1, k0+1): 1}
                c1 = {(i0, k0): 1, (i0+1, k0+1): 1}
                cols.append((
                    {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                    {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                    {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
                ))
                
                # M2
                a2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
                b2 = {(j0, k0): 1}
                c2 = {(i0+1, k0): 1, (i0+1, k0+1): -1}
                cols.append((
                    {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                    {j0*p+k0: 1},
                    {(k0)*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
                ))
                
                # M3
                a3 = {(i0, j0): 1}
                b3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
                c3 = {(i0, k0+1): 1, (i0+1, k0+1): 1}
                cols.append((
                    {i0*m+j0: 1},
                    {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                    {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
                ))
                
                # M4
                a4 = {(i0+1, j0+1): 1}
                b4 = {(j0+1, k0): 1, (j0, k0): -1}
                c4 = {(i0, k0): 1, (i0+1, k0): 1}
                cols.append((
                    {(i0+1)*m+(j0+1): 1},
                    {(j0+1)*p+k0: 1, j0*p+k0: -1},
                    {k0*n+i0: 1, k0*n+(i0+1): 1}
                ))
                
                # M5
                a5 = {(i0, j0): 1, (i0, j0+1): 1}
                b5 = {(j0+1, k0+1): 1}
                c5 = {(i0, k0): -1, (i0, k0+1): 1}
                cols.append((
                    {i0*m+j0: 1, i0*m+(j0+1): 1},
                    {(j0+1)*p+(k0+1): 1},
                    {k0*n+i0: -1, (k0+1)*n+i0: 1}
                ))
                
                # M6
                a6 = {(i0+1, j0): 1, (i0, j0): -1}
                b6 = {(j0, k0): 1, (j0, k0+1): 1}
                c6 = {(i0+1, k0+1): 1}
                cols.append((
                    {(i0+1)*m+j0: 1, i0*m+j0: -1},
                    {j0*p+k0: 1, j0*p+(k0+1): 1},
                    {(k0+1)*n+(i0+1): 1}
                ))
                
                # M7
                a7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
                b7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
                c7 = {(i0, k0): 1}
                cols.append((
                    {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                    {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                    {k0*n+i0: 1}
                ))
                
                # Recurse on remainders
                # 1. (n_curr-2) x m_curr x p_curr at (i0+2, j0, k0)
                if n_curr > 2:
                    recurse(n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0)
                
                # 2. 2 x (m_curr-2) x p_curr at (i0, j0+2, k0)
                # Note: The A block is 2 x (m_curr-2). The B block is (m_curr-2) x p_curr.
                # Wait, the B block for the remainder must align with the A block's columns.
                # A is rows i0..i0+1, cols j0+2..j0+m_curr-1.
                # B is rows j0+2..j0+m_curr-1, cols k0..k0+p_curr-1.
                # C is rows i0..i0+1, cols k0..k0+p_curr-1.
                if m_curr > 2:
                    recurse(2, m_curr - 2, p_curr, i0, j0 + 2, k0)
                
                # 3. 2 x 2 x (p_curr-2) at (i0, j0, k0+2)
                # A is 2x2 at (i0, j0).
                # B is 2x(p_curr-2) at (j0, k0+2).
                # C is 2x(p_curr-2) at (i0, k0+2).
                if p_curr > 2:
                    recurse(2, 2, p_curr - 2, i0, j0, k0 + 2)
                
                return

            # If we can't do 2x2x2, try other splits
            # If n_curr > 1, split n into 1 and n-1
            # If m_curr > 1, split m into 1 and m-1
            # If p_curr > 1, split p into 1 and p-1
            
            # We need to be careful to avoid infinite recursion.
            # Always reduce one dimension.
            
            if n_curr > 1:
                # Split into 1 x m x p and (n-1) x m x p
                recurse(1, m_curr, p_curr, i0, j0, k0)
                recurse(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0)
                return
            elif m_curr > 1:
                # Split into n x 1 x p and n x (m-1) x p
                recurse(n_curr, 1, p_curr, i0, j0, k0)
                recurse(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0)
                return
            elif p_curr > 1:
                # Split into n x m x 1 and n x m x (p-1)
                recurse(n_curr, m_curr, 1, i0, j0, k0)
                recurse(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1)
                return
            else:
                # 1x1x1
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

        recurse(n, m, p, 0, 0, 0)
        return cols

    cols = decompose(n, m, p)
    
    # Verify rank
    rank = len(cols)
    
    # Build U, V, W
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    # Quick check: does it work?
    # We can't easily check without random matrices, but we can check the tensor identity.
    # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Should be 1 for all i,j,k.
    
    # Let's spot check one term.
    # For 3x4x6:
    # n=3, m=4, p=6.
    # Recurse(3,4,6,0,0,0)
    # n>=2, m>=2, p>=2 -> Strassen 2x2x2 at (0,0,0)
    # Remainders:
    # 1. (3-2)=1 x 4 x 6 at (2,0,0)
    # 2. 2 x (4-2)=2 x 6 at (0,2,0)
    # 3. 2 x 2 x (6-2)=4 at (0,0,2)
    
    # Recurse(1,4,6,2,0,0)
    # n=1, so split n? No, n=1.
    # m=4>1 -> split m into 1 and 3.
    # Recurse(1,1,6,2,0,0) and Recurse(1,3,6,2,1,0)
    
    # Recurse(1,1,6,2,0,0)
    # n=1, m=1, p=6>1 -> split p into 1 and 5.
    # Recurse(1,1,1,2,0,0) -> Schoolbook 1 term.
    # Recurse(1,1,5,2,0,1) -> ...
    # This will result in 6 terms for this 1x1x6 block.
    
    # Total terms:
    # Strassen 2x2x2: 7 terms.
    # Remainder 1: 1x4x6.
    #   Splits to 1x1x6 (6 terms) + 1x3x6.
    #   1x3x6 splits to 1x1x6 (6 terms) + 1x2x6.
    #   1x2x6 splits to 1x1x6 (6 terms) + 1x1x6 (6 terms).
    #   Total for 1x4x6: 6 + 6 + 6 + 6 = 24 terms?
    #   Wait. 1x1x6 is 6 terms.
    #   1x3x6 -> 1x1x6 (6) + 1x2x6.
    #   1x2x6 -> 1x1x6 (6) + 1x1x6 (6).
    #   So 1x3x6 is 6+12=18.
    #   1x4x6 is 6+18=24.
    # Remainder 2: 2x2x6.
    #   n=2, m=2, p=6.
    #   Can do Strassen 2x2x2? Yes.
    #   Strassen 2x2x2 (7 terms) + Remainder 2x2x4.
    #   2x2x4 -> Strassen 2x2x2 (7) + Remainder 2x2x2 (7).
    #   Total for 2x2x6: 7 + 7 + 7 = 21 terms.
    # Remainder 3: 2x2x4.
    #   Same as above: 7 + 7 = 14 terms.
    
    # Total rank = 7 (first Strassen) + 24 (Rem 1) + 21 (Rem 2) + 14 (Rem 3) = 66.
    
    # This matches the knowledge: "reducing rank from 68 to 66".
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
