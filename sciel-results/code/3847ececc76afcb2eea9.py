import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    target = 1.0 if i == k else 0.0
                    if not np.isclose(val, target, atol=1e-6):
                        return False
        return True

    def build_decomposition(rank_terms):
        """Build U, V, W matrices from a list of (u_vals, v_vals, w_vals) tuples."""
        r = len(rank_terms)
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for t, (u_vals, v_vals, w_vals) in enumerate(rank_terms):
            for (i, j), val in u_vals.items():
                U[i*m+j, t] = val
            for (j, k), val in v_vals.items():
                V[j*p+k, t] = val
            for (k, i), val in w_vals.items():
                W[k*n+i, t] = val
        
        return U, V, W

    # --- Strassen 2x2 helper ---
    def strassen_2x2_terms(i0, i1, j0, j1, k0, k1):
        """Return list of 7 (u, v, w) tuples for Strassen 2x2 multiplication."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append((
            {(i0,j0):1, (i1,j1):1},
            {(j0,k0):1, (j1,k1):1},
            {(k0,i0):1, (k1,i1):1}
        ))
        # M2 = (A10+A11)B00
        terms.append((
            {(i1,j0):1, (i1,j1):1},
            {(j0,k0):1},
            {(k0,i1):1, (k1,i1):-1}
        ))
        # M3 = A00(B01-B11)
        terms.append((
            {(i0,j0):1},
            {(j0,k1):1, (j1,k1):-1},
            {(k1,i0):1, (k1,i1):1}
        ))
        # M4 = A11(B10-B00)
        terms.append((
            {(i1,j1):1},
            {(j1,k0):1, (j0,k0):-1},
            {(k0,i0):1, (k0,i1):1}
        ))
        # M5 = (A00+A01)B11
        terms.append((
            {(i0,j0):1, (i0,j1):1},
            {(j1,k1):1},
            {(k0,i0):-1, (k1,i0):1}
        ))
        # M6 = (A10-A00)(B00+B01)
        terms.append((
            {(i1,j0):1, (i0,j0):-1},
            {(j0,k0):1, (j0,k1):1},
            {(k1,i1):1}
        ))
        # M7 = (A01-A11)(B10+B11)
        terms.append((
            {(i0,j1):1, (i1,j1):-1},
            {(j1,k0):1, (j1,k1):1},
            {(k0,i0):1}
        ))
        return terms

    # --- Schoolbook for small blocks ---
    def schoolbook_terms(i_list, j_list, k_list):
        """Return list of (u, v, w) tuples for schoolbook multiplication of specified blocks."""
        terms = []
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    terms.append((
                        {(i,j):1},
                        {(j,k):1},
                        {(k,i):1}
                    ))
        return terms

    # --- Build the 2+3 block decomposition ---
    # Split indices: top={0,1}, bottom={2,3,4}
    top = [0, 1]
    bot = [2, 3, 4]
    
    all_terms = []
    
    # C22 (i in top, k in top) = A22*B22 + A23*B32
    # A22*B22: 2x2*2x2 -> Strassen
    all_terms.extend(strassen_2x2_terms(top[0], top[1], top[0], top[1], top[0], top[1]))
    # A23*B32: A is 2x3 (i in top, j in bot), B is 3x2 (j in bot, k in top)
    # Split j: j={2,3} and j={4}
    # A23a (2x2, j=2,3) * B32a (2x2, j=2,3): Strassen
    all_terms.extend(strassen_2x2_terms(top[0], top[1], 2, 3, top[0], top[1]))
    # A23b (2x1, j=4) * B32b (1x2, j=4): schoolbook 2*1*2 = 4
    all_terms.extend(schoolbook_terms(top, [4], top))
    
    # C23 (i in top, k in bot) = A22*B23 + A23*B33
    # A22*B23: A is 2x2 (i in top, j in top), B is 2x3 (j in top, k in bot)
    # Split k: k={2,3} and k={4}
    # A22 * B23a (2x2, k=2,3): Strassen
    all_terms.extend(strassen_2x2_terms(top[0], top[1], top[0], top[1], 2, 3))
    # A22 * B23b (2x1, k=4): schoolbook 2*2*1 = 4
    all_terms.extend(schoolbook_terms(top, top, [4]))
    
    # A23*B33: A is 2x3 (i in top, j in bot), B is 3x3 (j in bot, k in bot)
    # Split j: j={2,3} and j={4}
    # A23a (2x2, j=2,3) * B33a (2x2, j=2,3, k=2,3): Strassen
    all_terms.extend(strassen_2x2_terms(top[0], top[1], 2, 3, 2, 3))
    # A23a (2x2, j=2,3) * B33b (2x1, j=2,3, k=4): schoolbook 2*2*1 = 4
    all_terms.extend(schoolbook_terms(top, [2,3], [4]))
    # A23b (2x1, j=4) * B33b (1x3, j=4, k=2,3,4): schoolbook 2*1*3 = 6
    all_terms.extend(schoolbook_terms(top, [4], bot))
    
    # C32 (i in bot, k in top) = A32*B22 + A33*B32
    # A32*B22: A is 3x2 (i in bot, j in top), B is 2x2 (j in top, k in top)
    # Split i: i={2,3} and i={4}
    # A32a (2x2, i=2,3) * B22: Strassen
    all_terms.extend(strassen_2x2_terms(2, 3, top[0], top[1], top[0], top[1]))
    # A32b (1x2, i=4) * B22: schoolbook 1*2*2 = 4
    all_terms.extend(schoolbook_terms([4], top, top))
    
    # A33*B32: A is 3x3 (i in bot, j in bot), B is 3x2 (j in bot, k in top)
    # Split j: j={2,3} and j={4}
    # A33a (2x2, i=2,3; j=2,3) * B32a (2x2, j=2,3; k in top): Strassen
    all_terms.extend(strassen_2x2_terms(2, 3, 2, 3, top[0], top[1]))
    # A33b (2x1, i=2,3; j=4) * B32b (1x2, j=4; k in top): schoolbook 2*1*2 = 4
    all_terms.extend(schoolbook_terms([2,3], [4], top))
    # A33c (1x2, i=4; j=2,3) * B32a (2x2, j=2,3; k in top): schoolbook 1*2*2 = 4
    all_terms.extend(schoolbook_terms([4], [2,3], top))
    # A33d (1x1, i=4; j=4) * B32b (1x2, j=4; k in top): schoolbook 1*1*2 = 2
    all_terms.extend(schoolbook_terms([4], [4], top))
    
    # C33 (i in bot, k in bot) = A32*B23 + A33*B33
    # A32*B23: A is 3x2 (i in bot, j in top), B is 2x3 (j in top, k in bot)
    # Split i: i={2,3} and i={4}; split k: k={2,3} and k={4}
    # A32a (2x2, i=2,3) * B23a (2x2, k=2,3): Strassen
    all_terms.extend(strassen_2x2_terms(2, 3, top[0], top[1], 2, 3))
    # A32a (2x2, i=2,3) * B23b (2x1, k=4): schoolbook 2*2*1 = 4
    all_terms.extend(schoolbook_terms([2,3], top, [4]))
    # A32b (1x2, i=4) * B23a (2x2, k=2,3): schoolbook 1*2*2 = 4
    all_terms.extend(schoolbook_terms([4], top, [2,3]))
    # A32b (1x2, i=4) * B23b (2x1, k=4): schoolbook 1*2*1 = 2
    all_terms.extend(schoolbook_terms([4], top, [4]))
    
    # A33*B33: A is 3x3 (i in bot, j in bot), B is 3x3 (j in bot, k in bot)
    # Split i: i={2,3} and i={4}; split j: j={2,3} and j={4}; split k: k={2,3} and k={4}
    # C33a (i=2,3; k=2,3) = A33aa*B33aa + A33ab*B33ba
    # A33aa (2x2, i=2,3; j=2,3) * B33aa (2x2, j=2,3; k=2,3): Strassen
    all_terms.extend(strassen_2x2_terms(2, 3, 2, 3, 2, 3))
    # A33ab (2x1, i=2,3; j=4) * B33ba (1x2, j=4; k=2,3): schoolbook 2*1*2 = 4
    all_terms.extend(schoolbook_terms([2,3], [4], [2,3]))
    
    # C33b (i=2,3; k=4) = A33aa*B33ab + A33ab*B33bb
    # A33aa (2x2, i=2,3; j=2,3) * B33ab (2x1, j=2,3; k=4): schoolbook 2*2*1 = 4
    all_terms.extend(schoolbook_terms([2,3], [2,3], [4]))
    # A33ab (2x1, i=2,3; j=4) * B33bb (1x1, j=4; k=4): schoolbook 2*1*1 = 2
    all_terms.extend(schoolbook_terms([2,3], [4], [4]))
    
    # C33c (i=4; k=2,3) = A33ba*B33aa + A33bb*B33ba
    # A33ba (1x2, i=4; j=2,3) * B33aa (2x2, j=2,3; k=2,3): schoolbook 1*2*2 = 4
    all_terms.extend(schoolbook_terms([4], [2,3], [2,3]))
    # A33bb (1x1, i=4; j=4) * B33ba (1x2, j=4; k=2,3): schoolbook 1*1*2 = 2
    all_terms.extend(schoolbook_terms([4], [4], [2,3]))
    
    # C33d (i=4; k=4) = A33ba*B33ab + A33bb*B33bb
    # A33ba (1x2, i=4; j=2,3) * B33ab (2x1, j=2,3; k=4): schoolbook 1*2*1 = 2
    all_terms.extend(schoolbook_terms([4], [2,3], [4]))
    # A33bb (1x1, i=4; j=4) * B33bb (1x1, j=4; k=4): schoolbook 1*1*1 = 1
    all_terms.extend(schoolbook_terms([4], [4], [4]))

    rank = len(all_terms)
    U, V, W = build_decomposition(all_terms)
    
    # Verify the decomposition
    if not verify(U, V, W, rank):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_school = np.zeros((n * m, rank_schoolbook))
        V_school = np.zeros((m * p, rank_schoolbook))
        W_school = np.zeros((p * n, rank_schoolbook))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Return the verified decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
