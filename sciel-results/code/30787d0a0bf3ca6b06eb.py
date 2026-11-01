import numpy as np
import time

def construct():
    n, m, p = 2, 2, 3
    start_time = time.time()
    deadline = start_time + 140.0

    # Dimensions
    nm, mp, pn = n * m, m * p, p * n
    # nm=4, mp=6, pn=6

    # Target Tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Indices:
    # U index: a = i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
    # V index: b = j*p + k  (j in 0..1, k in 0..2) -> 0,1,2,3,4,5
    # W index: c = k*n + i  (k in 0..2, i in 0..1) -> 0,1,2,3,4,5

    # Strategy: Decompose into two parts
    # Part 1: k=0,1 (2x2x2 tensor). Use Strassen (rank 7).
    # Part 2: k=2 (2x2x1 tensor). Use Schoolbook (rank 4).
    # Total Rank = 7 + 4 = 11.

    # --- Part 1: 2x2x2 Strassen ---
    # Tensor T1[a, b, c] for k in {0,1}.
    # a = i*2 + j
    # b = j*2 + k  (Note: here p_eff=2 for this block, so b = j*2 + k_local where k_local in 0..1)
    # Wait, the global V index is b = j*p + k. For k in 0,1, b = j*3 + k.
    # Let's map local indices to global.
    # Local 2x2x2:
    # A is 2x2 (i,j). B is 2x2 (j,k_local). C is 2x2 (k_local,i).
    # Global indices:
    # U (i,j) -> a = i*2 + j. (Same as global)
    # V (j,k_local) -> b = j*3 + k_local. (Since global p=3, but we are only using k=0,1)
    # W (k_local,i) -> c = k_local*2 + i. (Since global n=2, and k_local corresponds to global k)
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to construct U, V, W columns for these 7 terms.
    # Term t corresponds to a linear combination of A entries in U, B entries in V, and contributes to C entries in W.
    
    U_strassen = np.zeros((nm, 7))
    V_strassen = np.zeros((mp, 7))
    W_strassen = np.zeros((pn, 7))

    # Helper to add a term
    # U_term: dict {(i,j): coeff}
    # V_term: dict {(j,k): coeff}
    # W_term: dict {(k,i): coeff}
    
    def add_strassen_term(idx, u_dict, v_dict, w_dict):
        for (i, j), coeff in u_dict.items():
            a = i * 2 + j
            U_strassen[a, idx] += coeff
        for (j, k), coeff in v_dict.items():
            # Global b index for k in {0,1} is j*3 + k
            b = j * 3 + k
            V_strassen[b, idx] += coeff
        for (k, i), coeff in w_dict.items():
            # Global c index is k*2 + i
            c = k * 2 + i
            W_strassen[c, idx] += coeff

    # M1 = (A11 + A22)(B11 + B22)
    # U: A11(0,0), A22(1,1) -> (0,0):1, (1,1):1
    # V: B11(0,0), B22(1,1) -> (0,0):1, (1,1):1
    # W: C11(0,0), C22(1,1) -> (0,0):1, (1,1):1
    add_strassen_term(0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})

    # M2 = (A21 + A22)B11
    # U: A21(1,0), A22(1,1) -> (1,0):1, (1,1):1
    # V: B11(0,0) -> (0,0):1
    # W: C21(0,1) [k=0,i=1], C22(1,1) [k=1,i=1]
    # Wait, C21 is (i=1, j=0, k=0)? No.
    # C indices are (k, i).
    # C21 means row 2, col 1 of C? No, standard notation C_ij.
    # My W index is c = k*n + i.
    # C21 corresponds to i=1, j=0? No, C is result of A*B.
    # A is n x m (2x2). B is m x p (2x2). C is n x p (2x2).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # My W index is c = k*n + i. This is column-major for C if we view C as (i,k)?
    # Let's re-read the task: "W[k*n+i,r]".
    # So W index c corresponds to output element C[i,k] where c = k*n + i.
    # In Strassen formulas:
    # C11 is C[0,0]. c = 0*2 + 0 = 0.
    # C12 is C[0,1]. c = 1*2 + 0 = 2.
    # C21 is C[1,0]. c = 0*2 + 1 = 1.
    # C22 is C[1,1]. c = 1*2 + 1 = 3.
    
    # Let's re-verify the Strassen mapping to W indices.
    # C11 (i=0,k=0) -> c=0
    # C12 (i=0,k=1) -> c=2
    # C21 (i=1,k=0) -> c=1
    # C22 (i=1,k=1) -> c=3

    # M1: C11 += 1, C22 += 1.
    # W: (0,0):1, (1,1):1. Correct in add_strassen_term call above?
    # My add_strassen_term used w_dict {(0,0):1, (1,1):1}.
    # (k=0,i=0) -> c=0. (k=1,i=1) -> c=3.
    # So M1 contributes to c=0 and c=3.
    # C11=c=0, C22=c=3. Correct.

    # M2: C21 += 1, C22 += 1.
    # C21 is c=1. C22 is c=3.
    # U: A21(i=1,j=0), A22(i=1,j=1).
    # V: B11(j=0,k=0).
    # W: (k=0,i=1):1, (k=1,i=1):1.
    add_strassen_term(1, {(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):1})

    # M3: C12 += 1, C21 += 1.
    # C12 is c=2. C21 is c=1.
    # U: A11(i=0,j=0).
    # V: B21(j=1,k=0), B22(j=1,k=1). Coeffs: 1, -1.
    # W: (k=1,i=0):1, (k=0,i=1):1.
    add_strassen_term(2, {(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (0,1):1})

    # M4: C11 += 1, C21 += 1.
    # C11 is c=0. C21 is c=1.
    # U: A22(i=1,j=1).
    # V: B12(j=0,k=1), B11(j=0,k=0). Coeffs: 1, -1.
    # W: (k=0,i=0):1, (k=0,i=1):1.
    add_strassen_term(3, {(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1})

    # M5: C11 -= 1, C12 += 1.
    # C11 is c=0. C12 is c=2.
    # U: A11(i=0,j=0), A12(i=0,j=1).
    # V: B22(j=1,k=1).
    # W: (k=0,i=0):-1, (k=1,i=0):1.
    add_strassen_term(4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})

    # M6: C22 += 1.
    # C22 is c=3.
    # U: A21(i=1,j=0), A11(i=0,j=0). Coeffs: 1, -1.
    # V: B11(j=0,k=0), B12(j=0,k=1).
    # W: (k=1,i=1):1.
    add_strassen_term(5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})

    # M7: C11 += 1.
    # C11 is c=0.
    # U: A12(i=0,j=1), A22(i=1,j=1). Coeffs: 1, -1.
    # V: B21(j=1,k=0), B22(j=1,k=1).
    # W: (k=0,i=0):1.
    add_strassen_term(6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})

    # --- Part 2: k=2 Schoolbook ---
    # Tensor T2 for k=2.
    # A is 2x2. B is 2x1 (column 2 of B). C is 2x1 (column 2 of C).
    # C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2].
    # Global indices:
    # U: a = i*2 + j.
    # V: b = j*3 + 2. (j=0 -> 2, j=1 -> 5).
    # W: c = 2*2 + i = 4 + i. (i=0 -> 4, i=1 -> 5).
    
    # 4 terms:
    # 1. A[0,0] * B[0,2] -> C[0,2]
    # 2. A[0,1] * B[1,2] -> C[0,2]
    # 3. A[1,0] * B[0,2] -> C[1,2]
    # 4. A[1,1] * B[1,2] -> C[1,2]
    
    U_school = np.zeros((nm, 4))
    V_school = np.zeros((mp, 4))
    W_school = np.zeros((pn, 4))

    # Term 0: A[0,0] B[0,2] -> C[0,2]
    U_school[0*2+0, 0] = 1
    V_school[0*3+2, 0] = 1
    W_school[2*2+0, 0] = 1

    # Term 1: A[0,1] B[1,2] -> C[0,2]
    U_school[0*2+1, 1] = 1
    V_school[1*3+2, 1] = 1
    W_school[2*2+0, 1] = 1

    # Term 2: A[1,0] B[0,2] -> C[1,2]
    U_school[1*2+0, 2] = 1
    V_school[0*3+2, 2] = 1
    W_school[2*2+1, 2] = 1

    # Term 3: A[1,1] B[1,2] -> C[1,2]
    U_school[1*2+1, 3] = 1
    V_school[1*3+2, 3] = 1
    W_school[2*2+1, 3] = 1

    # Combine
    U = np.hstack([U_strassen, U_school])
    V = np.hstack([V_strassen, V_school])
    W = np.hstack([W_strassen, W_school])

    # Verification
    # Compute tensor T_prime
    T_prime = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Build Target Tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Check exactness
    # Since entries are integers, we can check exact equality
    if not np.allclose(T_prime, T_target, atol=1e-12):
        # Fallback to schoolbook rank 12 if something went wrong
        rank_school = n * m * p
        U_sb = np.zeros((nm, rank_school))
        V_sb = np.zeros((mp, rank_school))
        W_sb = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
