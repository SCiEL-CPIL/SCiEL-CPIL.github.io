import numpy as np
import itertools
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    
    # We decompose M_{3,3,7} into M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # Rank = R333 + R333 + R331
    # R331 is 9 (schoolbook)
    
    # Search for low rank decomposition of M_{3,3,3}
    # Schoolbook rank is 27
    # Generate schoolbook terms
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = np.zeros(9)
                v = np.zeros(9)
                w = np.zeros(9)
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
                
    T = np.zeros(27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[k*9 + i*3 + j] = 1 # Indexing: w[k*3+i], u[i*3+j], v[j*3+k]
                # Wait, the tensor is T_{i,j,k} = delta_{i,k} delta_{j,l}?
                # C_{i,l} = sum_j A_{i,j} B_{j,l}
                # Term: A_{i,j} B_{j,l} C_{i,l}
                # u_{i,j}, v_{j,l}, w_{i,l}
                # My indexing:
                # u index: i*3+j
                # v index: j*3+k (where k is l)
                # w index: i*3+k (where k is l)
                # So T[i*3+j, j*3+k, i*3+k] = 1
                # Flatten: T[(i*3+j)*81 + (j*3+k)*9 + (i*3+k)]?
                # No, the tensor is 3x3x3.
                # T[i, j, k] = 1 if i==k? No.
                # The multiplication tensor T has T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
                # Yes, T_{i,j,k} = 1 for all i,j,k? No.
                # The tensor is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
                # So T[i, j, k] = 1 for all i,j,k?
                # No, the basis vectors are e_{i,j}, e_{j,k}, e_{i,k}.
                # The tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
                # So T has 27 non-zero entries, all 1.
                
    # Correct T
    T = np.zeros(27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[i*9 + j*3 + k] = 1
                
    # Check consistency for rank R
    def check_rank(R):
        if time.time() > deadline:
            return None
        # Try all subsets of size R
        # This is too slow for R=23 (27 choose 23 is large)
        # But for R=26, 27 choose 26 is 27.
        # For R=25, 27 choose 25 is 351.
        # For R=24, 27 choose 24 is 2925.
        # For R=23, 27 choose 23 is 17550.
        
        # We can use a greedy approach or random sampling
        # But let's try small R first
        
        # For now, try R=26
        # If found, return it
        pass

    # Simple approach: Use schoolbook for M_{3,3,3} if search fails
    # But we know rank is 23.
    # I will hardcode a known rank 23 decomposition if possible.
    # Since I don't have it, I will use the schoolbook for M_{3,3,3} (27)
    # and hope that 27+27+9 = 63 is not improved.
    # But I need to beat 63.
    
    # Let's try to find a rank 26 decomposition with integer coefficients.
    found = None
    for R in [26, 25]:
        if time.time() > deadline:
            break
        # Sample some subsets
        # For R=26, check all 27
        if R == 26:
            for omit in range(27):
                idx = [i for i in range(27) if i != omit]
                A_mat = np.zeros((27, R))
                for col, t_idx in enumerate(idx):
                    u, v, w = terms[t_idx]
                    # Tensor product
                    tv = np.kron(np.kron(u, v), w)
                    A_mat[:, col] = tv
                # Solve A_mat c = T
                c, residuals, rank, s = np.linalg.lstsq(A_mat, T, rcond=None)
                # Check if T is in span
                if np.allclose(A_mat @ c, T, atol=1e-9):
                    # Check if c is integer or half-integer
                    if np.allclose(c * 2, np.round(c * 2), atol=1e-9):
                        found = (idx, c)
                        break
        else:
            # Random sampling for R=25
            for _ in range(100):
                idx = np.random.choice(27, R, replace=False)
                A_mat = np.zeros((27, R))
                for col, t_idx in enumerate(idx):
                    u, v, w = terms[t_idx]
                    tv = np.kron(np.kron(u, v), w)
                    A_mat[:, col] = tv
                c, residuals, rank, s = np.linalg.lstsq(A_mat, T, rcond=None)
                if np.allclose(A_mat @ c, T, atol=1e-9):
                    if np.allclose(c * 2, np.round(c * 2), atol=1e-9):
                        found = (idx, c)
                        break
        if found:
            break

    if found:
        idx, c = found
        R333 = len(idx)
    else:
        idx = list(range(27))
        c = np.ones(27)
        R333 = 27

    # Construct M_{3,3,7}
    # U: 9 x R_total
    # V: 21 x R_total
    # W: 21 x R_total
    
    R331 = 9
    R_total = R333 + R333 + R331
    
    U = np.zeros((9, R_total))
    V = np.zeros((21, R_total))
    W = np.zeros((21, R_total))
    
    col = 0
    # Part 1: M_{3,3,3} for cols 0-2 of B
    for r_idx in idx:
        u, v, w = terms[r_idx]
        U[:, col] = c[idx.index(r_idx)] * u
        V[0:9, col] = c[idx.index(r_idx)] * v # v is for B_1 (3x3)
        # W for C_1 (3x3) -> indices 0-2 of C
        W[0:9, col] = c[idx.index(r_idx)] * w
        col += 1
        
    # Part 2: M_{3,3,3} for cols 3-5 of B
    for r_idx in idx:
        u, v, w = terms[r_idx]
        U[:, col] = c[idx.index(r_idx)] * u
        V[9:18, col] = c[idx.index(r_idx)] * v # v is for B_2a (3x3)
        # W for C_2a (3x3) -> indices 3-5 of C
        W[9:18, col] = c[idx.index(r_idx)] * w
        col += 1
        
    # Part 3: M_{3,3,1} for col 6 of B
    # Schoolbook for M_{3,3,1}
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1
            V[18+j, col] = 1 # B_2b (3x1) index j
            W[18+i, col] = 1 # C_2b (3x1) index i
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
