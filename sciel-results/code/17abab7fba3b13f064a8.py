"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 3, 7
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    def schoolbook():
        rank = n * m * p  # 42
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0

        R = np.einsum('a,b,c->abc', U, V, W)  # wrong, need sum over rank
        # Correct: R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])

        if np.max(np.abs(R - target)) > tol:
            return False

        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True

    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def strassen_2x2x2_block():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Strassen's 7 products:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B11)
        M4 = A11*(B01-B10)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B10)
        M7 = (A01-A11)*(B01+B11)
        
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        
        In our tensor decomposition:
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        For 2x2x2: n=2, m=2, p=2
        U indices: (i,j) -> i*m+j = i*2+j
        V indices: (j,k) -> j*p+k = j*2+k
        W indices: (k,i) -> k*n+i = k*2+i
        
        Each Strassen product r corresponds to a rank-1 term.
        """
        # 7 rank-1 terms for Strassen 2x2x2
        terms = []
        # M1 = (A00+A11)*(B00+B11)
        # A00: i=0,j=0; A11: i=1,j=1
        # B00: j=0,k=0; B11: j=1,k=1
        # C00 += M1: k=0,i=0; C11 += M1: k=1,i=1
        # U: A00=+1, A11=+1
        # V: B00=+1, B11=+1
        # W: C00=+1, C11=+1
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[0] = 1  # A00: i=0,j=0 -> 0*2+0=0
        u[3] = 1  # A11: i=1,j=1 -> 1*2+1=3
        v[0] = 1  # B00: j=0,k=0 -> 0*2+0=0
        v[3] = 1  # B11: j=1,k=1 -> 1*2+1=3
        w[0] = 1  # C00: k=0,i=0 -> 0*2+0=0
        w[3] = 1  # C11: k=1,i=1 -> 1*2+1=3
        terms.append((u, v, w))
        
        # M2 = (A10+A11)*B00
        # A10: i=1,j=0; A11: i=1,j=1
        # B00: j=0,k=0
        # C10 += M2: k=0,i=1
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[2] = 1  # A10: i=1,j=0 -> 1*2+0=2
        u[3] = 1  # A11: i=1,j=1 -> 1*2+1=3
        v[0] = 1  # B00: j=0,k=0 -> 0*2+0=0
        w[2] = 1  # C10: k=0,i=1 -> 0*2+1=1... wait k*n+i = 0*2+1=1
        # Actually W index is k*n+i = k*2+i
        # C10: k=0, i=1 -> 0*2+1 = 1
        w[1] = 1
        terms.append((u, v, w))
        
        # M3 = A00*(B10-B11)
        # A00: i=0,j=0
        # B10: j=1,k=0; B11: j=1,k=1
        # C01 += M3: k=1,i=0
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[0] = 1  # A00: i=0,j=0 -> 0
        v[2] = 1  # B10: j=1,k=0 -> 1*2+0=2
        v[3] = -1  # B11: j=1,k=1 -> 1*2+1=3
        w[0] = 1  # C01: k=1,i=0 -> 1*2+0=2... wait
        # W index: k*n+i = k*2+i
        # C01: k=1, i=0 -> 1*2+0 = 2
        w[2] = 1
        terms.append((u, v, w))
        
        # M4 = A11*(B01-B10)
        # A11: i=1,j=1
        # B01: j=0,k=1; B10: j=1,k=0
        # C00 += M4: k=0,i=0; C10 += M4: k=0,i=1
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[3] = 1  # A11: i=1,j=1 -> 3
        v[1] = 1  # B01: j=0,k=1 -> 0*2+1=1
        v[2] = -1  # B10: j=1,k=0 -> 1*2+0=2
        w[0] = 1  # C00: k=0,i=0 -> 0
        w[1] = 1  # C10: k=0,i=1 -> 1
        terms.append((u, v, w))
        
        # M5 = (A00+A01)*B11
        # A00: i=0,j=0; A01: i=0,j=1
        # B11: j=1,k=1
        # C00 -= M5: k=0,i=0; C01 += M5: k=1,i=0
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[0] = 1  # A00: i=0,j=0 -> 0
        u[1] = 1  # A01: i=0,j=1 -> 0*2+1=1
        v[3] = 1  # B11: j=1,k=1 -> 3
        w[0] = -1  # C00: k=0,i=0 -> 0 (negative)
        w[2] = 1  # C01: k=1,i=0 -> 2
        terms.append((u, v, w))
        
        # M6 = (A10-A00)*(B00+B10)
        # A10: i=1,j=0; A00: i=0,j=0
        # B00: j=0,k=0; B10: j=1,k=0
        # C11 += M6: k=1,i=1
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[2] = 1  # A10: i=1,j=0 -> 2
        u[0] = -1  # A00: i=0,j=0 -> 0
        v[0] = 1  # B00: j=0,k=0 -> 0
        v[2] = 1  # B10: j=1,k=0 -> 2
        w[3] = 1  # C11: k=1,i=1 -> 1*2+1=3
        terms.append((u, v, w))
        
        # M7 = (A01-A11)*(B01+B11)
        # A01: i=0,j=1; A11: i=1,j=1
        # B01: j=0,k=1; B11: j=1,k=1
        # C00 += M7: k=0,i=0
        u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
        u[1] = 1  # A01: i=0,j=1 -> 1
        u[3] = -1  # A11: i=1,j=1 -> 3
        v[1] = 1  # B01: j=0,k=1 -> 1
        v[3] = 1  # B11: j=1,k=1 -> 3
        w[0] = 1  # C00: k=0,i=0 -> 0
        terms.append((u, v, w))
        
        return terms

    def hybrid_strassen_schoolbook():
        """
        Hybrid decomposition: split m=3 into m1=2 (Strassen) + m2=1 (schoolbook).
        
        For j in {0,1}: use Strassen on 2x2x7 (three 2x2x2 blocks + one 2x2x1)
        For j=2: schoolbook (2x1x7 = 14 terms)
        
        Total: 7+7+7+4+14 = 39
        """
        all_terms = []  # list of (U_col, V_col, W_col) each of length nm, mp, pn
        
        # Part 1: j in {0,1} -> 2x2x7 multiplication
        # Split k into 3 blocks of 2 + 1 block of 1
        # For k in {0,1}: Strassen 2x2x2 (7 terms)
        # For k in {2,3}: Strassen 2x2x2 (7 terms)  
        # For k in {4,5}: Strassen 2x2x2 (7 terms)
        # For k=6: schoolbook 2x2x1 (4 terms)
        
        # Actually, let me be more careful.
        # The 2x2x7 part: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x7 (rows j=0,1; cols k=0..6)
        # C is 2x7 (rows i=0,1; cols k=0..6)
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        
        # We can decompose B into 3 blocks of width 2 + 1 block of width 1
        # For each 2x2x2 block, use Strassen (7 terms)
        # For the 2x2x1 block, use schoolbook (4 terms)
        
        # Block 1: k=0,1
        # Block 2: k=2,3
        # Block 3: k=4,5
        # Block 4: k=6
        
        for block_start in [0, 2, 4]:
            # Strassen 2x2x2 for k in {block_start, block_start+1}
            strassen_terms = strassen_2x2x2_block()
            for (su, sv, sw) in strassen_terms:
                # Map to full indices
                U_col = np.zeros(nm)  # 6
                V_col = np.zeros(mp)  # 21
                W_col = np.zeros(pn)  # 14
                
                # U: 2x2 block (i in 0,1; j in 0,1) -> indices i*3+j
                # su has indices: i*2+j for the local 2x2
                # Map local (i,j) to global (i, j) where j is 0 or 1
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        global_idx = li * m + lj  # li*3 + lj
                        U_col[global_idx] = su[local_idx]
                
                # V: 2x2 block (j in 0,1; k in {block_start, block_start+1})
                # sv has indices: j*2+k_local where k_local in 0,1
                # Map local (j, k_local) to global (j, block_start + k_local)
                for lj in range(2):
                    for lk in range(2):
                        local_idx = lj * 2 + lk
                        global_idx = lj * p + (block_start + lk)
                        V_col[global_idx] = sv[local_idx]
                
                # W: 2x2 block (k in {block_start, block_start+1}; i in 0,1)
                # sw has indices: k_local*2+i
                # Map local (k_local, i) to global (block_start + k_local, i)
                for lk in range(2):
                    for li in range(2):
                        local_idx = lk * 2 + li
                        global_idx = (block_start + lk) * n + li
                        W_col[global_idx] = sw[local_idx]
                
                all_terms.append((U_col, V_col, W_col))
        
        # Block 4: k=6, schoolbook 2x2x1 (4 terms)
        for i in range(2):
            for j in range(2):
                U_col = np.zeros(nm)
                V_col = np.zeros(mp)
                W_col = np.zeros(pn)
                U_col[i * m + j] = 1  # i*3 + j
                V_col[j * p + 6] = 1  # j*7 + 6
                W_col[6 * n + i] = 1  # 6*2 + i
                all_terms.append((U_col, V_col, W_col))
        
        # Part 2: j=2, schoolbook 2x1x7 (14 terms)
        for i in range(2):
            for k in range(7):
                U_col = np.zeros(nm)
                V_col = np.zeros(mp)
                W_col = np.zeros(pn)
                U_col[i * m + 2] = 1  # i*3 + 2
                V_col[2 * p + k] = 1  # 2*7 + k
                W_col[k * n + i] = 1  # k*2 + i
                all_terms.append((U_col, V_col, W_col))
        
        rank = len(all_terms)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for c, (U_col, V_col, W_col) in enumerate(all_terms):
            U[:, c] = U_col
            V[:, c] = V_col
            W[:, c] = W_col
        
        return U, V, W

    # Try the hybrid Strassen/Schoolbook decomposition first
    best_result = None
    best_rank = 42

    # Generate the hybrid decomposition
    try:
        U_hybrid, V_hybrid, W_hybrid = hybrid_strassen_schoolbook()
        if verify(U_hybrid, V_hybrid, W_hybrid, tol=1e-8):
            best_result = (U_hybrid, V_hybrid, W_hybrid)
            best_rank = U_hybrid.shape[1]
    except Exception:
        pass

    # If hybrid works, try to see if we can do better with search
    if best_result is not None and best_rank < 42:
        # Try to search for even lower rank
        # Theoretical lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
        # But realistically, for 2x3x7, the best known might be around 30-35
        # Try a few random searches at lower ranks
        for target_rank in range(best_rank - 1, max(21, best_rank - 5), -1):
            if time.time() > deadline - 20:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 20:
                    break
                if found:
                    break
                
                # Random search: fix U and V, solve for W
                U_rand = np.zeros((nm, target_rank))
                V_rand = np.zeros((mp, target_rank))
                
                for c in range(target_rank):
                    # Sparse random with half-integer values
                    for idx in range(nm):
                        if random.random() < 0.3:
                            U_rand[idx, c] = random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                    for idx in range(mp):
                        if random.random() < 0.2:
                            V_rand[idx, c] = random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                
                # Solve for W
                # Equations: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For each (a,b,c) where T=1: sum_r U[a,r]*V[b,r]*W[c,r] = 1
                # For each (a,b,c) where T=0: sum_r U[a,r]*V[b,r]*W[c,r] = 0
                
                # This is a linear system in W entries
                # Unknowns: W[c_idx, r] for c_idx in range(pn), r in range(target_rank)
                # Total unknowns: pn * target_rank
                
                equations = []
                # Non-zero entries
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c_idx = k * n + i
                            row = np.zeros(pn * target_rank)
                            for r in range(target_rank):
                                coeff = U_rand[a, r] * V_rand[b, r]
                                row[c_idx * target_rank + r] = coeff
                            equations.append((row, 1.0))
                
                # Zero entries (sample a subset for efficiency)
                valid_set = set()
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c_idx = k * n + i
                            valid_set.add((a, b, c_idx))
                
                zero_eqs = []
                for a in range(nm):
                    for b in range(mp):
                        for c_idx in range(pn):
                            if (a, b, c_idx) not in valid_set:
                                zero_eqs.append((a, b, c_idx))
                
                # Sample zero equations if too many
                if len(zero_eqs) > 200:
                    zero_eqs = random.sample(zero_eqs, 200)
                
                for (a, b, c_idx) in zero_eqs:
                    row = np.zeros(pn * target_rank)
                    for r in range(target_rank):
                        coeff = U_rand[a, r] * V_rand[b, r]
                        row[c_idx * target_rank + r] = coeff
                    equations.append((row, 0.0))
                
                A = np.array([eq[0] for eq in equations])
                rhs = np.array([eq[1] for eq in equations])
                
                try:
                    W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except Exception:
                    continue
                
                W_rand = W_flat.reshape(pn, target_rank)
                
                # Check if solution is valid
                if verify(U_rand, V_rand, W_rand, tol=1e-6):
                    best_result = (U_rand, V_rand, W_rand)
                    best_rank = target_rank
                    found = True
                    break
                
                # Try rounding
                W_rounded = round_to_half_integer(W_rand)
                if verify(U_rand, V_rand, W_rounded, tol=1e-6):
                    best_result = (U_rand, V_rand, W_rounded)
                    best_rank = target_rank
                    found = True
                    break

    # Fallback to schoolbook
    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
