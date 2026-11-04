"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 125  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_structured_search(target_rank, max_attempts=200, val_sets=None):
        """
        Uses the decoupled linear system structure.
        For fixed U, V, the system for W decouples into n*p independent sub-systems.
        Each sub-system is M @ w = 1, where M is m x rank.
        We solve for w and check if it's close to integer.
        """
        if val_sets is None:
            val_sets = [[0, 1], [-1, 0, 1], [0, 1, 2], [-2, -1, 0, 1, 2]]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Cycle through different value sets
            vs = val_sets[attempt % len(val_sets)]
            
            U = [[random.choice(vs) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice(vs) for _ in range(target_rank)] for _ in range(mp)]
            
            # Convert to numpy for easier manipulation
            U_np = np.array(U).reshape(n, m, target_rank)
            V_np = np.array(V).reshape(m, p, target_rank)
            
            W_solution = [[0] * target_rank for _ in range(pn)]
            valid = True
            
            # Solve for each (i, k)
            for i in range(n):
                for k in range(p):
                    # Construct M: m x target_rank
                    # M[j, c] = U[i, j, c] * V[j, k, c]
                    M = U_np[i, :, :] * V_np[:, k, :]
                    
                    # Solve M @ w = 1
                    try:
                        w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if w is close to integers
                    w_int = np.round(w)
                    if np.allclose(M @ w_int, np.ones(m), atol=1e-9):
                        # Store W[k, i, c]
                        for c in range(target_rank):
                            W_solution[k * n + i][c] = int(w_int[c])
                    else:
                        # Try half-integers too
                        w_half = (2 * np.round(2 * w) // 2)
                        if np.allclose(M @ w_half, np.ones(m), atol=1e-9):
                            for c in range(target_rank):
                                W_solution[k * n + i][c] = w_half[c]
                        else:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Final verification
                W_list = [[float(x) for x in row] for row in W_solution]
                if verify_fast(U, V, W_list, target_rank):
                    return {"U": U, "V": V, "W": W_list}
        
        return None

    def try_als_search(target_rank, max_restarts=50, max_iters=100):
        """
        ALS-based search for low-rank decomposition.
        """
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random sparse matrices
            U = np.random.randint(0, 2, size=(n, m, target_rank)).astype(float)
            V = np.random.randint(0, 2, size=(m, p, target_rank)).astype(float)
            W = np.random.randint(0, 2, size=(p, n, target_rank)).astype(float)
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((target_rank, p))
                        for k in range(p):
                            A[:, k] = V[j, k, :] * W[k, i, :]
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i, j, :] = u
                        except:
                            pass
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((target_rank, n))
                        for i in range(n):
                            A[:, i] = U[i, j, :] * W[k, i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j, k, :] = v
                        except:
                            pass
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((target_rank, m))
                        for j in range(m):
                            A[:, j] = U[i, j, :] * V[j, k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k, i, :] = w
                        except:
                            pass
            
            # After ALS, project to integers and verify
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_int, V_int, W_int)
            if np.all(T == 1):
                U_list = U_int.reshape(nm, target_rank).tolist()
                V_list = V_int.reshape(mp, target_rank).tolist()
                W_list = W_int.reshape(pn, target_rank).tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try to find direct sum decomposition: 2x3x7 = 2x3x3 + 2x3x4
    def find_sub_decomp(nn, mm, pp, target_rank, time_limit=30):
        """Find a rank-minimal decomposition for nn x mm x pp."""
        nmm = nn * mm
        mpp = mm * pp
        pnn = pp * nn
        start_time = time.time()
        
        val_sets = [[0, 1], [-1, 0, 1], [0, 1, 2], [-2, -1, 0, 1, 2], [0, 1, -1, 2, -2]]
        
        attempts = 0
        while time.time() - start_time < time_limit:
            attempts += 1
            if attempts > 50000:
                break
            
            vs = val_sets[attempts % len(val_sets)]
            
            U_sub = [[random.choice(vs) for _ in range(target_rank)] for _ in range(nmm)]
            V_sub = [[random.choice(vs) for _ in range(target_rank)] for _ in range(mpp)]
            
            U_np = np.array(U_sub).reshape(nn, mm, target_rank)
            V_np = np.array(V_sub).reshape(mm, pp, target_rank)
            
            W_solution = [[0] * target_rank for _ in range(pnn)]
            valid = True
            
            for i in range(nn):
                for k in range(pp):
                    M = U_np[i, :, :] * V_np[:, k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(mm), rcond=None)
                    except:
                        valid = False
                        break
                    
                    w_int = np.round(w)
                    if np.allclose(M @ w_int, np.ones(mm), atol=1e-9):
                        for c in range(target_rank):
                            W_solution[k * nn + i][c] = int(w_int[c])
                    else:
                        w_half = (2 * np.round(2 * w) // 2)
                        if np.allclose(M @ w_half, np.ones(mm), atol=1e-9):
                            for c in range(target_rank):
                                W_solution[k * nn + i][c] = w_half[c]
                        else:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Verify
                U_np2 = np.array(U_sub, dtype=np.int64).reshape(nn, mm, target_rank)
                V_np2 = np.array(V_sub, dtype=np.int64).reshape(mm, pp, target_rank)
                W_np2 = np.array(W_solution, dtype=np.int64).reshape(pp, nn, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_np2, V_np2, W_np2)
                if np.all(T == 1):
                    return {"U": U_sub, "V": V_sub, "W": W_solution}
        
        return None

    def combine_decomps(decomp1, rank1, n1, m1, p1, decomp2, rank2, n2, m2, p2):
        """Combine two decompositions via direct sum."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(nm)]
        V = [[0] * total_rank for _ in range(mp)]
        W = [[0] * total_rank for _ in range(pn)]
        
        # First part: n1 x m1 x p1
        for i in range(n1):
            for j in range(m1):
                for c in range(rank1):
                    U[i*m + j][c] = decomp1["U"][i*m1 + j][c]
        for j in range(m1):
            for k in range(p1):
                for c in range(rank1):
                    V[j*p + k][c] = decomp1["V"][j*p1 + k][c]
        for k in range(p1):
            for i in range(n1):
                for c in range(rank1):
                    W[k*n + i][c] = decomp1["W"][k*n1 + i][c]
        
        # Second part: n2 x m2 x p2
        for i in range(n2):
            for j in range(m2):
                for c in range(rank2):
                    U[(n1+i)*m + (m1+j)][rank1 + c] = decomp2["U"][i*m2 + j][c]
        for j in range(m2):
            for k in range(p2):
                for c in range(rank2):
                    V[(m1+j)*p + (p1+k)][rank1 + c] = decomp2["V"][j*p2 + k][c]
        for k in range(p2):
            for i in range(n2):
                for c in range(rank2):
                    W[(p1+k)*n + (n1+i)][rank1 + c] = decomp2["W"][k*n2 + i][c]
        
        return {"U": U, "V": V, "W": W}

    # Try direct sum: 2x3x3 (rank 7) + 2x3x4 (rank 14) = rank 21
    if time.time() < deadline - 30:
        sub1 = find_sub_decomp(2, 3, 3, 7, time_limit=25)
        if sub1 and time.time() < deadline - 25:
            sub2 = find_sub_decomp(2, 3, 4, 14, time_limit=25)
            if sub2:
                combined = combine_decomps(sub1, 7, 2, 3, 3, sub2, 14, 2, 3, 4)
                if verify_fast(combined["U"], combined["V"], combined["W"], 21):
                    best = combined
                    best_rank = 21

    # If direct sum didn't work, try direct search
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_structured_search(target_rank, max_attempts=300)
        if result:
            best = result
            best_rank = target_rank
            continue
        
        result = try_als_search(target_rank, max_restarts=30, max_iters=80)
        if result:
            best = result
            best_rank = target_rank
            continue
    
    return best
# EVOLVE-BLOCK-END

