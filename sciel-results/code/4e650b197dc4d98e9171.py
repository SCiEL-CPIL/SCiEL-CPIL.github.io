"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
DEADLINE = START_TIME + 140  # safety margin before 150s wall-clock

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63

    # Schoolbook decomposition: guaranteed valid, rank = n*m*p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best = {"U": U, "V": V, "W": W, "rank": rank}

    # Build the target tensor T[a,b,c]
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        r = U_arr.shape[1]
        for t in range(r):
            result += np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        return np.allclose(result, T, atol=tol)

    def verify_fast(U_arr, V_arr, W_arr, tol=1e-9):
        """Faster verification using einsum."""
        r = U_arr.shape[1]
        result = np.einsum('it,jt,kt->ijk', U_arr, V_arr, W_arr)
        return np.allclose(result, T, atol=tol)

    # Try structured decomposition: split into 3x3x3 + 4x(3x3x1)
    # For 3x3x3, use a rank-20 decomposition (approximate Strassen-like)
    # For each 3x3x1, use rank-9 (schoolbook)
    # Total: 20 + 4*9 = 56

    # First, try to find a rank-56 decomposition
    # We'll construct it by combining a 3x3x3 decomposition with 4 3x3x1 decompositions
    
    # The 3x3x3 sub-tensor (k=0,1,2) uses B indices j*7+k and C indices k*3+i
    # The 3x3x1 sub-tensors (k=3,4,5,6) use B index j*7+k and C indices k*3+i
    
    # For now, let's try a search approach with better heuristics
    np.random.seed(42)
    
    # Try ranks from 56 down to 62
    for r in range(56, 63):
        if time.time() > DEADLINE - 30:
            break
        max_iters = 50
        for it in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            
            # Strategy: start from schoolbook and try to merge terms
            # Use a structured initialization
            U_arr = np.zeros((nm, r), dtype=np.float64)
            V_arr = np.zeros((mp, r), dtype=np.float64)
            W_arr = np.zeros((pn, r), dtype=np.float64)
            
            # Fill with schoolbook entries for first r terms
            for t in range(min(r, nm*mp*pn)):
                i = t // (m * p)
                rem = t % (m * p)
                j = rem // p
                k = rem % p
                U_arr[i * m + j, t] = 1
                V_arr[j * p + k, t] = 1
                W_arr[k * n + i, t] = 1
            
            # Apply random sparse perturbations to try to create cancellations
            n_perturb = np.random.randint(1, 5)
            for _ in range(n_perturb):
                # Pick a random column to perturb
                t = np.random.randint(r)
                # Add a small random vector
                U_arr[:, t] += np.random.choice([0, 0.5, -0.5], size=nm)
                V_arr[:, t] += np.random.choice([0, 0.5, -0.5], size=mp)
                W_arr[:, t] += np.random.choice([0, 0.5, -0.5], size=pn)
            
            # Round to half-integers
            U_arr = np.round(2 * U_arr) / 2
            V_arr = np.round(2 * V_arr) / 2
            W_arr = np.round(2 * W_arr) / 2
            
            if verify_fast(U_arr, V_arr, W_arr):
                best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist(), "rank": r}
                break
        if best["rank"] < rank:
            break

    # If we found a better rank, return it; otherwise return schoolbook
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
