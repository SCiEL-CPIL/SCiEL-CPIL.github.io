import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices:
    # U[i*m + b, l]
    # V[c*p + d, l]
    # W[k*n + i, l]
    # Equation: sum_l U[a*m+b, l] V[c*p+d, l] W[k*n+i, l] = 1 if (a==i, d==k, b==c) else 0
    
    # Target tensor T
    # T is a 3-way tensor of size (n*m) x (m*p) x (p*n) ? No.
    # The bilinear map is defined by the sum over l.
    # We can flatten the equation into a linear system for one factor.
    # Let's define the "Target" as a set of equations.
    # For each l, we have a rank-1 term.
    # Let's try to solve for a specific rank R.
    
    def generate_target_equations():
        # Returns a list of tuples (U_idx, V_idx, W_idx, target_val)
        # But this is not linear in U, V, W simultaneously.
        # We will use the iterative approach.
        pass

    def check_solution(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check all (a,b,c,d,i,k)
        # This is O(n*m * m*p * p*n * R) which is 6*18*12*R = 1296*R.
        # For R=20, ~26k operations. Very fast.
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and d == k and b == c) else 0.0
                                s = 0.0
                                for l in range(U.shape[1]):
                                    s += U[a*m + b, l] * V[c*p + d, l] * W[k*n + i, l]
                                if abs(s - target) > 1e-9:
                                    return False
        return True

    def snap_to_half_integers(M):
        M = np.round(M * 2) / 2.0
        return M

    def solve_for_U(V, W, R):
        # Equations: sum_l U[u, l] * V[v, l] * W[w, l] = T[u, v, w]
        # For a fixed u, the equation is:
        # sum_l U[u, l] * (V[v, l] * W[w, l]) = T[u, v, w]
        # This is linear in U[u, :].
        # We have (m*p * p*n) equations for each u? No.
        # The tensor is defined such that T[u, v, w] is non-zero only for specific combinations.
        # Actually, the target is sparse.
        # Let's construct the matrix A_u such that A_u @ U[u, :] = b_u.
        # The rows of A_u correspond to (v, w) pairs.
        # A_u[v, w, l] = V[v, l] * W[w, l].
        # b_u[v, w] = Target[u, v, w].
        
        # Target[u, v, w]:
        # u = a*m + b
        # v = c*p + d
        # w = k*n + i
        # Target is 1 if a==i, d==k, b==c.
        
        U_sol = np.zeros((n*m, R))
        
        for u in range(n*m):
            a = u // m
            b = u % m
            
            # Construct A (size (m*p * p*n) x R)
            # We only need the rows where Target is potentially non-zero?
            # No, we need all rows to satisfy the zero constraints too.
            # But many rows have target 0.
            # Let's collect all (v, w) pairs.
            
            A = np.zeros((m*p * p*n, R))
            b_vec = np.zeros(m*p * p*n)
            
            idx = 0
            for v in range(m*p):
                c = v // p
                d = v % p
                for w in range(p*n):
                    k = w // n
                    i = w % n
                    
                    # Target
                    if a == i and d == k and b == c:
                        b_vec[idx] = 1.0
                    else:
                        b_vec[idx] = 0.0
                    
                    # A row
                    # A[idx, l] = V[v, l] * W[w, l]
                    A[idx, :] = V[v, :] * W[w, :]
                    idx += 1
            
            # Solve A @ U[u, :] = b_vec
            # Use least squares
            try:
                U_sol[u, :], residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
        return U_sol

    def solve_for_V(U, W, R):
        V_sol = np.zeros((m*p, R))
        
        for v in range(m*p):
            c = v // p
            d = v % p
            
            A = np.zeros((n*m * p*n, R))
            b_vec = np.zeros(n*m * p*n)
            
            idx = 0
            for u in range(n*m):
                a = u // m
                b = u % m
                for w in range(p*n):
                    k = w // n
                    i = w % n
                    
                    if a == i and d == k and b == c:
                        b_vec[idx] = 1.0
                    else:
                        b_vec[idx] = 0.0
                    
                    A[idx, :] = U[u, :] * W[w, :]
                    idx += 1
            
            try:
                V_sol[v, :], residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                return None
                
        return V_sol

    def solve_for_W(U, V, R):
        W_sol = np.zeros((p*n, R))
        
        for w in range(p*n):
            k = w // n
            i = w % n
            
            A = np.zeros((n*m * m*p, R))
            b_vec = np.zeros(n*m * m*p)
            
            idx = 0
            for u in range(n*m):
                a = u // m
                b = u % m
                for v in range(m*p):
                    c = v // p
                    d = v % p
                    
                    if a == i and d == k and b == c:
                        b_vec[idx] = 1.0
                    else:
                        b_vec[idx] = 0.0
                    
                    A[idx, :] = U[u, :] * V[v, :]
                    idx += 1
            
            try:
                W_sol[w, :], residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                return None
                
        return W_sol

    def refine(U, V, W, R, max_iters=50):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_new = solve_for_U(V, W, R)
            if U_new is None: return U, V, W
            V_new = solve_for_V(U_new, W, R)
            if V_new is None: return U, V, W
            W_new = solve_for_W(U_new, V_new, R)
            if W_new is None: return U, V, W
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if check_solution(U, V, W):
                return U, V, W
                
        return U, V, W

    # Fallback Rank 22 Solution
    # Constructed from two Rank 11 blocks for 2x3x3.
    # We need a valid Rank 11 decomposition for 2x3x3.
    # Known: mu(2,3,3) = 11?
    # Let's use a known Rank 12 or Rank 11 if available.
    # If I don't have a hardcoded Rank 11, I can try to find one quickly.
    # But for safety, let's use the Rank 22 from the seed if it was valid?
    # The seed program had a Rank 34 solution.
    # The "inspiration" says it uses two copies of Rank 11.
    # I will implement a quick search for Rank 11 for 2x3x3.
    # If not found, I will use Rank 12 (if I can find it) or fallback to a simpler construction.
    
    # Actually, I can just use the iterative solver to find a Rank 20, 19, 18 solution for the full 2x3x6.
    # If it fails, I will return a known valid solution.
    # Let's generate a valid Rank 36 solution (identity) as a last resort?
    # Rank 36: U, V, W are identity-like.
    # U[i*m+b, l] = 1 if l == i*m+b else 0.
    # V[c*p+d, l] = 1 if l == c*p+d else 0.
    # W[k*n+i, l] = 1 if l == k*n+i else 0.
    # This works but rank is 36.
    
    # Better fallback: Rank 24 (two Rank 12 blocks).
    # Or just return the Rank 34 from the seed?
    # The seed returned Rank 34.
    # I want to beat 34. Rank 22 is better.
    
    # Let's try to find a Rank 20 solution.
    # If that fails, try Rank 22.
    
    best_solution = None
    
    for R in [20, 21, 22]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Random init
            U = np.random.randint(-1, 2, size=(n*m, R)).astype(float)
            V = np.random.randint(-1, 2, size=(m*p, R)).astype(float)
            W = np.random.randint(-1, 2, size=(p*n, R)).astype(float)
            
            # Refine
            U, V, W = refine(U, V, W, R, max_iters=20)
            
            if check_solution(U, V, W):
                # Snap to half-integers
                U = snap_to_half_integers(U)
                V = snap_to_half_integers(V)
                W = snap_to_half_integers(W)
                
                if check_solution(U, V, W):
                    best_solution = (U, V, W, R)
                    found = True
                    break
            
            if found:
                break
                
        if found:
            break

    if best_solution:
        U, V, W, R = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback: Hardcoded Rank 22 (Two Rank 11 blocks)
    # I need a valid Rank 11 decomposition for 2x3x3.
    # If I don't have one, I will use a Rank 12 decomposition?
    # Or I will use the Rank 34 from the seed?
    # Let's try to construct a Rank 24 solution (two Rank 12 blocks).
    # Rank 12 for 2x3x3 is easy (Strassen 7 + 5? No, 12 is safe).
    # Actually, let's just return the Rank 34 from the seed to be safe, 
    # but try to improve it to Rank 24 if possible.
    
    # Let's try to find a Rank 12 for 2x3x3 quickly.
    # If I find one, I can make a Rank 24 for 2x3x6.
    
    def find_233_decomposition(R_target=12):
        # n=2, m=3, p=3
        nn, mm, pp = 2, 3, 3
        def check_233(U, V, W):
            for a in range(nn):
                for b in range(mm):
                    for c in range(mm):
                        for d in range(pp):
                            for i in range(nn):
                                for k in range(pp):
                                    target = 1.0 if (a == i and d == k and b == c) else 0.0
                                    s = 0.0
                                    for l in range(U.shape[1]):
                                        s += U[a*mm + b, l] * V[c*pp + d, l] * W[k*nn + i, l]
                                    if abs(s - target) > 1e-9:
                                        return False
            return True

        def solve_233_U(V, W, R):
            U_sol = np.zeros((nn*mm, R))
            for u in range(nn*mm):
                a = u // mm
                b = u % mm
                A = np.zeros((mm*pp * pp*nn, R))
                b_vec = np.zeros(mm*pp * pp*nn)
                idx = 0
                for v in range(mm*pp):
                    c = v // pp
                    d = v % pp
                    for w in range(pp*nn):
                        k = w // nn
                        i = w % nn
                        if a == i and d == k and b == c:
                            b_vec[idx] = 1.0
                        A[idx, :] = V[v, :] * W[w, :]
                        idx += 1
                try:
                    U_sol[u, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                except:
                    return None
            return U_sol

        def solve_233_V(U, W, R):
            V_sol = np.zeros((mm*pp, R))
            for v in range(mm*pp):
                c = v // pp
                d = v % pp
                A = np.zeros((nn*mm * pp*nn, R))
                b_vec = np.zeros(nn*mm * pp*nn)
                idx = 0
                for u in range(nn*mm):
                    a = u // mm
                    b = u % mm
                    for w in range(pp*nn):
                        k = w // nn
                        i = w % nn
                        if a == i and d == k and b == c:
                            b_vec[idx] = 1.0
                        A[idx, :] = U[u, :] * W[w, :]
                        idx += 1
                try:
                    V_sol[v, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                except:
                    return None
            return V_sol

        def solve_233_W(U, V, R):
            W_sol = np.zeros((pp*nn, R))
            for w in range(pp*nn):
                k = w // nn
                i = w % nn
                A = np.zeros((nn*mm * mm*pp, R))
                b_vec = np.zeros(nn*mm * mm*pp)
                idx = 0
                for u in range(nn*mm):
                    a = u // mm
                    b = u % mm
                    for v in range(mm*pp):
                        c = v // pp
                        d = v % pp
                        if a == i and d == k and b == c:
                            b_vec[idx] = 1.0
                        A[idx, :] = U[u, :] * V[v, :]
                        idx += 1
                try:
                    W_sol[w, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                except:
                    return None
            return W_sol

        def refine_233(U, V, W, R, max_iters=50):
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                U_new = solve_233_U(V, W, R)
                if U_new is None: return U, V, W
                V_new = solve_233_V(U_new, W, R)
                if V_new is None: return U, V, W
                W_new = solve_233_W(U_new, V_new, R)
                if W_new is None: return U, V, W
                U, V, W = U_new, V_new, W_new
                if check_233(U, V, W):
                    return U, V, W
            return U, V, W

        for R_try in [11, 12]:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                U = np.random.randint(-1, 2, size=(6, R_try)).astype(float)
                V = np.random.randint(-1, 2, size=(9, R_try)).astype(float)
                W = np.random.randint(-1, 2, size=(6, R_try)).astype(float)
                U, V, W = refine_233(U, V, W, R_try, max_iters=20)
                if check_233(U, V, W):
                    U = snap_to_half_integers(U)
                    V = snap_to_half_integers(V)
                    W = snap_to_half_integers(W)
                    if check_233(U, V, W):
                        return U, V, W, R_try
        return None

    block_sol = find_233_decomposition()
    
    if block_sol:
        U_b, V_b, W_b, R_b = block_sol
        R_total = 2 * R_b
        U_full = np.zeros((n*m, R_total))
        V_full = np.zeros((m*p, R_total))
        W_full = np.zeros((p*n, R_total))
        
        U_full[:, :R_b] = U_b
        U_full[:, R_b:] = U_b
        
        for j in range(m):
            for k in range(3):
                V_full[j*p + k, :R_b] = V_b[j*3 + k]
                V_full[j*p + 3 + k, R_b:] = V_b[j*3 + k]
                
        for k in range(3):
            for i in range(n):
                W_full[k*n + i, :R_b] = W_b[k*n + i]
                W_full[(k+3)*n + i, R_b:] = W_b[k*n + i]
                
        if check_solution(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }

    # Last resort: Rank 36 Identity
    R = n * m * p
    U = np.eye(n*m, R)
    V = np.eye(m*p, R)
    W = np.zeros((p*n, R))
    # Identity construction for matrix mult:
    # C[i,k] = sum_b A[i,b] B[b,k]
    # Term l corresponds to (i,b,k)? No, l corresponds to (i,b) and (b,k)?
    # Rank 1 term: A[i,b] * B[b,k] * 1.
    # So U[i*m+b, l] = 1 if l == i*m+b.
    # V[c*p+d, l] = 1 if l == c*p+d AND c==b AND d==k?
    # This is not a direct sum of rank 1 tensors in the standard way if we just use identity.
    # Standard Rank n*m*p decomposition:
    # For each i, b, k:
    # U[i*m+b, l_ijk] = 1
    # V[b*p+k, l_ijk] = 1
    # W[k*n+i, l_ijk] = 1
    # All other entries 0.
    
    U_id = np.zeros((n*m, R))
    V_id = np.zeros((m*p, R))
    W_id = np.zeros((p*n, R))
    
    l = 0
    for i in range(n):
        for b in range(m):
            for k in range(p):
                U_id[i*m + b, l] = 1
                V_id[b*p + k, l] = 1
                W_id[k*n + i, l] = 1
                l += 1
                
    if check_solution(U_id, V_id, W_id):
        return {
            "U": U_id.tolist(),
            "V": V_id.tolist(),
            "W": W_id.tolist()
        }

    # Should not reach here
    return {
        "U": U_id.tolist(),
        "V": V_id.tolist(),
        "W": W_id.tolist()
    }

if __name__ == "__main__":
    construct()
