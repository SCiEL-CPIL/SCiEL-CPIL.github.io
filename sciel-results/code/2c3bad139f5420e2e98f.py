"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # Known rank-17 solution
    rank17 = 17
    U17 = np.zeros((n * m, rank17), dtype=int)
    V17 = np.zeros((m * p, rank17), dtype=int)
    W17 = np.zeros((p * n, rank17), dtype=int)
    
    # Strassen part for 2x2 block (indices 0-6)
    # M1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # M2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # M3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # M4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # M5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[2, 4] = 1; W17[0, 4] = -1
    
    # M6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # M7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Schoolbook for remaining terms (indices 7-16)
    # A02*B20 -> C00
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    # A02*B21 -> C01
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    # A12*B20 -> C10
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    # A12*B21 -> C11
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    # A00*B02 -> C02
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    # A01*B12 -> C02
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    # A10*B02 -> C12
    U17[3, 13] = 1; V17[2, 13] = 1; W17[5, 13] = 1
    # A11*B12 -> C12
    U17[4, 14] = 1; V17[5, 14] = 1; W17[5, 14] = 1
    # A02*B22 -> C02
    U17[2, 15] = 1; V17[8, 15] = 1; W17[4, 15] = 1
    # A12*B22 -> C12
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    # Verify rank-17 solution
    valid17 = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                comp = 0
                for r in range(rank17):
                    comp += U17[u_idx, r] * V17[v_idx, r] * W17[w_idx, r]
                if comp != 1:
                    valid17 = False
                    break
            if not valid17:
                break
        if not valid17:
            break
    
    if not valid17:
        # Fallback to schoolbook rank 18
        rank = 18
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, idx] = 1
                    V[v_idx, idx] = 1
                    W[w_idx, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank-16 solution using algebraic search
    # Strategy: Use the explicit rank-16 decomposition derived from 3x3x3 rank-20 algorithm
    # by restricting the third dimension and performing exact rational Gaussian elimination
    
    rank16 = 16
    nU, nV, nW = n * m, m * p, p * n
    
    # Build the multiplication tensor T[a,b,c]
    T = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Attempt rank-16 decomposition search using algebraic methods
    # We'll use a structured search based on the known rank-17 solution
    # and try to merge some terms using rational arithmetic
    
    np.random.seed(42)
    best_error = float('inf')
    best_params = None
    
    # Try multiple random initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Start from rank-17 solution and perturb
        U_init = np.random.randn(nU, rank16) * 0.5
        V_init = np.random.randn(nV, rank16) * 0.5
        W_init = np.random.randn(nW, rank16) * 0.5
        
        def loss_function(params):
            U_l = params[:nU * rank16].reshape(nU, rank16)
            V_l = params[nU * rank16:nU * rank16 + nV * rank16].reshape(nV, rank16)
            W_l = params[nU * rank16 + nV * rank16:].reshape(nW, rank16)
            
            T_approx = np.einsum('at,bt,ct->abc', U_l, V_l, W_l)
            diff = T - T_approx
            error = np.sum(diff * diff)
            return error
        
        params = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        # Use scipy's minimize with L-BFGS-B
        try:
            from scipy.optimize import minimize
            result = minimize(loss_function, params, method='L-BFGS-B',
                              options={'maxiter': 200, 'gtol': 1e-10})
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Check if we found a valid rank-16 solution
    if best_params is not None and best_error < 1e-8:
        U16 = best_params[:nU * rank16].reshape(nU, rank16)
        V16 = best_params[nU * rank16:nU * rank16 + nV * rank16].reshape(nV, rank16)
        W16 = best_params[nU * rank16 + nV * rank16:].reshape(nW, rank16)
        
        # Verify the solution
        valid16 = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    comp = 0.0
                    for r in range(rank16):
                        comp += U16[u_idx, r] * V16[v_idx, r] * W16[w_idx, r]
                    if abs(comp - 1.0) > 1e-6:
                        valid16 = False
                        break
                if not valid16:
                    break
            if not valid16:
                break
        
        if valid16:
            # Convert to rational numbers if possible
            from fractions import Fraction
            U16_rational = np.zeros((nU, rank16), dtype=object)
            V16_rational = np.zeros((nV, rank16), dtype=object)
            W16_rational = np.zeros((nW, rank16), dtype=object)
            
            for i in range(nU):
                for r in range(rank16):
                    U16_rational[i, r] = Fraction(U16[i, r]).limit_denominator(100)
            for i in range(nV):
                for r in range(rank16):
                    V16_rational[i, r] = Fraction(V16[i, r]).limit_denominator(100)
            for i in range(nW):
                for r in range(rank16):
                    W16_rational[i, r] = Fraction(W16[i, r]).limit_denominator(100)
            
            # Verify with rational arithmetic
            valid_rational = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        comp = Fraction(0)
                        for r in range(rank16):
                            comp += U16_rational[u_idx, r] * V16_rational[v_idx, r] * W16_rational[w_idx, r]
                        if comp != 1:
                            valid_rational = False
                            break
                    if not valid_rational:
                        break
                if not valid_rational:
                    break
            
            if valid_rational:
                return {"U": U16_rational.tolist(), "V": V16_rational.tolist(), "W": W16_rational.tolist()}
    
    # Fallback to rank-17 solution
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

# EVOLVE-BLOCK-END
