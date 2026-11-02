"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    def schoolbook():
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if decomposition is valid."""
        rank = len(U[0]) if U else 0
        # Build tensor and check
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # For this to equal sum_j A[i,j]*B[j,k] for all A,B,
        # we need: for all i,k,j1,j2: sum_r W[k*n+i,r] * U[i*m+j1,r] * V[j2*p+k,r] = delta[j1,j2]
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(rank):
                            s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                        expected = 1 if j1 == j2 else 0
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    best = schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank by solving linear system
    # The constraint is: for all i,k,j1,j2: sum_r W[k*n+i,r] * U[i*m+j1,r] * V[j2*p+k,r] = delta[j1,j2]
    # This is a system of n*p*m*m equations in rank*(n*m + m*p + p*n) unknowns (with bilinear structure)
    
    # For small ranks, we can try random initialization and gradient-free search
    # But given time constraints, let's try a few specific constructions
    
    # Try rank = 35, 34, ... down to some minimum
    # Use the fact that we can perturb the schoolbook decomposition
    
    # Alternative: use known results. For 2x2xp, the rank is known for small p.
    # For p=1: 2x2x1 has rank 4 (just 4 multiplications)
    # For p=2: 2x2x2 has rank 7 (Strassen)
    # For p=3: 2x2x3 - unknown to me, but at most 4*3=12 schoolbook, or 7*3/2? 
    
    # Let's just try to optimize from schoolbook with small perturbations
    # Actually, let's try a direct approach: assume rank r, set up the bilinear system
    
    # Given time constraints, return schoolbook if we can't find better quickly
    if time.time() > deadline - 20:
        return best
    
    # Try a few random searches for lower rank
    np.random.seed(42)
    
    for target_rank in [35, 34, 33, 32, 31]:
        if time.time() > deadline - 15:
            break
        
        # Try random initialization and check if we can satisfy constraints
        # This is a hard problem, so just try a few random tries
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Check constraints
            valid = True
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = 0
                            for r in range(target_rank):
                                s += W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r]
                            expected = 1 if j1 == j2 else 0
                            if abs(s - expected) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in U[i]] for i in range(n*m)]
                V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in V[i]] for i in range(m*p)]
                W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in W[i]] for i in range(p*n)]
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
    
    return best
# EVOLVE-BLOCK-END
