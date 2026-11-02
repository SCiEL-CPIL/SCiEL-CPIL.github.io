"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from sympy import symbols, groebner, Poly

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 18
    
    # The multiplication tensor T has T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # for some valid (i,j,k), else 0.
    # We need: for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Lower bound: max(n*m, m*p, n*p) = max(6, 9, 6) = 9
    # Try ranks from 9 up to 17 (below schoolbook)
    
    # Build the list of (a,b,c) triples with T=1
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    one_set = set(ones)
    
    # Try to find rank-r decomposition using Groebner basis
    # For each rank r, create variables for U, V, W entries
    # and solve the system of polynomial equations
    
    for r in range(9, 18):
        if time.time() > deadline - 20:
            break
            
        # Create symbolic variables
        # U: (n*m) x r = 6r variables
        # V: (m*p) x r = 9r variables  
        # W: (p*n) x r = 6r variables
        total_vars = (n*m + m*p + p*n) * r
        
        var_names = []
        for idx in range(total_vars):
            var_names.append(symbols(f'x{idx}'))
        
        # Map variable indices
        # U[a, col] -> index = a * r + col
        # V[b, col] -> index = (n*m + b) * r + col
        # W[c, col] -> index = (n*m + m*p + c) * r + col
        
        def U_var(a, col):
            return var_names[a * r + col]
        
        def V_var(b, col):
            return var_names[(n*m + b) * r + col]
        
        def W_var(c, col):
            return var_names[(n*m + m*p + c) * r + col]
        
        # Build equations
        equations = []
        
        # For each triple (a,b,c), the equation is:
        # sum_col U[a,col]*V[b,col]*W[c,col] - T[a,b,c] = 0
        
        for (a, b, c) in all_triples:
            # sum over columns
            terms = []
            for col in range(r):
                terms.append(U_var(a, col) * V_var(b, col) * W_var(c, col))
            lhs = sum(terms)
            rhs = 1 if (a, b, c) in one_set else 0
            equations.append(lhs - rhs)
        
        # Try to solve using Groebner basis
        try:
            # Use a timeout by checking time periodically
            # Groebner can be very slow, so limit the number of equations we try
            # Actually, let's just try it and see
            
            # Create a Groebner basis
            G = groebner(equations, *var_names, order='lex')
            
            # Check if the system is consistent
            # If all variables reduce to constants, we have a solution
            # This is a heuristic - Groebner basis computation can be very expensive
            
            # For now, just check if we completed
            if time.time() > deadline - 10:
                break
                
        except Exception as e:
            # Groebner basis failed or timed out
            continue
    
    return best_result
# EVOLVE-BLOCK-END
