"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
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

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U,V,W give a valid tensor decomposition."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s - 1) > tol:
                            return False
                    else:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Structural idea: exploit the Kronecker-product structure.
    # The 2x3 * 3x7 multiplication tensor can be seen as a sum of 3 "slices"
    # (one per j), each being a 2x7 outer-product-like structure.
    # For each j, the contribution is A[:,j] (outer) B[j,:].
    # We can try to find a low-rank decomposition of each 2x7 slice
    # and combine them, potentially sharing rank across j's.
    
    # Key insight: The tensor T has the form:
    # T[a,b,c] = sum_j delta_{a,ij} delta_{b,jk} delta_{c,ki}
    # where a=im+j, b=jp+k, c=kn+i.
    
    # Let's try a parametric approach:
    # For each column r of the decomposition, we assign it a "type" (i,j,k)
    # or a combination. The schoolbook assigns one (i,j,k) per column.
    # We can try to have a column contribute to multiple (i,j,k) terms
    # by setting U, V, W entries appropriately.
    
    # A column r contributes to term (i,j,k) if and only if
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] != 0.
    # For the column to contribute exactly 1 to that term and 0 to others,
    # we need careful setting.
    
    # Let's try a different structural approach: use the fact that
    # for fixed j, the 2x7 matrix A[:,j] B[j,:] can be decomposed.
    # The rank of a 2x7 matrix is at most 2, so the outer product
    # A[:,j] (outer) B[j,:] has a specific structure.
    
    # Actually, let's try a systematic search with a better initialization.
    # Initialize U, V, W with a pattern that has some sharing built in.
    
    def structured_search(rank, max_iters=500, seed=42):
        """Search with structured initialization based on grouping."""
        random.seed(seed)
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Assign each of the 42 (i,j,k) terms to a column
            # with some probability of sharing a column
            assignments = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3 and rank < n*m*p:
                            # Try to share with an existing assignment
                            # Find a column that already has some terms
                            cols = list(assignments.keys())
                            if cols:
                                col = random.choice(cols)
                            else:
                                col = random.randint(0, rank - 1)
                        else:
                            col = random.randint(0, rank - 1)
                        
                        if col not in assignments:
                            assignments[col] = []
                        assignments[col].append((i, j, k))
            
            # Now set up U, V, W based on assignments
            # For each column, we need to set entries so that
            # the product U[a,r]*V[b,r]*W[c,r] gives the right values
            for col, terms in assignments.items():
                if not terms:
                    continue
                # For each term in this column, set the appropriate entries
                # If a column has multiple terms, we need to be careful
                # This is a simplified approach; in general, a column
                # with multiple terms may not be feasible.
                
                # For now, just set single-term columns
                if len(terms) == 1:
                    i, j, k = terms[0]
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                else:
                    # Multiple terms in one column: this is tricky.
                    # Let's skip for now and just use single-term columns.
                    pass
            
            # Add random perturbations to try to find valid multi-term columns
            for r in range(rank):
                for a in range(n * m):
                    if U[a][r] == 0 and random.random() < 0.1:
                        U[a][r] = random.choice([-1, -0.5, 0.5, 1])
                for b in range(m * p):
                    if V[b][r] == 0 and random.random() < 0.1:
                        V[b][r] = random.choice([-1, -0.5, 0.5, 1])
                for c in range(p * n):
                    if W[c][r] == 0 and random.random() < 0.1:
                        W[c][r] = random.choice([-1, -0.5, 0.5, 1])
            
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try structured search for ranks below schoolbook
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10.0:
            break
        result = structured_search(r, max_iters=200, seed=r * 7919)
        if result is not None:
            best = result
            best_rank = r
            break

    # Also try a pure random search for the best rank found so far
    def random_search(rank, max_iters=3000, seed=42):
        """Random search for a rank decomposition."""
        random.seed(seed)
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                return None
            entries = [-1, -0.5, 0, 0, 0, 0, 0.5, 1]
            U = [[random.choice(entries) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(entries) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(entries) for _ in range(rank)] for _ in range(p * n)]
            if verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        return None

    # Try to improve the best rank found
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 5.0:
            break
        result = random_search(r, max_iters=1000, seed=r * 104729)
        if result is not None:
            best = result
            best_rank = r
            break

    return best
# EVOLVE-BLOCK-END
