import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 36

    min_direct_rank = max(n * m, m * p, p * n)  # 18

    # Strategy 1: Try to find rank-18 decomposition with structured U, V
    # For each d = k*n+i, we need to solve B @ w_d = 1_m where B[j,c] = U[i_d*m+j,c]*V[j*p+k_d,c]
    # If we fix U and V, W is determined (if solvable).
    # Key: we need B to be "nice" so that w_d has small rational entries.

    # Try structured U, V where entries are from {-1, 0, 1}
    # and the resulting W has half-integer entries.

    # Approach: Use the fact that the system decouples by d.
    # For rank 18: U is 6x18, V is 18x18, W is 12x18.
    # For each of the 12 values of d, we have a 3x18 system.
    # We need each 3x18 system to have a solution with half-integer entries.

    # Let's try: make U and V such that for each d, B has a simple structure.
    # Specifically, if B has rank 3 (full row rank), the solution space is 15-dimensional.
    # We need a particular solution with half-integer entries.

    # Better approach: try specific structured patterns for U and V.

    # Pattern idea: Use Fourier-like or combinatorial designs.
    # For n=2, m=3, p=6, rank=18:
    # U: 6 rows (i*m+j for i in {0,1}, j in {0,1,2})
    # V: 18 rows (j*p+k for j in {0,1,2}, k in {0,...,5})
    # W: 12 rows (k*n+i for k in {0,...,5}, i in {0,1})

    # Let's try: U[i*m+j, c] depends on i, j, c in a structured way.
    # V[j*p+k, c] depends on j, k, c in a structured way.

    # Idea: Let c = 3*a + b where a in {0,1,2,3,4,5} (for p=6) and b in {0,1,2} (for m=3)
    # That's 18 values. So c indexes (a,b) pairs.
    
    # U[i*m+j, c=(a,b)] = delta_{j,b} * f(i, a)
    # V[j*p+k, c=(a,b)] = delta_{j,b} * g(a, k)
    # Then B[j,c] = U[i_d*m+j,c] * V[j*p+k_d,c]
    # For fixed d=(i_d, k_d): B[j, (a,b)] = delta_{j,b} * f(i_d, a) * delta_{j,b} * g(a, k_d)
    # = delta_{j,b} * f(i_d, a) * g(a, k_d)
    # So B is block-diagonal with 3 blocks (one for each j), each of size 1x6.
    # B[j, (a,b)] = delta_{j,b} * f(i_d, a) * g(a, k_d)
    # The equation B @ w = 1 becomes:
    # For each j: sum_{a=0}^{5} f(i_d, a) * g(a, k_d) * w[j, (a,j)] = 1
    # (since b must equal j)
    
    # So we need: for each i_d in {0,1}, k_d in {0,...,5}, j in {0,1,2}:
    #   sum_{a=0}^{5} f(i_d, a) * g(a, k_d) * w_{j,(a,j)} = 1
    
    # This is 12*3 = 36 equations in 12*3 = 36 unknowns w_{j,(a,j)}.
    # The system is: for each (i_d, k_d, j): sum_a f(i_d,a)*g(a,k_d)*w_{j,(a,j)} = 1
    
    # This decouples by j! For each j, we have 12 equations in 6 unknowns:
    # sum_a f(i_d,a)*g(a,k_d)*w_{j,(a,j)} = 1 for i_d in {0,1}, k_d in {0,...,5}
    
    # This is an overdetermined system (12 eq, 6 unknowns). It's solvable only if
    # the 12 equations are consistent.
    
    # Let's choose f and g to make this work.
    # Let f(i, a) = 1 for all i, a. Then the system becomes:
    # sum_a g(a, k_d) * w_{j,(a,j)} = 1 for all k_d
    # This means the vector g(:,k_d) dotted with w_{j,(·,j)} = 1 for all k_d.
    # If g has full column rank 6, this has a unique solution for each j.
    
    # Let g(a, k) = delta_{a,k} (identity). Then w_{j,(k,j)} = 1 for all j,k.
    # Check: sum_a delta_{a,k_d} * 1 = 1. Yes!
    
    # So: f(i,a) = 1, g(a,k) = delta_{a,k}, w_{j,(a,j)} = 1.
    # Let's verify the full decomposition:
    # U[i*m+j, (a,b)] = delta_{j,b} * 1 = delta_{j,b}
    # V[j*p+k, (a,b)] = delta_{j,b} * delta_{a,k}
    # W[k*n+i, (a,b)] = w_{b,(a,b)} = 1 (when the "j-index" in w matches b)
    # Wait, W is indexed by d=k*n+i and c=(a,b).
    # W[d, c] = W[k*n+i, (a,b)]
    # In our solution, w_{j,(a,j)} = 1, so W[k*n+i, (a,b)] = 1 if b is the "j" that matches...
    # Hmm, I need to be more careful.
    
    # Let me re-derive. W[d, c] where d=k*n+i, c=(a,b).
    # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # = sum_{a,b} U[i*m+j,(a,b)] * V[j*p+k,(a,b)] * W[k*n+i,(a,b)]
    # = sum_{a,b} delta_{j,b} * delta_{j,b} * delta_{a,k} * W[k*n+i,(a,b)]
    # = sum_{a,b} delta_{j,b} * delta_{a,k} * W[k*n+i,(a,b)]
    # = delta_{a=k} sum_b delta_{j,b} * W[k*n+i,(k,b)]
    # = sum_b delta_{j,b} * W[k*n+i,(k,b)]
    # = W[k*n+i,(k,j)]
    
    # So we need W[k*n+i,(k,j)] = 1 for all i,k,j.
    # And W[k*n+i,(a,b)] = 0 for a != k or b != j... wait, no.
    # We need: W[k*n+i,(k,j)] = 1 for all i,j,k.
    # What about W[k*n+i,(a,b)] when a != k? It doesn't appear in the constraint
    # for that particular (i,j,k) because delta_{a,k}=0. But it could appear in
    # other constraints.
    
    # Actually, for a given (i,j,k), only W[k*n+i,(k,j)] appears. So we need
    # W[k*n+i,(k,j)] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,5}.
    # That's 12*3 = 36 constraints on W, but W has 12*18 = 216 entries.
    # The remaining entries can be 0.
    
    # So: W[k*n+i, (a,b)] = 1 if a=k and b=j, else 0.
    # But (a,b) is the column index c, and a in {0,...,5}, b in {0,1,2}.
    # c = 3*a + b (if we index columns as 3*a+b).
    
    # Let me verify: U[i*m+j, c=3a+b] = delta_{j,b}
    # V[j*p+k, c=3a+b] = delta_{j,b} * delta_{a,k}
    # W[k*n+i, c=3a+b] = 1 if a=k and b=j, else 0
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_{a,b} delta_{j,b} * (delta_{j,b} * delta_{a,k}) * (1 if a=k,b=j else 0)
    # = sum_{a,b} delta_{j,b} * delta_{a,k} * (1 if a=k,b=j else 0)
    # The only non-zero term is when a=k and b=j:
    # = delta_{j,j} * delta_{k,k} * 1 = 1. 
    # 
    # And for "invalid" (a,b,d) combinations, the product is 0 because
    # at least one of U, V, W is 0.
    
    # This gives a valid rank-18 decomposition! All entries are 0 or 1.

    # Let me construct it:
    r = 18
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for a in range(p):
                for b in range(m):
                    c = 3 * a + b  # column index
                    if b == j:
                        U[i * m + j][c] = 1
                        V[j * p + a][c] = 1  # V[j*p+k, c] with k=a
                    # W[k*n+i, c] = 1 if a=k and b=j
                    # For W, the row is k*n+i, column is c=3a+b
                    # W[k*n+i, 3a+b] = 1 if a=k and b=j
                    # But this depends on k and j, which are the loop variables for U and V.
                    # Let me set W separately.
    
    # Set W: W[k*n+i, 3a+b] = 1 if a==k and b==j... but j is not a free variable here.
    # W[k*n+i, 3a+b] should be 1 if a==k and b is "any j" — no, it must be 1 specifically
    # when b equals the j that appears in the constraint.
    # Actually, from the derivation: W[k*n+i, (a,b)] = 1 iff a=k AND b=j where j is the
    # specific j from the constraint. But W doesn't depend on j!
    # 
    # I made an error. Let me re-examine.
    # The constraint for (i,j,k) gives: W[k*n+i, (k,j)] = 1.
    # So W[d, c] where d=k*n+i, c=3a+b: W[d,c] = 1 if a=k and b=j.
    # But j is determined by the constraint, not by d and c alone.
    # For a given d=k*n+i and c=3a+b, we have a and b. The constraint requires
    # W[k*n+i, 3a+b] = 1 when a=k and b=j for the specific j in the (i,j,k) triple.
    # But for a fixed d and c, there's only one j that makes the constraint require
    # W[d,c]=1, namely j=b. And the constraint for (i,b,k) requires W[k*n+i, 3k+b]=1.
    # So W[k*n+i, 3k+b] = 1 for all i, k, b.
    # And W[k*n+i, 3a+b] = 0 for a != k (since no constraint requires it to be 1,
    # and setting it to 0 is fine as long as it doesn't break other constraints).
    
    # Let me verify this doesn't break anything:
    # For constraint (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_{a,b} delta_{j,b} * delta_{j,b}*delta_{a,k} * W[k*n+i, 3a+b]
    # = sum_{a,b} delta_{j,b} * delta_{a,k} * W[k*n+i, 3a+b]
    # = delta_{a=k} sum_b delta_{j,b} * W[k*n+i, 3k+b]
    # = sum_b delta_{j,b} * W[k*n+i, 3k+b]
    # = W[k*n+i, 3k+j]
    # We need this = 1, so W[k*n+i, 3k+j] = 1. 
    # And for a != k, W[k*n+i, 3a+b] doesn't contribute. Good.
    
    # So the final construction:
    for i in range(n):
        for j in range(m):
            for a in range(p):
                for b in range(m):
                    c = 3 * a + b
                    if b == j:
                        U[i * m + j][c] = 1
                    if b == j and a == a:  # V[j*p+k, c] with k=a
                        V[j * p + a][c] = 1 if b == j else 0

    # Reset and do it cleanly:
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for a in range(p):
                for b in range(m):
                    c = 3 * a + b
                    # U[i*m+j, c] = 1 if b == j
                    if b == j:
                        U[i * m + j][c] = 1
                    # V[j*p+k, c] = 1 if b == j and a == k (here k=a)
                    if b == j:
                        V[j * p + a][c] = 1
                    # W[k*n+i, c] = 1 if a == k and b == j (here we set for all valid)
                    # W[a*n+i, 3a+b] = 1 if b == j... but j is the loop variable.
                    # We need W[k*n+i, 3a+b] = 1 when a==k. For a given (a,b),
                    # this means W[a*n+i, 3a+b] = 1 for all i.
                    # But it should only be 1 when b corresponds to a valid j.
                    # Since b in {0,1,2} = j values, all b are valid.
                    # So W[a*n+i, 3a+b] = 1 for all i, a, b.
                    W[a * n + i][c] = 1

    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = r

    return best_result


def _schoolbook_decomposition(n, m, p):
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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    r = U_np.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
