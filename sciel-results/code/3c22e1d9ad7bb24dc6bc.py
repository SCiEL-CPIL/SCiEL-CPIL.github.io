"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_exact(U, V, W, n, m, p, rank):
    """Verify exact tensor decomposition."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                valid = (i1 == i3 and j1 == j2 and k2 == k3)
                s = int(np.sum(U[a, :] * V[b, :] * W[c, :]))
                if valid and s != 1:
                    return False
                if not valid and s != 0:
                    return False
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2 x 2x2 Strassen multiplication.
    
    For C = A * B where A, B are 2x2:
    C[0,0] = M1 - M4 + M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M6 + M7
    
    where:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    In tensor form, each M corresponds to a rank-1 term:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For 2x2x2: n=m=p=2, indices i,j,k in {0,1}
    U has 4 rows, V has 4 rows, W has 4 rows, rank 7.
    """
    # Each rank-1 term: (i,j,k) -> U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # M1: (A00+A11)*(B00+B11)
    # This expands to: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In the tensor, we need the coefficient of each (i,j,k) term.
    # The rank-1 term contributes U[i*m+j]*V[j*p+k]*W[k*n+i] to T[i,j,k]
    # For M1, the "A part" is A00+A11, "B part" is B00+B11
    # So U[0*2+0]=1, U[1*2+1]=1 (i=0,j=0 and i=1,j=1)
    # V[0*2+0]=1, V[1*2+1]=1 (j=0,k=0 and j=1,k=1)
    # W[0*2+0]=1, W[1*2+1]=1 (k=0,i=0 and k=1,i=1)
    # But this would give product 1 for (0,0,0), (0,0,1), (1,1,0), (1,1,1)
    # That's not right for a rank-1 term.
    
    # Actually, in the Strassen algorithm, each multiplication M is a scalar product
    # of a linear combination of A entries with a linear combination of B entries.
    # The rank-1 tensor term for M1 is:
    # U[i*m+j, r] = coefficient of A[i,j] in the first factor
    # V[j*p+k, r] = coefficient of B[j,k] in the second factor
    # W[k*n+i, r] = coefficient of C[i,k] that M1 contributes to
    # 
    # Wait, that's not quite right either. Let me reconsider.
    # 
    # The tensor T satisfies: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    # No wait, the standard form is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # = sum_{j,r} A[i,j] * B[j,k] * delta
    # 
    # The bilinear form: C[i,k] = sum_{j,r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Wait no. Let me re-read the problem.
    # 
    # "U factor of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)"
    # "output factor follows W[k*n+i,r]"
    # 
    # So: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (something)
    # Actually the tensor decomposition is:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # And T[i*m+j, j*p+k, k*n+i] = 1 for the matrix multiplication tensor.
    # 
    # So for Strassen M1 = (A00+A11)*(B00+B11):
    # The contribution to C[i,k] from M1 is:
    # M1 contributes to C[0,0], C[0,1], C[1,0], C[1,1] with coefficients.
    # 
    # In the tensor form, the rank-1 term for M1 has:
    # U[i*m+j, r] = coefficient of A[i,j] in the "A-factor" of M1
    # V[j*p+k, r] = coefficient of B[j,k] in the "B-factor" of M1
    # W[k*n+i, r] = coefficient such that the product gives the right C[i,k] contribution
    # 
    # For M1: A-factor = A00 + A11, so U[0*2+0, r] = 1, U[1*2+1, r] = 1
    # B-factor = B00 + B11, so V[0*2+0, r] = 1, V[1*2+1, r] = 1
    # The product U*V gives: 1 for (i=0,j=0,k=0), (i=0,j=0,k=1) -- wait no.
    # U[0, r] * V[0, r] = 1*1 = 1 for a=0 (i=0,j=0), b=0 (j=0,k=0)
    # U[0, r] * V[2, r] = 1*0 = 0 for a=0 (i=0,j=0), b=2 (j=1,k=0) -- V[2] is j=1,k=0
    # 
    # Hmm, this is getting confused. Let me think more carefully.
    # 
    # In the tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # A rank-1 term is U[a,r]*V[b,r]*W[c,r].
    # For this to equal 1 at (a,b,c) = (i*m+j, j*p+k, k*n+i) for specific (i,j,k),
    # we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    # 
    # For Strassen, the rank-1 terms don't correspond to single (i,j,k) triples.
    # Instead, each rank-1 term has multiple nonzero entries in U, V, W.
    # 
    # Let me just directly construct the 7 rank-1 terms for 2x2x2.
    # 
    # The 2x2x2 tensor has T[i,j,k] = 1 for all i,j,k in {0,1} (8 entries).
    # We need 7 rank-1 terms that sum to this.
    # 
    # Rank-1 term r: U_r (4-dim), V_r (4-dim), W_r (4-dim)
    # U_r[i*2+j] * V_r[j*2+k] * W_r[k*2+i] should sum over r to delta.
    
    # Let me use the known Strassen decomposition:
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1] -- no, this doesn't work simply.
    
    # Actually, I think the cleanest way is to just verify by construction.
    # Let me define the 7 terms explicitly and verify.
    
    # Term 0 (M1): A-factor: A00+A11, B-factor: B00+B11
    # U[0]=1, U[3]=1 (i=0,j=0 and i=1,j=1)
    # V[0]=1, V[3]=1 (j=0,k=0 and j=1,k=1)  
    # W: need W[k*2+i] such that the product works.
    # The product U*V is nonzero at (a,b) where a in {0,3}, b in {0,3}
    # (a=0: i=0,j=0), (a=3: i=1,j=1)
    # (b=0: j=0,k=0), (b=3: j=1,k=1)
    # For (a=0,b=0): i=0,j=0,k=0 -> c=k*2+i=0 -> W[0]
    # For (a=0,b=3): i=0,j=1,k=1 -> but j must match: a gives j=0, b gives j=1. Mismatch!
    # So U[0]*V[3] = 1*1 = 1 but this corresponds to i=0,j=0 from U and j=1,k=1 from V.
    # The j indices don't match, so this shouldn't contribute.
    # 
    # I think I'm overcomplicating this. The tensor entry T[a,b,c] is only defined
    # when the indices are consistent: a=i*m+j, b=j*p+k, c=k*n+i with the same i,j,k.
    # For inconsistent (a,b,c), T=0.
    # 
    # So for the rank-1 term to be valid, U[a,r]*V[b,r]*W[c,r] must be 0 whenever
    # (a,b,c) is inconsistent, and 1 when consistent and part of the target.
    # 
    # For M1 with U[0]=U[3]=1, V[0]=V[3]=1:
    # (a=0,b=0): i=0,j=0 from a; j=0,k=0 from b. Consistent (j=0). c=k*2+i=0. Need W[0]=1.
    # (a=0,b=3): i=0,j=0 from a; j=1,k=1 from b. Inconsistent (j=0 vs j=1). Product must be 0.
    #   U[0]*V[3] = 1*1 = 1. Not 0! Problem.
    # 
    # So this simple assignment doesn't work. The Strassen algorithm works at the
    # matrix level, not directly as a tensor decomposition in this indexing scheme.
    # 
    # I need to think about this differently. The tensor decomposition for Strassen
    # requires that each rank-1 term has the property that U[a,r]*V[b,r] is nonzero
    # only when a and b are consistent (same j).
    
    # This means for each rank-1 term r, the support of U_r and V_r must be compatible:
    # if U_r[i*m+j] != 0 and V_r[j'*p+k] != 0, then j must equal j'.
    
    # This is a strong constraint. Let me reconsider.
    
    # Actually, I think the issue is that Strassen's algorithm, when viewed as a
    # tensor decomposition, does satisfy this constraint because the linear
    # combinations in A and B are structured to have compatible index patterns.
    
    # Let me look at M1 = (A00+A11)*(B00+B11) more carefully.
    # The "A part" involves A[0,0] and A[1,1].
    # The "B part" involves B[0,0] and B[1,1].
    # For the product to contribute to C[i,k], we need:
    # C[i,k] gets contribution from A[i,j]*B[j,k] for various j.
    # M1 contributes to C[0,0] via A[0,0]*B[0,0] and A[1,1]*B[1,1]
    # M1 contributes to C[0,1] via A[0,0]*B[0,1] -- but B[0,1] is not in M1's B-factor!
    # 
    # I think I'm confusing myself. Let me just give up on the Strassen embedding
    # and do a much more aggressive randomized search with better heuristics.
    pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    schoolbook_rank = n * m * p
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1

    random.seed(42)
    np.random.seed(42)
    alphabet = [-2, -1, 0, 1, 2]
    lower_bound = max(n * p, n * m, m * p)

    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline:
            break
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            W = np.zeros((p * n, target_rank), dtype=np.int64)
            density = 0.3
            for a in range(n * m):
                for r in range(target_rank):
                    if random.random() < density:
                        U[a, r] = random.choice(alphabet)
            for b in range(m * p):
                for r in range(target_rank):
                    if random.random() < density:
                        V[b, r] = random.choice(alphabet)
            for c in range(p * n):
                for r in range(target_rank):
                    if random.random() < density:
                        W[c, r] = random.choice(alphabet)
            if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
