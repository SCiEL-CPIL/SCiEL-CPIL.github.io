"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140

    def _build_terms(terms):
        """terms: list of (U_vec, V_vec, W_vec) each a flat list with half-integer entries."""
        rank = len(terms)
        U = np.array([[Fraction(t[0][i]) for t in terms] for i in range(n * m)], dtype=object)
        V = np.array([[Fraction(t[1][i]) for t in terms] for i in range(m * p)], dtype=object)
        W = np.array([[Fraction(t[2][i]) for t in terms] for i in range(p * n)], dtype=object)
        return U, V, W

    def _verify_exact(U, V, W, n, m, p):
        rank = U.shape[1]
        T = np.ones((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if val != 1:
                        return False
        return True

    def _is_half_integer(M):
        for row in M:
            for val in row:
                if val != val.numerator / 2 * (val.denominator == 2 or val.denominator == 1):
                    pass
                d = val.denominator
                if d != 1 and d != 2:
                    return False
        return True

    def _try_decomp(terms, n, m, p):
        U, V, W = _build_terms(terms)
        if _verify_exact(U, V, W, n, m, p) and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            def to_float_frac(M):
                return [[float(v) for v in row] for row in M]
            return {"U": to_float_frac(U), "V": to_float_frac(V), "W": to_float_frac(W)}
        return None

    # ---- Construction: use Winograd rank-10 for 2x2x3, Strassen rank-7 for 2x2x2, etc. ----
    # Strategy: 
    #   Part A: 2x2x3 (j=0,1; k=0,1,2) using Winograd rank-10
    #   Part B: 2x2x2 (j=0,1; k=3,4) using Strassen rank-7
    #   Part C: 2x1x5 (j=2; k=0..4) using rank-10 schoolbook
    # Total: 10 + 7 + 10 = 27

    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    # ---- Part A: Winograd 2x2x3 (rank 10) for j=0,1; k=0,1,2 ----
    # Winograd F(2,2,3): rank 10
    # Standard construction:
    #   For 2x2xn with n=3, rank = 2n+4 = 10? No, rank(2x2x3)=10 is known.
    #   Let me use the explicit Winograd formula.
    #
    # Winograd 2x2x3:
    #   A = [[a, b], [c, d]], B = [[e0,e1,e2], [f0,f1,f2]], C = A*B
    #   C[0,0] = a*e0 + b*f0
    #   C[0,1] = a*e1 + b*f1
    #   C[0,2] = a*e2 + b*f2
    #   C[1,0] = c*e0 + d*f0
    #   C[1,1] = c*e1 + d*f1
    #   C[1,2] = c*e2 + d*f2
    #
    # Winograd rank-10 for 2x2x3:
    #   m1 = a*e0, m2 = a*e1, m3 = a*e2, m4 = b*f0, m5 = b*f1, m6 = b*f2
    #   m7 = c*e0, m8 = c*e1, m9 = c*e2, m10 = d*f0
    #   That's schoolbook rank 12, not 10.
    #
    # Correct Winograd 2x2x3 rank-10:
    #   Let me look up the known decomposition.
    #   r1 = (a+b)(e0+f0) -> C00+=1, C10+=1, C01+=1, C11+=1 (wait, need to be careful)
    #
    # Actually, let me just use a known rank-10 Winograd for 2x2x3:
    # The 10 terms are:
    # 1: U=[1,1,0,0], V=[1,0,1,0,1,0], W=[1,0,1,0,1,0] -> (a+b)(e0+e1+e2+f0+f1+f2)? No that's wrong.
    #
    # Let me use a different approach: since rank(2x2x3)=10 is proven, 
    # I'll construct it explicitly.
    #
    # Winograd 2x2xn uses rank 2n+2 for n odd? For n=3, rank 8? No, rank(2x2x3)=10.
    # 
    # I'll use a verified rank-10 construction:
    # Terms for 2x2x3:
    winograd_2x2x3 = [
        # Term 0: (a+b)*(e0+f0) contributes to C00 and C10... let me think differently
        # Use the standard Winograd transform.
        # U transform: u0=a+b, u1=a-b, u2=c+d, u3=c-d, u4=a, u5=b, u6=c, u7=d
        # For each column k: V_k = e_k + f_k, W_k = e_k - f_k (for diagonal)
        # Plus cross terms.
    ]

    # Let me just use a known correct decomposition. The rank of 2x2x3 is 10.
    # I'll use the explicit terms from the literature:
    # m1=(a+b)(e0+f0), m2=(a-b)(e0-f0), m3=(c+d)(e0+f0), m4=(c-d)(e0-f0)
    # m5=a*e1, m6=b*f1, m7=c*e1, m8=d*f1
    # m9=a*e2, m10=b*f2 -> no this is getting complex.

    # Simplified approach: I'll use a known rank-10 Winograd pattern.
    # For 2x2x3, the Winograd scheme is:
    # Row sums/diffs of A, column sums/diffs of B for each k, plus per-element terms.
    # Actually, let me just use a verified set of 10 terms:

    # Term construction for Winograd 2x2x3:
    # I'll use the fact that we can write it as a combination of:
    # - (a+b)*(e_k+f_k) for k=0,1,2 (3 terms)
    # - (a-b)*(e_k-f_k) for k=0,1,2 (3 terms)  
    # - (c+d)*(e_k+f_k) for k=0,1,2 (3 terms) -> but that's 9, and we need 10
    # - plus one more term for the cross

    # Let me just verify a specific construction numerically:
    # I'll construct 10 terms and check:
    winograd_terms = []

    # I'll use a direct rank-10 construction for 2x2x3:
    # After some thought, here's a working rank-10 decomposition:
    # 
    # Let me use a simpler but still rank-10 approach:
    # Use the standard Winograd F(2,2,3) which has rank 10:
    # 
    # U factors (on a,b,c,d):
    # u1 = a+b, u2 = a-b, u3 = c+d, u4 = c-d, u5 = a, u6 = b, u7 = c, u8 = d
    # 
    # For each k, V and W combine:
    # v1_k = e_k + f_k, w1_k = 1 (contributes to both rows via a+b)
    # v2_k = e_k - f_k, w2_k = 1 (contributes via a-b)
    # v3_k = e_k + f_k, w3_k = 1 (contributes to both rows via c+d)
    # v4_k = e_k - f_k, w4_k = 1 (contributes via c-d)
    # But that's 4*3 = 12 terms, not 10.

    # OK, let me just use schoolbook for the 2x2x3 part (rank 12) and see if the total is better.
    # 12 + 7 + 10 = 29. Worse than the current 28.

    # Let me try a different partition:
    # 2x3x5 = (2x2x5) + (2x1x5)
    # 2x2x5: use Winograd-style. rank(2x2x5) = ?
    # Known: rank(2x2xn) = 2n+2 for n odd (Winograd), so rank(2x2x5) = 12
    # 2x1x5: rank 10
    # Total: 12 + 10 = 22!

    # Let me construct rank-12 Winograd for 2x2x5:
    # Winograd F(2,2,5) rank = 2*5+2 = 12
    # 
    # The Winograd 2x2xn (n odd) scheme:
    # U transforms:
    #   u_i for i=0..n: related to a,b,c,d
    #   u_n+1..n+2: a+b, a-b, c+d, c-d? 
    # 
    # Actually, the standard Winograd 2x2xn (n odd) has rank 2n+2:
    # 
    # Let n=5. Rank = 12.
    # 
    # U matrix (4 entries: a,b,c,d) for each of 12 terms:
    # Term 0: a, 0, 0, 0  -> multiplied by (e_k for all k in col 0 pattern)
    # 
    # I'll use the explicit Winograd F(2,2,5) from the literature.
    # The key idea: 
    #   C = AB where A is 2x2, B is 2x5, C is 2x5
    #   C[i,j] = sum over l=0,1 of A[i,l]*B[l,j]
    # 
    # Winograd uses:
    #   For each "block" of columns, use sums and differences.
    #   With n=5, we can use 5+1+1+... 
    # 
    # Let me just construct it explicitly:
    # 
    # The 12 terms for 2x2x5:
    # I'll use the pattern where we have:
    # - 5 terms for the "diagonal" (each k uses a*b pattern)
    # - But that's schoolbook rank 10 for 2x2x5.
    # - Plus 2 more terms for the sum/diff structure? 
    # 
    # Actually rank(2x2x5) is known to be 12 from Winograd.
    # 
    # Let me use a verified construction:
    # 
    # U (12x4), V (12x10), W (12x10) where V,W index over (j,k) for j=0,1 and k=0..4

    # I'll use a simpler strategy: just use schoolbook for everything but optimize the partition.
    # 
    # Partition: 2x3x5 = (2x3x2) + (2x3x2) + (2x3x1)
    # rank(2x3x2) = ? This is the rank of the tensor for 2x3 times 3x2.
    # Known: rank(2xnx2) for n>=2 is ... let me think.
    # For 2x3x2, we can use: 
    #   C[i,k] = sum over j A[i,j]*B[j,k] for i=0,1; j=0,1,2; k=0,1
    #   That's 2*3*2 = 12 schoolbook terms per 2x3x2 block.
    #   But we can do better.
    # 
    # Actually, 2x3x2 is equivalent to computing 2 vectors of length 3 dot products with 2 columns.
    # 
    # Let me just go with a clean construction:
    # 
    # Partition: 2x3x5 as three 2x3x? blocks won't help unless we reduce rank.
    # 
    # Best known for 2x3x5: Let me think about what's achievable.
    # 
    # Upper bound: rank <= n*m*p = 30 (schoolbook)
    # Using block Strassen:
    #   2x3x5 = 2x3x3 + 2x3x2
    #   2x3x3: rank? 
    #   2x3x2: rank?
    # 
    # I'll just use a clean, verified decomposition. Let me use:
    # 
    # Strategy: 2x3x5 = (2x2x3) + (2x2x2) + (2x2x1) + (2x1x3) + (2x1x2) + (2x1x1)
    # But that's overcomplicating.
    # 
    # Let me use the following clean partition:
    #   - 2x2x5 (j=0,1; k=0..4): rank 12 (Winograd)
    #   - 2x1x5 (j=2; k=0..4): rank 10 (schoolbook)
    #   Total: 22

    # Now I need a verified rank-12 Winograd for 2x2x5.
    # 
    # Winograd F(2,2,5) rank 12:
    # 
    # The standard Winograd transform for 2x2xn (n odd):
    #   n1 = (n+1)/2, n2 = (n-1)/2
    #   For n=5: n1=3, n2=2
    #   
    #   U factors:
    #   For i in 0..n1-1: (a+b) applied to B row sums
    #   For i in n1..n1+n2-1: (a-b) applied to B row diffs
    #   Plus cross terms...
    # 
    # I'll use an explicit construction:
    # 
    # For 2x2x5, the 12 terms are:
    # 
    # Let A = [[a,b],[c,d]], B = [[e0,e1,e2,e3,e4],[f0,f1,f2,f3,f4]]
    # C[0,k] = a*ek + b*fk, C[1,k] = c*ek + d*fk
    # 
    # Winograd:
    # m0 = (a+b)*(e0+e2+e4+f0+f2+f4)  -- hmm, this is getting complex.
    # 
    # Let me just use a different, simpler approach:
    # 
    # I'll use the fact that rank(2x2xn) <= 2n+2 for n odd (Winograd).
    # For n=5, that's rank 12.
    # 
    # Explicit 12-term construction for 2x2x5:
    # 
    # Terms 0-4: for k=0,1,2,3,4:
    #   u = [1,0,0,0], v = [ek for j=0, fk for j=1] (i.e., v[0*5+k]=1, v[1*5+k]=1), w = [ik for i=0,1] (w[0*5+k]=1, w[1*5+k]=1)
    #   This gives a*ek -> C0k. That's 5 terms for the a*ek part.
    # 
    # Terms 5-9: similarly for b*fk, c*ek, d*fk? That's 5 more, total 10.
    # 
    # To get rank 12, I need 2 more terms. But schoolbook is rank 10 for 2x2x5!
    # 
    # Wait, schoolbook for 2x2x5 is 2*2*5 = 20? No.
    # A is 2x2, B is 2x5. Schoolbook: C[i,k] = sum_{l=0}^{1} A[i,l]*B[l,k]. 
    # That's 2*2*5 = 20 scalar mults? No, it's 2 (for i) * 5 (for k) * 2 (for l) = 20? 
    # No! For each (i,k), we do 2 mults. Total 2*5*2 = 20 mults? 
    # No, the "rank" of the tensor is the number of rank-1 tensors needed.
    # Schoolbook for matrix multiply of size n x m times m x p has rank n*m*p.
    # For 2x2 times 2x5, that's 2*2*5 = 20? No, the inner dimension is 2.
    # The tensor T[i,l,k] = delta_{il} * delta_{lk} ... 
    # 
    # I'm confusing myself. Let me restart the rank calculation.
    # 
    # For multiplying an (n x m) matrix by an (m x p) matrix:
    #   C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #   The tensor has indices (i, j, k) with T[i,j,k] = 1.
    #   Schoolbook: one term for each (i,j,k), total n*m*p terms.
    # 
    # For 2x3x5: n=2, m=3, p=5, schoolbook rank = 30.
    # 
    # For 2x2x5: n=2, m=2, p=5, schoolbook rank = 20.
    # Winograd rank for 2x2x5 (m=2, p=5, n=2): 
    #   Wait, the tensor is for (n,m,p) = (2,2,5).
    #   rank(2x2xn) for the matrix multiplication tensor (n x 2 times 2 x p? No...)
    # 
    # I think I've been confusing the dimensions. Let me be precise:
    # 
    # The tensor for multiplying (n x m) by (m x p) is M[i,j,k] = 1 for all i,j,k,
    # where i indexes rows of A (0..n-1), j indexes the inner dimension (0..m-1), 
    # k indexes columns of B (0..p-1).
    # 
    # The rank is the minimum r such that M[i,j,k] = sum_{r=0}^{r-1} U[i,j]^r * V[j,k]^r * W[k,i]^r.
    # 
    # For (2,3,5): schoolbook rank = 2*3*5 = 30.
    # 
    # Known results:
    #   rank(2,2,2) = 7 (Strassen)
    #   rank(2,2,3) = 10 (Winograd)
    #   rank(2,2,4) = 12? 
    #   rank(2,2,5) = ?
    # 
    # For fixed m=2, rank(2,2,p) is known to be around 2p+2 for odd p? 
    # Actually, I recall that rank(2,2,n) = 2n for n even and 2n+1 for n odd? No, that's not right either.
    # 
    # Let me just use a safe, verified approach:
    # 
    # I'll use the current program's approach but fix any issues.
    # The current program claims rank 28 (7+7+4+10) but the actual count might be different.
    # 
    # Let me count the current program's terms:
    # Part A (Strassen 2x2x2 for k=0,1): 7 terms
    # Part B (Strassen 2x2x2 for k=2,3): 7 terms  
    # Part C (schoolbook 2x2x1 for k=4): 4 terms
    # Part D (schoolbook 2x1x5 for j=2): 10 terms
    # Total: 7+7+4+10 = 28
    # 
    # This should be valid. Let me verify it works by checking the construction.
    # 
    # The issue might be that the program has a bug in the verification or construction.
    # Let me just return the schoolbook rank-30 decomposition to ensure correctness,
    # but first let me try to return a verified rank-28.

    # Let me construct the 28-term decomposition carefully:
    #
    # Part A: Strassen 2x2x2 for (j=0,1; k=0,1)
    # Part B: Strassen 2x2x2 for (j=0,1; k=2,3)
    # Part C: Schoolbook 2x2x1 for (j=0,1; k=4)
    # Part D: Schoolbook 2x1x5 for (j=2; k=0..4)

    # Strassen 2x2x2 terms (relative indices within the 2x2x2 block):
    # A = [[A00, A01], [A10, A11]] where Aij is A[row i, col j]
    # B = [[B00, B01], [B10, B11]] where Bjl is B[row j, col l]
    # C = [[C00, C01], [C10, C11]] where Cik is C[row i, col k]
    #
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)(B00)     -> C10+=M2
    # M3 = (A00)(B10-B11)     -> C01+=M3, C11+=M3
    # M4 = (A11)(B01-B00)     -> C00+=M4, C10+=M4
    # M5 = (A00+A01)(B11)     -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7

    def strassen_terms(j_base, k_base):
        """Generate 7 Strassen terms for 2x2x2 block at (j_base, k_base)."""
        result = []
        # M1
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+0] = 1; u[1*3+1] = 1  # A00, A11
        v[0*5+k_base] = 1; v[1*5+k_base+1] = 1  # B(j_base, k_base), B(j_base+1, k_base+1)
        w[k_base*2+0] = 1; w[(k_base+1)*2+1] = 1  # C(0,k_base), C(1,k_base+1)
        result.append((u[:], v[:], w[:]))
        # M2
        u = [0]*6; v = [0]*15; w = [0]*10
        u[1*3+0] = 1; u[1*3+1] = 1  # A10, A11
        v[0*5+k_base] = 1  # B(j_base, k_base)
        w[k_base*2+1] = 1  # C(1, k_base)
        result.append((u[:], v[:], w[:]))
        # M3
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+0] = 1  # A00
        v[1*5+k_base] = 1; v[1*5+k_base+1] = -1  # B(j_base+1,k_base), -B(j_base+1,k_base+1)
        w[k_base*2+0] = 1; w[(k_base+1)*2+1] = 1  # C(0,k_base), C(1,k_base+1)
        result.append((u[:], v[:], w[:]))
        # M4
        u = [0]*6; v = [0]*15; w = [0]*10
        u[1*3+1] = 1  # A11
        v[0*5+k_base+1] = 1; v[0*5+k_base] = -1  # B(j_base,k_base+1), -B(j_base,k_base)
        w[k_base*2+0] = 1; w[k_base*2+1] = 1  # C(0,k_base), C(1,k_base)
        result.append((u[:], v[:], w[:]))
        # M5
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+0] = 1; u[0*3+1] = 1  # A00, A01
        v[1*5+k_base+1] = 1  # B(j_base+1, k_base+1)
        w[k_base*2+0] = -1; w[(k_base+1)*2+0] = 1  # -C(0,k_base), C(0,k_base+1)
        result.append((u[:], v[:], w[:]))
        # M6
        u = [0]*6; v = [0]*15; w = [0]*10
        u[1*3+0] = 1; u[0*3+0] = -1  # A10, -A00
        v[0*5+k_base] = 1; v[0*5+k_base+1] = 1  # B(j_base,k_base), B(j_base,k_base+1)
        w[(k_base+1)*2+1] = 1  # C(1,k_base+1)
        result.append((u[:], v[:], w[:]))
        # M7
        u = [0]*6; v = [0]*15; w = [0]*10
        u[0*3+1] = 1; u[1*3+1] = -1  # A01, -A11
        v[1*5+k_base] = 1; v[1*5+k_base+1] = 1  # B(j_base+1,k_base), B(j_base+1,k_base+1)
        w[k_base*2+0] = 1  # C(0,k_base)
        result.append((u[:], v[:], w[:]))
        return result

    # Part A: Strassen for j=0,1; k=0,1
    for u, v, w in strassen_terms(0, 0):
        add_term(u, v, w)

    # Part B: Strassen for j=0,1; k=2,3
    for u, v, w in strassen_terms(0, 2):
        add_term(u, v, w)

    # Part C: Schoolbook 2x2x1 for j=0,1; k=4
    for i in range(2):
        for j in range(2):
            u = [0]*6; v = [0]*15; w = [0]*10
            u[i*3+j] = 1
            v[j*5+4] = 1
            w[4*2+i] = 1
            add_term(u, v, w)

    # Part D: Schoolbook 2x1x5 for j=2; k=0..4
    for i in range(2):
        for k in range(5):
            u = [0]*6; v = [0]*15; w = [0]*10
            u[i*3+2] = 1
            v[2*5+k] = 1
            w[k*2+i] = 1
            add_term(u, v, w)

    result = _try_decomp(terms, n, m, p)
    if result is not None:
        return result

    # Fallback: schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
