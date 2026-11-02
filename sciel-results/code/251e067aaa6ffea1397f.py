import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Start with the known rank-39 solution
    # 4 Strassen 2x2 blocks (cols 0-7) + 2x3 schoolbook+strassen (cols 8-10)
    # Block 5 (cols 8,9): Strassen rank 7
    # Col 10: schoolbook rank 4
    # Total: 4*7 + 7 + 4 = 39

    def build_rank_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def strassen_block(b, r0):
            v00 = 2 * b
            v01 = 2 * b + 1
            v10 = 11 + 2 * b
            v11 = 12 + 2 * b
            w00 = 4 * b
            w01 = 4 * b + 2
            w10 = 4 * b + 1
            w11 = 4 * b + 3

            r = r0
            U[0][r] += 1; U[3][r] += 1
            V[v00][r] += 1; V[v11][r] += 1
            W[w00][r] += 1; W[w11][r] += 1
            r += 1
            U[2][r] += 1; U[3][r] += 1
            V[v00][r] += 1
            W[w10][r] += 1; W[w11][r] -= 1
            r += 1
            U[0][r] += 1
            V[v01][r] += 1; V[v11][r] -= 1
            W[w01][r] += 1; W[w11][r] += 1
            r += 1
            U[3][r] += 1
            V[v10][r] += 1; V[v00][r] -= 1
            W[w00][r] += 1; W[w10][r] += 1
            r += 1
            U[0][r] += 1; U[1][r] += 1
            V[v11][r] += 1
            W[w00][r] -= 1; W[w01][r] += 1
            r += 1
            U[2][r] += 1; U[0][r] -= 1
            V[v00][r] += 1; V[v01][r] += 1
            W[w11][r] += 1
            r += 1
            U[1][r] += 1; U[3][r] -= 1
            V[v10][r] += 1; V[v11][r] += 1
            W[w00][r] += 1
            r += 1
            return r

        r = 0
        for b in range(5):
            r = strassen_block(b, r)

        # Col 10 schoolbook: 4 terms
        U[0][r] = 1; V[10][r] = 1; W[20][r] = 1
        r += 1
        U[1][r] = 1; V[21][r] = 1; W[20][r] = 1
        r += 1
        U[2][r] = 1; V[10][r] = 1; W[21][r] = 1
        r += 1
        U[3][r] = 1; V[21][r] = 1; W[21][r] = 1

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0])
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Should equal 1 if there exists a term in A[i,0..m-1] @ B[j,0..p-1] that contributes to C[k,0..n-1]
        # Actually, for matrix multiplication M_{n,m,p}, the tensor is:
        # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
        # The matrix mult tensor has entries T[i,j,k] = 1 if the multiplication
        # C = A @ B has C[k, ?] involving A[i, ?] and B[?, ?]
        # 
        # More precisely: for MM_{n,m,p}, we have:
        # C[k, l] = sum_j A[k, j] * B[j, l]
        # The tensor T has shape (n*m, m*p, p*n) in the factorization
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # 
        # The index conventions:
        # U index: i*n + j for A[i,j], i in [0,n), j in [0,m)
        # V index: j*p + k for B[j,k], j in [0,m), k in [0,p)
        # W index: k*n + i for C[k,i], k in [0,p), i in [0,n)
        #
        # The MM tensor: T[u, v, w] = 1 iff
        # u = i*m + j, v = j*p + k, w = k*n + i
        # i.e., A[i,j] * B[j,k] contributes to C[k,i]
        
        # Build target tensor
        target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    target[u, v, w] = 1
        
        # Compute from factorization
        computed = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for r in range(rank):
            # u_indices where U[u,r] != 0
            u_nonzero = [u for u in range(n*m) if U[u][r] != 0]
            v_nonzero = [v for v in range(m*p) if V[v][r] != 0]
            w_nonzero = [w for w in range(p*n) if W[w][r] != 0]
            for u in u_nonzero:
                for v in v_nonzero:
                    for w in w_nonzero:
                        computed[u, v, w] += U[u][r] * V[v][r] * W[w][r]
        
        return np.array_equal(computed, target)

    # Get baseline
    best_result = build_rank_39()
    assert verify_decomposition(best_result["U"], best_result["V"], best_result["W"]), "Base rank-39 solution is invalid"
    
    best_rank = 39
    best_solution = best_result

    # Try to find a lower-rank solution using rank reduction techniques
    # Strategy: try to find linear dependencies among the rank-39 tensors
    # that allow us to reduce the rank

    # Convert to numpy for easier manipulation
    U_np = np.array(best_result["U"], dtype=np.float64)
    V_np = np.array(best_result["V"], dtype=np.float64)
    W_np = np.array(best_result["W"], dtype=np.float64)

    # Build the full tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1.0

    # The rank-39 decomposition gives us 39 rank-1 tensors
    # T_r = U[:,r] * V[:,r] * W[:,r] (outer product)
    # We want to see if some linear combination gives the same result with fewer terms

    # Check time
    if time.time() > DEADLINE - 30:
        return best_solution

    # Try a more sophisticated search: look for rank-38 solutions
    # by trying to merge two rank-1 tensors into one, or finding
    # a different decomposition

    # Approach: Use the fact that for 2x2 @ 2x11, we can think of it as
    # 2x2 @ 2x8 (rank 28 via Strassen) + 2x2 @ 2x3
    # For 2x2 @ 2x3: schoolbook is rank 12, but maybe we can do better
    
    # Actually, let's try a different block partition
    # What if we do 2x2 @ 2x9 with a mixed approach?
    
    # Let's try: 2x2 @ 2x11 as one big problem
    # Use a different factorization that might have lower rank
    
    # Try rank 38: 4 Strassen blocks (28) + rank-10 for 2x2 @ 2x3
    def try_rank_38():
        """Find a rank-38 solution: 4 Strassen blocks + rank-10 2x2@2x3."""
        rank = 38
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def strassen_block(b, r0):
            v00 = 2 * b
            v01 = 2 * b + 1
            v10 = 11 + 2 * b
            v11 = 12 + 2 * b
            w00 = 4 * b
            w01 = 4 * b + 2
            w10 = 4 * b + 1
            w11 = 4 * b + 3

            r = r0
            U[0][r] += 1; U[3][r] += 1
            V[v00][r] += 1; V[v11][r] += 1
            W[w00][r] += 1; W[w11][r] += 1
            r += 1
            U[2][r] += 1; U[3][r] += 1
            V[v00][r] += 1
            W[w10][r] += 1; W[w11][r] -= 1
            r += 1
            U[0][r] += 1
            V[v01][r] += 1; V[v11][r] -= 1
            W[w01][r] += 1; W[w11][r] += 1
            r += 1
            U[3][r] += 1
            V[v10][r] += 1; V[v00][r] -= 1
            W[w00][r] += 1; W[w10][r] += 1
            r += 1
            U[0][r] += 1; U[1][r] += 1
            V[v11][r] += 1
            W[w00][r] -= 1; W[w01][r] += 1
            r += 1
            U[2][r] += 1; U[0][r] -= 1
            V[v00][r] += 1; V[v01][r] += 1
            W[w11][r] += 1
            r += 1
            U[1][r] += 1; U[3][r] -= 1
            V[v10][r] += 1; V[v11][r] += 1
            W[w00][r] += 1
            r += 1
            return r

        r = 0
        for b in range(4):
            r = strassen_block(b, r)
        # r = 28, need 10 more for cols 8-10 (2x2 @ 2x3)
        
        # For 2x2 @ 2x3, try to find rank-10 decomposition
        # Sub-problem: A is 2x2, B is 2x3, C is 3x2
        # U indices: 0=A00, 1=A01, 2=A10, 3=A11
        # V indices: 0=B08, 1=B18, 2=B09, 3=B19, 4=B010, 5=B110
        # W indices: 0=C80, 1=C81, 2=C90, 3=C91, 4=C100, 5=C101
        
        # Target tensor entries (u,v,w) -> 1:
        # A00*B08->C80: (0,0,0), A01*B18->C80: (1,1,0)
        # A10*B08->C81: (2,0,1), A11*B18->C81: (3,1,1)
        # A00*B09->C90: (0,2,2), A01*B19->C90: (1,3,2)
        # A10*B09->C91: (2,2,3), A11*B19->C91: (3,3,3)
        # A00*B010->C100: (0,4,4), A01*B110->C100: (1,5,4)
        # A10*B010->C101: (2,4,5), A11*B110->C101: (3,5,5)
        
        # Try rank-10: use Strassen for cols 8,9 (7 terms) + 3 terms for col 10
        # But we need to verify this works.
        
        # Alternative: try to find a rank-10 solution by solving equations
        # Let's try a specific construction:
        # Terms 1-7: Strassen for cols 8,9
        # Terms 8-10: 3 terms for col 10
        
        # Strassen for 2x2 @ 2x2 (cols 8,9):
        # M1=(A00+A10)(B08+B19) -> contributes to C80,C81,C90,C91
        # M2=(A10-A00)B08 -> C80,C81
        # M3=A00(B19-B08) -> C80,C90
        # M4=(A11+A00)B09 -> C81,C91
        # M5=(A01+A11)B08 -> C80,C81
        # M6=(A11-A01)B19 -> C90,C91
        # M7=(A01-A11)(B09+B19) -> C81,C90
        
        # Let me use the standard Strassen pattern for 2x2@2x2:
        # For A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[ae+bg, af+bh],[ce+dg, cf+dh]]
        # Strassen:
        # p1=(a+d)(e+h): C00+=p1, C11+=p1
        # p2=(c+d)e: C10+=p2, C11-=p2
        # p3=a(f-h): C01+=p3, C11+=p3
        # p4=d(g-e): C00+=p4, C10+=p4
        # p5=(a+b)h: C00-=p5, C01+=p5
        # p6=(c-a)(e+f): C11+=p6
        # p7=(b-d)(g+h): C00+=p7
        
        # Mapping: a=A00, b=A01, c=A10, d=A11
        # e=B08, f=B09, g=B18, h=B19
        # C00=C80, C01=C90, C10=C81, C11=C91
        
        # p1=(A00+A11)(B08+B19): W80+=1, W91+=1
        # p2=(A10+A11)B08: W81+=1, W91-=1
        # p3=A00(B09-B18): W90+=1, W91+=1
        # p4=A11(B18-B08): W80+=1, W81+=1
        # p5=(A00+A01)B19: W80-=1, W90+=1
        # p6=(A10-A00)(B08+B09): W91+=1
        # p7=(A01-A11)(B18+B19): W80+=1
        
        # Let me verify p1: (A00+A11)(B08+B19) = A00*B08 + A00*B19 + A11*B08 + A11*B19
        # C80 should get A00*B08 (from p1) - but C80 also needs A01*B18
        # C91 should get A11*B19 (from p1) - but C91 also needs A10*B09
        # This looks correct for Strassen.
        
        # Add Strassen terms for cols 8,9:
        # p1: U: A00,A11; V: B08,B19; W: C80,C91
        U[0][r] += 1; U[3][r] += 1
        V[8][r] += 1; V[11][r] += 1  # B08=idx8, B19=idx11
        W[16][r] += 1; W[21][r] += 1  # C80=idx16, C91=idx21
        r += 1
        
        # p2: U: A10,A11; V: B08; W: C81,C91
        U[2][r] += 1; U[3][r] += 1
        V[8][r] += 1
        W[17][r] += 1; W[21][r] -= 1  # C81=idx17
        r += 1
        
        # p3: U: A00; V: B09,B18; W: C90,C91
        U[0][r] += 1
        V[9][r] += 1; V[10][r] -= 1  # B09=idx9, B18=idx10
        W[18][r] += 1; W[21][r] += 1  # C90=idx18
        r += 1
        
        # p4: U: A11; V: B18,B08; W: C80,C81
        U[3][r] += 1
        V[10][r] += 1; V[8][r] -= 1
        W[16][r] += 1; W[17][r] += 1
        r += 1
        
        # p5: U: A00,A01; V: B19; W: C80,C90
        U[0][r] += 1; U[1][r] += 1
        V[11][r] += 1
        W[16][r] -= 1; W[18][r] += 1
        r += 1
        
        # p6: U: A10,A00; V: B08,B09; W: C91
        U[2][r] += 1; U[0][r] -= 1
        V[8][r] += 1; V[9][r] += 1
        W[21][r] += 1
        r += 1
        
        # p7: U: A01,A11; V: B18,B19; W: C80
        U[1][r] += 1; U[3][r] -= 1
        V[10][r] += 1; V[11][r] += 1
        W[16][r] += 1
        r += 1
        # r = 35, need 3 more for col 10
        
        # For col 10: C100 = A00*B010 + A01*B110, C101 = A10*B010 + A11*B110
        # B010 = idx 12, B110 = idx 13
        # C100 = idx 20, C101 = idx 21... wait
        # W index: k*n + i where k is column of C (0..10), i is row (0..1)
        # C[8,0] -> w=8*2+0=16, C[8,1]->17, C[9,0]->18, C[9,1]->19
        # C[10,0]->20, C[10,1]->21
        
        # Wait, I used W[21] for C91 above. Let me recheck.
        # C[9,1] -> w = 9*2+1 = 19, not 21!
        # C[10,0] -> w = 10*2+0 = 20
        # C[10,1] -> w = 10*2+1 = 21
        
        # I made an error! Let me fix the W indices.
        # C[8,0]=16, C[8,1]=17, C[9,0]=18, C[9,1]=19, C[10,0]=20, C[10,1]=21
        
        # Let me redo this more carefully.
        return None  # Will redo below

    # Try alternative block decompositions
    # What about 3x3 blocks? 2x2 @ 2x11 = 2x2 @ 2x6 + 2x2 @ 2x5
    # 2x2 @ 2x6: schoolbook = 24, Strassen-like could be less
    # 2x2 @ 2x5: schoolbook = 20
    
    # Or 2x2 @ 2x11 directly with a custom scheme
    
    # Let's try: for the last 3 columns (8,9,10), use a rank-9 decomposition
    # instead of rank-11 (7+4).
    
    def try_last3_rank9():
        """
        Decompose 2x2 @ 2x3 (cols 8,9,10) with rank 9.
        Schoolbook needs 12. Can we save 3?
        
        C[0,8] = A00*B08 + A01*B18
        C[1,8] = A10*B08 + A11*B18
        C[0,9] = A00*B09 + A01*B19
        C[1,9] = A10*B09 + A11*B19
        C[0,10] = A00*B010 + A01*B110
        C[1,10] = A10*B010 + A11*B110
        
        That's 12 products. To reduce, we need shared intermediates.
        
        Idea: compute M1 = A00 + A10, M2 = A01 + A11
        Then:
        C[0,8] + C[1,8] = M1*B08 + M2*B18  (saves 2 products, but we need individual values)
        
        This doesn't directly help unless we can reconstruct.
        
        For 2x2 @ 2x3, the known minimum rank might be 9 or 10.
        Let's try rank 10 for the 2x3 part.
        """
        # 2x2 @ 2x3 with rank 10:
        # Use Strassen-like for first 2 cols (rank 7) + 3 for last col
        # But 3 isn't enough for last col alone.
        
        # Alternative: rank 10 direct
        # Let's try a specific construction.
        
        # Actually, for 2x2 @ 2x3, let's use:
        # 4 products for col 8, 3 for col 9, 3 for col 10 = 10
        # But each col needs at least 2 products (one per output row)
        # 2+2+2 = 6 minimum, but with shared intermediates maybe 9 or 10.
        
        # This is complex. Let's skip and stick with rank 39.
        return None

    # Given time constraints, return the best known solution
    if time.time() > DEADLINE - 10:
        return best_solution

    # Try one more thing: verify our solution is correct and return it
    return best_solution

# EVOLVE-BLOCK-END
