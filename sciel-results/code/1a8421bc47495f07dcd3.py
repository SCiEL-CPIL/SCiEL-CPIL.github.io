import time
import numpy as np

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 135  # safety margin before 150s kill


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 69

    U = [[0] * rank for _ in range(n * m)]  # 15 x 69
    V = [[0] * rank for _ in range(m * p)]  # 25 x 69
    W = [[0] * rank for _ in range(p * n)]  # 15 x 69

    col = 0

    # Schoolbook for a sub-block
    def schoolbook(i0, j0, k0, n_s, m_s, p_s, U, V, W, col):
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Strassen 2x2x2 block
    def strassen2(i0, j0, k0, U, V, W, col):
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1
        W[(k0+0)*n+(i0+1)][col] = 1; W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1
        V[(j0+0)*p+(k0+1)][col] = 1; V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+0)][col] = -1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+0)*n+(i0+1)][col] = 1
        col += 1
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = -1; W[(k0+1)*n+(i0+0)][col] = 1
        col += 1
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+0)][col] = -1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[(i0+0)*m+(j0+1)][col] = 1; U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1
        col += 1
        return col

    # Winograd 3x3x3 (rank 23 vs schoolbook 27)
    def winograd3(i0, j0, k0, U, V, W, col):
        # Winograd F(3) for 3x3x3
        # Uses 23 multiplications
        # The Winograd algorithm for 3x3 matrix multiplication
        # A(3x3) * B(3x3) = C(3x3)
        
        # We'll use the standard Winograd F(3) scheme
        # which requires 23 scalar multiplications
        
        # Define the Winograd factors
        # d0 = A[0,0], d1 = A[0,1], d2 = A[0,2]
        # d3 = A[1,0], d4 = A[1,1], d5 = A[1,2]
        # d6 = A[2,0], d7 = A[2,1], d8 = A[2,2]
        
        # e0 = B[0,0], e1 = B[1,0], e2 = B[2,0]
        # e3 = B[0,1], e4 = B[1,1], e5 = B[2,1]
        # e6 = B[0,2], e7 = B[1,2], e8 = B[2,2]
        
        # The 23 products are:
        # P1 = (d0+d4+d8)*(e0+e4+e8)
        # P2 = (d0-d4+d8)*(e0-e4+e8)
        # P3 = (d0+d4-d8)*(e0+e4-e8)
        # P4 = (d0-d4-d8)*(e0-e4-e8)
        # P5 = (d1+d5)*(e1+e5)
        # P6 = (d1-d5)*(e1-e5)
        # P7 = (d2+d6)*(e2+e6)
        # P8 = (d2-d6)*(e2-e6)
        # P9 = (d3+d7)*(e3+e7)
        # P10 = (d3-d7)*(e3-e7)
        # P11 = (d4+d8)*(e1+e5)
        # P12 = (d4-d8)*(e1-e5)
        # P13 = (d4+d8)*(e3+e7)
        # P14 = (d4-d8)*(e3-e7)
        # P15 = (d0+d4)*(e0+e4)
        # P16 = (d0-d4)*(e0-e4)
        # P17 = (d4+d8)*(e0+e4)
        # P18 = (d4-d8)*(e0-e4)
        # P19 = (d0+d4+d8)*(e1+e5)
        # P20 = (d0-d4+d8)*(e1-e5)
        # P21 = (d0+d4-d8)*(e3+e7)
        # P22 = (d0-d4-d8)*(e3-e7)
        # P23 = (d1+d5+d7)*(e2+e6)
        
        # Hmm, this is getting complex. Let me use a simpler approach.
        # Actually, let me just use schoolbook for 3x3x3 (27 ops) 
        # and instead focus on getting more 2x2x2 Strassen blocks.
        
        # For now, fall back to schoolbook
        return schoolbook(i0, j0, k0, 3, 3, 3, U, V, W, col)

    # Try different partitions to find minimum rank
    # Current best: [1,2] x [2,1,2] x [2,1,2] = 4 Strassen (28) + schoolbook (43) = 71
    
    # Let's try [2,1] x [2,1,2] x [2,1,2]
    # 2x2x2 blocks: where n_part=2, m_part=2, p_part=2
    # n_parts=[2,1], m_parts=[2,1,2], p_parts=[2,1,2]
    # Strassen blocks: (n=2,m=2,p=2) at positions (0,0,0), (0,0,2), (0,2,0), (0,2,2) = 4 blocks
    # Same as before.
    
    # Try [2,1] x [1,2,2] x [1,2,2]
    # n_parts=[2,1], m_parts=[1,2,2], p_parts=[1,2,2]
    # Strassen: n=2, m=2, p=2 at (0,1,1), (0,1,2), (0,2,1), (0,2,2) = 4 blocks
    # Same count.
    
    # Try [1,2] x [2,2,1] x [2,2,1]
    # Strassen: (1,0,0), (1,0,1), (1,1,0), (1,1,1) = 4 blocks
    # Same.
    
    # What about using Winograd 3x3x2?
    # Winograd 3x3x2 has rank 6 vs schoolbook 6. No gain.
    # Winograd 3x2x3 has rank 6 vs schoolbook 6. No gain.
    # Winograd 2x3x3 has rank 6 vs schoolbook 6. No gain.
    
    # What about 3x3x3 Winograd with rank 23?
    # If we can fit a 3x3x3 block, we save 4 ops (27-23=4)
    # Partition [3] x [3,2] x [3,2]:
    # 3x3x3 block: 1 block, saves 4
    # Remaining: 3x2x3 (schoolbook 6), 3x3x2 (schoolbook 6), 3x2x2 (schoolbook 4)
    # Total: 23 + 6 + 6 + 4 = 39
    # But wait, we need to check if this is valid.
    # n=3, m=5, p=5
    # m_parts=[3,2], p_parts=[3,2]
    # Blocks: (3,3,3), (3,3,2), (3,2,3), (3,2,2)
    # Schoolbook: 27+6+6+4 = 43
    # Winograd on 3x3x3: 23+6+6+4 = 39
    # That's a big improvement!
    
    # But wait, can we actually do Winograd 3x3x3? Let me implement it properly.
    
    # Actually, the Winograd F(3) algorithm for 3x3x3 is well-known.
    # Let me look it up properly.
    
    # The Winograd algorithm for 3x3 matrix multiply:
    # Uses 23 multiplications.
    
    # Let me implement it step by step.
    # A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
    # B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    # C = A*B
    
    # The 23 products are:
    # p0 = (a0+a4+a8) * (b0+b4+b8)
    # p1 = (a0-a4+a8) * (b0-b4+b8)  
    # p2 = (a0+a4-a8) * (b0+b4-b8)
    # p3 = (a0-a4-a8) * (b0-b4-b8)
    # p4 = (a1+a5) * (b3+b7)
    # p5 = (a1-a5) * (b3-b7)
    # p6 = (a2+a6) * (b6+b2)  # wait, need to be careful with indices
    # ...
    
    # This is getting complicated. Let me just use a different approach.
    # Instead of Winograd, let me try to find a better partition that
    # gives more 2x2x2 Strassen blocks.
    
    # Maximum number of 2x2x2 blocks in a 3x5x5 partition:
    # We need at least 2 rows, 2 cols, 2 rows in each dimension.
    # n=3: can have at most one block of size 2 (and one of size 1)
    # m=5: can have at most two blocks of size 2 (and one of size 1)
    # p=5: can have at most two blocks of size 2 (and one of size 1)
    # So max 1*2*2 = 4 Strassen blocks.
    
    # With 4 Strassen blocks (4*7=28), remaining schoolbook:
    # Total schoolbook if no Strassen: 3*5*5 = 75
    # Each 2x2x2 Strassen replaces 8 schoolbook with 7, saving 1.
    # 4 blocks save 4, so 75-4 = 71. That's what we have.
    
    # To do better, we need either:
    # 1. A 3x3x3 block with rank < 27 (Winograd gives 23, saving 4)
    # 2. Some other optimization
    
    # Let me try partition [3] x [2,3] x [2,3]:
    # Blocks: (3,2,2), (3,2,3), (3,3,2), (3,3,3)
    # If we use Winograd for 3x3x3 (rank 23) and schoolbook for others:
    # 3*2*2 + 3*2*3 + 3*3*2 + 23 = 12 + 18 + 18 + 23 = 71
    # Same as before!
    
    # What about [3] x [3,2] x [3,2]? Same thing.
    
    # Hmm, 71 seems hard to beat with simple block decompositions.
    
    # Let me try a completely different approach: use the known
    # rank of 3x5x5 matrix multiplication.
    
    # The rank of matrix multiplication tensor M_{n,m,p} satisfies:
    # r(M_{n,m,p}) <= n*m*p (schoolbook)
    # r(M_{n,m,p}) <= 7/8 * n*m*p for large n,m,p (Strassen)
    
    # For 3x5x5: 75 * 7/8 = 65.625, so at most 65?
    # But that's an asymptotic bound.
    
    # Actually, let me just try to implement Winograd 3x3x3 properly.
    
    # Winograd F(3) for 3x3x3:
    # Reference: Winograd, "Arithmetic complexity of controller design"
    
    # The algorithm uses the following structure:
    # Let A be 3x3, B be 3x3.
    
    # Preprocess A:
    # alpha0 = a00 + a11 + a22
    # alpha1 = a00 - a11 + a22
    # alpha2 = a00 + a11 - a22
    # alpha3 = a00 - a11 - a22
    # alpha4 = a01 + a12
    # alpha5 = a01 - a12
    # alpha6 = a02 + a10
    # alpha7 = a02 - a10
    # alpha8 = a20 + a21
    # alpha9 = a20 - a21
    # alpha10 = a11 + a22
    # alpha11 = a11 - a22
    # alpha12 = a00 + a11
    # alpha13 = a00 - a11
    # alpha14 = a01 + a11
    # alpha15 = a01 - a11
    # alpha16 = a11 + a22  (same as alpha10)
    # alpha17 = a11 - a22  (same as alpha11)
    # alpha18 = a00 + a22
    # alpha19 = a00 - a22
    # alpha20 = a01 + a02
    # alpha21 = a01 - a02
    
    # Hmm, this is getting too complex without a reference.
    
    # Let me just go with the current best of 71 and try a slightly
    # different partition to see if I can get 70.
    
    # Actually, let me reconsider. The problem says "rank-minimal exact
    # tensor decomposition". Maybe the true rank is lower than 71.
    
    # Let me try to use a 3x3x3 Winograd block. I'll implement it
    # using the known Winograd factors.
    
    # Winograd 3x3x3 (23 multiplications):
    # From the literature, the Winograd algorithm for 3x3 matrix
    # multiplication uses the following 23 products:
    
    # I'll use a simpler formulation. The key idea is:
    # C = A*B where A,B are 3x3.
    
    # Using the Winograd F(3) scheme:
    # The 23 scalar multiplications are:
    
    # Let me just use schoolbook for everything and accept rank 75,
    # then apply Strassen where possible.
    
    # Actually, I realize I should just keep the current approach
    # which gives 71, and try to verify it's correct.
    
    # Let me go back to the [1,2] x [2,1,2] x [2,1,2] partition
    # which gives 4 Strassen blocks + schoolbook = 71.
    
    # Reset and rebuild
    col = 0
    for i in range(len(U)):
        for j in range(rank):
            U[i][j] = 0
    for i in range(len(V)):
        for j in range(rank):
            V[i][j] = 0
    for i in range(len(W)):
        for j in range(rank):
            W[i][j] = 0
    
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_off = [0, 1]
    m_off = [0, 2, 3]
    p_off = [0, 2, 3]
    
    for bi in range(2):
        for bj in range(3):
            for bk in range(3):
                ns = n_parts[bi]
                ms = m_parts[bj]
                ps = p_parts[bk]
                i0 = n_off[bi]
                j0 = m_off[bj]
                k0 = p_off[bk]
                
                if ns == 2 and ms == 2 and ps == 2:
                    col = strassen2(i0, j0, k0, U, V, W, col)
                else:
                    col = schoolbook(i0, j0, k0, ns, ms, ps, U, V, W, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE:
        return None
    return {"U": U, "V": V, "W": W}
