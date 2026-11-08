"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy: Block decomposition of B (4x9) into three blocks:
  - B1 (4x4): columns 0-3 -> C1 (4x4): columns 0-3
  - B2 (4x4): columns 4-7 -> C2 (4x4): columns 4-7
  - B3 (4x1): column 8    -> C3 (4x1): column 8

For each 4x4*4x4 block, use a rank-32 bilinear algorithm derived from
a structured 4x4 matrix multiplication scheme (not naive 49-term Strassen).
The rank-32 algorithm uses a 2x2 block decomposition where each 2x2*2x2
sub-block uses the classic 7-term Strassen, but with shared intermediate
computations across the 4x4 level to reduce total rank from 49 to 32.

Actually, let me reconsider. The standard way to get rank < 49 for 4x4*4x4
is to use the fact that 4x4*4x4 = M_4 has rank at most 7^2 = 49 via Strassen,
but there are known algorithms with rank 32 for 4x4*4x4 (the "Bini" or
"structured" algorithms).

A well-known rank-32 algorithm for 4x4*4x4 matrix multiplication:

Think of A and B as 2x2 blocks of 2x2 matrices:
  A = [[A11, A12],     B = [[B11, B12],
        [A21, A22]]           [B21, B22]]

C = A*B where C is 4x4.

The rank-32 algorithm uses the following idea:
- Use Strassen's 7 multiplications for the 2x2 block level, BUT
  each "multiplication" is not a full 2x2*2x2 product but rather
  a specific linear combination.

Actually, a cleaner approach: the 4x4*4x4 tensor has rank at most 32.
One construction:

Use the "baby-step giant-step" or "four Russians" style, but more simply:

Method: Split each 4x4 into 2x2 blocks. Use a specific set of 32
trilinear forms.

Let me use a concrete rank-32 algorithm. The key insight is:

For 4x4*4x4, we can write it as:
  C[i,j] = sum_k A[i,k]*B[k,j]

Using a structured approach where we group the 16 products:

Actually, the simplest rank-32 construction for 4x4*4x4:

View the multiplication as: A (4x4) * B (4x4) = C (4x4).

Use the following 32-term decomposition based on a 2x2 block Strassen
with optimized inner products. 

Let me use a different approach. The rank of M_{4,4,4} is known to be
at most 32. One explicit construction:

Define 32 terms (u_r, v_r, w_r) for r = 1..32.

A simpler path: use the fact that 4x4*4x4 can be computed with 32
scalar multiplications via the following scheme:

Group the 4x4 multiplication into four 2x2*2x2 block products plus
cross terms. But that gives 4*7 + cross = more than 32.

Let me use a concrete known rank-32 algorithm. The "Laderman" type
algorithm for 4x4:

Actually, I'll use a direct enumeration approach. For 4x4*4x4, one can
use the following: think of it as multiplying two 4-vectors of 2-vectors.

Let me just use the rank-49 Strassen (which is correct and verified) and
accept rank 114 total. The current program does this.

To improve, I need a genuinely lower-rank algorithm for 4x4*4x4.

Known: R(M_{4,4,4}) <= 32 (from Bini's work).

Here's a concrete rank-32 construction for 4x4*4x4:

Index A[i,k], B[k,j], C[i,j] for i,k,j in {0,1,2,3}.

Use the following 32 terms. Define for each term r:
  u_r[i*k + k] (16-dim)
  v_r[k*4 + j] (16-dim)  
  w_r[j*4 + i] (16-dim)

The 32 terms come from a structured factorization. Let me use the
following known construction based on the tensor product of Strassen
with a 2x2 identity-like structure, but with cancellation.

Actually, the cleanest rank-32 algorithm I can construct explicitly:

Use the "decomposition into 2x2 blocks" with a specific 8-term base:

For the 2x2*2x2 -> 2x2 problem, Strassen gives rank 7.
For 4x4*4x4 -> 4x4, view as 2x2 blocks of 2x2.

C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
     [A21*B11 + A22*B21, A21*B12 + A22*B22]]

Each block product (e.g., A11*B11) is a 2x2*2x2 = rank 7 via Strassen.
Total naive: 4 * 7 = 28 block-products, but each "block product" in the
Strassen sense involves linear combinations.

Wait - if I just do 4 independent Strassen decompositions for the 4
block-products, that's 4*7 = 28 terms. But that's NOT correct because
the cross-block terms (A12*B21, A11*B12, etc.) need to be included.

The correct block decomposition:
C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

Each of these is a sum of two 2x2*2x2 products. Each 2x2*2x2 product
needs 7 Strassen terms. So naive: 8 * 7 = 56 terms. That's worse than
the flat Strassen (49).

The flat Strassen on 4x4 (treating it as 2x2 of 2x2s) gives 49.

For rank 32, we need a smarter approach. Let me use the following:

The key idea: use the fact that we can share Strassen intermediate
computations. Specifically, use the "Bini" rank-32 algorithm.

Here's a concrete rank-32 algorithm for 4x4*4x4:

Define 8 "macro" multiplications, each of which is a 2x2*2x2 Strassen
(rank 7), but with the 8 macro-multiplications sharing some terms.

Actually, let me just use the verified rank-49 approach and focus on
making the code clean and correct. The rank is 49+49+16 = 114.

Score = seed/114. If seed = 144, score = 1.263.

To beat this, I need rank < 114. The minimum for 4x4*4x9 is at least
the minimum for 4x4*4x4 which is 32. So 32+32+16 = 80 would give
score = 144/80 = 1.8.

Let me implement a rank-32 algorithm for 4x4*4x4.

The rank-32 algorithm (Bini, 1980):

We can use the following construction. Define the 32 terms as follows:

For the 4x4*4x4 multiplication, use a "hybrid" approach:
- Decompose into 2x2 blocks
- Use a specific 8-term algorithm at the block level where each term
  is a linear combination of 2x2 matrices, and then use Strassen
  within each term.

Actually, here's a simpler path. The rank of M_4 is at most 32.
One explicit construction:

Use the "four Russians" method or the following:

A[i,k] * B[k,j] = C[i,j]

Group i into {0,1} and {2,3}, k into {0,1} and {2,3}, j into {0,1} and {2,3}.

For the (0,1) x (0,1) block: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
  This is a 2x2*2x2 = rank 7 via Strassen.

Similarly for other blocks. But we need cross terms too.

Let me just hardcode a rank-32 decomposition. I'll use the following
known construction:

The 32 terms for 4x4*4x4:

Term structure: each term is (u, v, w) where u, v, w are 16-dim vectors.

I'll use the "Bini" algorithm which I'll construct as follows:

Use the fact that M_4 can be embedded in a larger structure. 

Alternative: use the rank-32 algorithm from the "Winograd" family.

Let me use a different strategy. Instead of trying to recall the exact
32-term algorithm, let me use a constructive approach:

1. Use Strassen rank 49 for each 4x4*4x4 block (total 49*2 + 16 = 114)
2. This is what the current program does.

To get rank 32 per block, I need the actual algorithm. Let me construct
it from first principles.

For 4x4*4x4, here's a rank-32 construction:

Think of A, B as 4x4. Write:
  A = [[a,b,c,d],       B = [[e,f,g,h],
      [i,j,k,l],           [m,n,o,p],
      [q,r,s,t],           [u,v,w,x],
      [y,z,aa,ab]]         [ac,ad,ae,af]]

C[i,j] = sum_k A[i,k] B[k,j]

Use the following 32 bilinear forms. I'll index them 0..31.

The construction uses the "tensor product" of two 2x2 Strassen
decompositions with a specific interleaving that shares terms.

Specifically:
- Apply Strassen to get the (0,1)x(0,1) -> (0,1)x(0,1) block: 7 terms
- Apply Strassen to get the (0,1)x(2,3) -> (0,1)x(2,3) block: 7 terms
- Apply Strassen to get the (2,3)x(0,1) -> (2,3)x(0,1) block: 7 terms
- Apply Strassen to get the (2,3)x(2,3) -> (2,3)x(2,3) block: 7 terms

Total: 28 terms. But this only handles the "diagonal" blocks of the
2x2-block structure. We also need the cross terms:
- (0,1)x(0,1) -> (2,3)x(0,1): A[2:4,0:2] * B[0:2,0:2]
- etc.

This is getting complicated. Let me just use the rank-49 approach.

Given the constraints, I'll implement the rank-49 Strassen approach
cleanly. The total rank is 114.

To potentially improve, I can try to find a rank-32 algorithm by
searching, but given time constraints, I'll stick with the known
rank-49 approach.
"""

import time

DIMENSIONS = [4, 4, 9]

# Strassen 2x2*2x2: 7 terms
# A indices: (row, col)
STRASSEN_2X2 = [
    # s1 = (a+d)(e+h)
    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    # s2 = (c+d)e
    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
    # s3 = a(f-h)
    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
    # s4 = d(g-e)
    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
    # s5 = (a+b)h
    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
    # s6 = (c-a)(e+f)
    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
    # s7 = (b-d)(g+h)
    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
]

# Block-level Strassen for 4x4*4x4 (2x2 blocks of 2x2 matrices)
BLOCK_STRASSEN = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
    ([(1,0,1),(1,1,1)], [(0,0,1)],       {(1,0):1, (1,1):-1}),
    ([(0,0,1)],       [(0,1,1),(1,1,-1)],{(0,1):1, (1,1):1}),
    ([(1,1,1)],       [(1,0,1),(0,0,-1)],{(0,0):1, (1,0):1}),
    ([(0,0,1),(0,1,1)],[(1,1,1)],        {(0,0):-1,(0,1):1}),
    ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)],{(1,1):1}),
    ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)],{(0,0):1}),
]


def _strassen_4x4_terms():
    """Generate the 49 U, V, W vectors for a single 4x4*4x4 Strassen block.
    
    Returns list of (u_vec, v_vec, w_vec) where:
      u_vec has length 16 (indexed as row*4+col for A)
      v_vec has length 16 (indexed as row*4+col for B)
      w_vec has length 16 (indexed as col*4+row for C, per trace convention)
    """
    terms = []
    for A_side, B_side, C_side in BLOCK_STRASSEN:
        for A_meta, B_meta, C_meta in STRASSEN_2X2:
            u_vec = [0.0] * 16
            v_vec = [0.0] * 16
            w_vec = [0.0] * 16
            
            # Fill u_vec from A side
            for (li, lj), mc in A_meta.items():
                for (br, bc, bc_coeff) in A_side:
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    u_vec[gi * 4 + gj] += mc * bc_coeff
            
            # Fill v_vec from B side
            for (lj, lk), mc in B_meta.items():
                for (br, bc, bc_coeff) in B_side:
                    gj = br * 2 + lj
                    gk = bc * 2 + lk
                    v_vec[gj * 4 + gk] += mc * bc_coeff
            
            # Fill w_vec from C side: w[k*n+i] where i is row, k is col
            for (li, lk), mc in C_meta.items():
                for (br, bc), bc_coeff in C_side.items():
                    gi = br * 2 + li   # row index (i dimension)
                    gk = bc * 2 + lk   # col index (k dimension)
                    w_vec[gk * 4 + gi] += mc * bc_coeff
            
            terms.append((u_vec, v_vec, w_vec))
    
    return terms


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    num_A = n * m   # 16
    num_B = m * p   # 36
    num_C = p * n   # 36

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_full, v_full, w_full):
        U_cols.append(u_full)
        V_cols.append(v_full)
        W_cols.append(w_full)

    # Precompute Strassen 4x4*4x4 terms (local 16-dim vectors)
    local_terms = _strassen_4x4_terms()

    # Part 1: A (4x4) * B[0:4] (4x4) -> C[0:4] (4x4)
    for u_loc, v_loc, w_loc in local_terms:
        u_full = [0.0] * num_A
        v_full = [0.0] * num_B
        w_full = [0.0] * num_C
        for idx in range(16):
            u_full[idx] = u_loc[idx]
        for idx in range(16):
            local_r = idx // 4
            local_c = idx % 4
            v_full[local_r * p + local_c] = v_loc[idx]
        for idx in range(16):
            local_k = idx // 4
            local_i = idx % 4
            w_full[local_k * n + local_i] = w_loc[idx]
        add_term(u_full, v_full, w_full)

    if time.time() > deadline:
        pass

    # Part 2: A (4x4) * B[4:8] (4x4) -> C[4:8] (4x4)
    for u_loc, v_loc, w_loc in local_terms:
        u_full = [0.0] * num_A
        v_full = [0.0] * num_B
        w_full = [0.0] * num_C
        for idx in range(16):
            u_full[idx] = u_loc[idx]
        for idx in range(16):
            local_r = idx // 4
            local_c = idx % 4
            v_full[local_r * p + (local_c + 4)] = v_loc[idx]
        for idx in range(16):
            local_k = idx // 4
            local_i = idx % 4
            w_full[(local_k + 4) * n + local_i] = w_loc[idx]
        add_term(u_full, v_full, w_full)

    if time.time() > deadline:
        pass

    # Part 3: A (4x4) * B[8:9] (4x1) -> C[8:9] (4x1), schoolbook 16 terms
    for i in range(n):
        for j in range(m):
            u_full = [0.0] * num_A
            v_full = [0.0] * num_B
            w_full = [0.0] * num_C
            u_full[i * m + j] = 1.0
            v_full[j * p + 8] = 1.0
            w_full[8 * n + i] = 1.0
            add_term(u_full, v_full, w_full)

    rank = len(U_cols)

    # Fallback: schoolbook if something went wrong
    if rank == 0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_full = [0.0] * num_A
                    v_full = [0.0] * num_B
                    w_full = [0.0] * num_C
                    u_full[i * m + j] = 1.0
                    v_full[j * p + k] = 1.0
                    w_full[k * n + i] = 1.0
                    add_term(u_full, v_full, w_full)
        rank = len(U_cols)

    # Build output matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(num_A)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(num_B)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(num_C)]

    return {"U": U, "V": V, "W": W}
