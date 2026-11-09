"""
Tensor decomposition for 4x4 * 4x6 matrix multiplication.

We use a block structure: partition the 4x6 matrix into two 4x3 blocks.
The 4x4 * 4x4 part uses Strassen's 7-term decomposition.
The 4x4 * 4x3 part can be handled by extending each Strassen term.

Actually, let's think more carefully. We want to compute C = A * B where
A is 4x4 and B is 4x6.

We can partition B into two 4x3 blocks: B = [B1 | B2] where B1, B2 are 4x3.
Then C = [A*B1 | A*B2], each a 4x3 matrix.

For a 4x4 * 4x3 multiplication, we can use a rank-10 decomposition or similar.
But let's try a different approach: use the known optimal or near-optimal
decompositions.

Known results:
- 2x2: rank 7 (Strassen)
- 3x3: rank 23 (or better, but 23 is standard)
- 4x4: rank 49 (best known) or we can use Strassen-based approaches

For 4x4 * 4x6, we can think of it as:
- Split the 4x6 into two 4x3 blocks
- Compute 4x4 * 4x3 twice, but share some computations

Actually, a cleaner approach: use the tensor structure directly.

The matrix multiplication tensor M_{n,m,p} has entries T[i,j,k; l,m,n] = 
1 if (i,j) corresponds to (l,m) in appropriate way...

Let me think about this differently. We want to find U, V, W such that
T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  ... no wait.

The tensor for matrix multiplication: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
So the tensor T has T[i,j,k] = 1 if we're computing the (i,k) entry using A[i,j] and B[j,k].

The rank-r decomposition means:
T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]

Wait, the task says W[k*n+i, r] - so W is indexed by (k, i) where k is the column of B (0..p-1) and i is the row of A (0..n-1). And n=4, m=4, p=6.

So:
- U has shape (n*m, rank) = (4*4, rank) = (16, rank), indexed by (i, j) for i in 0..3, j in 0..3
- V has shape (m*p, rank) = (4*6, rank) = (24, rank), indexed by (j, k) for j in 0..3, k in 0..5
- W has shape (p*n, rank) = (6*4, rank) = (24, rank), indexed by (k, i) for k in 0..5, i in 0..3

And the decomposition is:
For all i, j, k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

Where T[i,j,k] = 1 (it's the matrix multiplication tensor).

So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

This is equivalent to: the bilinear form (i,j,k) -> 1 can be written as a sum of rank-1 tensors.

Let's try to use a structured approach. We'll use the fact that we can decompose the 4x4 matrix into 2x2 blocks and use Strassen.

Actually, let me just try to find a good decomposition empirically. The schoolbook gives rank 96 (4*4*6). 

Let me try a block-based Strassen approach:
- Split A (4x4) into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
- Split B (4x6) into top 2x6 and bottom 2x6: B = [B0; B1] where B0 is 2x6, B1 is 2x6
- Result C (4x6) = [A00*B0 + A01*B1; A10*B0 + A11*B1]

For each 2x2 * 2x6 multiplication, we can use a Strassen-like approach.
2x2 * 2x6: split B into two 2x3 blocks.
2x2 * 2x3: we can use a rank-? decomposition.

Known: 2x2 * 2x2 has rank 7.
2x2 * 2x3: We can write it as 2x2 * 2x2 + 2x2 * 2x1, but that's not quite right for tensor rank.

Actually, for 2x2 * 2x3, we can partition the 2x3 into a 2x2 and a 2x1.
The 2x2 * 2x2 part needs rank 7.
The 2x2 * 2x1 part: (AB)_{i,2} = sum_j A_{i,j} B_{j,2}, which is just two dot products, rank 2.
But we need to share computations...

This is getting complex. Let me try a direct approach with a known decomposition.

For 4x4 * 4x4, the best known rank is 49. But implementing that is complex.

Let me try a simpler approach: use Strassen recursively.
4x4 * 4x4: split into 2x2 blocks, use 7 Strassen terms, each involving 2x2*2x2 multiplications.
That gives 7 * 7 = 49 scalar multiplications for 4x4*4x4.

For 4x4 * 4x6:
Split A into 2x2 blocks. Split B into top 2x6 and bottom 2x6.
Each 2x2 * 2x6 can be handled by splitting the 2x6 into two 2x3 blocks.
2x2 * 2x3: partition into 2x2 * 2x2 + 2x2 * 2x1.

Hmm, this is getting complicated. Let me try a different tactic.

Let me just use a randomized/local search approach to find a low-rank decomposition.

Actually, let me try the most straightforward approach first: use the 2x2 Strassen decomposition and extend it.

For 2x2 * 2x6:
A is 2x2, B is 2x6.
Split B into B1 (2x3) and B2 (2x3).
For 2x2 * 2x3:
  Use Strassen-like: 
  m1 = (A00+A11)*(B00+B11)  -- but B00, B11 are 1x3 vectors here
  This doesn't directly work because Strassen assumes square matrices.

Let me think about 2x2 * 2x3 more carefully.
C = A * B, A is 2x2, B is 2x3.
C00 = A00*B00 + A01*B10
C01 = A00*B01 + A01*B11
C02 = A00*B02 + A01*B12
C10 = A10*B00 + A11*B10
C11 = A10*B01 + A11*B11
C12 = A10*B02 + A11*B12

We can use 7 scalar multiplications? Let's see...
m1 = (A00+A11)*(B00+B10)  -- wait, these are scalars times... no, A00 is scalar, B00 is scalar.

Actually all entries are scalars. So:
m1 = (A00+A11)*(B00+B10)
m2 = (A10+A11)*B00
m3 = A00*(B01-B11)
m4 = A11*(B10-B00)
m5 = (A00+A01)*B12  -- hmm, this doesn't work cleanly

Let me just use a known result: the rank of 2x2*2x3 is at most 10, and I believe it's actually 8 or 9.

You know what, let me just implement a search. I'll use the fact that we can verify a decomposition easily, and do a local search or genetic algorithm to find a low-rank decomposition.

Given the time constraint (150s), let me implement:
1. Start with the schoolbook decomposition (rank 96)
2. Try to merge terms / reduce rank using local search
3. Or, use a structured decomposition based on block Strassen

Let me try the block Strassen approach carefully.

4x4 * 4x6:
A = [[A00, A01], [A10, A11]] where each Ai is 2x2
B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6
C = [A00*B_top + A01*B_bottom; A10*B_top + A11*B_bottom]

So we need 4 instances of 2x2 * 2x6.

For 2x2 * 2x6:
Split the 2x6 into three 2x2 blocks: B = [B1 | B2 | B3]
2x2 * 2x2 has rank 7 (Strassen).
So 2x2 * 2x6 = 2x2 * 2x2 + 2x2 * 2x2 + 2x2 * 2x2, rank 3*7 = 21?
But we can share some computations across the three blocks.

Actually, if we use Strassen for each 2x2*2x2 independently, that's 21 multiplications.
But we can do better by sharing the additions.

Alternatively, for 2x2 * 2x6, we can think of it as:
C = A * B where A is 2x2, B is 2x6.
We can write:
m1 = (A00+A11) * (B00+B10)  -- but B00, B10 are 1x6 vectors? No...

I think the cleanest approach is to just do a numerical search. Let me implement that.

Strategy:
1. Generate random rank-r decompositions
2. Verify they satisfy the tensor equation
3. Use local search to improve

But verifying is: for all i,j,k in range, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1

This is 4*4*6 = 96 constraints.

Let me try a different approach: use the known Strassen decomposition for 2x2 and build up.

Actually, I'll implement a direct search using the structure. Let me try rank 49 first (the known bound for 4x4*4x4, and 4x4*4x6 should be at most that times some factor).

Let me just implement a local search that starts from schoolbook and tries to zero out entries to reduce effective rank.
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 6]
TIME_LIMIT = 140.0  # safety margin from 150s

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def _construct_strassen_2x2_2x6():
    """
    Construct a decomposition for 2x2 * 2x6.
    Split 2x6 into three 2x2 blocks.
    Use Strassen (rank 7) for each, but try to share.
    
    Actually, let me just use 3 independent Strassen decompositions.
    That gives rank 21 for 2x2*2x6.
    """
    # Strassen for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # 
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For 2x2 * 2x6, split B into B1, B2, B3 each 2x2.
    # Use 3 independent Strassen: rank 21.
    
    # U indices: (i,j) for i in 0..1, j in 0..1 -> 4 entries
    # V indices: (j,k) for j in 0..1, k in 0..5 -> 6 entries  
    # W indices: (k,i) for k in 0..5, i in 0..1 -> 6 entries
    
    rank = 21
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    for block in range(3):
        base = block * 7
        # Strassen terms for block
        # m1: (A00+A11) * (B00+B10) where B00 = B[0, block*2], B10 = B[1, block*2]
        U[0][base+0] = 1  # A00
        U[3][base+0] = 1  # A11
        V[0][base+0] = 1  # B00
        V[2][base+0] = 1  # B10  (j=1, k=block*2)
        # W for C00 and C11
        W[block*2][base+0] = 1      # C00: k=block*2, i=0
        W[block*2+1][base+0] = 1    # C01: k=block*2+1, i=0
        W[block*2][base+0] += 0     # Wait, W is indexed by (k,i)
        # C00: i=0, k=block*2 -> W[block*2*2+0] = W[block*2]
        # C11: i=1, k=block*2+1 -> W[(block*2+1)*2+1] = W[block*2+3]
        
        # Let me redo W indexing: W[k*n+i] where n=2 for this subproblem
        # W[k*2 + i]
        W[block*2*2 + 0] = 1     # C[0, block*2]
        W[(block*2+1)*2 + 1] = 1  # C[1, block*2+1]
        
        # m2: (A10+A11) * B00
        U[2][base+1] = 1  # A10
        U[3][base+1] = 1  # A11
        V[0][base+1] = 1  # B00
        W[block*2*2 + 1] = 1     # C[1, block*2]
        W[(block*2+1)*2 + 1] = -1  # C[1, block*2+1]
        
        # m3: A00 * (B01 - B11)
        U[0][base+2] = 1
        V[1][base+2] = 1   # B01
        V[3][base+2] = -1  # B11
        W[(block*2+1)*2 + 0] = 1  # C[0, block*2+1]
        W[(block*2+1)*2 + 1] = 1  # C[1, block*2+1]
        
        # m4: A11 * (B10 - B00)
        U[3][base+3] = 1
        V[2][base+3] = 1   # B10
        V[0][base+3] = -1  # B00
        W[block*2*2 + 0] = 1   # C[0, block*2]
        W[block*2*2 + 1] = 1   # C[1, block*2]
        
        # m5: (A00+A01) * B11
        U[0][base+4] = 1
        U[1][base+4] = 1
        V[3][base+4] = 1   # B11
        W[block*2*2 + 0] = -1   # C[0, block*2]
        W[(block*2+1)*2 + 0] = 1  # C[0, block*2+1]
        
        # m6: (A10-A00) * (B00+B01)
        U[2][base+5] = 1
        U[0][base+5] = -1
        V[0][base+5] = 1   # B00
        V[1][base+5] = 1   # B01
        W[(block*2+1)*2 + 1] = 1  # C[1, block*2+1]
        
        # m7: (A01-A11) * (B10+B11)
        U[1][base+6] = 1
        U[3][base+6] = -1
        V[2][base+6] = 1   # B10
        V[3][base+6] = 1   # B11
        W[block*2*2 + 0] = 1   # C[0, block*2]
    
    # Verify
    valid, info = _verify_decomposition(U, V, W, 2, 2, 6, rank)
    if not valid:
        return None, rank
    
    return {'U': U, 'V': V, 'W': W}, rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
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
    
    best = {'U': U, 'V': V, 'W': W}
    best_rank = rank
    
    # Try to find better decomposition using local search
    # Strategy: try to eliminate terms by finding linear dependencies
    
    def _try_eliminate(decomp, current_rank, target_rank, deadline):
        """Try to reduce rank by finding linear dependencies."""
        U = [row[:] for row in decomp['U']]
        V = [row[:] for row in decomp['V']]
        W = [row[:] for row in decomp['W']]
        
        # Try to find a column c that can be expressed as a combination of others
        # This is complex, so let's try a simpler approach: random perturbation + reoptimization
        
        return None
    
    # Try structured decomposition based on block Strassen
    # 4x4 * 4x6: split into 2x2 blocks
    # This gives 4 instances of 2x2 * 2x6
    
    # For now, let's try a randomized search for low-rank decompositions
    def _random_decomposition(rank, n, m, p, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randint(-2, 3, size=(n*m, rank)).tolist()
        V = rng.randint(-2, 3, size=(m*p, rank)).tolist()
        W = rng.randint(-2, 3, size=(p*n, rank)).tolist()
        return {'U': U, 'V': V, 'W': W}
    
    # Try various ranks with random initializations
    for target_rank in range(50, 97, 2):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        for seed in range(10):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            decomp = _random_decomposition(target_rank, n, m, p, seed=1000 + target_rank + seed)
            valid, info = _verify_decomposition(decomp['U'], decomp['V'], decomp['W'], n, m, p, target_rank)
            if valid:
                best = decomp
                best_rank = target_rank
                return best
    
    return best

if __name__ == '__main__':
    result = construct()
    rank = len(result['U'][0])
    print(f"Best rank: {rank}")
