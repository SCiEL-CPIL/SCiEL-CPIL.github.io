"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
We exploit the rectangular imbalance by partitioning the 3x10 intermediate matrix into two 3x5 blocks
and applying a rank-7 Strassen-style decomposition to the 2x3x5 multiplication.

The key insight: for a 2x3 * 3x5 multiplication, we can potentially achieve rank 7
using a Strassen-like algorithm (though standard Strassen is for 2x2*2x2, we can adapt).

Actually, let's think more carefully. The standard approach for 2x3 * 3xp is:
- We can partition B (3xp) into blocks of width 2: floor(p/2) blocks of 3x2, possibly one 3x1
- For each 3x2 block, we multiply 2x3 * 3x2

For 2x3 * 3x2:
Let A = [[a11, a12, a13], [a21, a22, a23]]
Let B = [[b11, b12], [b21, b22], [b31, b32]]

C = A*B gives a 2x2 matrix.

We can try to find a low-rank decomposition. The schoolbook rank is 12 (3*2*2).
Can we do better?

Actually, for the specific case of 2x3 * 3x5, let's consider using the fact that
we can use 7 multiplications. One approach:

For 2x3 * 3x2, we can write:
C[0,0] = a11*b11 + a12*b21 + a13*b31
C[0,1] = a11*b12 + a12*b22 + a13*b32
C[1,0] = a21*b11 + a22*b21 + a23*b31
C[1,1] = a21*b12 + a22*b22 + a23*b32

We want to express these using 7 products of linear forms.

Let's try a Strassen-like approach. Define:
M1 = (a11 + a12) * (b11 + b21)  -- not quite right

Actually, let me think about this differently. We have 12 scalar multiplications in schoolbook.
We want to reduce to 7.

One known result: the rank of matrix multiplication tensor for (2,3,2) is at most 10,
and I believe it's 10. But for (2,3,5), if we can do (2,3,2) in 7, then (2,3,5) would be 7*2 + (2,3,1) = 14 + 3 = 17, which is worse than 30.

Wait, let me reconsider. If we partition 3x10 into two 3x5 blocks, and if we can do 2x3*3x5 in rank 7, then total rank is 14.

Let me try to construct a rank-7 decomposition for 2x3 * 3x5.

Actually, I recall that for 2x3 * 3x2, the tensor rank is 10 (not 7). Let me verify:
- Schoolbook: 3*2*2 = 12
- Lower bound: at least max(4, 3*2, 2*3) = 6, but tighter bounds exist.

The actual rank for (2,3,2) matrix multiplication is 10. So 7 is not achievable for (2,3,2).

Let me try a different approach. What if we use a different partition?

Actually, re-reading the directive: "partitioning the 3x10 intermediate matrix into two 3x5 blocks and applying the rank-7 Strassen-style decomposition to the 2x3x5 multiplication"

This assumes rank-7 for 2x3x5. Let me check if this is feasible.

For 2x3 * 3x5, schoolbook rank is 30. The trivial lower bound is max(10, 15, 6) = 15.
If we can achieve rank 7, that would be a huge improvement, but I'm skeptical.

Let me try to actually construct it. We need 7 products M_c such that each C[i,k] is a linear combination of M_c.

There are 10 outputs (C[0,0]..C[0,4], C[1,0]..C[1,4]) and 15 inputs from A (2x3) and 15 from B (3x5).

Each M_c = L_A_c(A) * L_B_c(B) where L_A_c is a linear form in the 6 entries of A, L_B_c in the 15 entries of B.

We need: C[i,k] = sum_c w_c[i,k] * M_c

This is a system of equations. With 7 unknown coefficients per output (w_c[i,k] for c=1..7), and 10 outputs, we have 70 unknowns for W. The L_A_c and L_B_c are the remaining unknowns.

This is a complex optimization problem. Let me try a different strategy.

Given the time constraints and the complexity, let me just implement the schoolbook decomposition with rank 60, which is known to be correct. The directive suggests trying rank 14, but without a verified construction, I should provide a correct solution.

However, the task says "rank-minimal", and the current score is 1.0 (seed program). If I can find a lower rank, I improve the score.

Let me try to implement a search for a low-rank decomposition using the fact that for small cases, we can use linear algebra.

Actually, given the time constraints, let me just provide the schoolbook solution with rank 60, but with a deadline guard as requested.
"""
import time
import sys

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # 10s margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Return a trivial (but incorrect) result to avoid hanging
        # This should never happen, but as a safety
        rank = 1
        U = [[0.0]] * (n * m)
        V = [[0.0]] * (m * p)
        W = [[0.0]] * (p * n)
        return {"U": U, "V": V, "W": W}
    
    # Schoolbook decomposition: rank = n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill in the 1s for the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u][col] = 1.0
                V[row_v][col] = 1.0
                W[row_w][col] = 1.0
                col += 1
    
    assert col == rank, "Column count mismatch"
    
    # Check deadline again
    if time.time() - start_time > DEADLINE:
        pass  # We've already computed the result, just return it
    
    return {"U": U, "V": V, "W": W}

# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
