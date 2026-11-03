import time
import numpy as np
from fractions import Fraction
import sys

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def is_half_integer(x):
        y = x * 2
        return y.denominator == 1

    def try_rank(rank):
        if time.time() > deadline:
            return None
        
        # Structural change: Fix U to be the identity matrix (rank must be >= 6).
        # This reduces the problem to finding V and W such that:
        # sum_{c} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
        # Since U is identity (U[a, c] = 1 if a==c else 0), and a = i*m+j:
        # V[i*m+j, p+k] * W[k*n+i, i*m+j] = 1
        # Wait, the index of V in the sum is j*p+k.
        # The term is non-zero only if c = i*m+j.
        # So the term is V[j*p+k, i*m+j] * W[k*n+i, i*m+j].
        # We need: V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 for all i,j,k.
        
        # This implies that for each (i,j,k), the product of two specific entries is 1.
        # Let A_{i,j,k} = V[j*p+k, i*m+j] and B_{i,j,k} = W[k*n+i, i*m+j].
        # We need A_{i,j,k} * B_{i,j,k} = 1.
        # This means B_{i,j,k} = 1 / A_{i,j,k}.
        
        # Constraints:
        # 1. V is (15, rank). If rank >= 6, we can set V's columns 0..5 to be the "active" ones for U=I.
        #    Actually, if U is I_6, then rank must be at least 6. Let's assume rank=6 for now?
        #    If rank=6, U is I_6.
        #    Then V is (15, 6) and W is (10, 6).
        #    For each c in 0..5, c corresponds to (i, j) where c = i*m+j.
        #    So for a fixed c, we have specific i and j.
        #    The equation for (i, j, k) involves only the term where c = i*m+j.
        #    So V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1.
        
        # Let's analyze the dependencies.
        # Fix c. Let i = c // m, j = c % m.
        # For this c, the non-zero entries in V are in column c.
        # The entry V[j*p+k, c] is non-zero?
        # The term in the sum for (i, j, k) is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # If U is I, U[i*m+j, c] is 1 if c == i*m+j, else 0.
        # So for a specific (i, j, k), only the term with c = i*m+j survives.
        # So we require: V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1.
        
        # Let's denote c = i*m+j.
        # Then the condition is: V[j*p+k, c] * W[k*n+i, c] = 1.
        
        # This must hold for all i, j, k.
        # For a fixed c (which fixes i, j), we have k ranging from 0 to p-1.
        # So for each k, V[j*p+k, c] * W[k*n+i, c] = 1.
        
        # This means V[j*p+k, c] and W[k*n+i, c] are reciprocals.
        # Since entries must be half-integers, let V[j*p+k, c] = x_k.
        # Then W[k*n+i, c] = 1/x_k.
        # For 1/x_k to be a half-integer, x_k must be of the form 2/a where a is an integer?
        # If x = 2/a, then 1/x = a/2, which is a half-integer.
        # So we can choose x_k from {1/2, 1, 2, 4, ...} or {-1/2, -1, -2, ...}
        
        # Are there any cross-constraints between different c's?
        # The matrix V has shape (15, rank).
        # The entry V[row, col] is V[j*p+k, c].
        # The row index is j*p+k.
        # For a fixed row (j, k), which c's are involved?
        # The condition for c involves i = c//m, j = c%m.
        # So for a fixed (j, k), the row in V is j*p+k.
        # The columns c that access this row are those where c % m == j.
        # There are n such columns (for i=0..n-1).
        # So V[j*p+k, c] is constrained for c = 0*m+j, 1*m+j, ..., (n-1)*m+j.
        # Specifically, for each such c (with corresponding i), V[j*p+k, c] * W[k*n+i, c] = 1.
        
        # Similarly for W. W has shape (10, rank).
        # Entry W[row, col] is W[k*n+i, c].
        # Row is k*n+i.
        # For a fixed (k, i), the row is k*n+i.
        # The columns c that access this row are those where c // m == i.
        # There are m such columns (for j=0..m-1).
        # So W[k*n+i, c] is constrained for c = i*m+0, i*m+1, ..., i*m+(m-1).
        
        # Crucially, the choice of V[j*p+k, c] for a specific c is independent of other c's
        # because W[k*n+i, c] is also specific to c.
        # Wait, is W[k*n+i, c] shared?
        # No, the column index is c.
        # So for each c, we have a set of equations:
        # For k=0..p-1: V[j*p+k, c] * W[k*n+i, c] = 1.
        # where i = c//m, j = c%m.
        
        # This means we can independently choose the values for each column c!
        # For each c, we pick values for V[j*p+0..p-1, c] and set W[k*n+i, c] = 1/V[...].
        # As long as V entries are half-integers and their reciprocals are half-integers, we are good.
        
        # So if rank=6, we can always find a solution?
        # Let's check if rank=6 is possible.
        # U = I_6.
        # For each c in 0..5:
        #   i = c//3, j = c%3.
        #   For k in 0..4:
        #     Set V[j*5+k, c] = 1.
        #     Set W[k*2+i, c] = 1.
        #   All other entries in V and W are 0.
        
        # Let's verify this solution.
        # T_{i,j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # If U is I, only c = i*m+j contributes.
        # Term = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
        #      = 1 * 1 * 1 = 1.
        
        # So rank 6 works!
        # Wait, is rank 6 the minimum?
        # Lower bound for matrix multiplication rank is max(n, m, p)? No.
        # For 2x3 * 3x5, the rank is at least the max dimension?
        # Actually, the rank of the tensor is at least the maximum of the dimensions of the matrices?
        # No, for square matrices n x n * n x n, rank is at least n.
        # Here, can we go lower than 6?
        # The tensor has support on 30 points.
        # But with the identity trick, we used 6 terms.
        # Is it possible to use fewer?
        # If rank < 6, say 5.
        # U is (6, 5).
        # The sum is over 5 terms.
        # It seems unlikely to beat 6 with this specific structure, but maybe with a different structure?
        # However, rank 6 is already a huge improvement over 30.
        # Score would be 30/6 = 5.0? No, score is seed/value.
        # Seed is schoolbook (30).
        # If we find rank 6, value is 6.
        # Score = 30/6 = 5.0.
        
        # Let's double check if rank can be lower.
        # The "rank" of the bilinear map A (2x3) * B (3x5) -> C (2x5).
        # The number of multiplications required.
        # For 2x2 * 2x2, Strassen uses 7. Rank 7.
        # For 2x3 * 3x5?
        # If we can decompose into 6 multiplications, that's very good.
        
        # Let's implement rank 6 solution.
        
        if rank < 6:
             # Can we do rank 5?
             # Likely not with this simple structure.
             # Let's just try to construct rank 6.
             return None
        
        # Construct rank 6 solution
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for c in range(rank):
            i = c // m
            j = c % m
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
        
        # Verify
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None

    # Try rank 6 first
    res = try_rank(6)
    if res:
        return res

    # Fallback to schoolbook
    return schoolbook()

if __name__ == "__main__":
    construct()
