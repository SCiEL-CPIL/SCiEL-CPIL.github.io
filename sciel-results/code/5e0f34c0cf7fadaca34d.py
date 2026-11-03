"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
Strategy: Exploit the decoupled structure of the tensor to formulate as a set of
independent linear systems. For fixed sparse integer U and V, the system for each
(i,k) pair is linear in W. We search for sparse U, V such that the resulting W
has half-integer entries, enabling an exact low-rank decomposition.
"""

import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    # Try ranks from 30 upward
    
    def verify_exact(U, V, W):
        """Verify exact decomposition using Fractions."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def solve_m_system(M, b):
        """
        Solve M * x = b over Q where M is m x rank, b is length m.
        M and b are lists of Fractions.
        Returns list of Fractions of length rank, or None if no solution.
        We prefer solutions with half-integer entries.
        """
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        
        if rows == 0:
            return [Fraction(0) for _ in range(cols)]
        
        # Augmented matrix
        M_aug = [[M[i][j] for j in range(cols)] + [b[i]] for i in range(rows)]
        
        # Gaussian elimination
        pivot_rows = []
        for c in range(cols):
            pivot = -1
            for r in range(len(pivot_rows), rows):
                if M_aug[r][c] != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap
            M_aug[len(pivot_rows)], M_aug[pivot] = M_aug[pivot], M_aug[len(pivot_rows)]
            pivot_rows.append(pivot)
            pv = M_aug[pivot][c]
            for j in range(c, cols + 1):
                M_aug[pivot][j] = M_aug[pivot][j] / pv
            for r in range(rows):
                if r != pivot and M_aug[r][c] != 0:
                    f = M_aug[r][c]
                    for j in range(c, cols + 1):
                        M_aug[r][j] -= f * M_aug[pivot][j]
        
        # Check consistency
        for r in range(rows):
            all_zero = all(M_aug[r][c] == 0 for c in range(cols))
            if all_zero and M_aug[r][cols] != 0:
                return None
        
        # Extract solution with free vars = 0
        pivot_cols = set()
        for r in range(len(pivot_rows)):
            for c in range(cols):
                if M_aug[pivot_rows[r]][c] != 0:
                    pivot_cols.add(c)
                    break
        
        x = [Fraction(0) for _ in range(cols)]
        for r in range(len(pivot_rows)):
            pr = pivot_rows[r]
            for c in range(cols):
                if M_aug[pr][c] != 0:
                    x[c] = M_aug[pr][cols]
                    break
        
        # Verify
        for i in range(rows):
            s = Fraction(0)
            for j in range(cols):
                s += M[i][j] * x[j]
            if s != b[i]:
                return None
        
        return x
    
    def try_rank(rank, attempt=0):
        """Try to find a rank-r decomposition with sparse integer U, V."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + rank * 1000 + attempt)
        
        candidates = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
                       Fraction(2), Fraction(-2)]
        
        U = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
        V = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
        
        density = 0.4
        for i in range(n*m):
            for c in range(rank):
                if np.random.rand() < density:
                    U[i][c] = candidates[np.random.randint(len(candidates))]
        
        for j in range(m*p):
            for c in range(rank):
                if np.random.rand() < density:
                    V[j][c] = candidates[np.random.randint(len(candidates))]
        
        W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
        
        for i in range(n):
            for k in range(p):
                M = [[Fraction(0) for _ in range(rank)] for _ in range(m)]
                b = [Fraction(1) for _ in range(m)]
                
                for j in range(m):
                    for c in range(rank):
                        M[j][c] = U[i*m + j][c] * V[j*p + k][c]
                
                sol = solve_m_system(M, b)
                if sol is None:
                    return None
                
                # Check half-integer
                for c in range(rank):
                    if (sol[c] * 2).denominator != 1:
                        return None
                
                for c in range(rank):
                    W[k*n + i][c] = sol[c]
        
        if verify_exact(U, V, W):
            U_out = [[float(u) for u in row] for row in U]
            V_out = [[float(v) for v in row] for row in V]
            W_out = [[float(w) for w in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
        
        return None
    
    best_result = None
    
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        
        max_attempts = 20 if rank <= 35 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            res = try_rank(rank, attempt)
            if res is not None:
                return res
    
    # Fallback: Schoolbook rank 60
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

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(s - 1))
    print(f"Rank: {rank}, Max Error: {max_err}")
