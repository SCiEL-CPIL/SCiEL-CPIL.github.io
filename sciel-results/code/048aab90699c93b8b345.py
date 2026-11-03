"""Find rank-minimal tensor decomposition for 2x3 * 3x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    num_entries = n * m * p  # 42
    num_u = n * m  # 6
    num_v = m * p  # 21
    num_w = p * n  # 14
    
    def schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(num_u)]
        V = [[Fraction(0)] * rank for _ in range(num_v)]
        W = [[Fraction(0)] * rank for _ in range(num_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True
    
    def solve_linear_system_exact(A, b, deadline):
        num_eq = len(A)
        if num_eq == 0:
            return [Fraction(0)] * (len(A[0]) if A else 0)
        num_var = len(A[0])
        aug = [A[i][:] + [b[i]] for i in range(num_eq)]
        pivot_cols = []
        for col in range(num_var):
            if time.time() > deadline:
                return None
            pivot = None
            for row in range(len(pivot_cols), num_eq):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            pr = len(pivot_cols)
            if pivot != pr:
                aug[pr], aug[pivot] = aug[pivot], aug[pr]
            pv = aug[pr][col]
            for j in range(col, num_var + 1):
                aug[pr][j] /= pv
            for row in range(num_eq):
                if row == pr:
                    continue
                f = aug[row][col]
                if f != 0:
                    for j in range(col, num_var + 1):
                        aug[row][j] -= f * aug[pr][j]
            pivot_cols.append(col)
        for row in range(len(pivot_cols), num_eq):
            if aug[row][num_var] != 0:
                return None
        x = [Fraction(0)] * num_var
        for row in range(len(pivot_cols)):
            for col in range(num_var):
                if aug[row][col] != 0:
                    x[col] = aug[row][num_var]
                    break
        return x
    
    def try_find_decomp(r, max_attempts=80):
        if r >= best_rank:
            return None
        num_w_vars = num_w * r
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            def rand_val():
                rv = random.random()
                if rv < 0.2:
                    return Fraction(0)
                elif rv < 0.55:
                    return Fraction(1)
                elif rv < 0.8:
                    return Fraction(-1)
                else:
                    return Fraction(random.randint(-2, 2))
            U = [[rand_val() for _ in range(r)] for _ in range(num_u)]
            V = [[rand_val() for _ in range(r)] for _ in range(num_v)]
            A = [[Fraction(0)] * num_w_vars for _ in range(num_entries)]
            b = [Fraction(1)] * num_entries
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            w_idx = (k * n + i) * r + c
                            coeff = U[i * m + j][c] * V[j * p + k][c]
                            A[eq_idx][w_idx] += coeff
                        eq_idx += 1
            w_sol = solve_linear_system_exact(A, b, deadline)
            if w_sol is None:
                continue
            W = [[Fraction(0)] * r for _ in range(num_w)]
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        W[k * n + i][c] = w_sol[(k * n + i) * r + c]
            if verify_decomp(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        return None
    
    for r in range(10, best_rank):
        if time.time() > deadline:
            break
        result = try_find_decomp(r, max_attempts=100)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
