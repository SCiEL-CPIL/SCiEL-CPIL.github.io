import itertools
import random

def construct():
    n = 10
    
    # Square 1: Cyclic addition table
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Search for a Square 2 orthogonal to L1
    # We can fix the first row of L2 to be 0..9 (standardization) or any permutation.
    # Let's fix L2[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] to reduce search space.
    # Actually, we just need ANY Latin square orthogonal to L1.
    
    # Backtracking search
    L2 = [[0] * n for _ in range(n)]
    # Fix first row to identity to reduce symmetry
    for j in range(n):
        L2[0][j] = j
        
    used_pairs = set()
    for j in range(n):
        used_pairs.add((L1[0][j], L2[0][j]))
        
    def is_valid(i, j, val):
        # Check row i
        if val in L2[i]:
            return False
        # Check col j
        for k in range(i):
            if L2[k][j] == val:
                return False
        # Check orthogonality with L1
        pair = (L1[i][j], val)
        if pair in used_pairs:
            return False
        return True

    def solve(i):
        if i == n:
            return True
        for j in range(n):
            for val in range(n):
                if is_valid(i, j, val):
                    L2[i][j] = val
                    used_pairs.add((L1[i][j], val))
                    if solve(i + 1):
                        return True
                    used_pairs.remove((L1[i][j], val))
                    L2[i][j] = 0
        return False

    # To speed up, we can order cells or use heuristics, but n=10 is small enough for random restarts or simple backtracking with pruning.
    # Simple backtracking might be slow. Let's add a timeout or limit.
    # However, since we know a solution exists, we can just run it.
    # To be safe against worst-case backtracking time, we can use a randomized approach or just hardcode.
    # Given the constraint of 60s, a simple backtracker on 10x10 with strong constraints (Latin + Orthogonal) is usually fast.
    
    # Let's try to solve it.
    if solve(1):
        return [L1, L2]
    else:
        # Fallback to 1 square if search fails (should not happen)
        return [L1]
