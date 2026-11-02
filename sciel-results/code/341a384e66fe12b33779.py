import time
import random
import itertools
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def schoolbook():
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

    def verify(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s - 1) > tol:
                            return False
                    else:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def verify_exact(U, V, W, n, m, p):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    best = schoolbook()
    best_rank = n * m * p

    # Search for lower rank
    # Try ranks 30 down to 26
    for target_rank in range(30, 25, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Random search for this rank
        # We will try to find a valid assignment of non-zero entries
        # This is a heuristic search
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize U, V, W with zeros
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # For each rank column, pick a random pattern of non-zeros
            # To make it work, the patterns must cover the 42 required entries
            # and cancel out the rest.
            
            # A better heuristic: Start with schoolbook and merge terms
            # Or, use a random assignment of the 42 triples to the target_rank columns
            # and then solve for coefficients?
            
            # Let's try a random assignment of triples to columns
            all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            assignments = [[] for _ in range(target_rank)]
            
            # Shuffle and assign
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for idx, triple in enumerate(shuffled):
                col = idx % target_rank
                assignments[col].append(triple)
                
            # Build U, V, W based on assignments
            # If a column has multiple triples, we set the corresponding entries to 1
            # This creates a "bilinear" term that might sum to 1 for the desired entries
            # and 0 for others if the sets are disjoint in a specific way
            # This is unlikely to work for rank < 42 directly without signs/coefficients
            
            # Instead, let's try to find a solution with coefficients in {-1, 0, 1}
            # by randomizing the signs
            
            for sign_seed in range(5):
                random.seed(attempt * 100 + sign_seed)
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                for col, triples in enumerate(assignments):
                    if not triples:
                        continue
                    for (i, j, k) in triples:
                        # Random sign
                        sign_u = 1 if random.random() < 0.5 else -1
                        sign_v = 1 if random.random() < 0.5 else -1
                        sign_w = 1 if random.random() < 0.5 else -1
                        
                        U[i * m + j][col] += sign_u
                        V[j * p + k][col] += sign_v
                        W[k * n + i][col] += sign_w
                
                if verify(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break

    return best
