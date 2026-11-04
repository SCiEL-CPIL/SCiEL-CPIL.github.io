"""Construct a finite set U to maximize sumset/difference-set exponent lower bound."""
import time
import math

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        n = len(U)
        sum_set = set()
        diff_set = set()
        for i in range(n):
            for j in range(i, n):
                sum_set.add(U[i] + U[j])
            for j in range(n):
                diff_set.add(U[i] - U[j])
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Powers of 2
    for n in range(5, 35):
        U = [2**i for i in range(n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Geometric base 3
    for n in range(4, 25):
        U = [3**i for i in range(n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Mixed constructions
    for n2 in range(3, 12):
        for n3 in range(1, 10):
            U = [2**i for i in range(n2)]
            start = 2**(n2-1)
            for i in range(1, n3+1):
                U.append(start * 3**i)
            U = sorted(set(U))
            if 0 not in U:
                continue
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Local search
    if time.time() < deadline - 15:
        U = best_U[:]
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            current_score = compute_score(U)
            improved = False
            for i in range(1, len(U)):
                if time.time() > deadline - 10:
                    break
                orig = U[i]
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    new_val = orig + delta
                    if new_val < 0 or new_val == 0:
                        continue
                    if new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    s = compute_score(new_U)
                    if s > current_score + 1e-12:
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return sorted(U)
