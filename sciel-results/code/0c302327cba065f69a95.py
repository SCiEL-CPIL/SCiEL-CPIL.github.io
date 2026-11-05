# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def time_left():
        return deadline - time.time()

    N = 192
    ALL = list(range(N, 0, -1))
    
    def is_valid(S):
        """Check if set S is valid."""
        S = list(S)
        n = len(S)
        if n < 4:
            return True
        for i in range(n):
            a = S[i]
            for j in range(n):
                if i == j: continue
                for k in range(n):
                    if i == k or j == k: continue
                    for l in range(n):
                        if i == l or j == l or k == l: continue
                        if (S[j] + S[k] + S[l]) % a == 0:
                            return False
        return True

    def can_add(x, S):
        """Check if x can be added to S."""
        # Check if x divides any sum of 3 from S
        n = len(S)
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if (S[i] + S[j] + S[k]) % x == 0:
                            return False
        # Check if any a in S divides x + y + z
        for a in S:
            for i in range(n):
                for j in range(i+1, n):
                    if (x + S[i] + S[j]) % a == 0:
                        return False
        return True

    def greedy():
        S = []
        for x in ALL:
            if can_add(x, S):
                S.append(x)
        return S

    def local_search(S, max_iters=100):
        S = list(S)
        best_size = len(S)
        for _ in range(max_iters):
            if time_left() < 2.0:
                break
            improved = False
            # Try add
            for x in ALL:
                if x not in S and can_add(x, S):
                    S.append(x)
                    best_size = len(S)
                    improved = True
                    break
            if improved:
                continue
                
            # Try swap (remove 1, add 1) - might not increase size but helps escape
            # Try remove 1, add 2
            for i in range(len(S)):
                S_temp = S[:i] + S[i+1:]
                for x1 in ALL:
                    if x1 in S: continue
                    if can_add(x1, S_temp):
                        S_temp.append(x1)
                        for x2 in ALL:
                            if x2 in S_temp: continue
                            if can_add(x2, S_temp):
                                S_temp.append(x2)
                                if len(S_temp) > best_size:
                                    S = S_temp
                                    best_size = len(S)
                                    improved = True
                                    break
                        if improved: break
                if improved: break
            if not improved:
                break
        return S

    # Run multiple greedy starts
    best = []
    for _ in range(5):
        S = greedy()
        S = local_search(S)
        if len(S) > len(best):
            best = S
            
    return best

# EVOLVE-BLOCK-END
