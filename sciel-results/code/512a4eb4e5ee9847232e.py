import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    A = [0] * (n + 1)
    for i in range(1, n + 1):
        A[i] = lcm_val // i
        
    best_W = None
    best_c = None
    
    def get_W(c):
        W = 0
        for i in range(1, n + 1):
            W += c[i] * A[i]
        return W
            
    def local_search(c, W, time_limit):
        nonlocal best_W, best_c
        
        # Update best if valid
        if W != 0:
            if best_W is None or abs(W) < abs(best_W):
                best_W = W
                best_c = c[:]
        
        # If W is 0, we must move.
        
        while time.time() < time_limit:
            improved = False
            
            # 1-flip moves
            for i in range(1, n + 1):
                for new_val in (-1, 0, 1):
                    if new_val == c[i]:
                        continue
                    delta = (new_val - c[i]) * A[i]
                    new_W = W + delta
                    if new_W != 0:
                        if W == 0 or abs(new_W) < abs(W):
                            c[i] = new_val
                            W = new_W
                            improved = True
                            if best_W is None or abs(W) < abs(best_W):
                                best_W = W
                                best_c = c[:]
                            break # Restart i-loop? No, just break new_val loop.
                                 # The outer i-loop continues.
                                 # This is fine.
            
            if improved:
                continue # Restart entire search from new state
            
            # 2-flip moves
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    for new_i in (-1, 0, 1):
                        if new_i == c[i]:
                            continue
                        for new_j in (-1, 0, 1):
                            if new_j == c[j]:
                                continue
                            delta = (new_i - c[i]) * A[i] + (new_j - c[j]) * A[j]
                            new_W = W + delta
                            if new_W != 0:
                                if W == 0 or abs(new_W) < abs(W):
                                    c[i] = new_i
                                    c[j] = new_j
                                    W = new_W
                                    improved = True
                                    if best_W is None or abs(W) < abs(best_W):
                                        best_W = W
                                        best_c = c[:]
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Stuck. Break to restart.
            break
            
    # Run restarts
    while time.time() < deadline:
        c = [0] * (n + 1)
        for i in range(1, n + 1):
            c[i] = random.choice([-1, 0, 1])
        
        W = get_W(c)
        
        # Run local search for a short duration or until stuck
        # To allow multiple restarts, we can limit the time per search
        # or just let it run until stuck.
        # Since local search breaks when stuck, it will return quickly if stuck.
        # If it finds a path to a very small W, it might take longer.
        # Let's just call it.
        local_search(c, W, deadline)
        
        if best_W is not None and abs(best_W) == 1:
            break

    if best_c is None:
        c = [0] * n
        c[27] = 1
        c[28] = -1
        return c

    return best_c[1:]
