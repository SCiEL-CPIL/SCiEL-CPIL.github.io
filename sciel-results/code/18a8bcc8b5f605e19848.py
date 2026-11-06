import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    """
    # Precompute bit masks for single states
    trans_a_bits = [1 << x for x in trans_a]
    trans_b_bits = [1 << x for x in trans_b]
    
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # Precompute transitions for all masks
    for mask in range(1 << N):
        ma = 0
        mb = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            ma |= trans_a_bits[i]
            mb |= trans_b_bits[i]
            m ^= lsb
        next_a[mask] = ma
        next_b[mask] = mb

    # BFS from FULL_MASK
    dist = [-1] * (1 << N)
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    
    while queue:
        cur = queue.popleft()
        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0 # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Classic Černý construction: a = cycle, b[i] = i-1 (with b[0]=0)
    # This is known to give reset length ~N*(N-1)/2 for N states
    # For N=12, the theoretical max is N*(N-1) = 132, but Černý gives N*(N-1)/2 = 66
    
    # Let's try multiple structured constructions and then do local search
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_config(a, b):
        nonlocal best_score, best_a, best_b
        score = compute_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
        return score
    
    # Construction 1: Classic Černý
    # a = cycle: a[i] = (i+1) % N
    # b[i] = i-1 for i>=1, b[0] = 0
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0] + [i - 1 for i in range(1, N)]
    try_config(a1, b1)
    
    # Construction 2: a = cycle, b[i] = (i-1) % N (same as above for i>=1)
    # Already tried
    
    # Construction 3: a = cycle, b maps all to a small set
    # b[i] = i % 2
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [i % 2 for i in range(N)]
    try_config(a3, b3)
    
    # Construction 4: a = cycle, b[i] = max(0, i-2)
    a4 = [(i + 1) % N for i in range(N)]
    b4 = [max(0, i-2) for i in range(N)]
    try_config(a4, b4)
    
    # Construction 5: a = cycle, b[i] = (i + k) % N for various k
    for k in range(N):
        a5 = [(i + 1) % N for i in range(N)]
        b5 = [(i + k) % N for i in range(N)]
        try_config(a5, b5)
    
    # Construction 6: a = reverse cycle, b = shift
    a6 = [(i - 1) % N for i in range(N)]
    for k in range(N):
        b6 = [(i + k) % N for i in range(N)]
        try_config(a6, b6)
    
    # Construction 7: a = cycle, b = "fold" pattern
    # b[0]=0, b[1]=0, b[2]=1, b[3]=1, b[4]=2, b[5]=2, ...
    a7 = [(i + 1) % N for i in range(N)]
    b7 = [i // 2 for i in range(N)]
    try_config(a7, b7)
    
    # Construction 8: a = cycle, b[i] = (N-1-i) (reverse)
    a8 = [(i + 1) % N for i in range(N)]
    b8 = [N - 1 - i for i in range(N)]
    try_config(a8, b8)
    
    # Now do local search from the best configuration found
    if best_a is None:
        return [a1, b1]
    
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score
    
    # Local search: try moving individual b destinations
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        improved = False
        
        # Try all single changes to b
        for i in range(N):
            old_val = cur_b[i]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                cur_b[i] = new_val
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                    improved = True
                    break
                else:
                    cur_b[i] = old_val
            
            if time.time() > deadline:
                break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Try a random perturbation to escape local optimum
                for _ in range(3):
                    i = random.randint(0, N-1)
                    cur_b[i] = random.randint(0, N-1)
                cur_score = compute_threshold(cur_a, cur_b)
                no_improve_count = 0
                if cur_score <= best_score:
                    # Revert to best
                    cur_a = best_a[:]
                    cur_b = best_b[:]
                    cur_score = best_score
                    break
        
        if time.time() > deadline:
            break
    
    # Also try local search on a
    if time.time() < deadline - 5:
        no_improve_a = 0
        max_no_improve_a = 50
        
        while time.time() < deadline - 5 and no_improve_a < max_no_improve_a:
            improved_a = False
            
            for i in range(N):
                old_val = cur_a[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    cur_a[i] = new_val
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score > cur_score:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]
                        improved_a = True
                        break
                    else:
                        cur_a[i] = old_val
                
                if time.time() > deadline - 5:
                    break
            
            if improved_a:
                no_improve_a = 0
            else:
                no_improve_a += 1
        
        # If a search didn't help, revert to best
        if cur_score <= best_score:
            cur_a = best_a[:]
            cur_b = best_b[:]
            cur_score = best_score
    
    # Ensure we return a synchronizing automaton
    if best_score == 0:
        return [a1, b1]
        
    return [best_a, best_b]
