import time
import random
from collections import deque

N = 12
MAX_STATES = 1 << N

def calculate_threshold_bitmask(a_trans, b_trans):
    # Precompute transitions for all subsets
    # next_a[mask] = result of applying a to all states in mask
    next_a = [0] * MAX_STATES
    next_b = [0] * MAX_STATES
    
    for mask in range(MAX_STATES):
        na = 0
        nb = 0
        for i in range(N):
            if mask & (1 << i):
                na |= (1 << a_trans[i])
                nb |= (1 << b_trans[i])
        next_a[mask] = na
        next_b[mask] = nb
        
    start = (1 << N) - 1
    if start == 0: return 0 # Not possible for N>0
    
    # BFS
    visited = [-1] * MAX_STATES
    q = deque()
    visited[start] = 0
    q.append(start)
    
    while q:
        curr = q.popleft()
        dist = visited[curr]
        
        if curr == 1 or curr == 2 or (curr & (curr - 1)) == 0: # Single bit
             # Check if it's a single state
             # Actually, any mask with popcount 1 is a reset
             if curr & (curr - 1) == 0:
                 return dist

        na = next_a[curr]
        if visited[na] == -1:
            visited[na] = dist + 1
            q.append(na)
            
        nb = next_b[curr]
        if visited[nb] == -1:
            visited[nb] = dist + 1
            q.append(nb)
            
    return -1
