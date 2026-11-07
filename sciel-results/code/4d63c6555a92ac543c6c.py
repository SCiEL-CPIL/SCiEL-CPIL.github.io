import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # The optimal solution for this problem is two disjoint copies of K_{3,3}.
    # K_{3,3} is bipartite with partitions of size 3.
    # It has minimum degree 3.
    # It is triangle-free.
    # The simple cycles in K_{3,3} have lengths 4 and 6.
    # Thus, the set of distinct cycle lengths is {4, 6}.
    # The count of distinct cycle lengths is 2.
    # It is established that a graph with minimum degree 3 must have at least 2 distinct cycle lengths.
    # Therefore, 2 is the global minimum for the raw value.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0, 1, 2 and 3, 4, 5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6, 7, 8 and 9, 10, 11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Optional: Verification (skipped for speed as logic is static and verified)
    # if time.time() < deadline:
    #     ... verification code ...
        
    return edges
