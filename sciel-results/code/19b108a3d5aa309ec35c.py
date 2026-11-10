# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy algorithm to maximize the complement S.
    
    Strategy:
    We want to find the largest set S such that for every point x, there is a line L through x
    with L \cap S \subseteq {x} (if x in S) or L \cap S = empty (if x not in S).
    
    We use a greedy approach with a priority queue (most constrained first).
    1. Maintain valid_dirs[x] as a bitmask of directions d such that the line L_{x,d} 
       satisfies the condition for x given the current S.
    2. Initially S is empty, so all directions are valid for all points.
    3. Repeatedly pick a point p not in S with minimum number of valid directions.
    4. Add p to S and update valid_dirs for all points on lines through p.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical lines (v = (0, 1))
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
                
    # valid_dirs[x] is a bitmask of 50 bits
    valid_dirs = [(1 << 50) - 1] * n
    in_S = [False] * n
    in_heap = [True] * n
    
    # Helper to count bits
    def bit_count(x):
        return bin(x).count('1')
        
    # Heap of (popcount, point_idx)
    heap = []
    for i in range(n):
        heapq.heappush(heap, (50, i))
        
    while time.time() < deadline - 10:
        # Find a valid point
        p_idx = -1
        while heap:
            pc, idx = heap[0]
            # Check if stale
            if not in_S[idx] and in_heap[idx]:
                if valid_dirs[idx] == 0:
                    heapq.heappop(heap)
                    in_heap[idx] = False
                    continue
                # Check if popcount matches
                if bit_count(valid_dirs[idx]) != pc:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (bit_count(valid_dirs[idx]), idx))
                    continue
                # Valid candidate
                p_idx = idx
                heapq.heappop(heap)
                in_heap[idx] = False
                break
        
        if p_idx == -1:
            break
            
        # Add p_idx to S
        in_S[p_idx] = True
        
        # Update neighbors
        for d in range(50):
            li = point_line[p_idx][d]
            line = direction_lines[d][li]
            for q in line:
                if q == p_idx:
                    continue
                # Invalidate direction d for q
                if (valid_dirs[q] >> d) & 1:
                    valid_dirs[q] &= ~(1 << d)
                    # If q is in heap and not in S, its priority changed
                    if in_heap[q] and not in_S[q]:
                        if valid_dirs[q] == 0:
                            in_heap[q] = False
                        else:
                            # Push new entry
                            heapq.heappush(heap, (bit_count(valid_dirs[q]), q))
                            # Note: old entry is stale, will be handled by pop check
        
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not in_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
