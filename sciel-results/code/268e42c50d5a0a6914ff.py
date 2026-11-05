import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3 and g == 16:
        return _construct_petersen_lift(deadline, start_time)
    else:
        return _construct_luw(k, g)

def _construct_petersen_lift(deadline, start_time):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    adj = [[] for _ in range(10)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    edge_indices = {}
    for i, (u, v) in enumerate(edges):
        edge_indices[(u, v)] = i
        edge_indices[(v, u)] = i
        
    walks = []
    for start in range(10):
        def gen_walks(v, prev, length, path):
            if length == 0:
                return
            # We generate walks of length L
            # This recursive approach is inefficient for large L.
            # Let's use iterative BFS.
            pass
        # Use iterative generation
        for L in range(5, 16):
            # State: (v, prev, path)
            # We want to find paths of length L ending at start
            # Start with all neighbors of start
            # Actually, simpler:
            # Generate all non-backtracking walks of length L from all vertices
            # and filter those that end at start.
            
            # Initialize paths of length 1
            paths = []
            for v in range(10):
                for next_v in adj[v]:
                    e_idx = edge_indices[(v, next_v)]
                    paths.append((next_v, v, [e_idx]))
            
            for length in range(1, L):
                new_paths = []
                for v, prev, path in paths:
                    for next_v in adj[v]:
                        if next_v == prev:
                            continue
                        e_idx = edge_indices[(v, next_v)]
                        new_paths.append((next_v, v, path + [e_idx]))
                paths = new_paths
                
            for v, prev, path in paths:
                if v == start:
                    walks.append(path)
                    
    # Convert walks to numpy array
    # Pad with 0s to same length? No, lengths vary.
    # We can just check each walk individually but use vectorization for sum?
    # Or just loop. The number of walks is small enough.
    
    # Let's estimate number of walks.
    # Petersen graph has 10 vertices.
    # Branching factor 2 (non-backtracking).
    # 2^15 = 32768.
    # 10 starts * 32768 = 327,680 walks of length 15.
    # Total walks for L=5..15 is roughly 10 * (2^5 + ... + 2^15) ~ 10 * 65536 = 655,360.
    # This is a lot for Python loops.
    
    # Optimization:
    # We only need to check if sum % m == 0.
    # We can pre-compute the "signature" of each walk?
    # No, voltages change.
    
    # Let's use a smarter search.
    # Instead of random voltages, we can fix voltages for some edges and solve for others?
    # Or use the fact that we just need to avoid 0.
    
    # Let's try a different approach:
    # Start with m=100.
    # Random voltages.
    # Check validity.
    # If invalid, find a violating walk and change a voltage in it to break it.
    # This is a local search.
    
    # But first, let's just try random with a smaller number of attempts and see.
    # If 650k walks, and each check is ~15 ops, that's 10M ops per attempt.
    # 100 attempts = 1B ops. Too slow in Python.
    
    # We need to speed up the check.
    # Use numpy.
    # Create a matrix W of shape (NumWalks, MaxLen)
    # And a mask for valid entries.
    # Then sums = W @ voltages
    # But lengths vary.
    
    # Alternative:
    # Group walks by length.
    # For each length L, create a matrix.
    
    walks_by_len = {L: [] for L in range(5, 16)}
    for w in walks:
        walks_by_len[len(w)].append(w)
        
    # Convert to numpy arrays
    walks_np = {}
    for L in range(5, 16):
        if walks_by_len[L]:
            arr = np.array(walks_by_len[L])
            walks_np[L] = arr
            
    # Now check
    def check_voltages(voltages, m):
        v = np.array(voltages)
        for L in range(5, 16):
            if L in walks_np:
                sums = walks_np[L] @ v
                # We need sums % m != 0
                # Note: sums can be negative? No, voltages are 0..m-1.
                # But we are doing mod m.
                # Actually, the walk sum is sum of voltages.
                # We need (sum % m) != 0.
                # If sum is 0, m, 2m, etc, it's invalid.
                # Since voltages are positive, sum > 0.
                # So we just need sum % m != 0.
                # But wait, if we use integers, sum can be large.
                # (sums % m) == 0 is the condition.
                
                mods = sums % m
                if 0 in mods:
                    return False
        return True

    # This is still slow if done in Python loop for each attempt.
    # But `sums % m` is vectorized.
    # `0 in mods` is a Python loop over the array.
    # `np.any(mods == 0)` is faster.
    
    def check_voltages_fast(voltages, m):
        v = np.array(voltages)
        for L in range(5, 16):
            if L in walks_np:
                sums = walks_np[L] @ v
                mods = sums % m
                if np.any(mods == 0):
                    return False
        return True

    # Now search
    for m in range(50, 200):
        if time.time() > deadline:
            break
            
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            voltages = np.random.randint(0, m, size=len(edges))
            
            if check_voltages_fast(voltages, m):
                # Construct lift
                n = 10 * m
                lift_edges = []
                for u, v in edges:
                    val = voltages[edge_indices[(u, v)]]
                    for o in range(m):
                        u_lift = u * m + o
                        v_lift = v * m + (o + val) % m
                        lift_edges.append((u_lift, v_lift))
                return n, lift_edges

    return _construct_luw(3, 16)

def _construct_luw(k, g, d=None):
    """LUW construction as fallback."""
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
