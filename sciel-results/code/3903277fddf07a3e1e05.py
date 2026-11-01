import time
import sys
import random
import numpy as np
from collections import deque
from itertools import product

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Map edges to indices for voltage assignment
    edge_index = {}
    for idx, (u, v) in enumerate(petersen_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    # Precompute all closed walks of length < g in the Petersen graph
    # We represent a walk as a list of edge indices
    # We only need walks that start and end at the same vertex
    # To reduce symmetry, we can fix the starting vertex to 0 and the first edge to a canonical one?
    # No, voltages are on edges, so we need all distinct edge sequences.
    # However, we can generate walks and check their voltage sum constraints.
    
    # Let's generate all closed walks of length L for L in 5..15
    # This might be large, but Petersen is small.
    # Number of walks of length L in 3-regular graph is approx 3 * 2^(L-1).
    # For L=15, 3 * 2^14 = 49152. Total for 5..15 is ~ 100k walks.
    # We can store them as tuples of edge indices.
    
    closed_walks = []
    for L in range(5, g):
        # Generate walks of length L starting from each vertex
        for start in range(10):
            # DFS to find closed walks
            def dfs_walk(current, length, path_edges):
                if length == L:
                    if current == start:
                        closed_walks.append(tuple(path_edges))
                    return
                for (u, v) in petersen_edges:
                    if u == current:
                        next_v = v
                    elif v == current:
                        next_v = u
                    else:
                        continue
                    # Avoid immediate backtracking? No, backtracking creates cycles of length 2, but we are looking for length >= 5.
                    # However, backtracking in the walk generation increases the count.
                    # We can prune if the last edge was the reverse of the current one? 
                    # Actually, for voltage sums, backtracking adds v_e + v_e = 2v_e.
                    # Let's just generate all walks.
                    path_edges.append(edge_index[(current, next_v)])
                    dfs_walk(next_v, length + 1, path_edges)
                    path_edges.pop()
            
            dfs_walk(start, 0, [])
            
    # Remove duplicates
    closed_walks = list(set(closed_walks))
    
    # We will search for m and voltages
    # n = 10 * m
    # Moore bound for k=3, g=16 is 512. So m >= 52.
    
    # Try m values
    # We use a backtracking solver for the voltages
    # Variables: v_0 ... v_14
    # Constraints: For each walk W, sum(v_e for e in W) != 0 mod m
    
    # To speed up, we can precompute the coefficient vector for each walk
    # coeff[walk_idx] = [c_0, ..., c_14] where c_e is count of edge e in walk
    walk_coeffs = []
    for w in closed_walks:
        coeffs = [0] * 15
        for e in w:
            coeffs[e] += 1
        walk_coeffs.append(coeffs)

    # Function to check if a partial assignment violates any constraint
    # We will use a simple backtracking with forward checking
    
    def find_voltages(m):
        # Use randomized backtracking
        # Order variables by degree (frequency in constraints)
        # All edges appear in many walks. Let's just use index order.
        
        # To make it faster, we can use a SAT-like approach or just try random assignments with pruning.
        # Given the small number of variables (15), we can try a few random permutations and backtracking.
        
        # Heuristic: Try to assign values such that sums are non-zero.
        
        # We will use a recursive search.
        # State: current assignment list, index of next variable
        # Pruning: For each constraint, if the sum of assigned parts + min/max possible of unassigned parts 
        # cannot be 0, it's safe. But "not 0" is easier: if sum is 0 and all unassigned are 0, it's bad.
        # Actually, we just need to ensure that for each walk, the total sum is not 0.
        
        # Let's try a simple randomized backtracking.
        
        def backtrack(idx, assignment):
            if idx == 15:
                # Check all constraints
                for coeffs in walk_coeffs:
                    s = 0
                    for i in range(15):
                        s += coeffs[i] * assignment[i]
                    if s % m == 0:
                        return None
                return assignment[:]
            
            # Try values for variable idx
            # Randomize order of values
            values = list(range(m))
            random.shuffle(values)
            
            for val in values:
                assignment[idx] = val
                # Forward checking: Check if any constraint is now impossible to satisfy?
                # This is hard for "not 0". 
                # Instead, we can check if any constraint is already 0 and the remaining variables are forced to be 0? No.
                # Let's just do a quick check: if there's a walk that only uses assigned variables and sum is 0, fail.
                valid = True
                for coeffs in walk_coeffs:
                    s = 0
                    unassigned_weight = 0
                    for i in range(15):
                        if i < idx:
                            s += coeffs[i] * assignment[i]
                        else:
                            unassigned_weight += coeffs[i]
                    if unassigned_weight == 0:
                        if s % m == 0:
                            valid = False
                            break
                if not valid:
                    continue
                    
                res = backtrack(idx + 1, assignment)
                if res is not None:
                    return res
            return None
        
        # Try a few times with different random seeds/shuffles
        for _ in range(20):
            random.seed()
            res = backtrack(0, [0] * 15)
            if res is not None:
                return res
        return None

    # Search for m
    # Start from Moore bound m=52
    for m in range(52, 200):
        if time.time() > deadline:
            break
            
        voltages = find_voltages(m)
        if voltages is not None:
            # Build the graph
            n = 10 * m
            edges = []
            for (u, v), shift in zip(petersen_edges, voltages):
                for i in range(m):
                    u_id = u * m + i
                    v_id = v * m + (i + shift) % m
                    edges.append((u_id, v_id))
            
            # Verify girth
            if _verify_girth_fast(n, edges, g):
                return n, edges

    # Fallback: Random search with larger m
    if time.time() < deadline:
        for m in range(200, 1000):
            if time.time() > deadline:
                break
            for _ in range(100):
                if time.time() > deadline:
                    break
                shifts_rand = [random.randint(0, m-1) for _ in range(15)]
                n, edges = _build_lift(petersen_edges, m, shifts_rand)
                if _verify_girth_fast(n, edges, g):
                    return n, edges

    # Final fallback
    m = 300
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(petersen_edges, m, shifts_rand)
    return n, edges


def _build_lift(base_edges, m, shifts):
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    seen_mark = np.zeros(n, dtype=np.int32)
    mark = 0
    half_g = g // 2

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

# EVOLVE-BLOCK-END
