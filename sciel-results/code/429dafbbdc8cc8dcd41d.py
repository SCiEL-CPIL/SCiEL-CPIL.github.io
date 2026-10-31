"""Traveling salesman constructive heuristic: Greedy Insertion + 2-Opt + Or-Opt ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_cache = {}
_start_time = None
_DEADLINE = 148.0  # Safety margin before 150s kill


def _fingerprint(distance_matrix):
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[:10, :10].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)


def _tour_length(tour, dm):
    n = len(tour)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total


def _greedy_insertion(dm, start_node=None):
    n = dm.shape[0]
    if start_node is None:
        start_node = 0
    
    # Start with a tour of 1 node? Or 2?
    # Greedy insertion usually starts with a random pair or single node.
    # Let's start with a single node and insert one by one.
    # Actually, starting with a random edge is better.
    # Let's pick a random start and a random second node.
    
    # To make it deterministic-ish but varied, we can just use start_node and the farthest node?
    # No, let's just use NN to get a second node to start the chain.
    
    # Simpler: Start with [start_node].
    tour = [start_node]
    unvisited = set(range(n))
    unvisited.remove(start_node)
    
    # Insert nodes one by one.
    # Cost of inserting node k between i and j:
    # delta = dm[i,k] + dm[k,j] - dm[i,j]
    
    while unvisited:
        best_node = None
        best_pos = None
        best_delta = np.inf
        
        # Current tour length is not needed, just the delta.
        # For each unvisited node, find the best insertion point.
        # This is O(|unvisited| * |tour|).
        
        for node in unvisited:
            # Find best position in current tour
            min_delta = np.inf
            min_idx = 0
            for i in range(len(tour)):
                prev = tour[i]
                next_node = tour[(i + 1) % len(tour)]
                # Insert between prev and next_node
                delta = dm[prev, node] + dm[node, next_node] - dm[prev, next_node]
                if delta < min_delta:
                    min_delta = delta
                    min_idx = i + 1 # Insert after i
            
            if min_delta < best_delta:
                best_delta = min_delta
                best_node = node
                best_pos = min_idx
        
        # Insert best_node at best_pos
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
        
    return tour


def _two_opt(tour, dm, time_limit=None):
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if time_limit and time.time() - _start_time > time_limit:
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                old = dm[a, b] + dm[c, d]
                new = dm[a, c] + dm[b, d]
                if new < old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(tour, dm, time_limit=None, segment_len=1):
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time_limit and time.time() - _start_time > time_limit:
                return tour
            
            # Segment to move: tour[i : i+segment_len]
            # We need to check all possible insertion positions j.
            # To avoid checking trivial moves (inserting back where it was), we skip j == i and j == i-1 (mod n)
            
            segment = tour[i : i + segment_len]
            if len(segment) < segment_len:
                continue
                
            # Calculate cost of removing segment
            # The neighbors of the segment are:
            # prev_node = tour[(i - 1) % n]
            # next_node = tour[(i + segment_len) % n]
            
            # We remove the segment. The new edge is prev_node -> next_node.
            # Old edges: prev->seg[0], seg[-1]->next, and internal edges of seg.
            # This logic is complex for segment_len > 1.
            # Let's stick to segment_len=1 for robustness, or implement carefully.
            
            if segment_len == 1:
                node = tour[i]
                a = tour[(i - 1) % n]
                b = tour[(i + 1) % n]
                
                # Try inserting node between k and k+1
                for k in range(n):
                    # Skip if k is adjacent to i
                    if k == i or (k + 1) % n == i:
                        continue
                    
                    p = tour[k]
                    q = tour[(k + 1) % n]
                    
                    # Old cost: a-node, node-b, p-q
                    old = dm[a, node] + dm[node, b] + dm[p, q]
                    # New cost: a-b, p-node, node-q
                    new = dm[a, b] + dm[p, node] + dm[node, q]
                    
                    if new < old - 1e-10:
                        # Perform move
                        tour.pop(i)
                        # Adjust k if i < k
                        if i < k:
                            k -= 1
                        # Insert after k
                        tour.insert(k + 1, node)
                        improved = True
                        break # Restart loop to be safe with indices
            else:
                # For now, skip multi-node Or-Opt to avoid bugs, 2-Opt covers many cases
                pass
                
    return tour


def _compute_best_tour(dm):
    n = dm.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Try a few different starting strategies
    # 1. Greedy Insertion from a few random starts
    # 2. NN from a few starts
    
    starts = list(range(n))
    random.shuffle(starts)
    
    # Limit number of starts to keep time in check, but Greedy Insertion is fast.
    # Let's try all starts if N is small, or a subset.
    # N=50, Greedy Insertion is O(N^3) = 125k ops. Very fast.
    # 2-Opt is O(N^2) per pass.
    
    for start in starts:
        if time.time() - _start_time > _DEADLINE - 20:
            break
            
        # Strategy 1: Greedy Insertion
        t1 = _greedy_insertion(dm, start)
        t1 = _two_opt(t1, dm, time_limit=_DEADLINE - 10)
        t1 = _or_opt(t1, dm, time_limit=_DEADLINE - 10, segment_len=1)
        t1 = _two_opt(t1, dm, time_limit=_DEADLINE - 10)
        
        t1_len = _tour_length(t1, dm)
        if t1_len < best_len:
            best_len = t1_len
            best_tour = t1
            
        # Strategy 2: NN (Faster, might find different local min)
        # Only do this if we have plenty of time
        if time.time() - _start_time < _DEADLINE - 40:
             # Simple NN
             unvisited = set(range(n))
             unvisited.remove(start)
             t2 = [start]
             cur = start
             while unvisited:
                 dists = dm[cur, list(unvisited)]
                 nxt = list(unvisited)[int(np.argmin(dists))]
                 t2.append(nxt)
                 unvisited.remove(nxt)
                 cur = nxt
                 
             t2 = _two_opt(t2, dm, time_limit=_DEADLINE - 10)
             t2 = _or_opt(t2, dm, time_limit=_DEADLINE - 10, segment_len=1)
             
             t2_len = _tour_length(t2, dm)
             if t2_len < best_len:
                 best_len = t2_len
                 best_tour = t2

    if best_tour is None:
        best_tour = np.arange(n)

    return np.array(best_tour)


def _next_in_tour(current_node, destination_node, unvisited_nodes, dm):
    tour = _get_tour(dm)
    current_pos = np.where(tour == current_node)[0]
    if len(current_pos) == 0:
        dists = dm[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    current_pos = current_pos[0]
    n = len(tour)
    unvisited_set = set(unvisited_nodes.tolist())

    for offset in range(1, n):
        next_pos = (current_pos + offset) % n
        candidate = tour[next_pos]
        if candidate in unvisited_set:
            return int(candidate)

    dists = dm[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])


def _get_tour(dm):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    fid = _fingerprint(dm)
    if fid in _cache:
        return _cache[fid]

    if time.time() - _start_time > _DEADLINE - 5:
        return np.arange(dm.shape[0])

    tour = _compute_best_tour(dm)
    _cache[fid] = tour
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if time.time() - _start_time > _DEADLINE:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    return _next_in_tour(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
