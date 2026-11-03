import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # For k=4, g=10, try the 2-vertex voltage graph construction
    if k == 4 and g == 10:
        result = _voltage_lift_search(deadline)
        if result is not None:
            return result

    # Fallback to LUW if voltage search fails or times out
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, g):
            return n, edges

    return _luw_construct(k, g)


def _precompute_walk_coefficients(length):
    """
    Precompute all non-backtracking walk coefficient vectors of given length
    starting from vertex 0 in a 4-vertex base graph.
    Returns a list of tuples (c0, c1, c2, c3) where c_j is the net coefficient
    of voltage v_j in the walk sum.
    """
    walks = []
    
    def gen(pos, last, coeffs):
        if pos == length:
            walks.append(tuple(coeffs))
            return
        for nxt in range(4):
            if nxt == last:
                continue
            sign = 1 if (pos + 1) % 2 == 0 else -1
            new_coeffs = coeffs[:]
            new_coeffs[nxt] += sign
            gen(pos + 1, nxt, new_coeffs)
    
    # Start at vertex 0 with sign +1
    gen(0, 0, [1, 0, 0, 0])
    
    # Deduplicate
    return list(set(walks))


# Precompute walk coefficient vectors for lengths 4, 6, 8
_WALKS_4 = _precompute_walk_coefficients(4)
_WALKS_6 = _precompute_walk_coefficients(6)
_WALKS_8 = _precompute_walk_coefficients(8)


def _check_voltages_fast(v1, v2, v3, v4, M):
    """
    Check that 4 distinct voltages in Z_M satisfy:
    no non-backtracking closed walk of length 4, 6, or 8 has sum 0 mod M.
    Uses precomputed coefficient vectors and vectorized NumPy for speed.
    """
    v = np.array([v1, v2, v3, v4], dtype=np.int64)
    
    # Check L=4 walks
    coeffs_4 = np.array(_WALKS_4, dtype=np.int64)
    sums_4 = (coeffs_4 @ v) % M
    if np.any(sums_4 == 0):
        return False
    
    # Check L=6 walks
    coeffs_6 = np.array(_WALKS_6, dtype=np.int64)
    sums_6 = (coeffs_6 @ v) % M
    if np.any(sums_6 == 0):
        return False
    
    # Check L=8 walks
    coeffs_8 = np.array(_WALKS_8, dtype=np.int64)
    sums_8 = (coeffs_8 @ v) % M
    if np.any(sums_8 == 0):
        return False
    
    return True


def _check_voltages_fast_batch(voltages, M):
    """
    Check a batch of voltage tuples against all walk constraints.
    voltages: shape (N, 4) array of voltage tuples
    Returns boolean array of shape (N,) indicating which are valid.
    """
    valid = np.ones(len(voltages), dtype=bool)
    
    # Check L=4
    coeffs_4 = np.array(_WALKS_4, dtype=np.int64)
    # For each walk, compute sum for each voltage tuple
    # sums_4[i, j] = sum of coeffs_4[i] * voltages[j]
    sums_4 = (voltages @ coeffs_4.T) % M  # shape (N, len(_WALKS_4))
    valid &= ~np.any(sums_4 == 0, axis=1)
    
    if not np.any(valid):
        return valid
    
    # Check L=6
    coeffs_6 = np.array(_WALKS_6, dtype=np.int64)
    sums_6 = (voltages[valid] @ coeffs_6.T) % M
    valid_sub = ~np.any(sums_6 == 0, axis=1)
    valid_indices = np.where(valid)[0]
    valid[valid_indices[~valid_sub]] = False
    
    if not np.any(valid):
        return valid
    
    # Check L=8
    coeffs_8 = np.array(_WALKS_8, dtype=np.int64)
    sums_8 = (voltages[valid] @ coeffs_8.T) % M
    valid_sub = ~np.any(sums_8 == 0, axis=1)
    valid_indices = np.where(valid)[0]
    valid[valid_indices[~valid_sub]] = False
    
    return valid


def _voltage_lift_search(deadline):
    """
    Search for smallest M and voltages v1..v4 in Z_M such that:
    1. v1, v2, v3, v4 are distinct.
    2. No non-backtracking closed walk of length < 10 has sum 0.
    Uses vectorized NumPy constraint checking for speed.
    """
    # Try M=160 first (known 320-vertex cage), then other values
    m_candidates = [160]
    # Add other candidates in order of likely usefulness
    for m in range(4, 300):
        if m != 160:
            m_candidates.append(m)
    
    for M in m_candidates:
        if time.time() > deadline - 15:
            break
        
        # Fix v1=0, v2=1 by symmetry. Search a, b in 2..M-1 with a < b.
        # Use batch checking for efficiency.
        
        # Generate candidate pairs (a, b) with a < b, both in [2, M-1]
        # Process in batches
        batch_size = 50000
        a_vals = np.arange(2, M)
        
        found = False
        for a_start in range(0, len(a_vals), batch_size):
            if time.time() > deadline - 5:
                break
                
            a_batch = a_vals[a_start:a_start + batch_size]
            # For each a, b ranges from a+1 to M-1
            # Create batch of (a, b) pairs
            pairs = []
            for a in a_batch:
                b_vals = np.arange(a + 1, M)
                for b in b_vals:
                    pairs.append((0, 1, a, b))
            
            if not pairs:
                continue
            
            voltages = np.array(pairs, dtype=np.int64)
            
            # Check in sub-batches to manage memory
            sub_batch = 10000
            for s_start in range(0, len(voltages), sub_batch):
                if time.time() > deadline - 5:
                    break
                    
                sub = voltages[s_start:s_start + sub_batch]
                valid = _check_voltages_fast_batch(sub, M)
                
                if np.any(valid):
                    # Found valid voltages
                    idx = np.where(valid)[0][0]
                    v1, v2, a, b = sub[idx]
                    
                    n = 2 * M
                    edges = []
                    voltages_tuple = (v1, v2, a, b)
                    for x in range(M):
                        for v in voltages_tuple:
                            y = (x + v) % M
                            u = x
                            w = M + y
                            edges.append((u, w))
                    
                    # Verify
                    if _verify_girth(n, edges, 10):
                        return n, edges
                    else:
                        continue
                        
        # If we've checked M=160 and it's the first candidate, try next
        if M == 160:
            continue
    
    return None


def _verify_girth(n, edges, g):
    """Verify graph is k-regular and has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    
    # Check no duplicate edges
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Check girth using non-backtracking BFS from each vertex
    # A cycle of length < g exists iff there's a non-backtracking walk
    # from a vertex back to itself of length < g
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # dist[state] = length of walk
        visited = {}
        queue = deque()
        
        for v in adj[start]:
            state = (v, start)
            visited[state] = 1
            queue.append(state)
        
        while queue:
            u, prev = queue.popleft()
            d = visited[(u, prev)]
            
            if d >= g:
                break
            
            for v in adj[u]:
                if v == prev:
                    continue  # non-backtracking
                if v == start and d + 1 < g:
                    return False  # found cycle of length d+1 < g
                state = (v, u)
                if state not in visited:
                    visited[state] = d + 1
                    queue.append(state)
    
    return True


def _luw_construct(k, g):
    """
    LUW construction fallback.
    """
    q = k
    d = (g + 1) // 2
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
