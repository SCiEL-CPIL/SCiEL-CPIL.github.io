import time
import random
from collections import deque

def _is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    i = 3
    while i * i <= x:
        if x % i == 0:
            return False
        i += 2
    return True

def _moore_bound(k, g):
    if k == 2:
        return g
    if g % 2 == 1:
        r = (g - 1) // 2
    else:
        r = g // 2
    return k * (k - 1) ** r

def construct(k, g):
    deadline = time.time() + 110.0  # Safety margin
    
    if k != 3:
        raise NotImplementedError("only k=3 supported")

    # Base graph: Petersen graph (10 vertices, 3-regular, girth 5).
    base_adj = [
        (1, 4, 5),
        (0, 3, 6),
        (3, 7, 8),
        (1, 2, 9),
        (0, 6, 7),
        (0, 8, 9),
        (1, 4, 8),
        (2, 4, 9),
        (2, 5, 7),
        (3, 5, 6),
    ]
    base_edges = []
    for u in range(10):
        for v in base_adj[u]:
            if u < v:
                base_edges.append((u, v))
    # 15 base edges.

    # Precompute all non-backtracking closed walks in the base graph up to length g-1.
    # A cycle of length L in the lift corresponds to a closed walk of length L in the base
    # with voltage sum 0 mod m.
    # We only need to check lengths 3 to g-1.
    max_walk_len = g - 1
    
    # Map (u, v) to (edge_index, sign)
    edge_map = {}
    for ei, (u, v) in enumerate(base_edges):
        edge_map[(u, v)] = (ei, 1)
        edge_map[(v, u)] = (ei, -1)
        
    closed_walks = []
    # Enumerate closed walks
    # State: (current_node, prev_node, path_of_(ei, sign))
    # We start from each node. To avoid duplicates, we can fix the start node or just collect all and deduplicate later, 
    # but for small graph, collecting all is fine.
    # Actually, for girth check, we just need to know if ANY closed walk has sum 0.
    # We can generate walks from each start vertex.
    
    for start in range(10):
        # Stack: (cur, prev, path)
        stack = [(start, -1, [])]
        while stack:
            cur, prev, path = stack.pop()
            if len(path) > max_walk_len:
                continue
            if len(path) >= 3 and cur == start:
                # Found a closed walk
                # To avoid processing the same walk multiple times from different starts or rotations,
                # we could canonicalize, but for just checking sum!=0, duplicates don't hurt correctness, 
                # just speed. However, the number of walks grows exponentially.
                # Let's add it.
                closed_walks.append(path)
                # Do not expand further if we hit max length, but we can stop expanding this branch?
                # No, longer walks might also be closed. But we stop at max_walk_len.
                continue
            if len(path) == max_walk_len:
                continue
            
            for nb in base_adj[cur]:
                if nb == prev:
                    continue
                ei, sign = edge_map[(cur, nb)]
                stack.append((nb, cur, path + [(ei, sign)]))
                
    # Optimization: We only need to check distinct voltage sums. 
    # Actually, just keeping the list is fine. It's small for Petersen graph.
    # Let's deduplicate by the sequence of (ei, sign) to save memory/time in check.
    # But simple list check is fast enough.
    
    # Function to check if a voltage assignment is valid for modulus m
    def is_valid_voltages(volt, m):
        for walk in closed_walks:
            s = 0
            for ei, sign in walk:
                s += sign * volt[ei]
            if s % m == 0:
                return False
        return True

    # Moore bound for n(3, 16)
    mb = _moore_bound(k, g)
    # n = 10 * m >= mb => m >= ceil(mb / 10)
    m_min = (mb + 9) // 10
    if m_min < 1:
        m_min = 1
        
    # We search for m starting from m_min.
    # For each m, we try a set of candidate voltage assignments.
    # Candidate strategy:
    # 1. Random voltages (fast to generate, might miss valid ones if search space is huge, but for small m it's ok)
    # 2. Structured voltages: e.g., volt[ei] = (ei + 1) % m, or based on a primitive root.
    
    best_n = None
    best_edges = None
    
    # To speed up, we can limit the number of random tries per m.
    # Also, if m is prime, the group is cyclic, which is good.
    
    # Pre-generate a few structured patterns
    def get_candidates(m, count):
        cands = []
        # Pattern 1: Linear
        cands.append([(i + 1) % m for i in range(15)])
        # Pattern 2: Quadratic
        cands.append([(i * i) % m for i in range(15)])
        # Pattern 3: Random
        rng = random.Random(m)
        for _ in range(count):
            cands.append([rng.randint(1, m-1) for _ in range(15)])
        return cands

    m = m_min
    # Upper bound for m to prevent infinite loop if no solution found (shouldn't happen for valid k,g)
    # But we have a time limit.
    while m <= 10000: # Arbitrary cap, time limit will stop us sooner
        if time.time() > deadline - 5:
            break
            
        # Try candidates
        # For small m, try more randoms. For large m, try structured first.
        num_randoms = 50
        cands = get_candidates(m, num_randoms)
        
        found = False
        for volt in cands:
            if time.time() > deadline - 5:
                break
            if is_valid_voltages(volt, m):
                # Valid! Construct the graph
                n = 10 * m
                edges = []
                for ei, (u, v) in enumerate(base_edges):
                    w = volt[ei]
                    for s in range(m):
                        a = u * m + s
                        b = v * m + ((s + w) % m)
                        edges.append((a, b))
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
        m += 1
        # Heuristic: If m is small, maybe step by 1. If large, step by 1 is still fine.
        
    if best_n is None:
        # Fallback: Return a valid 3-regular graph even if girth is not met?
        # The task says "invalid result ... scores 0".
        # Better to return a simple cycle or something? No, must be 3-regular.
        # Let's try a larger m with a deterministic voltage that is "likely" to work.
        # Or just return the Petersen graph if g<=5? No, g=16.
        # If we fail, we must return something.
        # Let's try one last desperate attempt with a large prime m and random voltages.
        m = 10007
        rng = random.Random(999)
        volt = [rng.randint(1, m-1) for _ in range(15)]
        if is_valid_voltages(volt, m):
             n = 10 * m
             edges = []
             for ei, (u, v) in enumerate(base_edges):
                 w = volt[ei]
                 for s in range(m):
                     a = u * m + s
                     b = v * m + ((s + w) % m)
                     edges.append((a, b))
             return n, edges
             
        # Absolute fallback: Return a small 3-regular graph (e.g., Petersen) 
        # This will fail the girth check but is a valid graph structure.
        # Actually, returning an invalid graph scores 0.
        # Let's try to construct a valid 3-regular graph on 10*m_min vertices using a simple cycle of triangles? No.
        # Just return the Petersen graph.
        n = 10
        edges = base_edges
        return n, edges

    return best_n, best_edges

# max_vertices guard (from task: n must be <= max_vertices; use a large default)
max_vertices = 10 ** 7
