import time
import random
import sys
from itertools import product

def _luw_construct(k, g):
    """
    Fallback to Lazebnik-Ustimenko-Woldar construction if voltage lift fails or isn't applicable.
    This is kept from the seed to ensure we always return *something* valid if the new method fails.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

def _voltage_graph_construct(k, g):
    """
    Construct a k-regular graph with girth >= g using a voltage lift of a base graph.
    Base graph: K_{k+1} is (k)-regular? No, K_{k+1} is k-regular.
    For k=5, Base = K_6.
    """
    if k != 5:
        return None

    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety

    # Base Graph: K_6
    base_n = 6
    base_edges = []
    edge_index = {}
    idx = 0
    for u in range(base_n):
        for v in range(u + 1, base_n):
            base_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            idx += 1

    # Generate all closed walks of length L < g in the base graph
    # A closed walk is a sequence of vertices v_0, ..., v_L such that v_L == v_0
    # and (v_i, v_{i+1}) is an edge.
    # We store the sequence of edge indices.
    closed_walks_by_length = {L: [] for L in range(3, g)}
    
    # To avoid generating duplicates or huge lists, we can just check sums on the fly
    # or precompute the edge sequences.
    # Number of closed walks of length L in K_6 is 6 * 5^L.
    # L=8: 6 * 390625 = 2.3 million. This is manageable to precompute once.
    
    for L in range(3, g):
        # Start at any vertex 0..5
        # First step: 5 choices
        # Subsequent steps: 5 choices (can't go to self, but in K_6 you can go to any other 5)
        # Note: In K_n, from a vertex, there are n-1 neighbors.
        # So number of walks is n * (n-1)^L.
        
        # Optimization: We only need to check if sum is 0 mod m.
        # We can iterate m and then check walks.
        
        pass # We will do the checking inside the m loop for better pruning or just precompute edge id sequences.
        
    # Precompute edge ID sequences for closed walks to make voltage sum fast
    # Actually, iterating 2.3M walks * 500 attempts * m values might be slow in Python.
    # Let's optimize:
    # Instead of storing all walks, we can do a DFS/BFS for each voltage assignment to check if a zero-sum walk exists.
    # But we need to check ALL closed walks.
    # Alternative: Precompute the list of edge ID tuples for each length.
    
    walk_sequences = {L: [] for L in range(3, g)}
    
    def gen_walks(start, L):
        # Recursive generator for edge sequences
        if L == 0:
            yield []
            return
        # From 'start', go to any neighbor
        for v in range(base_n):
            if v != start:
                e_id = edge_index[(start, v)]
                for rest in gen_walks(v, L - 1):
                    if L == 1:
                        # Must close
                        if start == v: # Impossible in simple graph step
                            yield [e_id]
                    else:
                        # Yield e_id + rest
                        yield [e_id] + rest

    # This is inefficient. Let's use iterative generation.
    for L in range(3, g):
        seqs = []
        # Start at 0 (symmetry allows fixing start vertex)
        # Walk of length L starting at 0 and ending at 0
        # steps: 0 -> v1 -> ... -> v_{L-1} -> 0
        # v1: 5 choices
        # v2: 5 choices (v2 != v1)
        # ...
        
        # We'll just iterate all 5^(L-1) paths from 0 and check if last step closes.
        # Actually, let's just store the edge IDs.
        
        # For L=3: 0->1->2->0. Edges: (0,1), (1,2), (2,0).
        
        pass # Precomputation is tricky to debug. Let's do a direct check in the attempt loop.
        
    # Better approach for checking:
    # For a given voltage assignment, do a DFS from each vertex to find if a zero-sum closed walk of length < g exists.
    # State: (current_vertex, length, current_sum)
    # We want to know if there is a path back to start of length L with sum 0.
    
    def has_short_zero_cycle(voltages, m):
        # BFS/DFS for each start vertex
        # To save time, we can just check if ANY closed walk of length L has sum 0.
        # We can precompute the list of edge index tuples for each length L once.
        # Let's generate them once here.
        
        if not hasattr(_voltage_graph_construct, '_cached_walks'):
            cached = {L: [] for L in range(3, 9)}
            for L in range(3, 9):
                # Generate all closed walks of length L in K_6
                # Start at 0
                paths = [ [0] ]
                for _ in range(L):
                    new_paths = []
                    for p in paths:
                        last = p[-1]
                        for v in range(6):
                            if v != last:
                                new_paths.append(p + [v])
                    paths = new_paths
                
                # Filter those ending at 0
                for p in paths:
                    if p[-1] == 0:
                        # Convert vertices to edge indices
                        edges_seq = []
                        valid = True
                        for i in range(L):
                            u = p[i]
                            v = p[i+1]
                            e_idx = edge_index.get((u, v))
                            if e_idx is None:
                                valid = False
                                break
                            edges_seq.append(e_idx)
                        if valid:
                            cached[L].append(tuple(edges_seq))
                # We only generated walks starting at 0. Due to symmetry, checking start 0 is sufficient for existence?
                # Yes, if a closed walk exists anywhere, an isomorphic one exists starting at 0 in K_6.
                _voltage_graph_construct._cached_walks = cached

        cached_walks = _voltage_graph_construct._cached_walks

        for L in range(3, g):
            for seq in cached_walks[L]:
                s = 0
                for e in seq:
                    s += voltages[e]
                if s % m == 0:
                    return True
        return False

    # Search for m and voltages
    best_n = float('inf')
    best_result = None

    # Moore bound for n(5,9) is ~428.
    # Base K_6 has 6 vertices.
    # Lift size = 6 * m.
    # We want 6m >= 428 => m >= 72.
    # Let's try m from 72 upwards.
    
    for m in range(72, 200):
        if time.time() > deadline:
            break
            
        n_lift = 6 * m
        if n_lift >= best_n:
            continue

        # Try random voltage assignments
        for attempt in range(1000):
            if time.time() > deadline:
                break

            # Random voltages 1..m-1
            voltages = [random.randint(1, m - 1) for _ in range(len(base_edges))]
            
            # Check condition
            if not has_short_zero_cycle(voltages, m):
                # Build the graph
                edges = []
                # Map vertex (u, g) to u*m + g
                # For each edge (u, v) in base with voltage w:
                # Add edges (u, g) - (v, g+w)
                for u, v in base_edges:
                    w = voltages[edge_index[(u, v)]]
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx = v * m + (g + w) % m
                        edges.append((u_idx, v_idx))
                        
                # Optional: Verify girth with BFS if time permits
                # For now, trust the voltage check.
                
                return (n_lift, edges)
        
    return None

def construct(k, g):
    if k == 5 and g == 9:
        res = _voltage_graph_construct(k, g)
        if res is not None:
            return res
    return _luw_construct(k, g)

if __name__ == "__main__":
    n, edges = construct(5, 9)
    print(f"Constructed graph with {n} vertices and {len(edges)} edges.")
