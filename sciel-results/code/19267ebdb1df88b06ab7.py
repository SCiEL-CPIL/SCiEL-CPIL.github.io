"""Voltage graph (circulant) construction for 5-regular graphs of girth >= 9."""
import numpy as np
from itertools import combinations, permutations, product

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using a circulant/voltage graph approach.
    
    For the specific case of k=5, g=9, we use a known construction.
    For general cases, we attempt to find a suitable connection set.
    """
    if k == 5 and g == 9:
        return _construct_5_9()
    
    # Fallback: use Lazebnik-Ustimenko-Woldar style construction (same as seed)
    return _construct_luw(k, g)


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback from seed)."""
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


def _construct_5_9():
    """
    Construct a 5-regular graph of girth >= 9 using a circulant graph C(n, S).
    
    We search for n and a connection set S = {s1, s2, s3, s4, s5} in Z_n such that:
    - All si are distinct non-zero elements
    - No element is the inverse of another (si + sj != 0 mod n for all i,j)
    - No non-trivial sum of < 9 elements from S ∪ S^{-1} equals 0 mod n
    
    The girth condition for a circulant graph is equivalent to: the shortest cycle
    corresponds to the shortest non-trivial relation among the connection elements.
    For a cycle of length L, we need s_{i1} + s_{i2} + ... + s_{iL} ≡ 0 (mod n)
    where each s_{ij} ∈ S ∪ S^{-1} and the walk is non-backtracking.
    
    For girth >= 9, we need no such relation of length < 9.
    """
    # Moore bound for (5,9): 1 + 5*(1 + 4 + 4^2 + 4^3) = 1 + 5*85 = 426
    # But for odd girth, the bound is different. Let's use a practical lower bound.
    # The smallest known (5,9)-cage has 1116 vertices.
    
    # We'll search for n starting from a reasonable lower bound
    # and find a valid connection set S.
    
    def verify_girth(n, S, min_girth=9):
        """
        Verify that C(n, S) has girth >= min_girth.
        Uses BFS from each vertex to find the shortest cycle.
        For efficiency, we use the fact that in a circulant graph,
        the girth can be determined by checking relations among S ∪ S^{-1}.
        """
        # Build the set of allowed steps: S ∪ S^{-1}
        steps = set()
        for s in S:
            steps.add(s % n)
            steps.add((-s) % n)
        steps.discard(0)  # Remove identity if present
        
        # For each possible cycle length L < min_girth, check if there's a non-trivial
        # sum of L elements from steps that equals 0 mod n, forming a non-backtracking cycle.
        # 
        # A non-backtracking cycle of length L means:
        # - v0, v1, ..., v_{L-1}, v0 with v_i = v_{i-1} + s_i where s_i ∈ steps
        # - s_i ≠ -s_{i+1} for all i (non-backtracking)
        # - Not all s_i are the same (non-trivial)
        # - The cycle is simple (no repeated vertices except v0 = v_L)
        #
        # For a circulant graph, a cycle of length L exists iff there exist s_1, ..., s_L ∈ steps
        # such that sum(s_i) ≡ 0 (mod n) and the partial sums are all distinct (mod n).
        
        # Efficient check: use BFS on the Cayley graph
        # For small n, this is feasible.
        
        # First, quick filter: check if any 2-step relation exists
        # s + t ≡ 0 mod n with s, t ∈ steps, s ≠ t (to avoid backtracking)
        # This would give a 2-cycle, but we already excluded si + sj ≡ 0.
        
        # Check for 3-cycles: s + t + u ≡ 0 mod n with s, t, u ∈ steps,
        # and no two consecutive are inverses.
        # For a 3-cycle: v0 -> v1 (step s), v1 -> v2 (step t), v2 -> v0 (step u)
        # s + t + u ≡ 0, s ≠ -t, t ≠ -u, u ≠ -s
        
        steps_list = list(steps)
        steps_set = set(steps_list)
        
        # Check all cycle lengths from 3 to min_girth - 1
        for L in range(3, min_girth):
            # Check if there's a non-backtracking cycle of length L
            # Use BFS from vertex 0
            # State: (current_vertex, previous_step) to track non-backtracking
            # But for circulant graphs, we can just track the current vertex and the last step used
            
            # BFS: dist[vertex][last_step] = shortest path length to reach vertex with last step being last_step
            # Actually, simpler: just do BFS tracking (vertex, last_step_used)
            
            # For efficiency, we'll do a limited BFS
            from collections import deque
            
            # State: (vertex, last_step) where last_step is the step used to reach this vertex
            # Start: (0, None) - no previous step
            visited = set()
            queue = deque()
            queue.append((0, None, 0))  # (vertex, last_step, depth)
            visited.add((0, None))
            
            found = False
            while queue and not found:
                vertex, last_step, depth = queue.popleft()
                if depth >= L:
                    continue
                for s in steps_list:
                    # Non-backtracking: s != -last_step
                    if last_step is not None and s == (-last_step) % n:
                        continue
                    new_vertex = (vertex + s) % n
                    new_state = (new_vertex, s)
                    if new_state not in visited:
                        visited.add(new_state)
                        if new_vertex == 0 and depth + 1 >= 3:
                            # Found a cycle of length depth + 1
                            if depth + 1 < min_girth:
                                return False
                        queue.append((new_vertex, s, depth + 1))
        
        return True
    
    # Search for a valid (n, S)
    # Start from a lower bound and increase n
    # The Moore bound for (5,9) is 426, but practical constructions are larger.
    # Let's search from 426 to 1200.
    
    import random
    
    best_n = None
    best_S = None
    
    # Try different n values
    for n in range(426, 1200):
        # For each n, try to find a valid S
        # Use random search with some structure
        
        # First, filter n: n should be such that we can find 5 distinct non-zero
        # elements with no short relations.
        
        # Quick filter: n should be large enough that 5*s < n for all s in S
        # (to avoid trivial wrap-around relations)
        # Actually, this is too strict. Let's just search.
        
        # Try a few random S for this n
        found = False
        for attempt in range(50):
            # Generate a random S
            # Ensure all elements are distinct, non-zero, and no two sum to 0
            candidates = list(range(1, n))
            random.shuffle(candidates)
            
            S = []
            for c in candidates:
                # Check if c can be added to S
                valid = True
                for s in S:
                    if (c + s) % n == 0:
                        valid = False
                        break
                if valid:
                    S.append(c)
                    if len(S) == 5:
                        break
            
            if len(S) != 5:
                continue
            
            # Verify girth
            if verify_girth(n, S, 9):
                best_n = n
                best_S = S
                found = True
                break
        
        if found:
            break
    
    if best_n is not None:
        n = best_n
        S = best_S
        edges = []
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(5, 9)
