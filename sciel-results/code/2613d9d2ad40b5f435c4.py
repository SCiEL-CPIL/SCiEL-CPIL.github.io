import time
import sys
import random
import itertools
import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)
    
    # For k=4, g=10: construct the 320-vertex cage via voltage graph lift of K_{4,4} over Z_40
    # K_{4,4} has vertices A = {a0..a3} and B = {b0..b3}, all edges a_i-b_j.
    # Lift over Z_40: 8*40 = 320 vertices.
    
    def _build_lift(voltages):
        """Build the voltage graph lift of K_{4,4} over Z_40."""
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k_val in range(40):
                    a_idx = i * 40 + k_val
                    b_idx = 160 + j * 40 + (k_val + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges
    
    def _check_4cycle_condition(voltages):
        """Check no 4-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for k in range(4):
                if i == k:
                    continue
                for j in range(4):
                    for l in range(4):
                        if j == l:
                            continue
                        s = (voltages[i][j] - voltages[k][j] + voltages[k][l] - voltages[i][l]) % 40
                        if s == 0:
                            return False
        return True
    
    def _check_6cycle_condition(voltages):
        """Check no 6-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if k == i:
                        continue
                    for l in range(4):
                        if l == j:
                            continue
                    for m in range(4):
                        if m == k or m == i:
                            continue
                        for n in range(4):
                            if n == l or n == j:
                                continue
                            s = (voltages[i][j] - voltages[k][j] + voltages[k][l] 
                                 - voltages[m][l] + voltages[m][n] - voltages[i][n]) % 40
                            if s == 0:
                                return False
        return True
    
    def _check_8cycle_condition(voltages):
        """Check no 8-closed-walk in base has voltage 0 mod 40."""
        # 8-cycles in K_{4,4} use all 4 A-vertices and all 4 B-vertices.
        # We need to check all cyclic permutations.
        # Fix a0=0 to break rotational symmetry.
        for perm_a_rest in itertools.permutations([1, 2, 3]):
            perm_a = [0] + list(perm_a_rest)
            for perm_b in itertools.permutations(range(4)):
                # Cycle: a0->b0->a1->b1->a2->b2->a3->b3->a0
                s = 0
                for idx in range(4):
                    a_cur = perm_a[idx]
                    b_cur = perm_b[idx]
                    a_next = perm_a[(idx + 1) % 4]
                    s += voltages[a_cur][b_cur]
                    s -= voltages[a_next][b_cur]
                if s % 40 == 0:
                    return False
        return True
    
    def _verify_girth_exact(n, edges, g, deadline):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_len = g - 1  # we check walks of length 1..g-1
        
        for start_v in range(n):
            if time.time() > deadline:
                return False
            
            # BFS over (vertex, prev) states
            states = set()
            for w in adj[start_v]:
                states.add((w, start_v))
            
            for d in range(2, max_len + 1):
                new_states = set()
                for (v, p) in states:
                    for w in adj[v]:
                        if w != p:
                            new_states.add((w, v))
                if any(v == start_v for (v, p) in new_states):
                    return False
                states = new_states
                if not states:
                    break
        
        return True

    # Try the specific candidate from the prompt first
    candidate_voltages = [[(i * j * 7 + i * 3 + j * 5) % 40 for j in range(4)] for i in range(4)]
    
    valid_voltages = None
    
    # Check the candidate
    if _check_4cycle_condition(candidate_voltages) and \
       _check_6cycle_condition(candidate_voltages) and \
       _check_8cycle_condition(candidate_voltages):
        valid_voltages = candidate_voltages
    
    if valid_voltages is None:
        # Optimized search using structured families
        # Family 1: volt[i][j] = (a*i + b*j + c*i*j) mod 40
        # We can vectorize the checks for speed
        
        # Pre-compute indices for 4-cycle checks
        # s = V[i][j] - V[k][j] + V[k][l] - V[i][l]
        # We want s != 0 mod 40
        
        # To speed up, we can just iterate and break early.
        # The previous loop was 16 * 40 * 40 = 25600 iterations.
        # Each iteration does ~144 + ~576 + ~144 checks. ~864 operations.
        # 25600 * 864 ~ 22 million operations. Python might take a few seconds.
        
        random.seed(42)
        
        # Try a few random matrices from the structured family
        for _ in range(1000):
            if time.time() > deadline - 30:
                break
            a = random.randint(0, 39)
            b = random.randint(0, 39)
            c = random.choice([1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39])
            voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                valid_voltages = voltages
                break
        
        if valid_voltages is None:
            # Full random search (low probability, but better than nothing)
            for attempt in range(50000):
                if time.time() > deadline - 30:
                    break
                voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
                if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                    valid_voltages = voltages
                    break

    if valid_voltages is not None:
        n, edges = _build_lift(valid_voltages)
        # Final verification with exact girth check
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
