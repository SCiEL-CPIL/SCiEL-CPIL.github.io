"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using random voltage lifts of K_{3,3} over Z_m.
    For k=3, the lift has n=6m vertices.
    """
    if k != 3:
        # Fallback for other k values
        return _fallback_construct(k, g)
    
    deadline = time.time() + 110  # Leave 10s buffer
    
    # Moore bound for 3-regular graph with even girth g:
    # n >= 3 * (2^(g/2 - 1) - 1)
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    def check_girth_fast(m, V, required_girth):
        """
        Fast girth check for K_{3,3} voltage lift over Z_m.
        V is a 3x3 list: V[i][j] = voltage on edge A_i -- B_j.
        Returns True if girth >= required_girth.
        
        Key insight: In the lifted graph, a cycle corresponds to a closed walk in the
        base graph K_{3,3}. Since K_{3,3} is bipartite, all cycles have even length.
        We can enumerate all closed walks of length up to required_girth-1 in the base
        graph and check if any lifts to a closed walk in the fiber.
        
        More efficiently: do BFS in the product space (base_vertex, shift) from each
        starting point, but we can be smarter.
        
        Actually, the most efficient approach for small m: build the full adjacency
        and do BFS from each vertex up to depth g/2.
        """
        n = 6 * m
        half = required_girth // 2
        
        # Build adjacency as list of lists
        # Vertex: (side, base, shift) -> index = side*3*m + base*m + shift
        adj = [[] for _ in range(n)]
        for i in range(3):
            for j in range(3):
                v = V[i][j]
                for s in range(m):
                    u = i * m + s  # side 0
                    w = 3 * m + j * m + (s + v) % m  # side 1
                    adj[u].append(w)
                    adj[w].append(u)
        
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= half:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        clen = du + dist[v] + 1
                        if clen < required_girth:
                            return False
        return True
    
    def check_girth_walks(m, V, required_girth):
        """
        Alternative: enumerate closed walks in base graph up to length required_girth-1.
        For each closed walk, compute the net voltage; if it's 0 mod m, we have a short cycle.
        This is much faster for small m.
        
        A closed walk in K_{3,3} of length 2L alternates between A and B vertices.
        It can be represented as a sequence of A-vertices a_0, a_1, ..., a_L and B-vertices b_0, ..., b_{L-1}
        where the walk goes a_0-b_0-a_1-b_1-...-a_{L-1}-b_{L-1}-a_L=a_0.
        The net voltage is sum of V[a_t][b_t] for t=0..L-1 (going A->B) plus
        the return voltages. Wait, let me think more carefully.
        
        Actually in the lift: edge from A_i at shift s goes to B_j at shift s+V[i][j].
        Edge from B_j at shift s goes to A_i at shift s-V[i][j] (reverse).
        
        So a walk A_{i0}(s0) -> B_{j0}(s0+V[i0][j0]) -> A_{i1}(s0+V[i0][j0]-V[i1][j0]) -> ...
        
        For a closed walk of length 2L: 
        s_final = s_initial + sum_{t=0}^{L-1} V[a_t][b_t] - sum_{t=0}^{L-1} V[a_{t+1}][b_t]
        where a_L = a_0.
        
        So net voltage = sum_{t=0}^{L-1} (V[a_t][b_t] - V[a_{t+1}][b_t])
        
        For this to be a cycle in the lift, net voltage must be 0 mod m.
        """
        Lmax = (required_girth - 1) // 2  # max half-length of closed walk to check
        
        # Enumerate all closed walks of length 2, 4, ..., 2*Lmax in K_{3,3}
        # A closed walk of length 2L: sequence (a0, b0, a1, b1, ..., a_{L-1}, b_{L-1}, a_L)
        # where a_L = a_0, and each (a_t, b_t) is an edge (always true in K_{3,3}).
        
        for L in range(1, Lmax + 1):
            # Enumerate all sequences of L pairs (a_t, b_t) with a_L = a_0
            # a_0 in {0,1,2}, b_t in {0,1,2}, a_t in {0,1,2} for t=1..L-1
            # Total: 3 * 3^L * 3^(L-1) = 3^(2L-1) walks
            # For L=7: 3^13 = 1.59M -- manageable
            for a0 in range(3):
                # Build all sequences
                def enumerate_walks(t, a_seq, b_seq):
                    if t == L:
                        # Check if a_L == a_0
                        if a_seq[-1] != a0:
                            return
                        # Compute net voltage
                        net = 0
                        for idx in range(L):
                            net += V[a_seq[idx]][b_seq[idx]] - V[a_seq[idx+1]][b_seq[idx]]
                        if net % m == 0:
                            # Found a short cycle
                            return False
                        return True
                    
                    for a_next in range(3):
                        for b_next in range(3):
                            a_seq.append(a_next)
                            b_seq.append(b_next)
                            if not enumerate_walks(t + 1, a_seq, b_seq):
                                a_seq.pop()
                                b_seq.pop()
                                return False
                            a_seq.pop()
                            b_seq.pop()
                    return True
                
                # Start: a_seq = [a0], then we need L more a's and L b's
                # Actually the walk is: a0, b0, a1, b1, ..., a_{L-1}, b_{L-1}, a_L=a0
                # So we choose b0, a1, b1, ..., a_{L-1}, b_{L-1}
                # That's L b's and (L-1) intermediate a's
                a_seq = [a0]
                b_seq = []
                
                def enum2(step, a_seq, b_seq):
                    # step: 0 means choose b0, then a1, b1, ..., a_{L-1}, b_{L-1}
                    # After choosing all, check a_L=a0 (already enforced by structure)
                    if step == 2 * L:
                        # We have a0, b0, a1, b1, ..., a_{L-1}, b_{L-1}
                        # a_L should be a0 -- but we haven't chosen it yet
                        # Actually the walk ends at a_L which must equal a0
                        # The last edge is a_{L-1} -> b_{L-1} -> a_L
                        # So we need to verify the walk closes: a_L = a0
                        # But in our enumeration, we just chose b_{L-1} as the last item
                        # The closing vertex a_L is implicit and must be a0
                        # Net voltage: sum over t=0..L-1 of (V[a_t][b_t] - V[a_{t+1}][b_t])
                        # where a_L = a0
                        net = 0
                        for idx in range(L):
                            a_t = a_seq[idx]
                            a_t1 = a_seq[idx+1] if idx+1 < len(a_seq) else a0
                            b_t = b_seq[idx]
                            net += V[a_t][b_t] - V[a_t1][b_t]
                        if net % m == 0:
                            return False
                        return True
                    
                    if step % 2 == 0:
                        # Choose b_{step//2}
                        for b in range(3):
                            b_seq.append(b)
                            if not enum2(step + 1, a_seq, b_seq):
                                b_seq.pop()
                                return False
                            b_seq.pop()
                    else:
                        # Choose a_{(step+1)//2}
                        for a in range(3):
                            a_seq.append(a)
                            if not enum2(step + 1, a_seq, b_seq):
                                a_seq.pop()
                                return False
                            a_seq.pop()
                    return True
                
                if not enum2(0, a_seq, b_seq):
                    return False
        
        return True
    
    # Try increasing m values
    random.seed(42)
    best_result = None
    
    for m in range(m_min, m_min + 150):
        if time.time() > deadline - 15:
            break
        if 6 * m > 2000:
            break
        
        # Try multiple random voltage assignments
        num_trials = 30 if m < 100 else 15
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Generate random voltages, ensuring some diversity
            V = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            # Quick sanity: check for 4-cycles first (fast)
            # A 4-cycle in K_{3,3} lift: a0-b0-a1-b1-a0
            # Net voltage: V[a0][b0] - V[a1][b0] + V[a1][b1] - V[a0][b1] = 0 mod m
            has_4cycle = False
            for a0 in range(3):
                for a1 in range(3):
                    if a0 == a1:
                        continue
                    for b0 in range(3):
                        for b1 in range(3):
                            if b0 == b1:
                                continue
                            net = V[a0][b0] - V[a1][b0] + V[a1][b1] - V[a0][b1]
                            if net % m == 0:
                                has_4cycle = True
                                break
                        if has_4cycle:
                            break
                    if has_4cycle:
                        break
                if has_4cycle:
                    break
            if has_4cycle:
                continue
            
            # Full girth check
            if g <= 10:
                ok = check_girth_walks(m, V, g)
            else:
                # For larger girth, use BFS-based check but only if m is small enough
                if m <= 80:
                    ok = check_girth_fast(m, V, g)
                else:
                    # Use walk enumeration if feasible
                    if g <= 14:
                        ok = check_girth_walks(m, V, g)
                    else:
                        # For g=16 and large m, use a hybrid: check walks up to length 8
                        # and rely on random properties
                        ok = check_girth_walks(m, V, g) if (g-1)//2 <= 7 else check_girth_fast(m, V, g)
            
            if ok:
                n = 6 * m
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = V[i][j]
                        for s in range(m):
                            u = i * m + s
                            w = 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges
    
    # Fallback
    return _fallback_construct(k, g)

def _fallback_construct(k, g):
    """Lazebnik-Ustimenko-Woldar style construction as fallback."""
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
# EVOLVE-BLOCK-END
