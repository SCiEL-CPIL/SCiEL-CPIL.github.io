import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _csp_backtracking(k, g, max_n, deadline, seed=42):
    """
    Constraint satisfaction approach: build a k-regular graph with girth >= g
    using backtracking with dynamic girth checking.
    
    Strategy: Start with a tree of depth floor((g-1)/2) from a root, then close
    the graph by connecting leaves in a way that maintains girth and regularity.
    
    For 3-regular graphs, this is equivalent to finding a "cage" construction.
    We'll use a randomized approach: build a random 3-regular graph and check girth,
    but with smart pruning to avoid short cycles early.
    """
    if k != 3:
        return None
    
    random.seed(seed)
    
    # Moore bound for 3-regular, girth 16: 1 + 3 + 3*2 + 3*2^2 + ... + 3*2^7 = 1 + 3*(2^8 - 1) = 1 + 765 = 766
    # But the actual Moore bound formula is: n >= 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i for odd girth
    # For even girth g=16: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 * (2^8 - 1)/(2-1) = 2*255 = 510
    # Actually for even girth: n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2-1))
    # = 2 * (2^8 - 1)/(2-1) = 2 * 255 = 510
    
    # Best known is 936, lower bound is 512.
    # Let's try to find something between 512 and 936.
    
    # Approach: Random 3-regular graph generation with girth checking
    # Use configuration model + rewiring to increase girth
    
    def generate_random_3reg(n):
        """Generate a random 3-regular simple graph on n vertices."""
        if n % 2 != 0:
            return None
        if n < 4:
            return None
        
        # Configuration model: each vertex has 3 stubs
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        
        # Randomly pair up stubs
        edges = set()
        valid = True
        for attempt in range(100):
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (min(u, v), max(u, v)) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            if valid:
                break
        
        if not valid:
            return None
        return list(edges)
    
    def increase_girth(edges, n, target_girth, max_iters=5000):
        """Try to increase girth by rewiring edges."""
        adj = _build_adj(n, edges)
        edge_list = list(edges)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if _check_girth_fast(n, edge_list, target_girth):
                return edge_list
            
            # Find a short cycle and try to break it
            # Pick a random edge and try to swap
            idx1 = random.randint(0, len(edge_list) - 1)
            u1, v1 = edge_list[idx1]
            
            # Find another edge that can be swapped
            # We need to find u2, v2 such that swapping (u1,v1) and (u2,v2) 
            # with (u1,u2) and (v1,v2) maintains 3-regularity and simplicity
            attempts = 0
            while attempts < 20:
                idx2 = random.randint(0, len(edge_list) - 1)
                if idx2 == idx1:
                    attempts += 1
                    continue
                u2, v2 = edge_list[idx2]
                
                # Try swap: (u1,v1) -> (u1,u2), (u2,v2) -> (v1,v2)
                # Check simplicity
                if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
                    attempts += 1
                    continue
                
                new_edge1 = (min(u1, u2), max(u1, u2))
                new_edge2 = (min(v1, v2), max(v1, v2))
                
                if new_edge1 == new_edge2:
                    attempts += 1
                    continue
                if new_edge1 in edge_list or new_edge2 in edge_list:
                    attempts += 1
                    continue
                
                # Apply swap
                edge_list.remove((min(u1, v1), max(u1, v1)))
                edge_list.remove((min(u2, v2), max(u2, v2)))
                edge_list.append(new_edge1)
                edge_list.append(new_edge2)
                
                # Check if this improved girth (or at least didn't make it worse)
                # Quick check: just accept if still valid
                attempts += 1
        
        return edge_list
    
    # Try different sizes
    # Start from a reasonable size and try to find a valid graph
    # The best known is 936, so let's try sizes around there
    
    # For 3-regular, n must be even
    # Try sizes: 512, 514, 516, ..., up to 936
    
    # Actually, let's be smarter. The LUW construction gives 13122 for d=8.
    # That's way too big. Let's try to find something much smaller.
    
    # Random search approach:
    # 1. Generate random 3-regular graphs of various sizes
    # 2. Use local search (edge swaps) to increase girth
    # 3. Check if girth >= 16
    
    best_result = None
    best_n = float('inf')
    
    # Try sizes from 512 to 1000 in steps of 2
    # But this might be too slow. Let's be more strategic.
    
    # Actually, for a 120s budget, we can't do extensive random search.
    # Let's try a few specific sizes with multiple restarts.
    
    candidate_sizes = [512, 514, 516, 518, 520, 522, 524, 526, 528, 530,
                       532, 534, 536, 538, 540, 542, 544, 546, 548, 550,
                       552, 554, 556, 558, 560, 562, 564, 566, 568, 570,
                       572, 574, 576, 578, 580, 582, 584, 586, 588, 590,
                       592, 594, 596, 598, 600, 602, 604, 606, 608, 610,
                       612, 614, 616, 618, 620, 622, 624, 626, 628, 630,
                       632, 634, 636, 638, 640, 642, 644, 646, 648, 650,
                       652, 654, 656, 658, 660, 662, 664, 666, 668, 670,
                       672, 674, 676, 678, 680, 682, 684, 686, 688, 690,
                       692, 694, 696, 698, 700, 702, 704, 706, 708, 710,
                       712, 714, 716, 718, 720, 722, 724, 726, 728, 730,
                       732, 734, 736, 738, 740, 742, 744, 746, 748, 750,
                       752, 754, 756, 758, 760, 762, 764, 766, 768, 770,
                       772, 774, 776, 778, 780, 782, 784, 786, 788, 790,
                       792, 794, 796, 798, 800, 802, 804, 806, 808, 810,
                       812, 814, 816, 818, 820, 822, 824, 826, 828, 830,
                       832, 834, 836, 838, 840, 842, 844, 846, 848, 850,
                       852, 854, 856, 858, 860, 862, 864, 866, 868, 870,
                       872, 874, 876, 878, 880, 882, 884, 886, 888, 890,
                       892, 894, 896, 898, 900, 902, 904, 906, 908, 910,
                       912, 914, 916, 918, 920, 922, 924, 926, 928, 930,
                       932, 934, 936]
    
    # We can't try all of these. Let's try a subset.
    # Focus on sizes near the known optimum.
    # Try every 10th size, plus some near 936.
    
    # Actually, let's just try a few sizes with multiple restarts.
    # The key insight: random 3-regular graphs have girth ~ log(n), so for n~1000,
    # girth is around 7-8. We need girth 16, which is much higher.
    # This means random search is unlikely to work directly.
    
    # Better approach: Use the voltage graph / covering space idea more carefully.
    # Or use a different algebraic construction.
    
    # For now, let's try a limited random search with local improvement.
    # If it fails, fall back to LUW.
    
    # Try sizes: 512, 600, 700, 800, 900, 936
    test_sizes = [512, 520, 540, 560, 580, 600, 620, 640, 660, 680,
                  700, 720, 740, 760, 780, 800, 820, 840, 860, 880,
                  900, 920, 936]
    
    for n in test_sizes:
        if time.time() > deadline - 10:
            break
        
        # Try multiple restarts for each size
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            edges = generate_random_3reg(n)
            if edges is None:
                continue
            
            # Try to improve girth
            edges = increase_girth(edges, n, g, max_iters=1000)
            
            if _check_girth_fast(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                break  # Found a valid graph for this size, move to next size
    
    if best_result is not None:
        return best_result
    
    return None


def _heisenberg_cayley(k, g, deadline):
    """
    Construct a Cayley graph of a finite Heisenberg group over a field.
    For k=3, use F_p with 3 generators (and their inverses handled by symmetry).
    Actually, for 3-regular, we need 3 generators where each generator is its own inverse
    or we use a set of generators {a, b, c} where a^-1, b^-1, c^-1 are also in the set.
    Simplest: use an abelian group (Z_m)^d with generators that are involutions? No, Z_m doesn't have involutions for odd m.
    
    Let's use the Heisenberg group H_q over F_q for odd prime q.
    Elements: (x, y, z) with x, y, z in F_q.
    Multiplication: (x1, y1, z1) * (x2, y2, z2) = (x1+x2, y1+y2, z1+z2 + x1*y2)
    Inverse: (x, y, z)^-1 = (-x, -y, -z - x*y)
    
    We need 3 generators S = {g1, g2, g3} such that the Cayley graph is 3-regular.
    This requires that the generators are distinct from identity and we don't include inverses separately?
    No, a Cayley graph on a set S is k-regular if |S| = k.
    If the group is not abelian, the graph is undirected iff S = S^-1.
    So we need a symmetric generating set of size 3.
    This is impossible for a non-abelian group if the group has odd order?
    In H_q, the center has order q. If q is odd, the center elements have odd order, so no involutions in the center.
    However, we can pick generators g1, g2, g3 such that {g1, g2, g3} = {g1^-1, g2^-1, g3^-1}.
    This implies each generator is an involution.
    Does H_q have involutions?
    (x, y, z)^2 = (2x, 2y, 2z + xy).
    For this to be identity (0,0,0), we need 2x=0 => x=0 (since q odd), 2y=0 => y=0, 2z=0 => z=0.
    So H_q has no involutions for odd q.
    
    Therefore, we cannot form a 3-regular undirected Cayley graph on H_q with symmetric generators of size 3.
    
    Alternative: Use a bipartite double cover? Or use a different group?
    Or use a voltage graph approach which is more flexible.
    
    Let's stick to the voltage graph approach but optimize it.
    The Petersen lift was a good direction.
    Let's try lifting other base graphs.
    The dodecahedron graph is 3-regular with girth 5.
    Lifting the dodecahedron might give better girth/size ratio?
    
    Actually, the Ramanujan-Margulis construction via Cayley graphs of SL(2, F_q) is the gold standard.
    But implementing that is complex.
    
    Let's refine the Petersen lift.
    The issue with the previous attempt was likely finding voltages.
    Let's use a more systematic search for voltages or a different base graph.
    
    How about lifting the 3-prism (two triangles connected)? No, girth 3.
    How about lifting the cube (Q3)? Girth 4.
    
    Let's try to implement a simple random search for a 3-regular graph with high girth,
    but that is very hard to get girth 16.
    
    Let's go back to the voltage graph idea.
    Base graph: Petersen (girth 5).
    We need girth >= 16.
    This means we need to avoid cycles of length 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 in the lift.
    A cycle in the lift corresponds to a closed walk in the base graph.
    If the base walk has length L and voltage sum V, the lift has a cycle of length L if V=0 mod m.
    If V != 0, the lift has no cycle of length L, but might have a cycle of length k*L?
    Actually, the lift is a covering. A closed walk in the base of length L lifts to a cycle of length L in the fiber if and only if the voltage sum is 0 mod m.
    If the voltage sum is V != 0, the lift of that walk is a path from fiber 0 to fiber V. It is not a cycle.
    However, if we take the walk L/m times (if m divides L?), no.
    The cycle length in the lift is the smallest k*L such that k*V = 0 mod m.
    So if V != 0, the cycle length is m/gcd(V, m) * L? No.
    The lift of a base cycle of length L with voltage sum V is a cycle in the lift iff V = 0 mod m.
    If V != 0, the lift is not a cycle. It is a path.
    To get a cycle, we need to traverse the base cycle multiple times?
    If we traverse the base cycle k times, the total length is k*L and the total voltage is k*V.
    This is a cycle in the lift if k*V = 0 mod m.
    The smallest such k is m/gcd(V, m).
    So the cycle length in the lift is (m/gcd(V, m)) * L.
    
    So, for a base cycle of length L, the lift has a cycle of length (m/gcd(V, m)) * L.
    We need this to be >= g for all base cycles.
    If L >= g, we are fine (since m/gcd >= 1).
    If L < g, we need (m/gcd(V, m)) * L >= g.
    Since m/gcd(V, m) >= 1, if L >= g, ok.
    If L < g, we need m/gcd(V, m) >= ceil(g/L).
    This is a weaker condition than V != 0.
    V != 0 implies m/gcd(V, m) >= 2 (if m is prime, then m).
    If m is prime, V != 0 implies m/gcd = m.
    So cycle length is m*L.
    We need m*L >= g.
    Since L >= 5 (girth of Petersen), we need 5m >= g.
    For g=16, m >= 16/5 = 3.2 => m >= 4.
    So any prime m >= 5 should work if we ensure V != 0 for all base cycles of length < g.
    
    Wait, are there base cycles of length < 5? No, girth is 5.
    So we only need to check cycles of length 5, 6, ..., g-1.
    For the Petersen graph, the cycles are of length 5, 6, 8, 9, 10.
    We need to assign voltages to the 15 edges such that for every cycle C of length L < g, the sum of voltages on C is not 0 mod m.
    
    This is a constraint satisfaction problem.
    Let's try m = 5, 7, 11, 13, 17...
    For m=5, we need to avoid sum=0 mod 5 for cycles of length 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.
    But Petersen has no cycles of length 7 or 11, 13, 15.
    It has cycles of length 5, 6, 8, 9, 10.
    So we need to avoid sum=0 mod 5 for cycles of length 5, 6, 8, 9, 10.
    
    Let's write a function to find such voltages using backtracking or random search.
    """
    if k != 3:
        return None

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    # Find all cycles in Petersen up to length g-1
    cycles = []
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length >= g:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    # We need to find voltages for m such that sum(v[C]) != 0 mod m for all C in cycles.
    # Try small primes m.
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        # Random search for voltages
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                return n, edges
    return None


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin

    if k != 3:
        return _luw_construct(k, g)

    # First try the CSP backtracking approach (new structural change)
    result = _csp_backtracking(k, g, 1000, deadline - 20, seed=42)
    if result is not None:
        n, edges = result
        # Verify
        if _check_girth_fast(n, edges, g):
            return n, edges
        else:
            # Should not happen
            pass

    # Try Heisenberg/Cayley approach (actually voltage lift of Petersen)
    result = _heisenberg_cayley(k, g, deadline - 10)
    if result is not None:
        n, edges = result
        # Verify
        if _check_girth_fast(n, edges, g):
            return n, edges
        else:
            # Should not happen
            pass

    # Fallback to LUW
    return _luw_construct(k, g)
