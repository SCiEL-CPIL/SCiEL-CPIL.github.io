import time
import itertools
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Fallback: LUW construction (Current program logic)
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Try to find a circulant graph C_n(S) with girth >= g
    # n must be even for bipartite (girth even)
    # S must contain 4 odd numbers for connected bipartite circulant
    
    def check_girth_circulant(n, S):
        # S is a set of 4 odd integers
        # BFS on (pos, last_step) to find shortest non-backtracking cycle
        # Start at 0, last_step=None
        # We want to return to 0 with a cycle length >= 3
        # Since graph is undirected and simple, no 2-cycles.
        # Shortest cycle is the girth.
        
        # visited[pos][last_step] = min_dist
        # last_step can be any s in S or None
        # To save memory, we can just store dist for each (pos, last_step)
        
        # Actually, we can just do a BFS from 0 in the "non-backtracking" graph.
        # Nodes are (v, u) where u is the neighbor we came from.
        # But for circulant, state is (pos, last_dir).
        
        # Let's use a dictionary for visited states to be safe with n up to 400.
        # State: (pos, last_s) where last_s is the step taken to get here.
        # Start: (0, None)
        
        from collections import deque
        q = deque()
        q.append((0, None, 0)) # pos, last_step, dist
        visited = {}
        visited[(0, None)] = 0
        
        while q:
            pos, last_s, dist = q.popleft()
            for s in S:
                if s == -last_s:
                    continue
                new_pos = (pos + s) % n
                new_dist = dist + 1
                if new_pos == 0 and last_s is not None:
                    # Found a cycle
                    # Ensure it's a simple cycle?
                    # Non-backtracking BFS guarantees the path is simple?
                    # Not necessarily, but the first time we hit 0 is the shortest non-backtracking walk.
                    # In a graph with girth G, the shortest non-backtracking closed walk is G.
                    return new_dist
                if (new_pos, s) not in visited:
                    visited[(new_pos, s)] = new_dist
                    q.append((new_pos, s, new_dist))
        
        return float('inf') # Should not happen for connected graph

    # Search for n
    # Moore bound for (4,10) is 242.
    # Try n from 242 to 400, step 2 (must be even)
    # Prioritize n=320
    
    # To speed up search, we can restrict S to small numbers first?
    # Or just iterate combinations.
    
    # We will try n=320 first.
    target_ns = [320]
    # Add other candidates if 320 fails
    for n in range(242, 402, 2):
        if n != 320:
            target_ns.append(n)
    # Sort by distance to 320? Or just try 320 first, then others.
    
    # We need to generate S.
    # Odds in 0..n-1
    # To reduce search, we can fix s1=1 (by scaling/rotation symmetry of circulant? No, S is a set, but graph is isomorphic if we multiply S by a unit in Z_n.
    # If gcd(a, n)=1, C_n(S) is isomorphic to C_n({a*s for s in S}).
    # So we can assume 1 in S?
    # Yes, if S is a valid connection set, and there exists a unit u such that 1 is in u*S, then we can transform.
    # Not all S can be transformed to contain 1.
    # But for small n, many can.
    # Let's just iterate all combinations for safety, but prune with 4-cycle check.
    
    def has_4_cycle(n, S):
        # S is a set of 4 elements
        # 4-cycle if a+b = c+d mod n
        # with {a,b} != {c,d} and a!=c, a!=d, b!=c, b!=d?
        # Actually, just check if any sum of 2 distinct elements equals another sum of 2 distinct elements.
        # Or a+a = b+c?
        # Let's just check all pairs.
        sums = {}
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i == j:
                    continue
                s = (a + b) % n
                if s in sums:
                    # Check if they are the same pair (reverse)
                    # sums[s] stores the pair (a,b)
                    # If (a,b) is not (b,a) of the previous one?
                    # Actually, if we find a collision, we need to ensure it's not a backtracking 4-cycle (which is just a 2-edge path back and forth, not a simple cycle).
                    # A 4-cycle v0-v1-v2-v3-v0 requires 4 distinct vertices.
                    # This implies s1, s2, s3, s4 such that s1+s2+s3+s4=0.
                    # s3 = -s2, s4 = -s1 is backtracking.
                    # So we need s1, s2, s3, s4 with no consecutive inverses.
                    # The sum check a+b=c+d is a necessary condition for a 4-cycle (v0=0, v1=a, v2=a+b, v3=a+b-c, v0=0 => c-d = -a-b? No).
                    # Let's just use the BFS girth check, it's fast enough for n=320.
                    pass
        return False

    # Given time constraints, we will just run the BFS girth check on candidates.
    # To limit candidates, we will only try S containing 1.
    # This is a heuristic.
    
    found_n = None
    found_S = None
    
    for n in target_ns:
        if time.time() > deadline:
            break
        
        odds = [i for i in range(1, n, 2)]
        # Heuristic: Try sets containing 1
        # Generate combinations of 3 other odds
        # This reduces search space by factor of 4 (approx)
        
        # To make it faster, we can limit the number of combinations checked.
        # But for n=320, binom(159, 3) ~ 6.7 million.
        # 6.7M * BFS(320) is too slow.
        
        # We need a better filter.
        # Check 4-cycles quickly.
        # A 4-cycle in C_n(S) exists if there are a,b,c,d in S such that a+b = c+d mod n.
        # We can precompute sums for a given S.
        
        count = 0
        for comb in itertools.combinations(odds, 4):
            count += 1
            if count > 50000: # Limit per n
                break
            
            S = set(comb)
            # Quick 4-cycle check
            has_4 = False
            sums = {}
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    s = (a + b) % n
                    if s in sums:
                        # Check if it's a valid 4-cycle
                        # The previous pair was (a', b')
                        # If (a,b) is not (b',a') and not (a',b'), it's a potential 4-cycle.
                        # Actually, if a+b = a'+b', and {a,b} != {a',b'}, we have a 4-cycle.
                        # Unless a=b' and b=a' (same pair).
                        prev_a, prev_b = sums[s]
                        if (a, b) != (prev_b, prev_a) and (a, b) != (prev_a, prev_b):
                            has_4 = True
                            break
                    else:
                        sums[s] = (a, b)
                if has_4:
                    break
            
            if has_4:
                continue
            
            # Check 6-cycles?
            # Just run BFS girth
            girth = check_girth_circulant(n, S)
            if girth >= g:
                found_n = n
                found_S = S
                break
        
        if found_n:
            break

    if found_n:
        n = found_n
        S = list(found_S)
        edges = []
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.append((i, j))
        return n, edges
    else:
        # Fallback to LUW
        return luw_construct(k, g)
