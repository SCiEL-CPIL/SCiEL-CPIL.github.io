import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 145.0  # Safety margin

    def check_deadline():
        return time.time() > deadline

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Get LUW result as fallback
    n_luw, e_luw = luw_family()
    best_n = n_luw
    best_e = e_luw

    # --- Search for Circulant Graphs ---
    # We only attempt this for the specific task parameters to avoid overhead
    if k == 5 and g == 9:
        # Moore bound for k=5, g=9 is 426
        # n must be even
        # We search up to n_luw to ensure we only return an improvement
        # If we don't find one, we return LUW
        
        # Fast girth check for circulant graphs using step sequences
        def circulant_girth_ok_fast(n, S, g):
            """
            Checks if circulant graph with step set S on Z_n has girth >= g.
            It enumerates all sequences of steps of length L < g.
            If any sequence forms a simple cycle (distinct prefix sums, final sum 0), return False.
            """
            # S is a list of steps
            # We check lengths 3 to g-1
            for L in range(3, g):
                # Use a set to track visited prefix sums
                # Start with sum 0 at depth 0
                # We want to see if we can reach 0 at depth L with all intermediate sums non-zero and distinct.
                
                # To be efficient, we can just check if there exists a sequence s1...sL
                # such that sum=0 and prefix sums are distinct.
                
                # For small L, we can just iterate.
                # But 5^L is large.
                # Optimization: 
                # If L=3: s1+s2+s3=0.
                # If L=4: s1+s2+s3+s4=0.
                
                # Let's implement a simple recursive check with pruning.
                # Actually, for L up to 8, 5^8 is 390k. 
                # If we do this for many candidates, it's slow.
                # We need to prune candidates BEFORE this check.
                
                # Pre-check for L=3:
                # Check if any s_i + s_j + s_k == 0
                S_set = set(S)
                # 3-sum check
                for s1 in S:
                    for s2 in S:
                        target = (-s1 - s2) % n
                        if target in S_set:
                            # Found a cycle of length 3 (or multiple of 3, but 3 is the smallest)
                            # Check if it's a simple cycle: s1, s1+s2, s1+s2+s3=0
                            # Prefix sums: s1, s1+s2.
                            # If s1 != 0, s1+s2 != 0, and s1 != s1+s2 (i.e. s2 != 0), it's simple.
                            # Since S contains no 0, s1, s2, s3 are non-zero.
                            # If s1+s2 == 0, then s3=0, which is not in S.
                            # So any 3-sum to 0 is a cycle of length 3.
                            return False
                
                # Pre-check for L=4:
                # Check if any s1+s2+s3+s4 == 0
                # We can do this by checking pairs (s1+s2) and (s3+s4)
                pair_sums = {}
                for s1 in S:
                    for s2 in S:
                        p = (s1 + s2) % n
                        if p not in pair_sums:
                            pair_sums[p] = []
                        pair_sums[p].append((s1, s2))
                
                for s3 in S:
                    for s4 in S:
                        target = (-s3 - s4) % n
                        if target in pair_sums:
                            for s1, s2 in pair_sums[target]:
                                # Sequence: s1, s2, s3, s4
                                # Check if simple cycle
                                # Prefixes: s1, s1+s2, s1+s2+s3
                                # Must be non-zero and distinct.
                                p1 = s1
                                p2 = (s1 + s2) % n
                                p3 = (s1 + s2 + s3) % n
                                # p4 is 0
                                # If p1, p2, p3 are distinct and non-zero, it's a simple cycle of length 4.
                                # Also need to ensure the path doesn't close earlier.
                                # p1 != 0 (s1 != 0)
                                # p2 != 0 (s1+s2 != 0 -> s2 != -s1. If s2==-s1, then s3+s4=0. 
                                #   If s3=-s4, then we have s1, -s1, s3, -s3. 
                                #   Prefixes: s1, 0... wait.
                                #   If s1+s2=0, then prefix sum at step 2 is 0. 
                                #   This means the walk returned to origin at step 2.
                                #   This is a cycle of length 2? No, s1, s2. 
                                #   In a simple graph, we don't have 2-cycles (no multi-edges).
                                #   But s1, -s1 is a back-and-forth. 
                                #   Is it a cycle? No, a cycle must be simple.
                                #   A walk s1, -s1 is not a simple cycle.
                                #   So if p2=0, it's not a simple cycle of length 4.
                                #   We require p1, p2, p3 != 0 and distinct.
                                if p1 != 0 and p2 != 0 and p3 != 0 and p1 != p2 and p1 != p3 and p2 != p3:
                                    return False
                
                # If we passed 3 and 4, we might need to check 5, 6, 7, 8.
                # For now, if we pass 3 and 4, we assume it's likely good or do a BFS.
                # Given the time constraint, let's do a BFS for the rest.
                
                # BFS for girth
                # This is O(n*k)
                depth = [-1] * n
                depth[0] = 0
                q = deque([0])
                max_depth = g - 2
                S_list = list(S)
                
                while q:
                    u = q.popleft()
                    if depth[u] >= max_depth:
                        continue
                    for s in S_list:
                        v = (u + s) % n
                        if depth[v] == -1:
                            depth[v] = depth[u] + 1
                            if depth[v] + 1 < g and v in S_set:
                                # Reaching a neighbor of 0 at depth d < g-1 creates cycle d+1 < g
                                # Wait, if v is in S, then edge (v,0) exists.
                                # Path 0->...->v (len d) + edge (v,0) = cycle len d+1.
                                return False
                            q.append(v)
                        else:
                            # If v is already visited, and depth[v] != depth[u] + 1
                            # Cycle length = depth[u] + depth[v] + 1
                            # But this logic is tricky in BFS.
                            # The "reaching S" check above is sufficient for cycles involving 0.
                            # For cycles not involving 0, BFS from 0 will find them as back edges?
                            # No, BFS from 0 finds shortest paths.
                            # If there is a cycle of length L < g, it must be found?
                            # Actually, the standard way is: 
                            # If we find an edge (u,v) such that depth[u] + depth[v] + 1 < g AND the edge is not a tree edge, 
                            # we have a cycle.
                            # But in a Cayley graph, it's easier to just check if the BFS tree has "conflicts".
                            
                            # Let's stick to the "reaching S" check. 
                            # Is it sufficient?
                            # If there is a cycle of length L < g, does it imply we reach a node in S at depth < g-1?
                            # Not necessarily. The cycle might not include 0.
                            # However, if the graph is connected, a cycle of length L exists.
                            # The BFS check `if depth[v] + 1 < g and v in S_set` only catches cycles passing through 0.
                            # We need a more robust check.
                            
                            # Let's refine the BFS check:
                            # When we visit u, for each neighbor v:
                            # If depth[v] == -1: set depth.
                            # Else: 
                            #   If depth[v] <= depth[u]:
                            #     Cycle length = depth[u] + depth[v] + 1
                            #     If this < g, return False.
                            #   This works for undirected graphs.
                            
                            if depth[v] <= depth[u]:
                                cycle_len = depth[u] + depth[v] + 1
                                if cycle_len < g:
                                    return False
                                    
                return True

        # Search loop
        # We'll try a range of n and random (a,b) pairs
        # To be efficient, we'll limit the number of (a,b) trials per n
        max_trials_per_n = 20
        # We'll search n from Moore bound to n_luw (exclusive)
        # Step 2 to keep n even
        for n in range(428, n_luw, 2):
            if check_deadline():
                break
            
            half_n = n // 2
            # We need S = {n/2, a, -a, b, -b}
            # a, b in 1..n/2-1, a != b, a != n/2, b != n/2
            # Also a != -a (n/2 is the only one)
            
            # Generate random candidates
            # We'll use a set to avoid duplicates
            tried = set()
            for _ in range(max_trials_per_n):
                if check_deadline():
                    break
                a = random.randint(1, half_n - 1)
                b = random.randint(1, half_n - 1)
                if a == b:
                    continue
                if (a, b) in tried:
                    continue
                tried.add((a, b))
                
                S = {half_n, a, n - a, b, n - b}
                if len(S) != 5:
                    continue
                
                if circulant_girth_ok_fast(n, S, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    best_n = n
                    best_e = edges
                    return best_n, best_e

    return best_n, best_e
