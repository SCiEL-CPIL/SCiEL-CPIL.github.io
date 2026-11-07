"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def construct():
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists, deadline, rng):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains as bitmasks
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        assignment = [-1] * n
        sys.setrecursionlimit(10000)

        # AC-3 with trail-based undo for efficiency
        def arc_consistency(initial_queue=None, trail=None):
            """Apply AC-3. Returns False if any domain becomes empty, True on success, 'timeout' on timeout.
               If trail is provided, append (var, old_domain) tuples for undo."""
            if initial_queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((v, u))
            else:
                queue = deque(initial_queue)
            
            while queue:
                if time.time() > deadline:
                    return 'timeout'
                
                xi, xj = queue.popleft()
                
                if domains[xj] == 0:
                    return False

                # If Dom(xj) is a singleton {c}, then c cannot be in Dom(xi).
                if domains[xj] & (domains[xj] - 1) == 0:
                    c_bit = domains[xj]
                    if domains[xi] & c_bit:
                        old_dom = domains[xi]
                        new_dom = old_dom & ~c_bit
                        if new_dom == 0:
                            return False
                        if trail is not None:
                            trail.append((xi, old_dom))
                        domains[xi] = new_dom
                        for xk in adj[xi]:
                            if xk != xj:
                                queue.append((xk, xi))
            
            return True

        def undo_trail(trail):
            for var, old_dom in reversed(trail):
                domains[var] = old_dom

        # Helper to get count of bits
        def popcount(x):
            return bin(x).count('1')

        # Initial AC-3
        ac_result = arc_consistency()
        if ac_result is False:
            return None
        if ac_result == 'timeout':
            return None

        # Backtracking search with trail-based undo
        def backtrack(depth=0):
            if time.time() > deadline:
                return 'timeout'
            
            # Find MRV with randomized tie-breaking
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            candidates = []
            for v in range(n):
                if assignment[v] == -1:
                    count = popcount(domains[v])
                    if count == 0:
                        return False
                    if count < best_count:
                        best_count = count
                        candidates = [(v, degree[v])]
                        best_deg = degree[v]
                    elif count == best_count:
                        candidates.append((v, degree[v]))
            
            if best_v == -1 and not candidates:
                return True  # All assigned
            
            if candidates:
                # Pick from candidates with tie-breaking: prefer higher degree, but randomize among top
                candidates.sort(key=lambda x: -x[1])
                # Take up to 3 candidates and pick randomly for diversity
                top = candidates[:min(3, len(candidates))]
                best_v = top[rng.randrange(len(top))][0]
            
            dom = domains[best_v]
            
            # Extract colors and shuffle for randomized search order
            colors = []
            temp = dom
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                colors.append(c)
                temp ^= lsb
            rng.shuffle(colors)
            
            for c in colors:
                if time.time() > deadline:
                    return 'timeout'
                
                lsb = 1 << c
                
                # Save state
                old_dom_v = domains[best_v]
                old_assignment = assignment[best_v]
                
                # Assign
                assignment[best_v] = c
                domains[best_v] = lsb
                
                # Propagate: Remove c from neighbors' domains, record trail
                trail = []
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if domains[u] & lsb:
                            old_u = domains[u]
                            new_dom_u = old_u & ~lsb
                            if new_dom_u == 0:
                                valid = False
                                break
                            trail.append((u, old_u))
                            domains[u] = new_dom_u
                
                if valid:
                    # Run AC-3 on affected arcs with trail
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected, trail)
                    
                    if ac_res is True:
                        result = backtrack(depth + 1)
                        if result is True:
                            return True
                        if result == 'timeout':
                            undo_trail(trail)
                            domains[best_v] = old_dom_v
                            assignment[best_v] = old_assignment
                            return 'timeout'
                        # result is False, undo and try next color
                    elif ac_res is False:
                        pass
                    else:  # timeout
                        undo_trail(trail)
                        domains[best_v] = old_dom_v
                        assignment[best_v] = old_assignment
                        return 'timeout'
                
                # Undo
                undo_trail(trail)
                domains[best_v] = old_dom_v
                assignment[best_v] = old_assignment
            
            return False

        result = backtrack()
        if result is True:
            return assignment
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 145.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.time() > hard_deadline - 5.0:
            out.append([0] * n)
            continue
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.9
        
        colouring = None
        rng_solver = Random(0)
        
        # Try to solve with multiple randomized attempts
        for attempt in range(5):
            if time.time() > instance_deadline:
                break
            rng_attempt = Random(attempt * 12345 + seed)
            colouring = solve_instance(n, edges, lists, instance_deadline, rng_attempt)
            if colouring is not None:
                break
        
        if colouring is None:
            colouring = [0] * n
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
