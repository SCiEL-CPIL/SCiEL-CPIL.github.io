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
        
        # Degree for variable ordering
        degree = [len(adj[i]) for i in range(n)]

        # Use sets for domains for efficient operations
        domains = [set(lst) for lst in lists]
        assignment = [-1] * n

        sys.setrecursionlimit(10000)

        # AC-3: Arc consistency
        def arc_consistency(initial_queue=None):
            """Apply AC-3. Returns False if any domain becomes empty."""
            if initial_queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((v, u))
            else:
                queue = deque(initial_queue)
            
            while queue:
                (xi, xj) = queue.popleft()
                if time.time() > deadline:
                    return 'timeout'
                
                revised = False
                for c in list(domains[xi]):
                    if len(domains[xj]) == 0:
                        domains[xi].discard(c)
                        revised = True
                    elif len(domains[xj]) == 1 and list(domains[xj])[0] == c:
                        domains[xi].discard(c)
                        revised = True
                
                if revised:
                    if len(domains[xi]) == 0:
                        return False
                    for xk in adj[xi]:
                        if xk != xj:
                            queue.append((xk, xi))
            
            return True

        # Try AC-3 first
        ac_result = arc_consistency()
        if ac_result is False:
            return None
        if ac_result == 'timeout':
            return None

        # Fast greedy pre-pass: assign vertices with domain size 1, then try MRV greedily
        # This can solve many instances without backtracking
        assigned_count = 0
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if assignment[v] == -1 and len(domains[v]) == 1:
                    c = next(iter(domains[v]))
                    assignment[v] = c
                    assigned_count += 1
                    changed = True
                    for u in adj[v]:
                        if assignment[u] == -1:
                            domains[u].discard(c)
                            if len(domains[u]) == 0:
                                return None
        
        # Greedy MRV pass for remaining vertices
        while True:
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = len(domains[v])
                    if count == 0:
                        return None
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return assignment  # All assigned
            
            # Try each color in domain, pick first that works with forward checking
            found = False
            for c in list(domains[best_v]):
                # Check if c is compatible with all assigned neighbors
                ok = True
                for u in adj[best_v]:
                    if assignment[u] != -1 and assignment[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Tentatively assign
                assignment[best_v] = c
                domains[best_v] = {c}
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if c in domains[u]:
                            domains[u].discard(c)
                            if len(domains[u]) == 0:
                                valid = False
                                break
                
                if valid:
                    found = True
                    break
                else:
                    # Undo
                    assignment[best_v] = -1
                    domains[best_v].add(c)
                    for u in adj[best_v]:
                        if assignment[u] == -1 and c not in domains[u]:
                            # Can't easily undo, so just continue
                            pass
            
            if not found:
                # Greedy failed, need backtracking
                # Restore: we can't easily undo, so break and use backtracking
                # Reset domains for unassigned vertices
                for v in range(n):
                    if assignment[v] == -1:
                        domains[v] = set(lists[v])
                break
        
        if found and all(assignment[v] != -1 for v in range(n)):
            return assignment

        # Backtracking with trail-based undo
        trail = []  # List of (vertex, removed_color) for undo

        def record_removal(v, c):
            trail.append((v, c))

        def undo_to_mark(mark):
            while len(trail) > mark:
                v, c = trail.pop()
                domains[v].add(c)

        def backtrack():
            if time.time() > deadline:
                return 'timeout'
            
            # Find unassigned vertex with smallest domain, break ties by higher degree
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = len(domains[v])
                    if count == 0:
                        return False
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return True  # All assigned
            
            # Try each color in domain
            for c in sorted(domains[best_v]):
                if time.time() > deadline:
                    return 'timeout'
                
                mark = len(trail)
                
                # Assign
                assignment[best_v] = c
                record_removal(best_v, c)  # Mark for undo (domain becomes {c})
                # Actually, we need to save the old domain state
                old_domain_size = len(domains[best_v])
                old_domain = set(domains[best_v])
                domains[best_v] = {c}
                
                # Propagate: remove c from neighbors' domains
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if c in domains[u]:
                            domains[u].discard(c)
                            record_removal(u, c)
                            if len(domains[u]) == 0:
                                valid = False
                                break
                
                if valid:
                    # Run AC-3 on affected arcs
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected)
                    if ac_res is True:
                        result = backtrack()
                        if result is True:
                            return True
                        if result == 'timeout':
                            undo_to_mark(mark)
                            assignment[best_v] = -1
                            domains[best_v] = old_domain
                            return 'timeout'
                    elif ac_res is False:
                        pass  # Fall through to undo
                    else:  # timeout
                        undo_to_mark(mark)
                        assignment[best_v] = -1
                        domains[best_v] = old_domain
                        return 'timeout'
                
                # Undo
                undo_to_mark(mark)
                assignment[best_v] = -1
                domains[best_v] = old_domain
            
            return False

        # Shuffle domains for randomized restart
        for i in range(n):
            if assignment[i] == -1:
                lst = list(domains[i])
                rng.shuffle(lst)
                domains[i] = set(lst)

        result = backtrack()
        if result is True:
            return assignment
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 145.0  # 5s margin before 150s kill

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
        colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
        
        if colouring is None:
            for retry_seed in range(1, 10):
                if time.time() > instance_deadline:
                    break
                rng_solver = Random(retry_seed)
                colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
                if colouring is not None:
                    break
        
        if colouring is None:
            colouring = [0] * n
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
