"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

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

        # Initial domains as lists (order matters for determinism, but we'll shuffle for retry)
        domains = [list(lst) for lst in lists]
        assignment = [-1] * n
        assigned = [False] * n

        sys.setrecursionlimit(10000)

        # Forward checking: remove color c from neighbor's domain
        def propagate(v, c):
            """Assign v=c, propagate to neighbors. Returns False if any domain becomes empty."""
            for u in adj[v]:
                if not assigned[u]:
                    if c in domains[u]:
                        domains[u].remove(c)
                        if not domains[u]:
                            return False
            return True

        def unpropagate(v, c):
            """Undo assignment of v=c, restore c to neighbors' domains."""
            for u in adj[v]:
                if not assigned[u]:
                    if c not in domains[u]:
                        domains[u].append(c)

        def backtrack():
            # MRV: pick unassigned vertex with fewest available colors
            min_count = float('inf')
            min_v = -1
            for v in range(n):
                if not assigned[v]:
                    count = len(domains[v])
                    if count < min_count:
                        min_count = count
                        min_v = v
                        if count == 0:
                            return False
                        if count == 1:
                            # Can't get lower than 1 for valid, but 0 is handled above
                            pass

            if min_v == -1:
                return True  # All assigned

            # Try each color in domain
            # Use a copy to iterate safely (domain might be modified by propagate? No, we remove from neighbors)
            # Actually, domains[min_v] is fixed during this iteration.
            for c in domains[min_v][:]:
                if time.time() > deadline:
                    return 'timeout'
                
                assignment[min_v] = c
                assigned[min_v] = True
                
                # Temporarily remove c from min_v's domain? No, min_v is assigned.
                # We just need to propagate to neighbors.
                if propagate(min_v, c):
                    result = backtrack()
                    if result is True:
                        return True
                    if result == 'timeout':
                        # Undo before returning timeout
                        unpropagate(min_v, c)
                        assigned[min_v] = False
                        assignment[min_v] = -1
                        return 'timeout'
                
                # Undo
                unpropagate(min_v, c)
                assigned[min_v] = False
                assignment[min_v] = -1

            return False

        # Initial propagation? No, no assignments yet.
        # Just start backtracking.
        # Note: The first call to backtrack will pick an MRV vertex.
        
        # Shuffle domains to break symmetry/tie-breaking for retry
        for i in range(n):
            rng.shuffle(domains[i])

        result = backtrack()
        if result is True:
            return assignment
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 145.0  # 5s margin before 150s kill

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Time check before generation
        if time.time() > hard_deadline:
            # Can't even generate, but we need correct length.
            # We must generate to know n. Generation is fast.
            pass

        n, edges, lists = instance(splits, seed)

        # If time is tight, we might skip solving and return zeros?
        # But we need to know n first.
        # Generation is fast. Solving is the bottleneck.
        
        if time.time() > hard_deadline - 5.0:
             # Safety margin: if we are close to limit, return dummy for remaining
             # We have already generated n.
             out.append([0] * n)
             continue
        
        # Solve with a per-instance deadline
        # Allocate remaining time / remaining instances
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.9  # 90% of allocated
        
        colouring = None
        # Try deterministic first (seed 0 shuffle)
        rng_solver = Random(0)
        colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
        
        if colouring is None:
            # Try a few randomized restarts if time permits
            for retry_seed in range(1, 5):
                if time.time() > instance_deadline:
                    break
                rng_solver = Random(retry_seed)
                colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
                if colouring is not None:
                    break
        
        if colouring is None:
            # Failed or timeout. Return dummy.
            colouring = [0] * n
        
        # Ensure types are Python int
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
