"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize domains as sets for O(1) membership and removal
        domains = [set(lst) for lst in lists]

        # AC-3 Arc Consistency
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))

        def revise(x, y):
            dx = domains[x]
            dy = domains[y]
            revised = False
            # For constraint x != y:
            # A value c in dx is removed if it is not supported by dy.
            # c is not supported if for all d in dy, c == d.
            # This happens if and only if dy is a subset of {c}, i.e., dy == {c}.
            if len(dy) == 1:
                only_y = next(iter(dy))
                if only_y in dx:
                    dx.remove(only_y)
                    revised = True
            return revised

        # Run AC-3
        while queue:
            x, y = queue.popleft()
            if revise(x, y):
                if not domains[x]:
                    return None # Unsatisfiable
                for z in adj[x]:
                    if z != y:
                        queue.append((z, x))

        # Check if any domain is empty
        for i in range(n):
            if not domains[i]:
                return None

        # Backtracking Search with Forward Checking, MRV, and LCV
        assignment = [None] * n
        
        def backtrack(assignment, domains):
            # Check time limit
            if time.time() > deadline:
                return None

            # Find unassigned variable with MRV
            unassigned = [i for i in range(n) if assignment[i] is None]
            if not unassigned:
                return assignment[:] # Success

            # MRV: Minimum Remaining Values
            # Tie-breaker: Degree (most constrained)
            min_size = float('inf')
            var = -1
            for u in unassigned:
                size = len(domains[u])
                if size < min_size:
                    min_size = size
                    var = u
                elif size == min_size:
                    # Tie-breaker: Higher degree
                    if len(adj[u]) > len(adj[var]):
                        var = u
            
            if var == -1:
                return None

            # LCV: Least Constraining Value
            # For each value in domains[var], count how many neighbors would lose this value.
            # We want to minimize this count.
            values = list(domains[var])
            
            # Pre-calculate unassigned neighbors for efficiency
            unassigned_neighbors = [nb for nb in adj[var] if assignment[nb] is None]
            
            scored_values = []
            for val in values:
                # Count how many unassigned neighbors have 'val' in their domain
                # If we assign var=val, we must remove 'val' from all neighbors' domains.
                # The "cost" is the number of neighbors that actually have 'val' in their domain.
                # A lower cost means we constrain the neighbors less.
                cost = 0
                for nb in unassigned_neighbors:
                    if val in domains[nb]:
                        cost += 1
                scored_values.append((cost, val))
            
            # Sort by cost (ascending), then by value (for determinism)
            scored_values.sort()
            
            for _, val in scored_values:
                # Check if val is still in domain (should be, but safe)
                if val not in domains[var]:
                    continue
                
                # Forward Checking
                consistent = True
                changes = []
                for neighbor in adj[var]:
                    if assignment[neighbor] is not None:
                        # Already assigned, just check constraint
                        if assignment[neighbor] == val:
                            consistent = False
                            break
                    else:
                        # Unassigned, remove val from neighbor's domain
                        if val in domains[neighbor]:
                            domains[neighbor].remove(val)
                            changes.append(neighbor)
                            if not domains[neighbor]:
                                consistent = False
                                break
                
                if consistent:
                    assignment[var] = val
                    result = backtrack(assignment, domains)
                    if result is not None:
                        return result
                    assignment[var] = None
                
                # Revert forward checking changes
                for neighbor in changes:
                    domains[neighbor].add(val)

            return None

        return backtrack(assignment, domains)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_csp(n, edges, lists)
        if colouring is None:
            # Fallback: should not happen
            out.append([0] * n)
        else:
            # Ensure types are Python int
            out.append([int(c) for c in colouring])
            
    return out
# EVOLVE-BLOCK-END
