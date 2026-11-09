"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0

    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve_instance(n, edges, lists, rng_solver, time_limit):
        """
        Solve using min-conflicts local search with random restarts.
        This is a different paradigm: instead of systematic backtracking,
        we start with a random valid assignment and iteratively fix conflicts.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to sets for fast membership testing
        list_sets = [set(lst) for lst in lists]
        
        def count_conflicts(assignment):
            """Count the number of violated edges (where both endpoints have same color)."""
            conflicts = 0
            for u, v in edges:
                if assignment[u] == assignment[v]:
                    conflicts += 1
            return conflicts
        
        def is_valid(assignment):
            """Check if assignment satisfies all list and edge constraints."""
            for v in range(n):
                if assignment[v] not in list_sets[v]:
                    return False
            for u, v in edges:
                if assignment[u] == assignment[v]:
                    return False
            return True
        
        def min_conflicts(max_iters=50000):
            """
            Min-conflicts local search.
            Returns a valid assignment or None.
            """
            # Initialize: assign each vertex a random color from its list
            assignment = [rng_solver.choice(lst) for lst in lists]
            
            # Track conflicts per vertex
            # conflict_count[v] = number of neighbors with same color as v
            conflict_count = [0] * n
            for u, v in edges:
                if assignment[u] == assignment[v]:
                    conflict_count[u] += 1
                    conflict_count[v] += 1
            
            # Check if already valid
            if all(c == 0 for c in conflict_count):
                if is_valid(assignment):
                    return assignment[:]
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    return None
                
                # Find all vertices with conflicts
                conflicted = [v for v in range(n) if conflict_count[v] > 0]
                
                if not conflicted:
                    if is_valid(assignment):
                        return assignment[:]
                    break
                
                # Pick a random conflicted vertex
                v = rng_solver.choice(conflicted)
                
                # Try all colors in its list, pick the one that minimizes conflicts
                best_color = assignment[v]
                best_conflicts = conflict_count[v]
                
                # Try each color in the list
                for c in lists[v]:
                    if c == assignment[v]:
                        continue
                    # Compute conflict count if we assign color c to v
                    # Current conflicts for v: conflict_count[v]
                    # If we change v's color from old to c:
                    #   - Remove conflicts where old color matched a neighbor
                    #   - Add conflicts where c matches a neighbor
                    old_color = assignment[v]
                    new_conflicts = 0
                    for u in adj[v]:
                        if assignment[u] == c:
                            new_conflicts += 1
                    # Also need to account for the fact that some neighbors might
                    # have been conflicting with old_color. But for min-conflicts,
                    # we just want to minimize total conflicts. The standard approach:
                    # new_conflicts_for_v = number of neighbors with color c
                    # But we also need to consider that changing v affects neighbors' conflict counts.
                    # For simplicity in min-conflicts, we just pick the color that minimizes
                    # the number of neighbors sharing that color.
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_color = c
                
                if best_color == assignment[v]:
                    # No improvement possible with current neighbors; stay or random restart
                    continue
                
                # Update conflict counts
                old_color = assignment[v]
                for u in adj[v]:
                    if assignment[u] == old_color:
                        conflict_count[u] -= 1
                    if assignment[u] == best_color:
                        conflict_count[u] += 1
                
                assignment[v] = best_color
                # Recompute conflict_count[v]
                conflict_count[v] = 0
                for u in adj[v]:
                    if assignment[u] == best_color:
                        conflict_count[v] += 1
                
                # Check if solved
                if all(c == 0 for c in conflict_count):
                    if is_valid(assignment):
                        return assignment[:]
            
            return None
        
        # Try multiple restarts
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            result = min_conflicts(max_iters=30000)
            if result is not None:
                return result
        
        # Fallback: greedy with backtracking-lite
        # Try greedy coloring in random order multiple times
        for r in range(50):
            if time.time() > time_limit:
                break
            order = list(range(n))
            rng_solver.shuffle(order)
            assignment = [-1] * n
            success = True
            for v in order:
                used = set()
                for u in adj[v]:
                    if assignment[u] != -1:
                        used.add(assignment[u])
                free = [c for c in lists[v] if c not in used]
                if free:
                    assignment[v] = rng_solver.choice(free)
                else:
                    success = False
                    break
            if success and is_valid(assignment):
                return assignment
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            pass

        n, edges, lists = instance(splits, seed)
        
        rng_solver = random.Random(seed + 42)
        
        time_per_instance = 25.0
        t_start = time.time()
        
        coloring = solve_instance(n, edges, lists, rng_solver, t_start + time_per_instance)
        
        if coloring is None:
            # Last resort: greedy by degree
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            col = [0] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    taken.add(col[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    col[v] = free[0]
                else:
                    col[v] = lists[v][0]
            coloring = col
        
        # Ensure all elements are Python int
        coloring = [int(c) for c in coloring]
        out.append(coloring)
        
    return out
# EVOLVE-BLOCK-END
