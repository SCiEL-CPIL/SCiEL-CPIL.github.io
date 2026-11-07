"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock kill

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

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by descending degree for better greedy/backtracking performance
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        # First try a fast greedy with backtracking (limited depth)
        color = [-1] * n
        # Assign in order
        for idx, v in enumerate(order):
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                color[v] = free[0]
            else:
                # Greedy failed, need backtracking
                break
        else:
            # All assigned successfully
            return color
        
        # Backtracking search
        # Use a recursive backtracker with time checks
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(order):
                return True
            v = order[pos]
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            for c in lists[v]:
                if c not in taken:
                    color[v] = c
                    if backtrack(pos + 1):
                        return True
                    color[v] = -1
            return False
        
        # Reset colors that were assigned greedily but may need reassignment
        # Actually, the greedy assignments for earlier vertices are fine if we keep them,
        # but backtracking might need to change them. Let's reset all and backtrack from 0.
        color = [-1] * n
        if backtrack(0):
            return color
        else:
            # Should not happen due to Alon-Tarsi, but return best effort
            return color

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 10:
            # Not enough time for this instance, break
            break
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        # Ensure types are Python int
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    
    # If we broke early due to deadline, pad with invalid colorings? 
    # The task says wrong length makes result invalid. So we must return 4 lists.
    # If we don't have time, we can't produce valid colorings for remaining.
    # But the evaluator kills at 150s. If we return fewer, it's invalid.
    # So we must ensure we return 4 lists. If we can't solve them in time, we might as well
    # just return what we have? No, "wrong length ... makes the whole result invalid".
    # So we MUST return 4 lists. If we run out of time, we are stuck.
    # However, the greedy + backtracking is very fast for these sizes (n <= 408).
    # 408 vertices, backtracking is likely very fast.
    
    # If out has less than 4 items, we have a problem.
    # Let's assume it finishes in time. If not, we can't do much.
    
    # Convert to list of lists of ints
    return [list(map(int, c)) for c in out]

# EVOLVE-BLOCK-END
