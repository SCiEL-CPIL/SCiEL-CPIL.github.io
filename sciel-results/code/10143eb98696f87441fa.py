"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    """
    Solves the four list-coloring instances using a CSP backtracking solver 
    with MRV (Minimum Remaining Values) heuristic and forward checking.
    """
    start_time = time.time()
    # Safety margin: aim to finish within 140s to be safe against 150s kill
    deadline = start_time + 140.0

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
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Convert lists to sets for faster lookup
        domains = [set(lst) for lst in lists]
        
        # Initial check: if any domain is empty, no solution
        for i in range(n):
            if not domains[i]:
                return None

        # We will use a recursive backtracking approach.
        # To avoid recursion depth issues and allow for time checking, 
        # we use a stack-based approach or increase recursion limit.
        # Given n <= 408, recursion depth is manageable, but let's be safe.
        sys.setrecursionlimit(2000)

        color = [-1] * n
        unassigned = set(range(n))
        
        def backtrack():
            # Check deadline
            if time.time() > deadline:
                return False

            if not unassigned:
                return True

            # Select vertex with smallest domain (MRV)
            best_v = -1
            best_rem = None
            best_count = float('inf')

            for v in unassigned:
                taken = {color[u] for u in adj[v] if color[u] != -1}
                rem = [c for c in lists[v] if c not in taken]
                if not rem:
                    return False # Dead end
                if len(rem) < best_count:
                    best_count = len(rem)
                    best_v = v
                    best_rem = rem
                    if best_count == 1:
                        break # Can't get better than 1

            if best_v == -1:
                return False

            unassigned.remove(best_v)

            # Try colors for best_v
            # Heuristic: try colors that eliminate the most options for neighbors (Least Constraining Value)
            # For simplicity and speed, we just iterate through the remaining colors.
            # Given small domain sizes (max 3), this is cheap.
            
            for c in best_rem:
                color[best_v] = c
                if backtrack():
                    return True
                color[best_v] = -1

            unassigned.add(best_v)
            return False

        if backtrack():
            return color
        return None

    out = []
    params = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for splits, seed in params:
        # Check time before starting a new instance
        if time.time() > deadline - 5: # Reserve 5s for finishing up
            # If we run out of time, we must return something.
            # The problem states "wrong length or type makes the whole result invalid".
            # So we must return a list of 4 lists.
            # If we fail to solve, we return a dummy valid structure? 
            # No, "Any broken rule makes the whole result invalid".
            # So we must try our best to solve all.
            # If we can't solve one, we can't really "skip" it without invalidating the result.
            # However, the CSP solver is very fast for these sizes.
            # Let's assume we finish. If not, we return a partial/invalid result which scores 0.
            # To be safe, if we time out, we should probably return the best we can, 
            # but since partial credit is 0, we just return what we have.
            # Actually, if we return a list of 4 items, but one is wrong, score is 0 for that instance.
            # If we return a list of length < 4, score is 0 (invalid).
            # So we must always return 4 lists.
            
            # If we run out of time, we fill remaining with a dummy valid structure?
            # No, we just let the loop run and hope. The solver is O(2^N) worst case but very fast in practice for this structure.
            pass

        n, edges, lists = instance(splits, seed)
        sol = solve_csp(n, edges, lists)
        
        if sol is None:
            # Fallback: shouldn't happen with correct solver and valid instances.
            # Return a dummy list of correct length to maintain type/length validity, 
            # though it will score 0 for this instance.
            sol = [0] * n
            
        # Ensure types are correct (Python int)
        # sol is a list of ints.
        out.append(sol)

    return out

# EVOLVE-BLOCK-END
