"""Model-and-optimize: parameterise the colouring (one int per vertex) and fit
each instance directly against the objective (list-respecting, edge-different
colouring) using a backtracking constraint solver with forward checking."""
# EVOLVE-BLOCK-START
import time
import heapq
import sys
from random import Random

sys.setrecursionlimit(10000)

def construct():
    t0 = time.time()
    LIMIT = 120.0          # hard wall-clock cap for the evaluator
    MARGIN = 10.0          # safety margin before the cap (return early)

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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

    def solve_coloring(n, adj, lists):
        # Initialize available colors for each vertex
        # Using a list of sets for fast removal and checking
        avail = [set(l) for l in lists]
        color = [-1] * n
        colored = [False] * n
        num_colored = 0
        
        # We use a heap to always pick the vertex with the fewest available colors (MRV heuristic)
        # Tuple: (num_available, vertex_id)
        # We need to handle updates to the heap. Since N is small, we can just rebuild or use lazy deletion.
        # Lazy deletion is efficient enough.
        
        heap = []
        for i in range(n):
            if avail[i]:
                heapq.heappush(heap, (len(avail[i]), i))
        
        def backtrack():
            nonlocal num_colored
            if num_colored == n:
                return True
            
            # Find a valid uncolored vertex
            v = -1
            while heap:
                cnt, u = heapq.heappop(heap)
                if not colored[u] and len(avail[u]) == cnt:
                    v = u
                    break
            
            if v == -1:
                # No uncolored vertices found in heap? 
                # This implies all uncolored vertices have avail size > cnt or were pruned.
                # If no vertex is available, we must be in a dead end or done.
                # But if num_colored < n, there must be uncolored vertices.
                # If their avail sets changed, the heap entries are stale.
                # We need to re-push them if we didn't process them.
                # Actually, if the heap is empty or only contains stale entries for uncolored vertices,
                # it means we might have a conflict or need to re-evaluate.
                # However, if an uncolored vertex has empty avail, it should have been detected 
                # when we colored its neighbor.
                
                # Let's check if any uncolored vertex has empty avail.
                for i in range(n):
                    if not colored[i] and not avail[i]:
                        return False
                        
                # If we reach here, it means the heap was empty but there are uncolored vertices.
                # This can happen if all heap entries were stale.
                # Rebuild heap from uncolored vertices.
                # This is rare but safe.
                for i in range(n):
                    if not colored[i] and avail[i]:
                        heapq.heappush(heap, (len(avail[i]), i))
                return backtrack() # Try again

            # Assign colors to v
            # Sort colors by how few options they leave for neighbors (LCV heuristic)
            # For simplicity and speed with small N, we can just iterate avail[v]
            # But to be efficient, let's pick the color that minimizes the sum of remaining options for neighbors
            # Given N is small, simple iteration is likely fast enough.
            
            # Save state for backtracking
            # We need to restore avail sets of neighbors and v itself
            
            for c in list(avail[v]): # Copy to allow modification during loop
                # Try color c
                color[v] = c
                colored[v] = True
                num_colored += 1
                
                # Check neighbors
                valid = True
                affected = []
                for nb in adj[v]:
                    if colored[nb]:
                        # If neighbor is colored, we should have already removed c from avail[nb]
                        # But we are coloring v now, so neighbors are uncolored or colored.
                        # If neighbor is colored, we don't need to update avail[nb] for future coloring,
                        # but we must ensure c != color[nb].
                        if color[nb] == c:
                            valid = False
                            break
                    else:
                        # Neighbor is uncolored. Remove c from its avail set.
                        if c in avail[nb]:
                            # Check if removing c empties the set
                            if len(avail[nb]) == 1:
                                # This will cause a conflict when we pick nb next
                                # We can mark it immediately or let the next pick handle it
                                pass
                            avail[nb].remove(c)
                            affected.append(nb)
                            # Push updated nb to heap
                            if avail[nb]:
                                heapq.heappush(heap, (len(avail[nb]), nb))
                            else:
                                # Conflict: neighbor has no colors left
                                valid = False
                                break
                        
                if valid:
                    if backtrack():
                        return True
                
                # Undo changes
                num_colored -= 1
                colored[v] = False
                color[v] = -1
                
                # Restore avail sets of affected neighbors
                for nb in affected:
                    avail[nb].add(c)
                    # We don't need to push to heap here because the old entry is still in heap
                    # and we will re-evaluate when we pop.
                    
                # Note: We did not remove c from avail[v] because we are trying all c in avail[v].
                # But wait, if we backtrack, we need to try the next c in avail[v].
                # The loop `for c in list(avail[v])` handles this.
                
            # If no color worked, backtrack to parent
            return False

        # Initial check
        for i in range(n):
            if not avail[i]:
                return None

        if backtrack():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            break
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        col = solve_coloring(n, adj, lists)
        
        if col is None:
            # Fallback: This should not happen per task description, but just in case
            # Use a simple random assignment that respects lists (might break edges)
            # Or return a dummy valid structure to avoid crash, though it will score 0 for this instance.
            # Better: try a greedy once more with different random seed?
            # Given the theorem, it should always be solvable.
            # Let's just return an empty list or dummy to prevent crash, but we expect non-None.
            col = [lists[i][0] for i in range(n)]
            
        # Ensure types are Python int
        out.append([int(c) for c in col])

    return out
# EVOLVE-BLOCK-END
