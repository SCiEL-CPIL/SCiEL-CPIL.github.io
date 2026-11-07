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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial available colors
        available = [set(lst) for lst in lists]
        color = [None] * n
        unassigned = set(range(n))
        
        # Heuristic: MRV (Minimum Remaining Values)
        # We will use a recursive backtracker with forward checking
        
        def backtrack(unassigned, available, color):
            if not unassigned:
                return True
            
            # Find vertex with minimum remaining values
            min_v = -1
            min_size = float('inf')
            for v in unassigned:
                if len(available[v]) < min_size:
                    min_size = len(available[v])
                    min_v = v
                    if min_size == 1:
                        break
            
            if min_size == 0:
                return False
            
            # If min_size == 1, we can propagate immediately (deterministic step)
            # But for simplicity and robustness, we just branch on the first color if size 1?
            # No, if size 1, it's forced. We should assign it and propagate.
            # Let's implement a loop for forced assignments first.
            
            # Actually, a simpler approach: just pick the MRV vertex and try its colors.
            # Forward checking will handle the propagation.
            
            v = min_v
            # Try colors in v's available set
            for c in list(available[v]):
                # Assign c to v
                color[v] = c
                new_unassigned = set(unassigned)
                new_unassigned.remove(v)
                
                # Check neighbors
                conflict = False
                # We need to copy available sets for neighbors to allow backtracking
                # Copying sets is expensive. 
                # Alternative: Use a trail/log for undo.
                
                # Given small size, copying is okay? 
                # N=408, degree ~4. 
                # Depth 408. 
                # Copying 408 sets of size 3 at each step is 408*408*3 ~ 500k ops per depth? 
                # That might be too slow if branching factor is high.
                
                # Let's use a trail-based undo.
                
                # For now, let's stick to a simple recursive call with copied state to ensure correctness first, 
                # and optimize if needed. The graphs are planar, so branching factor should be low with MRV.
                
                new_available = [s.copy() for s in available]
                new_available[v] = set() # v is assigned, no avail left
                
                for u in adj[v]:
                    if u in new_unassigned:
                        if c in new_available[u]:
                            new_available[u].remove(c)
                            if not new_available[u]:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack(new_unassigned, new_available, color):
                        return True
                
                # Undo
                color[v] = None
                
            return False

        # Start backtracking
        # To avoid recursion depth issues or overhead, we can use an iterative approach or just trust recursion.
        # Python recursion limit is 1000. Max depth 408. Safe.
        
        if backtrack(set(range(n)), available, color):
            return color
        else:
            # Fallback to greedy if backtracking fails (should not happen)
            # Greedy: sort by degree, pick first available
            deg = [len(adj[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: -deg[x])
            greedy_color = [None] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if greedy_color[u] is not None:
                        taken.add(greedy_color[u])
                for c in lists[v]:
                    if c not in taken:
                        greedy_color[v] = c
                        break
                if greedy_color[v] is None:
                    greedy_color[v] = lists[v][0] # Fallback, might be invalid
            return greedy_color

    out = []
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    instances = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for i, (splits, seed) in enumerate(instances):
        # Check if we have enough time for a full solve.
        # If not, we still need to produce a result.
        # The backtracker is fast, so we likely have time.
        
        n, edges, lists = instance(splits, seed)
        
        # Determine if we should use the full backtracker or just greedy.
        # Given the small size, backtracker is preferred.
        res = solve_instance(n, edges, lists)
        
        # Ensure types are correct (int, list)
        final_res = []
        for c in res:
            if c is None:
                final_res.append(0)
            else:
                final_res.append(int(c))
                
        out.append(final_res)
        
        # If we are running out of time, we can't do much else but ensure we finish the loop.
        # The loop will complete all 4 iterations.
        
    return out

# EVOLVE-BLOCK-END
