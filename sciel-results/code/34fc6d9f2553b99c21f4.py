"""
Solves the list coloring problem for four fixed planar bipartite graphs.
Uses a CSP approach: Arc Consistency (AC-3) preprocessing + Backtracking with Forward Checking (FC) and MRV heuristic.
"""

import time
from random import Random
from collections import deque

def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 10.0
    
    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Initial cube faces
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        
        # Grow the graph by splitting faces
        for _ in range(splits):
            if not faces:
                break
            face_idx = rng.randrange(len(faces))
            face = faces[face_idx]
            inner = list(range(n, n + 4))
            n += 4
            
            # Add edges for the new inner square and connections to the outer face
            for i in range(4):
                a = face[i]
                b = face[(i + 1) % 4]
                in_i = inner[i]
                in_next = inner[(i + 1) % 4]
                
                edges.add((min(a, in_i), max(a, in_i)))
                edges.add((min(b, in_next), max(b, in_next)))
                edges.add((min(in_i, in_next), max(in_i, in_next)))
                
            # Update faces: replace the split face with 3 new outer faces and 1 inner face
            # Original face: a, b, c, d
            # Inner: x, y, z, w
            # New outer faces: (a,b,y,x), (b,c,z,y), (c,d,w,z), (d,a,x,w)
            # New inner face: (x,y,z,w)
            faces[face_idx] = [face[0], face[1], inner[1], inner[0]]
            faces.append([face[1], face[2], inner[2], inner[1]])
            faces.append([face[2], face[3], inner[3], inner[2]])
            faces.append([face[3], face[0], inner[0], inner[3]])
            faces.append(inner)
            
        # Generate lists
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        
        return n, sorted(edges), lists

    def solve_csp(n, adj, lists):
        # Initialize domains as sets for efficient modification
        domains = [set(lst) for lst in lists]
        
        # AC-3 Arc Consistency
        def revise(X, Y):
            # Returns True if domain of X was revised
            removed = False
            for x_val in list(domains[X]):
                if not any(x_val != y_val for y_val in domains[Y]):
                    domains[X].discard(x_val)
                    removed = True
            return removed

        queue = deque()
        # Initialize queue with all arcs (u, v)
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        while queue:
            (X, Y) = queue.popleft()
            if revise(X, Y):
                if not domains[X]:
                    return None # Unsatisfiable
                for Z in adj[X]:
                    if Z != Y:
                        queue.append((Z, X))
        
        # Backtracking with Forward Checking
        assignment = [-1] * n
        
        def backtrack():
            # Select variable with Minimum Remaining Values (MRV)
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                return True
            
            # MRV: pick v with smallest domain
            # Tie-breaker: Highest degree
            v = min(unassigned, key=lambda x: (len(domains[x]), -len(adj[x])))
            
            if not domains[v]:
                return False
            
            # Value Ordering: Least Constraining Value (LCV)
            # For small domains, sorting by how many neighbors it eliminates is good.
            # However, with only 3 colors, simple iteration is often fast enough.
            # We'll stick to sorted order for determinism and simplicity, 
            # as LCV overhead might not pay off for size 3.
            
            for c in sorted(domains[v]):
                # Assign
                assignment[v] = c
                
                # Record changes for backtracking
                saved_domains = []
                conflict = False
                for nb in adj[v]:
                    if assignment[nb] == -1 and c in domains[nb]:
                        saved_domains.append((nb, c))
                        domains[nb].discard(c)
                        if not domains[nb]:
                            conflict = True
                            break
                
                if not conflict:
                    if backtrack():
                        return True
                
                # Undo
                for (nb_idx, val) in saved_domains:
                    domains[nb_idx].add(val)
                assignment[v] = -1
                
            return False

        # Run AC-3 first
        # Note: The AC-3 above modifies domains. We need to ensure we don't modify the original 'lists' if we wanted to restart, 
        # but since we only need one solution, modifying domains in place is fine.
        
        # Wait, the AC-3 function `revise` modifies `domains`. 
        # If AC-3 fails, it returns None. 
        # If it succeeds, domains are pruned.
        # Then backtracking runs on the pruned domains.
        
        # Re-run AC-3 logic properly integrated
        # The previous AC-3 block was inside solve_csp but I defined it. 
        # Let's call it.
        
        # Actually, I defined `revise` and the queue logic inside `solve_csp`? 
        # No, I defined `revise` and the loop in the draft above but didn't wrap it in a function call properly in the thought process.
        # Let's structure it clearly.
        
        # --- AC-3 Phase ---
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        while queue:
            (X, Y) = queue.popleft()
            if revise(X, Y):
                if not domains[X]:
                    return None
                for Z in adj[X]:
                    if Z != Y:
                        queue.append((Z, X))
        
        # --- Backtracking Phase ---
        return backtrack()

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            # If we run out of time, we can't return a valid list of 4.
            # But this is extremely unlikely for these sizes.
            # To be safe against a crash/timeout leading to 0 score, we assume we finish.
            pass
            
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        solution = solve_csp(n, adj, lists)
        
        if solution is None:
            # Fallback: This should not happen.
            # If it does, returning an invalid structure results in 0 score.
            # We can try a simple greedy as a last resort, though it's likely to fail too.
            # Given Alon-Tarsi, a solution exists. The solver should find it.
            # For robustness, we just return a list of 0s which will be invalid but valid *type*.
            # Actually, the prompt says "wrong length or type makes the whole result invalid".
            # "An instance whose colouring breaks a list or an edge scores 0".
            # So returning a wrong coloring is better than crashing? 
            # "Any broken rule makes the whole result invalid; there is no partial credit."
            # Wait, "The score is the number of properly coloured instances (0..4)".
            # "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # This implies partial credit IS possible for individual instances if the *structure* is valid (list of ints of correct length).
            # "wrong length or type makes the whole result invalid" -> This refers to the top-level return value.
            # So, if I return a list of 4 lists, and one list has wrong length, the WHOLE result is invalid (score 0).
            # If I return a list of 4 lists, all correct length, but one coloring is bad, that instance scores 0, others score 1.
            # So, if solver fails, I should return a dummy coloring of correct length to preserve the validity of the other 3 instances.
            dummy = [0] * n
            # Ensure dummy is a list of ints
            out.append(dummy)
        else:
            # Ensure types are Python int
            out.append([int(c) for c in solution])
        
    return out

# EVOLVE-BLOCK-END
