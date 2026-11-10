import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We try to find a valid coloring for n=34, then 33, etc.
    for n in range(34, 0, -1):
        # Try multiple random restarts for each n
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Randomize the order of vertices to break symmetry
            # Actually, we just randomize the choice when both colors are valid
            # and potentially the order of processing triples if we want more variety.
            # But the greedy order (by largest vertex) is fixed for efficiency.
            
            colors = {}
            # We'll store colors in a dictionary or a flat list.
            # Flat list is faster.
            # Index: (a, b, c) -> index
            # We can use a helper to get index.
            # Or just use a 3D array or nested dict.
            # For speed, a flat list with a precomputed index map is best.
            
            # Precompute index for all triples
            index_map = {}
            triples = list(combinations(range(n), 3))
            for i, t in enumerate(triples):
                index_map[t] = i
            
            color_array = [None] * len(triples)
            
            # We process triples in a specific order.
            # Order: by max vertex, then mid, then min.
            # This ensures that when we process (a,b,c) with a<b<c,
            # all triples with vertices < c have been processed? No.
            # Triples like (a,b,d) with d<c are processed earlier if we sort by max vertex.
            # Let's sort triples by (max, mid, min).
            sorted_triples = sorted(triples, key=lambda t: (t[2], t[1], t[0]))
            
            valid = True
            for a, b, c in sorted_triples:
                idx = index_map[(a, b, c)]
                
                # Check if Red is allowed
                red_allowed = True
                # A red K4 is formed if there exists d < a such that
                # (a,b,d), (a,c,d), (b,c,d) are all Red.
                # Note: d must be < a because a is the smallest in (a,b,c)?
                # No, a<b<c. The 4th vertex d can be any other vertex.
                # If d > c, then (a,b,d) etc. are not processed yet?
                # Wait, if we process in increasing order of max vertex,
                # then when processing (a,b,c), any triple with max vertex < c is processed.
                # Triples with max vertex > c are not processed.
                # A K4 on {a,b,c,d} involves edges (a,b,c), (a,b,d), (a,c,d), (b,c,d).
                # If d > c, then (a,b,d), (a,c,d), (b,c,d) have max vertex d > c.
                # They are not processed yet. So they are "unknown".
                # We cannot determine if they will be Red.
                # However, if we are coloring (a,b,c) Red, we must ensure that
                # we *can* color the other edges later to avoid a Red K4.
                # This is complex.
                
                # Standard Greedy CSP:
                # We only check for conflicts with *already assigned* edges.
                # If a conflict *could* be formed with future edges, we don't know yet.
                # But if a conflict is *already* formed by assigned edges, we must avoid it.
                
                # So, for Red (a,b,c):
                # Check if there exists d such that (a,b,d), (a,c,d), (b,c,d) are ALL RED.
                # This requires d to be such that these edges are already assigned.
                # Since we process by increasing max vertex:
                # - If d < c, then max(a,b,d) <= b < c, max(a,c,d) = c? No.
                # Let's be precise.
                # Edge (u,v,w) is processed when we reach max(u,v,w).
                # If we are at (a,b,c) with a<b<c, it is processed at step c.
                # Edges with max vertex < c are assigned.
                # Edges with max vertex = c are being assigned now.
                # Edges with max vertex > c are unassigned.
                
                # For a Red K4 on {a,b,c,d}:
                # Case 1: d < c.
                # Then max(a,b,d) < c, max(a,c,d) = c, max(b,c,d) = c.
                # (a,b,d) is assigned.
                # (a,c,d) and (b,c,d) have max vertex c. They are being assigned in the same batch (step c).
                # If we process triples with max vertex c in some order, we might not have assigned (a,c,d) yet.
                # This makes the greedy check tricky.
                
                # To simplify, we can process all triples with max vertex c *before* checking for K4s involving c?
                # No, we assign them one by one.
                
                # Alternative: Process vertices one by one.
                # When adding vertex c, we assign all triples (a,b,c) for a<b<c.
                # We can check for Red K4s involving c.
                # A Red K4 involving c is {a,b,d,c}.
                # Edges: (a,b,c), (a,d,c), (b,d,c), (a,b,d).
                # (a,b,d) is already assigned (max d < c).
                # (a,d,c) and (b,d,c) are being assigned now (max c).
                # If we assign (a,b,c) Red, we must ensure that we can assign (a,d,c) and (b,d,c) such that not all are Red.
                # If (a,b,d) is Red, and we *must* assign (a,d,c) and (b,d,c) Red (because Blue is forbidden for them), then we have a conflict.
                
                # So, the condition for (a,b,c) being Red is:
                # There is no d < c such that:
                # 1. (a,b,d) is Red.
                # 2. (a,d,c) is FORCED to be Red (i.e., Blue is forbidden).
                # 3. (b,d,c) is FORCED to be Red (i.e., Blue is forbidden).
                
                # This requires knowing if Blue is forbidden for (a,d,c) and (b,d,c).
                # Blue is forbidden for (x,y,c) if there exists e < c such that the other 8 edges of {x,y,c,e} are Blue?
                # No, Blue K5.
                
                # This dependency chain is deep.
                # A simpler heuristic:
                # Just check for *existing* conflicts.
                # If (a,b,c) is Red, and there is d < c such that (a,b,d), (a,c,d), (b,c,d) are *all already assigned* Red, then conflict.
                # But (a,c,d) and (b,c,d) are not assigned yet if we process (a,b,c) first.
                # So we should process triples in an order such that for any K4, the "other" edges are processed before the "new" one?
                # Impossible for all K4s.
                
                # Let's use a different strategy: **Randomized Local Search**.
                # It is more robust to such dependencies.
                pass
            
            # If greedy fails or is too complex, use Local Search.
            # Let's switch to Local Search.
            break # Break out of attempt loop if we decide to use LS
            
    # Implement Local Search
    for n in range(34, 0, -1):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize random coloring
            num_triples = len(list(combinations(range(n), 3)))
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Precompute index map
            index_map = {}
            triples = list(combinations(range(n), 3))
            for i, t in enumerate(triples):
                index_map[t] = i
                
            # Function to calculate energy (number of conflicts)
            def get_energy(colors):
                red_k4 = 0
                blue_k5 = 0
                # Check Red K4
                for a, b, c, d in combinations(range(n), 4):
                    if (colors[index_map[(a,b,c)]] == 0 and 
                        colors[index_map[(a,b,d)]] == 0 and 
                        colors[index_map[(a,c,d)]] == 0 and 
                        colors[index_map[(b,c,d)]] == 0):
                        red_k4 += 1
                # Check Blue K5
                for a, b, c, d, e in combinations(range(n), 5):
                    # 10 edges
                    edges = [
                        (a,b,c), (a,b,d), (a,b,e),
                        (a,c,d), (a,c,e),
                        (a,d,e),
                        (b,c,d), (b,c,e),
                        (b,d,e),
                        (c,d,e)
                    ]
                    all_blue = True
                    for edge in edges:
                        if colors[index_map[edge]] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        blue_k5 += 1
                return red_k4 + blue_k5

            # Local Search Loop
            current_energy = get_energy(colors)
            best_energy = current_energy
            best_colors = colors.copy()
            
            # Simple Hill Climbing with Random Restarts
            # We try to flip one edge at a time.
            # To speed up, we can sample a few random edges to flip.
            
            steps = 1000
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to flip
                idx = random.randint(0, num_triples - 1)
                old_color = colors[idx]
                colors[idx] = 1 - old_color
                
                new_energy = get_energy(colors)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors.copy()
                        if best_energy == 0:
                            found = True
                            break
                else:
                    # Revert
                    colors[idx] = old_color
                    
            if found:
                break
                
        if found:
            return {'n': n, 'colors': best_colors}
            
    # If no solution found for n>=1, return n=1
    return {'n': 1, 'colors': []}
