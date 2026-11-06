import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    for n in range(47, 38, -1):
        d_min = n - 39
        
        if d_min <= 0:
            return {'n': n, 'edges': []}
            
        max_attempts = 5000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            deficient = list(range(n))
            random.shuffle(deficient)
            
            stuck = False
            
            while deficient:
                if time.time() > deadline:
                    stuck = True
                    break
                    
                # Pick a random deficient vertex
                idx = random.randint(0, len(deficient) - 1)
                u = deficient[idx]
                
                # Remove u from deficient temporarily to avoid re-picking same one if it stays deficient?
                # No, we want to fix it.
                # Swap and pop
                deficient[idx] = deficient[-1]
                deficient.pop()
                
                # Try to add an edge to u
                success = False
                # Candidates are non-neighbors
                # We can iterate through all v and shuffle? 
                # Or just pick random v until we find a valid one?
                # Number of non-neighbors is n - 1 - deg[u].
                # Initially large. Later small.
                
                # Let's collect candidates
                candidates = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
                random.shuffle(candidates)
                
                for v in candidates:
                    common = adj[u] & adj[v]
                    if common.bit_count() <= 1:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        degrees[u] += 1
                        degrees[v] += 1
                        
                        # Update deficient list
                        if degrees[u] < d_min:
                            deficient.append(u)
                        if degrees[v] < d_min:
                            # Check if v is already in deficient?
                            # We removed u. v might be in deficient or not.
                            # If v was in deficient, it's still there.
                            # If v was NOT in deficient (deg >= d_min), it shouldn't be added.
                            # But deg[v] increased, so if it was >= d_min, it is still >= d_min.
                            # If it was < d_min, it is in deficient.
                            # So we only add v if it was NOT in deficient?
                            # No, if deg[v] < d_min, it MUST be in deficient.
                            # Is it possible v is not in deficient but deg[v] < d_min?
                            # Only if we made a mistake.
                            # We start with all in deficient.
                            # We only remove when we process.
                            # We only add back if still deficient.
                            # So invariant: v in deficient iff deg[v] < d_min.
                            # So if deg[v] < d_min, v is in deficient.
                            # We don't need to add it.
                            pass
                        
                        success = True
                        break
                        
                if not success:
                    # u could not be connected to any valid v
                    # Put u back? No, we are stuck.
                    stuck = True
                    break
                
            if not stuck and not deficient:
                if min(degrees) >= d_min:
                    return {'n': n, 'edges': edges}
                    
    return {'n': 39, 'edges': []}
