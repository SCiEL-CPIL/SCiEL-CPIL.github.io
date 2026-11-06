import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] that share a factor with 255
    # i.e., divisible by 3, 5, or 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # 255 must be in the set
    # Build adjacency: two candidates are adjacent if gcd > 1
    n = len(candidates)
    # Use bitset for adjacency
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # Greedy maximum clique heuristic with 255 forced in
    # Start with 255, greedily add vertices that are connected to all current members
    
    # Multiple randomized restarts for better results
    import random
    random.seed(42)
    
    best_clique = {idx_255}
    best_size = 1
    
    def try_greedy(forced_set, remaining_candidates):
        """Greedy: start with forced_set, add vertices connected to all current members."""
        current = set(forced_set)
        # Available vertices: those in remaining_candidates that connect to all in current
        available = set()
        for v in remaining_candidates:
            if v in current:
                continue
            # Check if v connects to all in current
            ok = True
            for u in current:
                if not (adj[u] & (1 << v)):
                    ok = False
                    break
            if ok:
                available.add(v)
        
        while available:
            # Pick vertex with highest degree within available (greedy)
            best_v = None
            best_deg = -1
            for v in available:
                # Count neighbors in available
                deg = 0
                for u in available:
                    if u != v and (adj[v] & (1 << u)):
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            current.add(best_v)
            available.discard(best_v)
            # Update available: remove vertices not connected to best_v
            new_available = set()
            for v in available:
                if adj[v] & (1 << best_v):
                    new_available.add(v)
            available = new_available
        
        return current
    
    # Strategy 1: Pure multiples of 3 (baseline)
    s3 = set(i for i, x in enumerate(candidates) if x % 3 == 0)
    if idx_255 in s3:
        best_clique = s3
        best_size = len(s3)
    
    # Strategy 2: Greedy starting from 255
    remaining = set(range(n))
    clique = try_greedy({idx_255}, remaining)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)
    
    # Strategy 3: Multiple randomized greedy restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        # Random subset of candidates to start with (always include 255)
        # Try different starting sets
        # Approach: start with 255, then greedily add from random ordering
        current = {idx_255}
        # Get all vertices connected to 255
        connected_to_255 = set()
        for v in range(n):
            if v != idx_255 and (adj[idx_255] & (1 << v)):
                connected_to_255.add(v)
        
        # Randomize order
        verts_list = list(connected_to_255)
        random.shuffle(verts_list)
        
        available = set(verts_list)
        while available and time.time() < deadline:
            # Try each vertex in random order, add if connects to all current
            added = False
            for v in list(available):
                ok = True
                for u in current:
                    if not (adj[u] & (1 << v)):
                        ok = False
                        break
                if ok:
                    current.add(v)
                    available.discard(v)
                    added = True
                    # Remove from available any vertex not connected to v
                    to_remove = set()
                    for w in available:
                        if not (adj[w] & (1 << v)):
                            to_remove.add(w)
                    available -= to_remove
                    break
            if not added:
                break
        
        if len(current) > best_size:
            best_clique = current
            best_size = len(current)
    
    # Strategy 4: Local search / swap heuristic on the best clique found
    if time.time() < deadline:
        # Try to improve by swapping elements
        current_best = set(best_clique)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # For each element in clique, try removing it and adding a better one
            for v in list(current_best):
                if time.time() > deadline:
                    break
                # Try removing v
                temp = current_best - {v}
                # Find vertices not in temp that connect to all in temp
                for w in range(n):
                    if w in temp:
                        continue
                    ok = True
                    for u in temp:
                        if not (adj[u] & (1 << w)):
                            ok = False
                            break
                    if ok and w not in current_best:
                        # Adding w and removing v keeps same size, but might enable further additions
                        # Actually, let's just try: new_clique = temp | {w}
                        new_clique = temp | {w}
                        # Try to extend new_clique
                        extended = set(new_clique)
                        avail = set()
                        for x in range(n):
                            if x in extended:
                                continue
                            ok2 = True
                            for y in extended:
                                if not (adj[y] & (1 << x)):
                                    ok2 = False
                                    break
                            if ok2:
                                avail.add(x)
                        # Greedy extend
                        while avail:
                            best_x = None
                            best_d = -1
                            for x in avail:
                                d = 0
                                for y in avail:
                                    if x != y and (adj[x] & (1 << y)):
                                        d += 1
                                if d > best_d:
                                    best_d = d
                                    best_x = x
                            if best_x is None:
                                break
                            extended.add(best_x)
                            avail.discard(best_x)
                            new_avail = set()
                            for x in avail:
                                if adj[x] & (1 << best_x):
                                    new_avail.add(x)
                            avail = new_avail
                        
                        if len(extended) > len(current_best):
                            current_best = extended
                            improved = True
                            break
            if improved:
                break
        
        if len(current_best) > best_size:
            best_clique = current_best
            best_size = len(current_best)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    result.sort()
    
    # Verify
    assert 255 in result
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) > 1, f"Failed: gcd({result[i]},{result[j]})=1"
    
    return result
