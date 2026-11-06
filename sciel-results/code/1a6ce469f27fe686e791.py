"""Snake-in-the-box in the 9-dimensional hypercube - ILS with Min-Next-Next Heuristic."""
N = 9
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Precompute neighbors as lists of ints for fast access
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    bit_masks = [1 << b for b in range(N)]
    
    def _get_score(v, forbidden_mask, neighbors_v):
        """
        Calculate the Min-Next-Next score for candidate v.
        Returns a tuple (min_onward, total_onward) where higher is better.
        min_onward: The minimum number of valid next moves among v's valid neighbors.
                   If v has no valid neighbors, this is -1.
                   If a neighbor u has no valid next moves, this is 0.
        total_onward: Sum of valid next moves among v's valid neighbors (tie-breaker).
        """
        new_forbidden = forbidden_mask | (1 << v)
        
        valid_nexts = []
        for u in neighbors_v:
            if not (new_forbidden & (1 << u)):
                valid_nexts.append(u)
                
        if not valid_nexts:
            return (-1, 0)
            
        min_onward = float('inf')
        total_onward = 0
        
        for u in valid_nexts:
            # Count valid moves from u
            onward_count = 0
            for w in neighbors[u]:
                if not (new_forbidden & (1 << w)):
                    # Note: we don't mark v as forbidden for w's neighbors in this simple check?
                    # Actually, if we go v->u->w, then v is used.
                    # The forbidden mask for w's next step would include v and u.
                    # But for a quick heuristic, we just check if w is not in current new_forbidden.
                    # This is an approximation.
                    onward_count += 1
            total_onward += onward_count
            if onward_count < min_onward:
                min_onward = onward_count
                
        if min_onward == float('inf'):
            return (-1, 0)
            
        return (min_onward, total_onward)

    def _greedy_construct(start_vertex, rng, temperature=0.0, max_time=None):
        """
        Construct a path using greedy selection with Min-Next-Next heuristic.
        temperature: 0.0 = deterministic (best score), >0 = random probability.
        """
        if max_time is None:
            max_time = deadline - 1.0
            
        path = [start_vertex]
        forbidden_mask = (1 << start_vertex)
        
        while time.time() < max_time:
            cur = path[-1]
            candidates = []
            
            for v in neighbors[cur]:
                if forbidden_mask & (1 << v):
                    continue
                score = _get_score(v, forbidden_mask, neighbors[v])
                candidates.append((score, v))
                
            if not candidates:
                break
                
            # Sort by score (descending). Higher min_onward is better.
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Selection
            best_score = candidates[0][0]
            if temperature <= 0.0:
                # Always pick best
                _, v = candidates[0]
            else:
                # Softmax-like selection or random among top-k
                # Simple: pick best with prob (1-temp), else random from top half
                if rng.random() < (1 - temperature):
                    _, v = candidates[0]
                else:
                    # Pick from top 50% or random from all
                    k = max(1, len(candidates) // 2)
                    _, v = candidates[rng.randint(0, k - 1)][1]
                    
            path.append(v)
            forbidden_mask |= (1 << v)
            
        return path

    def _try_extend(path, deadline_limit, rng, direction=1):
        """Try to extend the path from the end (direction=1) or start (direction=-1)."""
        if not path:
            return path
            
        # We need to reconstruct forbidden mask if extending from start?
        # It's easier to just re-greedy from the end of the current path.
        # If direction == -1, we reverse, extend, and reverse back.
        
        if direction == -1:
            path = path[::-1]
            
        cur = path[-1]
        forbidden_mask = 0
        for v in path:
            forbidden_mask |= (1 << v)
        for v in path[:-1]:
            forbidden_mask |= (1 << neighbors[v][0]) # Placeholder, need proper mask
            
        # Reconstruct forbidden mask properly
        forbidden_mask = 0
        for v in path:
            forbidden_mask |= (1 << v)
        for v in path[:-1]:
            for nb in neighbors[v]:
                forbidden_mask |= (1 << nb)
                
        extended = False
        while time.time() < deadline_limit:
            candidates = []
            for v in neighbors[cur]:
                if forbidden_mask & (1 << v):
                    continue
                score = _get_score(v, forbidden_mask, neighbors[v])
                candidates.append((score, v))
                
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, v = candidates[0]
            
            path.append(v)
            forbidden_mask |= (1 << v)
            cur = v
            extended = True
            
        if direction == -1:
            path = path[::-1]
            
        return path

    best_path = None
    best_len = 0
    rng = random.Random(42)
    
    # Phase 1: Multi-start Greedy with different temperatures
    starts = [0, 1, (1 << N) - 1, 0x155, 0x1AA, 0x0FF, 0x1F0, 0x123, 0x1AB]
    for _ in range(50):
        starts.append(rng.randint(0, (1 << N) - 1))
        
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    for sv in starts:
        if time.time() > deadline - 10:
            break
        for temp in temperatures:
            if time.time() > deadline - 5:
                break
            path = _greedy_construct(sv, rng, temp)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
    # Phase 2: Iterated Local Search (Perturb and Re-Optimize)
    if best_path:
        # We have a good base. Let's try to improve it.
        # Strategy: Truncate a random amount from the end, then re-greedy.
        # Or: Break a random edge and reconnect? Hard.
        # Let's do: Truncate to L-k, then greedy extend.
        
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Perturbation: truncate
            truncate_amount = rng.randint(1, 10)
            if len(best_path) <= truncate_amount:
                continue
                
            prefix = best_path[:-truncate_amount]
            
            # Re-greedy from the end of prefix
            # We need to pass the state or just let _greedy_construct handle it?
            # _greedy_construct starts from scratch. We need a function that extends an existing path.
            # Let's write a helper to extend an existing path using the heuristic.
            
            def _extend_existing(path, rng, temp=0.0):
                cur = path[-1]
                forbidden_mask = 0
                for v in path:
                    forbidden_mask |= (1 << v)
                for v in path[:-1]:
                    for nb in neighbors[v]:
                        forbidden_mask |= (1 << nb)
                        
                while time.time() < deadline - 2:
                    candidates = []
                    for v in neighbors[cur]:
                        if forbidden_mask & (1 << v):
                            continue
                        score = _get_score(v, forbidden_mask, neighbors[v])
                        candidates.append((score, v))
                        
                    if not candidates:
                        break
                        
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    
                    if temp <= 0.0:
                        _, v = candidates[0]
                    else:
                        if rng.random() < (1 - temp):
                            _, v = candidates[0]
                        else:
                            k = max(1, len(candidates) // 2)
                            _, v = candidates[rng.randint(0, k - 1)][1]
                            
                    path.append(v)
                    forbidden_mask |= (1 << v)
                    cur = v
                return path

            new_path = _extend_existing(prefix[:], rng, 0.1)
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = new_path
                
            # Also try extending from the beginning (by reversing)
            rev_prefix = prefix[::-1]
            new_path_rev = _extend_existing(rev_prefix[:], rng, 0.1)
            if len(new_path_rev) > best_len:
                best_len = len(new_path_rev)
                best_path = new_path_rev[::-1]

    # Final check
    if best_path is None or len(best_path) < 2:
        best_path = [0, 1]
        
    return best_path
