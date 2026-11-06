"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest snake in Q_9 using Warnsdorff-style backtracking.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_len = 0
    best_path = []
    
    # Precompute neighbors for all vertices
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(512)]
    
    def _warnsdorff_search(start, max_depth=200, time_limit=None):
        """
        Warnsdorff-style backtracking search.
        At each step, prefer the candidate with fewest onward options.
        Backtrack when stuck to try alternative choices.
        Returns the longest path found.
        """
        if time_limit is None:
            time_limit = deadline
        
        best_local = [start]
        
        def _count_options(v, used, blocked, prev):
            """Count how many valid next moves from v."""
            cnt = 0
            for nb in neighbors[v]:
                if nb != prev and nb not in used and nb not in blocked:
                    cnt += 1
            return cnt
        
        def _search(path, used, blocked, depth):
            nonlocal best_len, best_path
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            if len(path) >= max_depth:
                return
            
            if time.time() > time_limit:
                return
            
            cur = path[-1]
            
            # Get candidates
            candidates = []
            for nb in neighbors[cur]:
                if nb not in used and nb not in blocked:
                    candidates.append(nb)
            
            if not candidates:
                return
            
            # Warnsdorff: sort by fewest onward options first
            # But add some randomness to avoid getting stuck in patterns
            scored = []
            for cand in candidates:
                opts = _count_options(cand, used, blocked, cur)
                scored.append((opts, random.random(), cand))
            
            # Sort by (fewest options, random tiebreak)
            scored.sort(key=lambda x: (x[0], x[1]))
            
            # Try candidates in order, with backtracking
            for opts, _, cand in scored:
                if time.time() > time_limit:
                    return
                
                # Try this candidate
                path.append(cand)
                used.add(cand)
                
                # Update blocked: all neighbors of cur except cand become blocked
                old_blocked = set()
                for nb in neighbors[cur]:
                    if nb != cand and nb not in used:
                        old_blocked.add(nb)
                        blocked.add(nb)
                
                _search(path, used, blocked, depth + 1)
                
                # Backtrack
                path.pop()
                used.discard(cand)
                for nb in old_blocked:
                    blocked.discard(nb)
                
                # If we found a very long path, stop exploring
                if len(path) >= max_depth:
                    return
            
            # If we exhausted all candidates at this depth, try limited backtracking
            # This is implicit in the recursive structure
        
        path = [start]
        used = {start}
        blocked = set()
        
        _search(path, used, blocked, 0)
        
        return best_path
    
    def _greedy_extend(start):
        """
        Fast greedy extension using O(1) blocked set updates.
        Used for quick restarts to find decent baselines.
        """
        path = [start]
        used = {start}
        blocked = set()
        
        while time.time() < deadline:
            cur = path[-1]
            candidates = []
            for nb in neighbors[cur]:
                if nb not in used and nb not in blocked:
                    candidates.append(nb)
            
            if not candidates:
                break
            
            # Warnsdorff: pick candidate with fewest onward options
            if len(candidates) > 1:
                best_cand = None
                best_score = float('inf')
                for cand in candidates:
                    score = 0
                    for nb in neighbors[cand]:
                        if nb != cur and nb not in used and nb not in blocked:
                            score += 1
                    if score < best_score:
                        best_score = score
                        best_cand = cand
                nxt = best_cand
            else:
                nxt = candidates[0]
            
            path.append(nxt)
            used.add(nxt)
            
            # Update blocked set
            for nb in neighbors[cur]:
                if nb != nxt:
                    blocked.add(nb)
        
        return path
    
    # Phase 1: Quick greedy restarts to establish a baseline
    while time.time() < deadline - 30:
        start = random.randint(0, 511)
        path = _greedy_extend(start)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 2: Warnsdorff backtracking from promising starts
    # Use the best path found so far as a guide for where to search
    if best_path:
        # Try Warnsdorff from the start of the best path
        warnsdorff_result = _warnsdorff_search(best_path[0], max_depth=best_len + 30)
        if len(warnsdorff_result) > best_len:
            best_len = len(warnsdorff_result)
            best_path = warnsdorff_result
    
    # Phase 3: More Warnsdorff searches with different starts
    while time.time() < deadline - 5:
        start = random.randint(0, 511)
        # Use a moderate max_depth to allow backtracking
        max_d = min(250, best_len + 50)
        warnsdorff_result = _warnsdorff_search(start, max_depth=max_d)
        
        if len(warnsdorff_result) > best_len:
            best_len = len(warnsdorff_result)
            best_path = list(warnsdorff_result)
    
    return best_path
# EVOLVE-BLOCK-END
