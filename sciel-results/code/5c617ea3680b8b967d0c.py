"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest snake in Q_9.
    Combines fast blocked-set greedy with limited backtracking polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_len = 0
    best_path = []
    
    # Precompute neighbors for all vertices to speed up iteration
    # neighbors[v] is a list of 9 integers
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(512)]
    
    def _greedy_run(start, bit_order=None):
        """
        Fast greedy snake construction using blocked set.
        Returns the path found.
        """
        if bit_order is None:
            bit_order = list(range(N))
        
        path = [start]
        used = {start}
        blocked = set()
        cur = start
        
        while True:
            if time.time() > deadline:
                return path
                
            candidates = []
            for b in bit_order:
                nb = cur ^ (1 << b)
                if nb not in used and nb not in blocked:
                    candidates.append(nb)
            
            if not candidates:
                return path
            
            # Pick a candidate randomly for diversity
            nxt = random.choice(candidates)
            
            path.append(nxt)
            used.add(nxt)
            
            # Update blocked set: all neighbors of cur except nxt become blocked
            for nb in neighbors[cur]:
                if nb != nxt:
                    blocked.add(nb)
                    
            cur = nxt
            
    def _backtrack_polish(path, depth_limit=50):
        """
        Limited backtracking to try to extend the path.
        Uses the blocked-set logic implicitly via candidate checking.
        Returns the best path found.
        """
        best_path = list(path)
        
        if time.time() > deadline:
            return best_path
            
        cur = path[-1]
        used = set(path)
        
        # We need to reconstruct the 'blocked' set for the current path state
        # to do fast candidate checks, or just use the slow check.
        # Given the path is long (~120+), the slow check is O(L).
        # Let's reconstruct blocked set for speed.
        blocked = set()
        for i in range(len(path) - 1):
            v = path[i]
            for nb in neighbors[v]:
                if nb not in path[i+1]: # Actually, just add all neighbors of non-consecutive
                    # Wait, the blocked set logic: 
                    # When we added path[i], we blocked neighbors of path[i-1] except path[i].
                    # Easier: just compute valid candidates from scratch using the slow check 
                    # for the polish step, as it's only called a few times.
                    pass
        
        # Simpler approach for polish: just check candidates against the whole path
        # This is O(L) per candidate, which is fine for a few steps.
        
        def _get_valid_candidates(current_vertex, current_path, current_used):
            cands = []
            for nb in neighbors[current_vertex]:
                if nb in current_used:
                    continue
                # Check distance >= 2 from all non-consecutive vertices
                # Non-consecutive means indices 0 to len(current_path)-2
                valid = True
                for i in range(len(current_path) - 1):
                    if (nb ^ current_path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    cands.append(nb)
            return cands

        # Recursive backtracking
        def _search(path, used, depth):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = list(path)
            if depth >= depth_limit:
                return
            
            cur = path[-1]
            cands = _get_valid_candidates(cur, path, used)
            random.shuffle(cands)
            
            for v in cands:
                if time.time() > deadline:
                    return
                path.append(v)
                used.add(v)
                _search(path, used, depth + 1)
                path.pop()
                used.remove(v)
                
        _search(path, used, 0)
        return best_path

    # Phase 1: Run many fast greedy instances
    # We want to maximize the number of restarts.
    # Each greedy run is very fast.
    
    # First, try a few deterministic starts to establish a baseline quickly
    for start in [0, 1, 255, 256]:
        p = _greedy_run(start)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    # Main loop: random restarts
    # Estimate time per run. 
    # Let's just run until deadline.
    
    while time.time() < deadline - 10.0:  # Leave 10s for polish
        start = random.randint(0, 511)
        # Random bit order to diversify
        bo = list(range(N))
        random.shuffle(bo)
        
        p = _greedy_run(start, bo)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    # Phase 2: Polish the best path found with backtracking
    if best_path and time.time() < deadline - 1.0:
        # Try to extend the best path
        polished = _backtrack_polish(best_path, depth_limit=20)
        if len(polished) > best_len:
            best_len = len(polished)
            best_path = polished
            
        # Also try polishing a few other good paths if we have time?
        # For now, just polish the best one.
        
    return best_path
# EVOLVE-BLOCK-END
