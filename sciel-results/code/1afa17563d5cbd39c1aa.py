"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TIME_LIMIT = 145  # safety margin below 150s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_path = None
    best_len = 0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return TIME_LIMIT - elapsed()

    def get_neighbors(v):
        """Generate all neighbors of v in Q_N."""
        for b in range(N):
            yield v ^ (1 << b)

    def compute_forbidden(path):
        """Compute forbidden set from a path: neighbors of all non-terminal vertices."""
        forbidden = set()
        for i in range(len(path) - 1):
            v = path[i]
            for nb in get_neighbors(v):
                forbidden.add(nb)
        return forbidden

    def warnsdorff_count(v, used, forbidden):
        """Count how many valid extensions v would have (Warnsdorff heuristic)."""
        count = 0
        for b in range(N):
            v2 = v ^ (1 << b)
            if v2 not in used and v2 not in forbidden:
                count += 1
        return count

    def try_warnsdorff_backtrack(start_vertex, max_depth=200, backtrack_limit=5000):
        """
        Warnsdorff heuristic with limited backtracking.
        At each step, pick the neighbor with fewest remaining extensions (Warnsdorff rule).
        If stuck, backtrack up to backtrack_limit times to try alternative moves.
        """
        path = [start_vertex]
        used = {start_vertex}
        forbidden = set()
        
        # Stack of (path_copy, used_copy, forbidden_copy, next_candidates) for backtracking
        # To save memory/time, we'll do iterative deepening with a simple backtracking stack
        # Each stack entry: (vertex, list_of_remaining_candidates, used_snapshot, forbidden_snapshot)
        
        # Simpler approach: recursive with manual stack, but limit total backtracks
        backtrack_count = 0
        max_backtracks = backtrack_limit
        
        # We'll use a stack-based approach
        # Stack entry: (current_vertex, candidates_list, used_set, forbidden_set)
        # But copying sets is expensive. Let's use a different approach.
        
        # Approach: Keep the main path, and when stuck, backtrack by popping the last few
        # vertices and trying alternative moves.
        
        # Let's do a simpler but effective approach:
        # 1. Greedy Warnsdorff forward
        # 2. When stuck, if we have time, try to "repair" by backtracking a few steps
        
        cur = start_vertex
        steps = 0
        max_steps = 512  # theoretical max is 512 vertices
        
        while steps < max_steps and time_left() > 2.0:
            cur = path[-1]
            
            # Find all valid candidates
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    # Warnsdorff score: how many extensions would v have?
                    wc = warnsdorff_count(v, used, forbidden)
                    candidates.append((wc, random.random(), v))  # random tiebreaker
            
            if not candidates:
                # Stuck! Try to backtrack
                if len(path) > 1 and backtrack_count < max_backtracks:
                    # Pop last vertex and try alternative
                    backtrack_count += 1
                    # Remove last vertex from path
                    last_v = path.pop()
                    used.discard(last_v)
                    # Recompute forbidden (expensive but necessary for correctness)
                    forbidden = compute_forbidden(path)
                    # Now try to find a different move from the new last vertex
                    new_cur = path[-1]
                    alt_candidates = []
                    for b in range(N):
                        v = new_cur ^ (1 << b)
                        if v not in used and v not in forbidden:
                            wc = warnsdorff_count(v, used, forbidden)
                            alt_candidates.append((wc, random.random(), v))
                    
                    if alt_candidates:
                        # Sort by Warnsdorff count (fewest first - "most constrained first")
                        alt_candidates.sort(key=lambda x: (x[0], x[1]))
                        # Try a few alternatives
                        tried = 0
                        for wc, _, v in alt_candidates[:3]:
                            if tried > 0:
                                break
                            path.append(v)
                            used.add(v)
                            # Update forbidden: remove old forbidden related to last_v, add new
                            # Simpler: recompute forbidden
                            forbidden = compute_forbidden(path)
                            cur = v
                            steps += 1
                            tried += 1
                        continue
                    else:
                        # No alternatives, continue backtracking (pop more)
                        continue
                else:
                    break
            
            # Sort by Warnsdorff count (fewest extensions first - "most constrained first")
            candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Pick the most constrained (fewest extensions), with some randomization
            # Sometimes pick from top 2-3 for diversity
            if random.random() < 0.7:
                chosen = candidates[0][2]
            else:
                top_k = min(3, len(candidates))
                chosen = random.choice([c[2] for c in candidates[:top_k]])
            
            path.append(chosen)
            used.add(chosen)
            # Update forbidden: add neighbors of the previous last vertex
            prev_last = path[-2]
            for nb in get_neighbors(prev_last):
                forbidden.add(nb)
            # Remove chosen from forbidden if it was there (it shouldn't be, but just in case)
            forbidden.discard(chosen)
            steps += 1
        
        return path

    def greedy_extend(start_v):
        """Simple greedy extension for baseline."""
        path = [start_v]
        used = {start_v}
        forbidden = set()
        while time_left() > 2.0:
            cur = path[-1]
            found = False
            # Try bits in random order
            bits = list(range(N))
            random.shuffle(bits)
            for b in bits:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    # Get a baseline with greedy
    baseline = greedy_extend(0)
    best_path = baseline
    best_len = len(baseline)
    
    # Now run Warnsdorff backtracking with multiple random starts
    attempts = 0
    while time_left() > 5.0:
        attempts += 1
        sv = random.randint(0, 2**N - 1)
        
        # Vary the backtrack limit based on time remaining
        remaining = time_left()
        if remaining > 60:
            bt_limit = 10000
        elif remaining > 30:
            bt_limit = 5000
        else:
            bt_limit = 2000
        
        p = try_warnsdorff_backtrack(sv, max_depth=512, backtrack_limit=bt_limit)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    return best_path
# EVOLVE-BLOCK-END
