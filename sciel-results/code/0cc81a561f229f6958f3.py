"""Snake-in-the-box in Q_9: bidirectional growth with Warnsdorff's rule + backtracking."""
import time
import random
import numpy as np

N = 9
SEED_TIME = 150.0  # evaluator wall-clock kill; keep margin
DEADLINE = SEED_TIME - 20.0  # 130s safety margin
T0 = time.time()

def _elapsed():
    return time.time() - T0

def _deadline_hit():
    return _elapsed() >= DEADLINE

def _is_valid_extension(v, path, used_set, end_index):
    """Check if v can be appended at position end_index (0 or 1).
    end_index=0 means prepend, end_index=1 means append.
    path is a list of current path vertices.
    Fast Python loop with early exit."""
    if v in used_set:
        return False
    # v must differ in >=2 bits from all path vertices except the one it connects to
    if end_index == 1:
        # Connecting vertex is path[-1]; check all others
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
    else:
        # Connecting vertex is path[0]; check all others
        for p in path[1:]:
            if (v ^ p).bit_count() == 1:
                return False
    return True

def _count_safe_options(v, path, used_set):
    """Count how many neighbors of v are still safe to extend (Warnsdorff heuristic).
    Fewer safe options = more urgent to place this vertex."""
    count = 0
    for b in range(N):
        u = v ^ (1 << b)
        if u in used_set:
            continue
        # Check u against all path vertices
        ok = True
        for p in path:
            if (u ^ p).bit_count() == 1:
                ok = False
                break
        if ok:
            count += 1
    return count

def _bidirectional_grow(start_vertex, bit_order, rng, max_backtrack_depth=3):
    """Grow a snake bidirectionally from start_vertex using Warnsdorff's rule with backtracking."""
    path = [start_vertex]
    used = {start_vertex}
    
    # Alternate growing left and right
    grow_left = False  # False means grow right (append)
    
    # Backtracking state: list of (path_snapshot, used_snapshot, grow_left) for undo
    backtrack_stack = []
    backtrack_depth = 0
    max_consecutive_fails = 0
    
    while not _deadline_hit():
        # Decide which end to grow
        if grow_left:
            cur = path[0]
            end_index = 0
        else:
            cur = path[-1]
            end_index = 1
        
        # Find all valid neighbors
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            if not _is_valid_extension(v, path, used, end_index):
                continue
            candidates.append(v)
        
        if not candidates:
            # Stuck: try backtracking
            if backtrack_stack and backtrack_depth < max_backtrack_depth:
                # Pop the last saved state
                prev_path, prev_used, prev_grow_left, tried_set = backtrack_stack.pop()
                backtrack_depth -= 1
                max_consecutive_fails = 0
                
                # Mark current vertex as tried at this position
                tried_set.add((cur, end_index))
                
                # Restore state
                path = prev_path
                used = prev_used
                grow_left = prev_grow_left
                
                # Find alternative candidates excluding the one we just tried
                alt_candidates = []
                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v in used:
                        continue
                    if (cur, end_index) in tried_set:
                        # Skip the vertex we already tried from this position
                        if v in tried_set:
                            continue
                    if not _is_valid_extension(v, path, used, end_index):
                        continue
                    alt_candidates.append(v)
                
                if alt_candidates:
                    # Warnsdorff: pick candidate with fewest safe options
                    scored = []
                    for v in alt_candidates:
                        opts = _count_safe_options(v, path, used)
                        scored.append((opts, rng.random(), v))
                    scored.sort(key=lambda x: (x[0], x[1]))
                    v = scored[0][2]
                    
                    if end_index == 1:
                        path.append(v)
                    else:
                        path.insert(0, v)
                    used.add(v)
                    
                    grow_left = not grow_left
                    continue
                else:
                    # No alternatives, continue backtracking
                    continue
            else:
                # No more backtracking possible
                break
        
        # Warnsdorff: pick candidate with fewest safe options
        # Shuffle ties randomly for diversity
        scored = []
        for v in candidates:
            opts = _count_safe_options(v, path, used)
            scored.append((opts, rng.random(), v))
        scored.sort(key=lambda x: (x[0], x[1]))
        
        v = scored[0][2]
        
        # Save state for potential backtracking (only if we have multiple candidates)
        if len(candidates) > 1 and backtrack_depth < max_backtrack_depth:
            backtrack_stack.append((path[:], used.copy(), grow_left, set()))
            backtrack_depth += 1
        
        if end_index == 1:
            path.append(v)
        else:
            path.insert(0, v)
        used.add(v)
        
        grow_left = not grow_left
        max_consecutive_fails = 0
    
    return path

def _validate(path):
    """Validate that path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return False
    seen = set(path)
    if len(seen) != n:
        return False
    for i in range(n):
        if path[i] < 0 or path[i] >= (1 << N):
            return False
    for i in range(n - 1):
        d = (path[i] ^ path[i+1]).bit_count()
        if d != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def construct():
    global T0
    T0 = time.time()
    
    best_path = None
    best_len = 0
    
    # First: run the original greedy to establish baseline
    path_greedy, used_greedy = [0], {0}
    while not _deadline_hit():
        cur = path_greedy[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_greedy:
                continue
            # Check against all previous vertices
            ok = True
            for p in path_greedy[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            break
        path_greedy.append(nxt)
        used_greedy.add(nxt)
    
    if len(path_greedy) > best_len:
        best_len = len(path_greedy)
        best_path = path_greedy[:]
    
    # Now run multiple bidirectional growth attempts with different seeds
    rng = random.Random(42)
    
    # Try different starting vertices and bit orders
    start_vertices = [0, (1 << N) - 1]  # 0 and all-ones
    # Add some random starts
    for _ in range(4):
        start_vertices.append(rng.randint(0, (1 << N) - 1))
    
    attempts = 0
    for sv in start_vertices:
        if _deadline_hit():
            break
        # Try multiple bit orderings
        for trial in range(8):
            if _deadline_hit():
                break
            bit_order = list(range(N))
            rng.shuffle(bit_order)
            
            path = _bidirectional_grow(sv, bit_order, rng)
            if len(path) > best_len:
                # Validate before accepting
                if _validate(path):
                    best_len = len(path)
                    best_path = path[:]
            attempts += 1
    
    # If we have time, do more randomized restarts
    while not _deadline_hit():
        sv = rng.randint(0, (1 << N) - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path = _bidirectional_grow(sv, bit_order, rng)
        if len(path) > best_len:
            if _validate(path):
                best_len = len(path)
                best_path = path[:]
        attempts += 1
        if attempts > 5000:  # increased safety cap
            break
    
    # Final safety: if deadline is very close, return immediately
    if best_path is None:
        return [0]
    
    # Final validation
    if not _validate(best_path):
        # Fallback to greedy
        return path_greedy
    
    return best_path
