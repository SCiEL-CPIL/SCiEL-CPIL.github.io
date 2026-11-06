"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from array import array

N = 9

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)
    
    # Precompute bit positions for speed
    bit_masks = [1 << b for b in range(N)]
    
    # Precompute popcount for all 9-bit values (0 to 511)
    popcount_table = bytearray(512)
    for i in range(512):
        popcount_table[i] = bin(i).count('1')
    
    def get_candidates_fast(cur, used, path):
        """Get valid next vertices from cur using precomputed popcount table."""
        candidates = []
        # Use list for path[:-1] to avoid numpy overhead
        path_nonadj = path[:-1]
        
        for b in range(N):
            v = cur ^ bit_masks[b]
            if v in used:
                continue
            # Conflict check using precomputed popcount
            conflict = False
            for p in path_nonadj:
                if popcount_table[p ^ v] == 1:
                    conflict = True
                    break
            if not conflict:
                candidates.append(v)
        return candidates
    
    def count_future_options_fast(v, used, path):
        """Count how many options v would leave for the next step."""
        new_used = used | {v}
        count = 0
        path_nonadj = path  # all current path vertices are non-adjacent to w
        
        for b in range(N):
            w = v ^ bit_masks[b]
            if w in new_used:
                continue
            # Check if w conflicts with any vertex in path
            conflict = False
            for p in path_nonadj:
                if popcount_table[p ^ w] == 1:
                    conflict = True
                    break
            if not conflict:
                count += 1
        return count
    
    def backtrack(path, used, depth_limit, max_depth=50):
        """Constraint-based backtracking with fail-first heuristic and time limit."""
        if time.time() > deadline:
            return False
        if len(path) >= depth_limit:
            return True
        
        cur = path[-1]
        candidates = get_candidates_fast(cur, used, path)
        
        if not candidates:
            return False
        
        # Fail-first: sort candidates by future options (ascending)
        scored = []
        for v in candidates:
            opts = count_future_options_fast(v, used, path)
            scored.append((opts, v))
        scored.sort()
        
        for opts, v in scored:
            path.append(v)
            used.add(v)
            if backtrack(path, used, depth_limit):
                return True
            path.pop()
            used.remove(v)
            if time.time() > deadline:
                return False
        return False
    
    def greedy_extend(path, used, randomize_prob=0.0):
        """Greedy extension with optional randomization."""
        while True:
            if time.time() > deadline:
                break
            cur = path[-1]
            candidates = get_candidates_fast(cur, used, path)
            if not candidates:
                break
            
            if randomize_prob > 0 and rng.random() < randomize_prob:
                v = rng.choice(candidates)
            else:
                # Fail-first: pick candidate with fewest future options
                best_v = min(candidates, key=lambda v: count_future_options_fast(v, used, path))
                v = best_v
            
            path.append(v)
            used.add(v)
    
    def local_search_extend(path, used, max_iterations=100):
        """Try to extend by backtracking from different positions."""
        best_len = len(path)
        best_path = list(path)
        
        for start_pos in range(max(0, len(path) - 10), len(path)):
            if time.time() > deadline - 5:
                break
            
            # Truncate path to start_pos
            truncated_path = path[:start_pos + 1]
            truncated_used = set(truncated_path)
            
            # Try to extend from here with backtracking
            target = best_len + 1
            if backtrack(truncated_path, truncated_used, target):
                if len(truncated_path) > best_len:
                    best_len = len(truncated_path)
                    best_path = list(truncated_path)
                    path[:] = truncated_path
                    used.clear()
                    used.update(truncated_used)
        
        return best_path, best_len
    
    # Phase 1: Greedy extension with fail-first
    path = [0]
    used = {0}
    greedy_extend(path, used)
    best_path = list(path)
    best_len = len(path)
    
    # Phase 2: Backtracking extension from the end
    target_len = best_len + 1
    while target_len <= best_len + 30 and time.time() < deadline - 10:
        if backtrack(path, used, target_len):
            best_path = list(path)
            best_len = len(path)
            target_len = best_len + 1
        else:
            break
    
    # Phase 3: Local search - try extending from different positions
    if time.time() < deadline - 15:
        best_path, best_len = local_search_extend(path, used)
    
    # Phase 4: Multiple diverse random restarts
    restart_strategies = [
        (0.0, "pure_greedy"),
        (0.2, "mild_random"),
        (0.4, "moderate_random"),
        (0.6, "high_random"),
    ]
    
    for restart in range(15):
        if time.time() > deadline - 5:
            break
        
        # Cycle through different randomization levels
        rand_prob, _ = restart_strategies[restart % len(restart_strategies)]
        
        path = [0]
        used = {0}
        greedy_extend(path, used, randomize_prob=rand_prob)
        
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Try backtracking extension from this point
        if time.time() < deadline - 10:
            target = best_len + 1
            if backtrack(path, used, target):
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
        
        # Try local search from this point
        if time.time() < deadline - 10 and len(path) >= best_len - 5:
            best_path, best_len = local_search_extend(path, used)
    
    # Phase 5: Final validation
    def validate(path):
        if len(path) < 2:
            return True
        n = len(path)
        # Check consecutive differ by exactly 1 bit
        for i in range(n - 1):
            if popcount_table[path[i] ^ path[i+1]] != 1:
                return False
        # Check non-consecutive differ by at least 2 bits
        for i in range(n):
            for j in range(i+2, n):
                if popcount_table[path[i] ^ path[j]] == 1:
                    return False
        return True
    
    if not validate(best_path):
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            candidates = get_candidates_fast(cur, used, path)
            if not candidates:
                break
            path.append(candidates[0])
            used.add(candidates[0])
        best_path = path
    
    return best_path
# EVOLVE-BLOCK-END
