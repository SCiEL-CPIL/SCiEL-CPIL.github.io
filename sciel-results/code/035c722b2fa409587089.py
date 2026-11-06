"""Snake-in-the-box in Q_9: greedy + backtracking + local search."""
import time
import numpy as np
import random

N = 9
VERTEX_COUNT = 1 << N
bit_masks = [1 << b for b in range(N)]

# Precompute popcount for all possible XOR values (0 to 2^9-1)
_popcount = np.zeros(VERTEX_COUNT, dtype=np.int8)
for i in range(VERTEX_COUNT):
    _popcount[i] = i.bit_count()

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)
    
    def get_candidates(cur, used, path):
        """Get valid next vertices from cur.
        A vertex v is valid if:
        - v not in used
        - for all path[i] where i < len(path)-1 (non-adjacent to cur), popcount(cur^v XOR path[i]) != 1
          Actually: v must not be at Hamming distance 1 from any path vertex except the last one (cur).
        """
        candidates = []
        path_len = len(path)
        # Non-adjacent vertices are path[0..path_len-2]
        non_adj = path[:path_len-1]
        
        for b in range(N):
            v = cur ^ bit_masks[b]
            if v in used:
                continue
            # Check against all non-adjacent path vertices
            valid = True
            for u in non_adj:
                if _popcount[v ^ u] == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        return candidates
    
    def count_future_options(v, used, path):
        """Count how many options v would leave for the next step."""
        new_used = used | {v}
        # Non-adjacent to v are path[0..len(path)-1] (all of path, since v is new)
        # Actually when we add v, the non-adjacent set for the NEXT vertex w is path + [v] minus the last element
        # For counting options from v: candidates w must not conflict with path[0..len(path)-1] (all existing path vertices)
        # because v is adjacent to w, so w can be at distance 1 from v (that's the edge), but not from any other path vertex.
        count = 0
        for b in range(N):
            w = v ^ bit_masks[b]
            if w in new_used:
                continue
            valid = True
            for u in path:
                if _popcount[w ^ u] == 1:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    def backtrack(path, used, target_len):
        """Try to extend path to target_len using backtracking with fail-first heuristic."""
        if time.time() > deadline:
            return False
        if len(path) >= target_len:
            return True
        
        cur = path[-1]
        candidates = get_candidates(cur, used, path)
        
        if not candidates:
            return False
        
        # Fail-first: sort by fewest future options
        scored = []
        for v in candidates:
            opts = count_future_options(v, used, path)
            scored.append((opts, v))
        scored.sort()
        
        for opts, v in scored:
            path.append(v)
            used.add(v)
            if backtrack(path, used, target_len):
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
            candidates = get_candidates(cur, used, path)
            if not candidates:
                break
            
            if randomize_prob > 0 and rng.random() < randomize_prob:
                v = rng.choice(candidates)
            else:
                # Fail-first: pick candidate with fewest future options
                best_v = min(candidates, key=lambda v: count_future_options(v, used, path))
                v = best_v
            
            path.append(v)
            used.add(v)
    
    def local_search_extend(path, used, max_iterations=20):
        """Try to extend by backtracking from different positions."""
        best_len = len(path)
        best_path = list(path)
        
        for start_pos in range(max(0, len(path) - 15), len(path)):
            if time.time() > deadline - 5:
                break
            
            truncated_path = path[:start_pos + 1]
            truncated_used = set(truncated_path)
            
            target = best_len + 1
            if backtrack(truncated_path, truncated_used, target):
                if len(truncated_path) > best_len:
                    best_len = len(truncated_path)
                    best_path = list(truncated_path)
                    path[:] = truncated_path
                    used.clear()
                    used.update(truncated_used)
        
        return best_path, best_len
    
    def validate(path):
        if len(path) < 2:
            return True
        n = len(path)
        for i in range(n - 1):
            if _popcount[path[i] ^ path[i+1]] != 1:
                return False
        for i in range(n):
            for j in range(i+2, n):
                if _popcount[path[i] ^ path[j]] == 1:
                    return False
        return True
    
    # Phase 1: Greedy extension with fail-first
    path = [0]
    used = {0}
    greedy_extend(path, used)
    best_path = list(path)
    best_len = len(path)
    
    # Phase 2: Backtracking extension from the end
    target_len = best_len + 1
    while target_len <= best_len + 40 and time.time() < deadline - 10:
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
        (0.15, "mild_random"),
        (0.3, "moderate_random"),
        (0.5, "high_random"),
        (0.7, "very_high_random"),
    ]
    
    for restart in range(25):
        if time.time() > deadline - 5:
            break
        
        rand_prob = restart_strategies[restart % len(restart_strategies)][0]
        
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
    if not validate(best_path):
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            candidates = get_candidates(cur, used, path)
            if not candidates:
                break
            path.append(candidates[0])
            used.add(candidates[0])
        best_path = path
    
    return best_path
