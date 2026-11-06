"""Combined snake-in-the-box search: Warnsdorff DFS with splicing and bitmask acceleration."""
import time
import random
import sys

N = 9
TIME_LIMIT = 115.0  # Safety margin below 120s evaluator limit
SEED_LENGTH = 104
FULL_SIZE = 1 << N

# Pre-compute neighbors for all vertices in Q_9
NEIGHBORS = [[] for _ in range(FULL_SIZE)]
for v in range(FULL_SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# EVOLVE-BLOCK-START

def _neighbors_mask(v):
    """Return a 512-bit mask with bits set for all N neighbors of v."""
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    return m

def _count_valid_next(v, used_set, exclusion_set):
    """Count how many valid moves are available from vertex v."""
    count = 0
    for w in NEIGHBORS[v]:
        if w not in used_set and w not in exclusion_set:
            count += 1
    return count

def _warnsdorff_dfs(path, used, exclusion, rng, time_budget, max_backtrack_depth=3):
    """
    Perform a DFS with Warnsdorff heuristic and limited backtracking.
    Returns the longest path found.
    """
    best_path = list(path)
    best_len = len(path)
    
    def dfs(current_path, current_used, current_exclusion, depth_remaining):
        nonlocal best_path, best_len
        
        if time.time() > time_budget:
            return
        
        if len(current_path) > best_len:
            best_len = len(current_path)
            best_path = list(current_path)
            
        cur = current_path[-1]
        
        # Generate candidates and score them using Warnsdorff
        candidates = []
        for v in NEIGHBORS[cur]:
            if v in current_used or v in current_exclusion:
                continue
            # Warnsdorff score: number of valid next moves from v
            score = _count_valid_next(v, current_used, current_exclusion)
            candidates.append((score, v))
        
        if not candidates:
            return
        
        # Sort by score (fewest exits first)
        candidates.sort(key=lambda x: x[0])
        
        # Add small random noise to break ties and allow variation
        for i in range(len(candidates)):
            candidates[i] = (candidates[i][0] + rng.random() * 0.1, candidates[i][1])
        candidates.sort(key=lambda x: x[0])
        
        # Limit backtracking depth to avoid explosion
        num_to_try = min(len(candidates), max_backtrack_depth + 1)
        
        for i in range(num_to_try):
            if time.time() > time_budget:
                return
            
            score, v = candidates[i]
            
            # Move to v
            new_used = current_used | {v}
            new_exclusion = set(current_exclusion)
            # Add neighbors of cur to exclusion (cur becomes non-terminal)
            for nb in NEIGHBORS[cur]:
                new_exclusion.add(nb)
            
            new_path = current_path + [v]
            
            dfs(new_path, new_used, new_exclusion, depth_remaining - 1)
            
            # Backtrack: try next candidate if we have budget
            if i < num_to_try - 1 and depth_remaining > 0:
                continue
            else:
                break
        
    # Start DFS
    dfs(path, used, exclusion, max_backtrack_depth)
    
    return best_path

def _fast_greedy_bitmask(rng, time_budget):
    """
    A fast greedy search using bitmask representation for speed.
    """
    path = [0]
    used_mask = 1  # bit 0 set
    exclusion_mask = 0
    
    while time.time() < time_budget:
        cur = path[-1]
        nm = _neighbors_mask(cur)
        valid_mask = nm & ~used_mask & ~exclusion_mask
        
        if valid_mask == 0:
            break
        
        # Extract candidate indices from valid_mask
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Use Warnsdorff-like heuristic: pick candidate with fewest future options
            best_score = float('inf')
            best_cands = []
            for c in candidates:
                # Quick score: count valid neighbors of c
                nm_c = _neighbors_mask(c)
                valid_c = nm_c & ~(used_mask | (1 << c)) & ~exclusion_mask
                score = 0
                vm2 = valid_c
                while vm2:
                    lsb2 = vm2 & (-vm2)
                    score += 1
                    vm2 ^= lsb2
                if score < best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        
        path.append(v)
        used_mask |= (1 << v)
        exclusion_mask |= nm
    
    return path

def _try_splice(path1, path2, rng, time_budget):
    """
    Try to splice two paths together to create a longer path.
    This is a structural improvement over single-run limits.
    """
    # Convert paths to sets for quick lookup
    set1 = set(path1)
    set2 = set(path2)
    
    # Try to find a valid connection between end of path1 and start of path2,
    # or between end of path2 and start of path1
    # For simplicity, try direct concatenation if compatible
    
    # Check if we can connect end of path1 to start of path2
    end1 = path1[-1]
    start2 = path2[0]
    
    # Check if start2 is a valid next move from end1
    used = set(path1)
    exclusion = set()
    for p in path1[:-1]:
        for nb in NEIGHBORS[p]:
            exclusion.add(nb)
    
    if start2 in NEIGHBORS[end1] and start2 not in used and start2 not in exclusion:
        # Try to extend path1 with path2
        combined = list(path1) + list(path2[1:])
        # Validate the combined path
        if _validate_path(combined):
            return combined
    
    # Try reverse: end of path2 to start of path1
    end2 = path2[-1]
    start1 = path1[0]
    
    used2 = set(path2)
    exclusion2 = set()
    for p in path2[:-1]:
        for nb in NEIGHBORS[p]:
            exclusion2.add(nb)
    
    if start1 in NEIGHBORS[end2] and start1 not in used2 and start1 not in exclusion2:
        combined = list(path2) + list(path1[1:])
        if _validate_path(combined):
            return combined
    
    # If direct splice doesn't work, try to find a better connection
    # by searching for a valid extension from the end of one path into the other
    for path_a, path_b in [(path1, path2), (path2, path1)]:
        end_a = path_a[-1]
        used_a = set(path_a)
        exclusion_a = set()
        for p in path_a[:-1]:
            for nb in NEIGHBORS[p]:
                exclusion_a.add(nb)
        
        # Try each neighbor of end_a that's in path_b
        for nb in NEIGHBORS[end_a]:
            if nb in used_a or nb in exclusion_a:
                continue
            # Check if nb is in path_b
            if nb in set(path_b):
                # Find position of nb in path_b
                idx = path_b.index(nb)
                # Take the suffix of path_b starting from nb
                suffix = path_b[idx:]
                # Combine path_a with suffix
                combined = list(path_a) + list(suffix[1:])
                if _validate_path(combined):
                    return combined
    
    return None

def _validate_path(path):
    """Validate that the path is a valid snake (induced path)."""
    if len(path) < 2:
        return True
    
    # Check consecutive vertices differ in exactly one bit
    for i in range(len(path) - 1):
        diff = path[i] ^ path[i + 1]
        if diff == 0 or (diff & (diff - 1)) != 0:
            return False
    
    # Check non-consecutive vertices differ in at least two bits
    used = set()
    for i, v in enumerate(path):
        if v in used:
            return False  # Duplicate vertex
        used.add(v)
        # Check against all non-consecutive vertices
        for j in range(i - 2):
            diff = v ^ path[j]
            if diff == 0 or (diff & (diff - 1)) == 0:
                return False
    
    return True

def _try_extend_bitmask(path, rng, time_budget):
    """
    Try to extend an existing path further using greedy lookahead with bitmasks.
    """
    used_mask = 0
    for p in path:
        used_mask |= (1 << p)
    
    # Build exclusion from non-terminal vertices
    exclusion_mask = 0
    for p in path[:-1]:
        exclusion_mask |= _neighbors_mask(p)
    
    while time.time() < time_budget:
        cur = path[-1]
        nm = _neighbors_mask(cur)
        valid_mask = nm & ~used_mask & ~exclusion_mask
        
        if valid_mask == 0:
            break
        
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Use Warnsdorff-like heuristic
            best_score = float('inf')
            best_cands = []
            for c in candidates:
                nm_c = _neighbors_mask(c)
                valid_c = nm_c & ~(used_mask | (1 << c)) & ~exclusion_mask
                score = 0
                vm2 = valid_c
                while vm2:
                    lsb2 = vm2 & (-vm2)
                    score += 1
                    vm2 ^= lsb2
                if score < best_score:
                    best_score = score
                    best_cands = [c]
                elif score == best_score:
                    best_cands.append(c)
            v = rng.choice(best_cands)
        
        path.append(v)
        used_mask |= (1 << v)
        exclusion_mask |= nm
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    sys.setrecursionlimit(10000)
    
    # First run: deterministic greedy to get baseline using bitmask
    rng_init = random.Random(42)
    path = _fast_greedy_bitmask(rng_init, start_time + 2.0)
    best_path = path
    best_len = len(path)
    
    # Strategy 1: Fast randomized greedy with Warnsdorff heuristic - many restarts
    while time.time() < deadline - 30.0:
        rng = random.Random(int(time.time() * 1000) % 1000000)
        p = _fast_greedy_bitmask(rng, deadline - 29.0)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 2: Warnsdorff DFS with backtracking
    while time.time() < deadline - 15.0:
        rng = random.Random(int(time.time() * 1000) % 1000000)
        # Start from a random vertex
        start_v = rng.randint(0, FULL_SIZE - 1)
        p_init = [start_v]
        used = {start_v}
        exclusion = set()
        
        p = _warnsdorff_dfs(p_init, used, exclusion, rng, deadline - 14.0, max_backtrack_depth=3)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Strategy 3: Splicing - try to combine two good paths
    if best_path is not None and time.time() < deadline - 10.0:
        # Generate another path to try splicing with
        rng = random.Random(12345)
        p2 = _fast_greedy_bitmask(rng, deadline - 9.0)
        
        if len(p2) > 10:
            spliced = _try_splice(list(best_path), list(p2), rng, deadline - 8.0)
            if spliced is not None and len(spliced) > best_len:
                best_len = len(spliced)
                best_path = spliced
    
    # Final: try to extend the best path found
    if best_path is not None and time.time() < deadline - 2.0:
        rng = random.Random(99999)
        extended = _try_extend_bitmask(list(best_path), rng, deadline - 1.0)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    return best_path

# EVOLVE-BLOCK-END
