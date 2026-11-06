import time
import random
import sys

N = 9
MAX_TIME = 115.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Precompute neighbors for all 512 vertices
    all_vertices = list(range(1 << N))
    neighbors = [[] for _ in range(1 << N)]
    for v in all_vertices:
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute bit_count for XOR results (0 to 511)
    bit_count_table = [i.bit_count() for i in range(1 << N)]
    
    def is_valid_extension(path, used_mask, new_v):
        """Check if new_v can be appended to path.
        path is a list of vertices. used_mask is a bitmask of used vertices.
        new_v must not be in used_mask and must not be adjacent to path[i] for i < len(path)-2.
        """
        if used_mask & (1 << new_v):
            return False
        # Check adjacency to all vertices except the last two (last one is adjacent by construction, second-to-last is allowed)
        # Actually, we need to check all i such that |i - (len(path))| > 1
        # new_v is adjacent to path[-1] (index len(path)-1).
        # We need dist(new_v, path[i]) >= 2 for all i where i < len(path) - 2.
        for i in range(len(path) - 2):
            if bit_count_table[new_v ^ path[i]] == 1:
                return False
        return True
    
    def greedy_extend(start, time_limit):
        """Greedy extension with MRV heuristic."""
        path = [start]
        used_mask = 1 << start
        
        while time.time() < time_limit:
            cur = path[-1]
            candidates = []
            for v in neighbors[cur]:
                if used_mask & (1 << v):
                    continue
                if is_valid_extension(path, used_mask, v):
                    candidates.append(v)
            
            if not candidates:
                break
            
            # MRV: sort by number of valid extensions from v
            def score_candidate(v):
                count = 0
                for nv in neighbors[v]:
                    if used_mask & (1 << nv):
                        continue
                    # Check if nv is valid extension from path + [v]
                    # This requires checking nv against path[0:-2] (since v is the new last)
                    # Actually, we check nv against path[i] for i < len(path) - 1
                    valid = True
                    for i in range(len(path) - 1):
                        if bit_count_table[nv ^ path[i]] == 1:
                            valid = False
                            break
                    if valid:
                        count += 1
                return count
            
            candidates.sort(key=score_candidate)
            
            # Pick the first candidate (most constrained)
            v = candidates[0]
            path.append(v)
            used_mask |= (1 << v)
        
        return path
    
    def backtrack_improve(path, used_mask, depth, max_depth, deadline):
        """Backtracking search to improve path length."""
        best = path.copy()
        
        if time.time() > deadline:
            return best
        
        cur = path[-1]
        candidates = []
        for v in neighbors[cur]:
            if used_mask & (1 << v):
                continue
            if is_valid_extension(path, used_mask, v):
                candidates.append(v)
        
        if not candidates or depth >= max_depth:
            return best
        
        # MRV ordering
        def score_candidate(v):
            count = 0
            for nv in neighbors[v]:
                if used_mask & (1 << nv):
                    continue
                valid = True
                for i in range(len(path) - 1):
                    if bit_count_table[nv ^ path[i]] == 1:
                        valid = False
                        break
                if valid:
                    count += 1
            return count
        
        candidates.sort(key=score_candidate)
        
        for v in candidates:
            new_path = path + [v]
            new_used = used_mask | (1 << v)
            
            improved = backtrack_improve(new_path, new_used, depth + 1, max_depth, deadline)
            if len(improved) > len(best):
                best = improved
            
            if time.time() > deadline:
                break
        
        return best
    
    best_path = [0]
    
    # Try multiple starting points and randomizations
    random.seed(42)
    
    # First, get a good greedy snake
    greedy_path = greedy_extend(0, start_time + 10)
    if len(greedy_path) > len(best_path):
        best_path = greedy_path
    
    # Now try to improve with backtracking
    # Use a limited depth to avoid explosion
    max_depth = 5
    improved = backtrack_improve(best_path, (1 << N) - 1 if False else sum(1 << v for v in best_path), 0, max_depth, start_time + 60)
    if len(improved) > len(best_path):
        best_path = improved
    
    # Try random restarts with different starting vertices
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        start_v = random.randint(0, (1 << N) - 1)
        greedy_path = greedy_extend(start_v, time.time() + 5)
        if len(greedy_path) > len(best_path):
            best_path = greedy_path
            # Try to improve
            used_mask = 0
            for v in best_path:
                used_mask |= (1 << v)
            improved = backtrack_improve(best_path, used_mask, 0, max_depth, time.time() + 20)
            if len(improved) > len(best_path):
                best_path = improved
    
    return best_path

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
