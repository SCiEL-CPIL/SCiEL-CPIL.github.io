import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]
# Precompute neighbor masks for fast forbidden computation
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for u in NEIGHBORS[v]:
        m |= (1 << u)
    NEIGHBOR_MASKS[v] = m

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_path = []
    best_len = 0
    
    def greedy_extend(start_v, time_limit):
        """Greedy extension using Warnsdorff's heuristic."""
        used_mask = 1 << start_v
        forbidden_mask = 0
        path = [start_v]
        
        while time.time() < time_limit:
            cur = path[-1]
            best_candidate = -1
            best_score = float('inf')
            
            for u in NEIGHBORS[cur]:
                bit = 1 << u
                if used_mask & bit or forbidden_mask & bit:
                    continue
                # Count onward moves from u
                onward = 0
                for w in NEIGHBORS[u]:
                    wbit = 1 << w
                    if not (used_mask & wbit) and not (forbidden_mask & wbit):
                        onward += 1
                if onward < best_score:
                    best_score = onward
                    best_candidate = u
            
            if best_candidate < 0:
                break
            
            path.append(best_candidate)
            used_mask |= (1 << best_candidate)
            forbidden_mask |= NEIGHBOR_MASKS[cur]
        
        return path

    # Get initial seed
    p = greedy_extend(0, start_time + 2.0)
    best_len = len(p)
    best_path = list(p)
    
    # Backtracking search with proper state management
    def backtrack_search(start_v, time_limit, depth_limit=300):
        """DFS backtracking with bitmask state and Warnsdorff ordering."""
        nonlocal best_len, best_path
        
        # State: path list, used_mask, forbidden_mask
        path = [start_v]
        used_mask = 1 << start_v
        forbidden_mask = 0
        
        def dfs():
            nonlocal best_len, best_path
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            if time.time() > time_limit:
                return False  # Time's up
            
            cur = path[-1]
            
            # Gather candidates
            candidates = []
            for u in NEIGHBORS[cur]:
                bit = 1 << u
                if not (used_mask & bit) and not (forbidden_mask & bit):
                    # Count onward moves for Warnsdorff
                    onward = 0
                    for w in NEIGHBORS[u]:
                        wbit = 1 << w
                        if not (used_mask & wbit) and not (forbidden_mask & wbit):
                            onward += 1
                    candidates.append((onward, u))
            
            if not candidates:
                return True  # Dead end, backtrack
            
            # Sort by onward moves (Warnsdorff)
            candidates.sort()
            
            # Pruning: if we can't beat best even with all remaining
            remaining = SIZE - bin(used_mask).count('1')
            if len(path) + remaining <= best_len:
                return True
            
            for onward, u in candidates:
                if time.time() > time_limit:
                    return False
                
                # Make move
                path.append(u)
                old_used = used_mask
                old_forbidden = forbidden_mask
                used_mask |= (1 << u)
                forbidden_mask |= NEIGHBOR_MASKS[cur]
                
                dfs()
                
                # Undo
                path.pop()
                used_mask = old_used
                forbidden_mask = old_forbidden
            
            return True
        
        dfs()
        return best_path
    
    # Phase 1: Run multiple greedy extensions to get a good baseline
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        start_v = random.randint(0, SIZE - 1)
        p = greedy_extend(start_v, time.time() + 1.0)
        if len(p) > best_len:
            best_len = len(p)
            best_path = list(p)
    
    # Phase 2: Backtracking search with time slices
    # Use the best path's start as one option, plus random starts
    starts = [best_path[0]] if best_path else [0]
    for _ in range(5):
        starts.append(random.randint(0, SIZE - 1))
    
    for i, start_v in enumerate(starts):
        if time.time() > deadline - 5:
            break
        # Allocate remaining time proportionally
        remaining_time = deadline - 5 - time.time()
        slice_time = max(2.0, remaining_time / (len(starts) - i))
        search_limit = min(time.time() + slice_time, deadline - 5)
        
        backtrack_search(start_v, search_limit)
        
        # After each search, try to extend the best path found
        if time.time() < deadline - 10:
            if best_path:
                # Try extending from the end of best_path
                ext_limit = min(time.time() + 5.0, deadline - 10)
                # Reconstruct state from best_path
                used_mask = 0
                forbidden_mask = 0
                for v in best_path:
                    used_mask |= (1 << v)
                if len(best_path) >= 2:
                    forbidden_mask = NEIGHBOR_MASKS[best_path[-2]]
                
                # Try to extend
                cur = best_path[-1]
                extended = list(best_path)
                for u in NEIGHBORS[cur]:
                    bit = 1 << u
                    if not (used_mask & bit) and not (forbidden_mask & bit):
                        extended.append(u)
                        used_mask |= bit
                        forbidden_mask |= NEIGHBOR_MASKS[cur]
                        cur = u
                        break  # Only one extension step for safety
                
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = extended
    
    # Final phase: try more backtracking if time remains
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            start_v = random.randint(0, SIZE - 1)
            search_limit = min(time.time() + 3.0, deadline - 5)
            backtrack_search(start_v, search_limit)
    
    return best_path

if __name__ == "__main__":
    # Quick test
    p = construct()
    print(f"Length: {len(p)}")
    # Verify
    if len(p) > 1:
        valid = True
        for i in range(len(p)-1):
            if (p[i] ^ p[i+1]).bit_count() != 1:
                print(f"Error: consecutive {p[i]} and {p[i+1]} not adjacent")
                valid = False
                break
        if valid:
            for i in range(len(p)):
                for j in range(i+2, len(p)):
                    if (p[i] ^ p[j]).bit_count() == 1:
                        print(f"Error: non-consecutive {p[i]} and {p[j]} adjacent")
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            print("Valid snake")
