import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_path = []
    best_len = 0
    
    # We will use a DFS approach with backtracking.
    # State: current path, used set, forbidden set.
    # To make it fast, we use a recursive function with a global counter for time checks.
    # Since recursion depth can be up to 512, we might hit recursion limits, but 104-190 is fine.
    # However, iterative DFS is safer and allows better control.
    
    # Let's use an iterative DFS stack.
    # Each stack item: (path, used, forbidden, current_end, next_index_in_neighbors)
    # This is memory heavy if we store full paths.
    # Alternative: Store only the path, and recompute used/forbidden? No, too slow.
    # Store: path (list), used (set), forbidden (set).
    # To save memory, we can store the path as a tuple or list, and used/forbidden as sets.
    # For N=9, sets are small.
    
    # Optimization: 
    # 1. Start with a good greedy seed to have a high baseline quickly.
    # 2. Then use DFS to try to extend or find longer paths.
    
    def greedy_extend(start_v, time_limit):
        used = {start_v}
        forbidden = set()
        path = [start_v]
        
        while time.time() < time_limit:
            cur = path[-1]
            neighbors = NEIGHBORS[cur]
            
            candidates = []
            for u in neighbors:
                if u in used or u in forbidden:
                    continue
                candidates.append(u)
            
            if not candidates:
                break
            
            # Heuristic: pick the one with fewest options (fail-first) to see if it works, 
            # or most options? 
            # For finding *a* long path, "most options" is usually better for greedy.
            # But for DFS, we want to explore.
            # Let's just pick random for the seed to get variety.
            next_v = random.choice(candidates)
            
            path.append(next_v)
            used.add(next_v)
            
            for n in NEIGHBORS[cur]:
                forbidden.add(n)
        
        return path

    # Get a decent initial seed
    p = greedy_extend(0, start_time + 1.0)
    best_len = len(p)
    best_path = p
    
    # Now, perform DFS from various starting points or extending the seed.
    # A pure DFS from 0 might get stuck in local minima quickly.
    # Strategy: 
    # 1. Use the greedy path as a starting point for DFS? No, DFS needs to build from scratch or backtrack.
    # 2. Run DFS from random starts, but with a time limit per run.
    
    # DFS Implementation:
    # We want to find the longest path.
    # We can use a "branch and bound" style: if current_len + (SIZE - len(used)) <= best_len, prune.
    # But SIZE - len(used) is an overestimate.
    
    # Let's implement a recursive DFS with manual stack to avoid recursion limit issues.
    # Stack items: (path, used, forbidden)
    # We will explore neighbors in a randomized order to escape local minima.
    
    def dfs_search(start_v, time_limit):
        nonlocal best_len, best_path
        
        # Initial state
        path = [start_v]
        used = {start_v}
        forbidden = set()
        
        # Stack of states to explore
        # Each state: (path, used, forbidden, next_neighbor_index)
        # To save memory, we can store the path and recompute used/forbidden? No.
        # Storing sets in the stack is expensive.
        # Alternative: Recursive DFS with backtracking (undo operations).
        
        # Let's use recursion. Depth ~ 200 is fine for Python if we increase limit?
        # sys.setrecursionlimit(10000)
        
        def dfs(path, used, forbidden):
            nonlocal best_len, best_path
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
            if time.time() > time_limit:
                return
            
            cur = path[-1]
            neighbors = NEIGHBORS[cur]
            
            # Find valid candidates
            candidates = []
            for u in neighbors:
                if u not in used and u not in forbidden:
                    candidates.append(u)
            
            if not candidates:
                return
            
            # Randomize order for diversity
            random.shuffle(candidates)
            
            for u in candidates:
                # Check time
                if time.time() > time_limit:
                    return
                
                # Pruning: if even using all remaining vertices we can't beat best_len
                # Remaining vertices = SIZE - len(used)
                # Max possible new length = len(path) + (SIZE - len(used) - 1) ? 
                # Actually, we add 1 vertex.
                # Upper bound on final length from this state:
                # len(path) + (SIZE - len(used))
                if len(path) + (SIZE - len(used)) <= best_len:
                    continue
                
                # Make move
                path.append(u)
                used.add(u)
                
                # Update forbidden
                # Add neighbors of cur to forbidden
                old_forbidden_size = len(forbidden)
                for n in NEIGHBORS[cur]:
                    forbidden.add(n)
                
                # Recurse
                dfs(path, used, forbidden)
                
                # Undo move
                path.pop()
                used.remove(u)
                
                # Undo forbidden
                # This is tricky with sets. We need to remove only the elements we added.
                # Since we only added neighbors of cur, and they might have been there before,
                # we can't just remove all neighbors of cur.
                # We need to know which ones were NEW.
                
                # Better approach: Pass a copy of forbidden? 
                # Sets are small (max 512). Copying is O(N).
                # Let's pass a copy to simplify backtracking logic.
                
                # Actually, let's refactor to pass copies. It's safer.
        
        # The above recursion with undo is complex due to forbidden set updates.
        # Let's use the copy approach in a recursive function.
        
        def dfs_copy(path, used, forbidden):
            nonlocal best_len, best_path
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
            if time.time() > time_limit:
                return
            
            cur = path[-1]
            neighbors = NEIGHBORS[cur]
            
            candidates = []
            for u in neighbors:
                if u not in used and u not in forbidden:
                    candidates.append(u)
            
            if not candidates:
                return
            
            # Heuristic: sort candidates by "fewest options" first? 
            # Or random? Random helps with global search.
            random.shuffle(candidates)
            
            for u in candidates:
                if time.time() > time_limit:
                    return
                
                # Pruning
                if len(path) + (SIZE - len(used)) <= best_len:
                    continue
                
                # Create new states
                new_path = path + [u]
                new_used = used | {u}
                new_forbidden = forbidden | set(NEIGHBORS[cur])
                
                dfs_copy(new_path, new_used, new_forbidden)
                
        # Start DFS
        dfs_copy([start_v], {start_v}, set())

    # Run multiple DFS searches with different starts
    # To avoid getting stuck in the same local minima, use different starts.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        start_v = random.randint(0, SIZE - 1)
        
        # Give each DFS search a small time slice, or let it run until deadline?
        # If it runs until deadline, we only do 1 search.
        # Better: run for a fixed time slice, e.g., 2 seconds, then restart with new start.
        # This allows exploring different regions of the state space.
        
        search_limit = time.time() + 2.0
        if search_limit > deadline:
            search_limit = deadline
            
        dfs_search(start_v, search_limit)
        
        # If we are close to the deadline, do one final long search
        if time.time() > deadline - 10:
            dfs_search(random.randint(0, SIZE - 1), deadline)
            break

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
