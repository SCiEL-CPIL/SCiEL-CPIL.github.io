import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    N = 256
    # Perfect squares up to 2*N
    squares = set(i*i for i in range(1, int(math.sqrt(2*N)) + 2))
    
    # Pre-filter: a is valid if 2*a is not a perfect square
    valid_numbers = [a for a in range(1, N + 1) if 2*a not in squares]
    
    # Precompute conflicts: conflicts[u] is a set of v such that u+v is a perfect square
    # Note: We only care about v in valid_numbers potentially, but for simplicity, 
    # we can check on the fly or precompute for all 1..N.
    # Precomputing for all 1..N is fast enough.
    conflicts = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u, N + 1): # u <= v
            if u + v in squares:
                conflicts[u].add(v)
                if u != v:
                    conflicts[v].add(u)
    
    def is_conflict(a, b):
        return b in conflicts[a]
        
    def can_add(candidate, current_set):
        if 2 * candidate in squares:
            return False
        for x in current_set:
            if is_conflict(candidate, x):
                return False
        return True

    best_size = 0
    best_set = set()
    
    def greedy_init(randomize=True):
        S = set()
        candidates = valid_numbers[:]
        if randomize:
            random.shuffle(candidates)
        for c in candidates:
            if can_add(c, S):
                S.add(c)
        return S

    def local_search(S, deadline):
        S = set(S)
        size = len(S)
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1-swap: remove one, add one
            # Try to find a pair (u in S, v not in S) such that size increases (impossible for 1-1 swap unless we swap for same size but better structure? No, we want to maximize size. 
            # A 1-swap that keeps size same might help escape local minima for 2-swaps, but standard hill climbing for max size usually only accepts if size increases.
            # However, if we are stuck, we might need to allow same-size moves to reach a state where a 2-swap increases size.
            # Let's stick to size-increasing moves first. If stuck, we can do random same-size swaps.
            
            # Actually, if we only accept size increases, we will get stuck at a local maximum.
            # To escape, we should allow "sideways" moves or use simulated annealing.
            # Given the time, let's try 2-swaps which can increase size by 2, or 1-swap which can increase by 1? 
            # Wait, if we remove 1 and add 1, size stays same. 
            # If we remove 1 and add 2, size increases by 1.
            # If we remove 2 and add 3, size increases by 1.
            
            # Let's implement a move that tries to increase size.
            # Candidate moves:
            # 1. Add a single element v not in S. (Size +1)
            # 2. Remove u in S, add v not in S. (Size 0, but might enable future adds) -> This is tricky.
            # 3. Remove u in S, add v, w not in S. (Size +1)
            
            # Let's try adding single elements first.
            for v in valid_numbers:
                if v in S:
                    continue
                if can_add(v, S):
                    S.add(v)
                    size += 1
                    improved = True
                    break # Restart search after improvement
            
            if improved:
                no_improve_count = 0
                continue
            
            # If no single add works, try 1-remove, 2-add (net +1)
            # This is expensive. O(N^3).
            # Let's try 2-remove, 3-add (net +1) or 1-remove 1-add (net 0) to escape?
            
            # Alternative: Simulated Annealing style.
            # For now, let's try random 1-swap (size 0) to shuffle the set, then try adds.
            if random.random() < 0.1: # Occasional shuffle
                # Pick a random u in S and a random v not in S
                u_list = list(S)
                if u_list:
                    u = random.choice(u_list)
                    # Find a v that can replace u (i.e., v doesn't conflict with S\{u})
                    # This is complex. Let's just do a random valid swap if possible.
                    # Or simply: remove u, add v if v is valid with S\{u}.
                    S_temp = S - {u}
                    for v in valid_numbers:
                        if v in S_temp:
                            continue
                        if can_add(v, S_temp):
                            S = S_temp
                            S.add(v)
                            break
                    # If no v found, revert (S is already S_temp if we didn't add, but we want to keep size same)
                    # If we didn't add, S is S_temp which is smaller. We must revert.
                    if len(S) < size:
                        S.add(u)
            
            # Try 1-remove, 2-add
            # This is likely too slow in pure Python for 256 elements if done naively.
            # Let's rely on the initial greedy + single adds.
            # If stuck, do a restart.
            
            no_improve_count += 1
            
            # If stuck for a while, force a restart
            if no_improve_count > 100:
                break
                
        return S

    # Main loop: Multiple restarts
    restarts = 0
    while time.time() < timeout:
        S = greedy_init(randomize=True)
        S = local_search(S, timeout - 2.0)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            # If we found a very good solution, we can reduce restarts or stop
            # Max possible size is bounded. 
            # 256 numbers. Half are invalid (2a square). ~128 valid.
            # Conflicts are dense. 
            # Let's just keep searching until time runs out.
        
        restarts += 1
        if best_size > 80: # Heuristic threshold to stop early if we hit a good number
             # We can continue, but maybe reduce effort?
             pass

    # Return as list
    return list(best_set)
# EVOLVE-BLOCK-END
