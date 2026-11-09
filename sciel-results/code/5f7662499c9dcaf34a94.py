import random
import time

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Precompute conflicts
    # conflicts[x] is a list of tuples (a, b, c) such that {a, b, c, x} is a 4-AP
    conflicts = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            terms = (a, a + d, a + 2 * d, a + 3 * d)
            # For each term, the other three form a conflict
            for i in range(4):
                x = terms[i]
                others = [terms[j] for j in range(4) if j != i]
                conflicts[x].append(tuple(others))
    
    def is_valid_add(S, x):
        for t in conflicts[x]:
            if t[0] in S and t[1] in S and t[2] in S:
                return False
        return True

    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        S = set()
        # Heuristic: Sort by degree (number of conflicts) descending?
        # Or random?
        # Let's try sorting by degree descending to place "constrained" elements first.
        # But random restarts help.
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try to add any element not in S
            # Or try swaps
            # Let's try: For each x not in S, if it's blocked by exactly 1 element y, 
            # try swapping y for x.
            
            # First, try simple additions (in case greedy missed something due to order? 
            # Greedy adds if valid, so no simple additions left.)
            
            # Try 1-for-1 swaps
            # Iterate through x not in S
            for x in range(1, N + 1):
                if x in S:
                    continue
                if time.time() > deadline:
                    return S
                
                # Check if x is valid
                if is_valid_add(S, x):
                    S.add(x)
                    improved = True
                    break # Restart scan
                    
                # If not valid, find blockers
                # We want to find a y in S such that S\{y} allows x
                # This is equivalent to: y is in the intersection of all blocked triples?
                # No, y must hit all blocked triples.
                
                blocked_triples = []
                for t in conflicts[x]:
                    if t[0] in S and t[1] in S and t[2] in S:
                        blocked_triples.append(t)
                
                if not blocked_triples:
                    # Should have been caught by is_valid_add
                    S.add(x)
                    improved = True
                    break
                
                # Try removing each y in S
                # Optimization: Only try y that are in at least one blocked triple
                candidate_removals = set()
                for t in blocked_triples:
                    candidate_removals.update(t)
                
                for y in candidate_removals:
                    if y not in S:
                        continue
                    # Check if removing y makes x valid
                    # This requires checking conflicts again
                    # S_temp = S - {y}
                    # We can simulate by checking if any blocked triple remains fully in S\{y}
                    
                    valid_after_removal = True
                    for t in blocked_triples:
                        # If y is in t, t is broken
                        if y in t:
                            continue
                        # If y is not in t, t is still fully in S
                        # So if there is any blocked triple not containing y, removal of y doesn't help
                        valid_after_removal = False
                        break
                    
                    if valid_after_removal:
                        # Perform swap
                        S.remove(y)
                        S.add(x)
                        improved = True
                        break # Restart scan for this x? Or break outer?
                        # Breaking inner loop, but we should restart the whole process
                        # because S changed
                        break
                
                if improved:
                    break
            
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple greedy restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        S = greedy_random(seed=i)
        
        # Local search
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Try a few more restarts if time permits
        if time.time() < deadline - 10:
             # Try a different heuristic: sort by degree
             pass

    # Final check and return
    # Ensure no 4-AP
    # The construction guarantees it, but let's be safe?
    # No, the logic is sound.
    
    return sorted(best_set)
