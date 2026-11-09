import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    initial_set = set(prescribed)
    initial_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            initial_sums.add(prescribed[i] + prescribed[j])
            
    # Candidates are 1..284
    # We want to add as many as possible.
    # Max possible size is 20 (given in prompt).
    # Current size is 4. We need to find 16 more.
    
    best_found_size = len(prescribed)
    best_found_set = list(prescribed)
    
    # Backtracking search with MRV (Minimum Remaining Values)
    # State: current_set, used_sums
    # We will try to build the set incrementally.
    
    # To make MRV work, we need to estimate for each candidate x,
    # how many "slots" it conflicts with.
    # Actually, a simpler heuristic for Sidon sets is to pick the candidate
    # that leaves the most options for future additions.
    # Or, we can just use a deterministic ordering based on "greediness" 
    # but backtrack if stuck.
    
    # Let's implement a recursive DFS with pruning.
    # To avoid recursion depth issues (max depth 16), we can use recursion limit or stack.
    # Depth 16 is safe.
    
    # Optimization: Precompute conflicts? 
    # Conflict between x and y if (x+y) in used_sums OR (x+x) in used_sums etc.
    # But used_sums changes.
    
    # Let's use a global variable for the current best to update it during search.
    
    # Heuristic for ordering candidates:
    # Sort candidates by how "tight" they are.
    # A candidate x is "tight" if it creates sums that are close to other sums?
    # Actually, standard Sidon construction often prefers larger numbers to spread out sums.
    # But we have fixed small numbers.
    
    # Let's try a "most constrained" approach:
    # For each candidate x, count how many existing sums s in used_sums
    # would be "blocked" or rather, how many future candidates y would be invalidated
    # if we pick x? This is expensive.
    
    # Alternative: 
    # Try to fill the set using a greedy approach first to get a baseline, 
    # then use backtracking to improve.
    
    # Let's do a pure backtracking with a good ordering.
    # Ordering: Try candidates in descending order? Or ascending?
    # In Sidon sets, large gaps are good.
    
    # Let's use a heuristic:
    # Score(x) = - (number of sums x+y that are "popular" or close to existing sums)
    # Simpler: Just try all candidates, but order them by 
    # "freedom" = number of valid pairs (x, z) for z in candidates \ {x}
    # This is O(N^2) per step. N=280. 280^2 = 78400. 
    # Depth 16. 16 * 78400 = 1.2M ops per path. 
    # If branching factor is low, this is fast.
    
    # However, computing "freedom" dynamically is costly.
    # Let's start with a simpler heuristic: 
    # Try candidates in an order that minimizes the "density" of sums.
    
    # Actually, let's just use a randomized restart of a *deterministic* backtracker 
    # with a specific ordering, or just a single deep backtracker with MRV.
    
    # Let's implement MRV properly.
    # For a given state (current_set, used_sums):
    # For each candidate x not in current_set:
    #   Check if x can be added (no sum conflict).
    #   If yes, calculate a "cost" or "priority".
    #   Priority: How many other candidates y become invalid if we pick x?
    #   This is the "degree" of conflict.
    
    # To speed this up, we can approximate.
    
    def can_add(x, current_set, used_sums):
        # Check if x conflicts with any element in current_set
        # Sums to check: x + y for y in current_set, and x + x
        # If any of these are in used_sums, return False
        # Note: x+x is a sum. x+y is a sum.
        # We must ensure x is not in current_set (handled by caller)
        for y in current_set:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True

    def get_valid_candidates(current_set, used_sums, candidates):
        valid = []
        for x in candidates:
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                valid.append(x)
        return valid

    # Heuristic: 
    # We want to pick the candidate that "blocks" the fewest other candidates.
    # For each valid candidate x:
    #   Count how many other valid candidates y would become invalid if x is added.
    #   new_used_sums = used_sums U {x+y for y in current_set} U {2x}
    #   For each y in valid_candidates \ {x}:
    #       if not can_add(y, current_set U {x}, new_used_sums):
    #           count++
    #   Score = count
    #   Pick x with minimum score.
    
    # This is O(V^2 * S) where V is number of valid candidates, S is size of set.
    # V <= 280, S <= 20.
    # 280^2 * 20 = 1.5M ops per node.
    # If depth is 16, and branching is small, this might be okay.
    # But if branching is high, it's slow.
    
    # Let's try a simpler heuristic first: 
    # Pick the candidate that maximizes the minimum distance to existing sums?
    # Or just pick the largest candidate?
    
    # Let's try "Largest First" with backtracking.
    # Sidon sets tend to have large elements.
    
    def solve(current_set, used_sums, candidates, depth):
        nonlocal best_found_size, best_found_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) > best_found_size:
            best_found_size = len(current_set)
            best_found_set = list(current_set)
            
        if len(current_set) == 20: # Max theoretical
            return
            
        # Get valid candidates
        valid_candidates = get_valid_candidates(current_set, used_sums, candidates)
        
        if not valid_candidates:
            return
            
        # Heuristic: Sort valid candidates
        # Let's try sorting by value descending (larger numbers are usually better for spreading sums)
        # Or ascending?
        # Let's try descending.
        valid_candidates.sort(reverse=True)
        
        # To avoid symmetry or get stuck, we can randomize the order slightly 
        # or try different orderings in restarts.
        # But for now, let's stick to deterministic descending.
        
        # Optimization: If we are close to deadline, stop exploring deep branches
        # unless they look promising.
        
        for x in valid_candidates:
            if time.time() > deadline:
                return
                
            new_set = current_set | {x}
            new_sums = set(used_sums)
            for y in current_set:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            
            # Recurse
            solve(new_set, new_sums, candidates, depth + 1)
            
            # Backtrack
            # No need to explicitly undo since we pass copies (sets are immutable-ish in this usage pattern if we don't modify in place, but we create new sets)
            # Creating new sets is O(S). S=20. Very fast.
            
    # Initial candidates: 1..284
    # We must include prescribed.
    # We will search for additions.
    
    # To improve search, we can use "Randomized Restart" with different orderings.
    # Ordering 1: Descending
    # Ordering 2: Ascending
    # Ordering 3: Random
    
    # Let's run multiple strategies.
    
    # Strategy 1: Descending
    candidates = list(range(284, 1, -1)) # 284 down to 2
    # Note: 1 is prescribed. 2, 5, 11 are prescribed.
    # We should filter out prescribed from candidates to add?
    # Yes, they are already in current_set.
    
    # Let's filter candidates to exclude prescribed
    prescribed_set = set(prescribed)
    candidates = [x for x in candidates if x not in prescribed_set]
    
    solve(set(prescribed), initial_sums, candidates, 0)
    
    # Strategy 2: Ascending
    if time.time() < deadline - 5:
        best_found_size = len(prescribed) # Reset to allow finding better in this branch? No, keep global best.
        # But we need to restart search.
        # The function updates nonlocal best.
        candidates_asc = [x for x in range(2, 285) if x not in prescribed_set]
        solve(set(prescribed), initial_sums, candidates_asc, 0)
        
    # Strategy 3: Randomized orderings (Multiple restarts)
    # If we have time, do a few random restarts.
    import random
    if time.time() < deadline - 10:
        for _ in range(10): # 10 restarts
            if time.time() > deadline - 5:
                break
            candidates_rand = [x for x in range(2, 285) if x not in prescribed_set]
            random.shuffle(candidates_rand)
            solve(set(prescribed), initial_sums, candidates_rand, 0)
            
    return best_found_set

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res), res)
    # Verify
    s = sorted(res)
    sums = set()
    valid = True
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            if sm in sums:
                valid = False
            sums.add(sm)
    print("Valid:", valid)
    print("Contains prescribed:", all(x in res for x in [1, 2, 5, 11]))
