import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s
    
    REQUIRED = set(range(1, 31))
    MAX_SIZE = 32
    
    best_diameter = [10**6]
    best_set = [None]
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    def solve():
        # Backtracking: build sorted list starting with [0]
        # state: current set (sorted), used_differences, remaining required
        # We add elements in increasing order.
        
        def backtrack(current, used_diffs, remaining_req, depth):
            if not check_time():
                return False
            
            # Check if we've covered all required differences
            if not remaining_req:
                diameter = current[-1] if current else 0
                if diameter < best_diameter[0]:
                    best_diameter[0] = diameter
                    best_set[0] = list(current)
                return True
            
            # If we can't add more elements, fail
            if len(current) >= MAX_SIZE:
                return False
            
            # Pruning: check if remaining required differences can still be covered
            # Each new element x added after current[-1] can create at most 
            # (len(current)+1) new differences (x - a_i for each a_i in current, plus possibly x - 0)
            # Actually, when we add x, it creates differences x - a_i for all a_i in current.
            # That's len(current) new differences.
            # We have (MAX_SIZE - len(current)) slots left.
            # Max new differences = sum over slots of (len(current) + slot_index) roughly
            # More precisely: if we add k more elements, the i-th new element creates (len(current) + i) differences
            # Total max new diffs = sum_{i=1}^{k} (len(current) + i) = k*len(current) + k*(k+1)/2
            # But we only need |remaining_req| more differences.
            k = MAX_SIZE - len(current)
            max_new_diffs = k * len(current) + k * (k + 1) // 2
            if max_new_diffs < len(remaining_req):
                return False
            
            # Determine candidate values for the next element
            # The next element must be > current[-1]
            # It should be chosen to cover as many remaining required differences as possible
            
            # Generate candidates: for each remaining required difference d,
            # if we add x such that x - a_i = d for some a_i in current, then x = a_i + d
            # Also try x = current[-1] + 1, current[-1] + 2, etc. as fallbacks
            
            candidates = set()
            last = current[-1]
            
            # Candidates from covering required differences
            for d in remaining_req:
                for a_i in current:
                    x = a_i + d
                    if x > last:
                        candidates.add(x)
            
            # Also add some sequential candidates
            for offset in range(1, 20):
                candidates.add(last + offset)
            
            # Sort candidates by how many remaining required differences they cover
            def score_candidate(x):
                if x <= last:
                    return -1
                covered = 0
                valid = True
                for a_i in current:
                    diff = x - a_i
                    if diff in used_diffs:
                        valid = False
                        break
                    if diff in remaining_req:
                        covered += 1
                if not valid:
                    return -1
                return covered
            
            scored = []
            for x in candidates:
                if x <= last:
                    continue
                # Quick check: does x create any duplicate difference?
                valid = True
                covered = 0
                for a_i in current:
                    diff = x - a_i
                    if diff in used_diffs:
                        valid = False
                        break
                    if diff in remaining_req:
                        covered += 1
                if valid:
                    scored.append((covered, x))
            
            # Sort by covered (descending), then by x (ascending)
            scored.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates for speed
            max_candidates = 50
            if len(scored) > max_candidates:
                scored = scored[:max_candidates]
            
            for covered, x in scored:
                if not check_time():
                    return False
                
                new_used = set(used_diffs)
                new_remaining = set(remaining_req)
                valid = True
                for a_i in current:
                    diff = x - a_i
                    if diff in new_used:
                        valid = False
                        break
                    new_used.add(diff)
                    if diff in new_remaining:
                        new_remaining.discard(diff)
                
                if not valid:
                    continue
                
                if backtrack(current + [x], new_used, new_remaining, depth + 1):
                    return True
            
            return False
        
        # Start with [0]
        initial_used = set()
        initial_remaining = set(REQUIRED)
        backtrack([0], initial_used, initial_remaining, 0)
    
    # Try multiple strategies with different random seeds for diversity
    random.seed(42)
    
    # First, try a structured search
    solve()
    
    if best_set[0] is not None:
        return best_set[0]
    
    # Fallback: greedy with backtracking restarts
    def greedy_construct():
        a = [0]
        used_diffs = set()
        remaining = set(REQUIRED)
        
        for _ in range(MAX_SIZE - 1):
            if not remaining:
                break
            if not check_time():
                break
            
            best_x = None
            best_covered = -1
            
            last = a[-1]
            # Try candidates
            candidates = set()
            for d in remaining:
                for ai in a:
                    x = ai + d
                    if x > last:
                        candidates.add(x)
            for off in range(1, 30):
                candidates.add(last + off)
            
            for x in sorted(candidates):
                if x <= last:
                    continue
                valid = True
                covered = 0
                for ai in a:
                    diff = x - ai
                    if diff in used_diffs:
                        valid = False
                        break
                    if diff in remaining:
                        covered += 1
                if valid and covered > best_covered:
                    best_covered = covered
                    best_x = x
            
            if best_x is None:
                # Try larger values
                x = last + 1
                while True:
                    valid = True
                    for ai in a:
                        if (x - ai) in used_diffs:
                            valid = False
                            break
                    if valid:
                        best_x = x
                        break
                    x += 1
            
            a.append(best_x)
            for ai in a[:-1]:
                diff = best_x - ai
                used_diffs.add(diff)
                if diff in remaining:
                    remaining.discard(diff)
        
        return a
    
    result = greedy_construct()
    if result and result[-1] < best_diameter[0]:
        best_diameter[0] = result[-1]
        best_set[0] = result
    
    if best_set[0] is not None:
        return best_set[0]
    
    # Last resort: return a valid but possibly suboptimal set
    return [0]

# EVOLVE-BLOCK-END
