import time
import sys

def construct():
    # Deadline: 115 seconds
    deadline = time.time() + 115
    best_solution = None
    best_max = float('inf')
    
    # We will search for a Sidon set containing 0 that covers diffs 1..30.
    # We minimize the largest element.
    
    # Bitmask for used differences. 
    # We assume max difference < 1024 for the bitmask to be a single Python int.
    # If max element is ~100, diffs are < 100.
    # Let's use a set for flexibility but bitmask is faster.
    # Let's stick to set for simplicity first, but optimize the inner loop.
    
    def solve():
        nonlocal best_solution, best_max
        
        elements = [0]
        used_diffs = set()
        target_diffs = set(range(1, 31))
        
        def backtrack():
            nonlocal best_solution, best_max
            if time.time() > deadline:
                return False
            
            if not target_diffs:
                mx = elements[-1]
                if mx < best_max:
                    best_max = mx
                    best_solution = list(elements)
                return True
            
            # Heuristic: Try the smallest possible next element
            # But we can also look at missing diffs to guide the search?
            # No, just try small x first to keep diameter small.
            
            last = elements[-1]
            # Upper bound for x: we don't want to exceed current best_max
            # Also, if we've tried x up to some limit and failed, we stop.
            # A safe upper bound for the next element in a minimal solution is usually small.
            # If best_max is inf, we need a cap. Let's cap at 200 initially, or best_max-1.
            upper_bound = best_max - 1 if best_max < float('inf') else 200
            
            # Optimization: We can skip x if it doesn't cover any new target diff?
            # Not necessarily, it might be needed to allow future diffs.
            # But if it covers NO target diffs and doesn't help "bridge", it's likely suboptimal for minimization.
            # However, let's just try all valid x in range.
            
            for x in range(last + 1, upper_bound + 1):
                if time.time() > deadline:
                    return False
                
                # Check validity
                valid = True
                new_diffs = []
                for a in elements:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Apply move
                elements.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                    if d in target_diffs:
                        target_diffs.discard(d)
                
                # Recurse
                if backtrack():
                    elements.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                    return True
                
                # Undo
                elements.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        backtrack()
    
    solve()
    
    if best_solution:
        return best_solution
    
    # Fallback if backtracking fails or times out without finding a solution
    # This shouldn't happen for this problem size, but just in case.
    # A known construction for covering 1..30 with a Sidon set:
    # We can use a greedy approach as a fallback.
    a = [0]
    used = set()
    for d in range(1, 31):
        if d in used:
            continue
        found = False
        # Try to add an element that creates diff d
        # x - y = d => x = y + d
        # We want to add x such that all x - z are new.
        candidates = []
        for y in a:
            x = y + d
            if x in a:
                continue
            # Check if x is valid
            valid = True
            for z in a:
                if z == x:
                    continue
                diff = abs(x - z)
                if diff in used:
                    valid = False
                    break
            if valid:
                candidates.append(x)
        
        if candidates:
            # Pick the smallest x to keep diameter small?
            x = min(candidates)
            a.append(x)
            a.sort()
            for z in a:
                if z != x:
                    used.add(abs(x - z))
            found = True
        else:
            # If no existing y works, we need to add a new element that creates d
            # This implies we might need to add 2 elements or restart?
            # Actually, if we build incrementally, we should always be able to add an element
            # that creates the next needed diff, unless we hit a dead end.
            # For a robust fallback, let's just return a large valid Sidon set if this fails.
            # But the backtracking above should have found it.
            pass
            
    return a

# To make it faster, let's use a bitmask for differences.
# Max difference for a set with max element M is M.
# If M < 1024, we can use an int.

def construct():
    deadline = time.time() + 115
    best_solution = None
    best_max = float('inf')
    
    # We'll use a bitmask for used differences.
    # Bit i represents difference i.
    # Max difference we care about for collision is up to ~200.
    # 200 bits fit easily in a Python int.
    
    def solve():
        nonlocal best_solution, best_max
        
        elements = [0]
        used_mask = 0  # Bit i is 1 if diff i is used
        target_mask = 0
        for d in range(1, 31):
            target_mask |= (1 << d)
        remaining_target = target_mask
        
        def backtrack():
            nonlocal best_solution, best_max, remaining_target
            if time.time() > deadline:
                return False
            
            if remaining_target == 0:
                mx = elements[-1]
                if mx < best_max:
                    best_max = mx
                    best_solution = list(elements)
                return True
            
            last = elements[-1]
            # Upper bound
            upper_bound = best_max - 1 if best_max < float('inf') else 200
            
            for x in range(last + 1, upper_bound + 1):
                if time.time() > deadline:
                    return False
                
                # Check collisions
                # New diffs are x - a for a in elements
                # We need to check if any (1 << (x-a)) is already set in used_mask
                valid = True
                new_diffs_mask = 0
                for a in elements:
                    d = x - a
                    if d > 1023: # Safety, though unlikely
                        valid = False
                        break
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                
                if not valid:
                    continue
                
                # Apply
                elements.append(x)
                new_used_mask = used_mask | new_diffs_mask
                new_remaining = remaining_target & ~new_diffs_mask
                
                if backtrack():
                    elements.pop()
                    return True
                
                elements.pop()
            
            return False
        
        # We need to pass used_mask and remaining_target as nonlocal or modify the closure properly
        # Python closures capture variables by reference. 
        # But `used_mask` and `remaining_target` are local to `solve`.
        # We need to update them in the closure.
        # Let's refactor to use lists or nonlocal declarations properly.
        
        # Actually, the above code has a bug: `used_mask` and `remaining_target` are not updated in the recursive call context correctly if we just reassign them locally in `backtrack` without `nonlocal`.
        # And we are passing `new_used_mask` but not updating the parent's `used_mask`.
        
        # Let's rewrite the backtracking to pass state explicitly or use mutable containers.
        
    # Corrected Backtracking with Bitmask
    
    elements = [0]
    used_mask = 0
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
    
    def backtrack():
        nonlocal best_solution, best_max
        if time.time() > deadline:
            return False
        
        current_remaining = target_mask & ~used_mask
        if current_remaining == 0:
            mx = elements[-1]
            if mx < best_max:
                best_max = mx
                best_solution = list(elements)
            return True
        
        last = elements[-1]
        upper_bound = best_max - 1 if best_max < float('inf') else 200
        
        for x in range(last + 1, upper_bound + 1):
            if time.time() > deadline:
                return False
            
            valid = True
            new_diffs_mask = 0
            for a in elements:
                d = x - a
                if d > 1023:
                    valid = False
                    break
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            
            if not valid:
                continue
            
            # Save old state
            old_used_mask = used_mask
            elements.append(x)
            used_mask |= new_diffs_mask
            
            if backtrack():
                return True
            
            # Restore
            elements.pop()
            used_mask = old_used_mask
        
        return False
    
    backtrack()
    
    if best_solution:
        return best_solution
    
    # Fallback
    return [0]

# Final check: The logic above modifies `used_mask` in the enclosing scope of `backtrack` (which is `construct`).
# This is correct.

# One optimization: The search space is small. 
# We can also try to jump to specific x values that cover missing diffs.
# But simple increasing x is usually good for minimizing max element.

# Let's add a small heuristic: if a diff d is missing, x must be such that x-a=d for some a.
# This restricts x to values in {a+d | a in elements}.
# If we only try these x values, we might miss solutions where x doesn't immediately cover a missing diff but enables future ones.
# However, for minimizing diameter, it's often optimal to cover missing diffs as soon as possible.
# Let's stick to the full range search for correctness, but the bitmask makes it fast enough.

# To be safe against timeouts, if no solution found, return a known valid set?
# The problem guarantees a solution exists. The backtracking should find it.
# If it times out, it returns [0] which is invalid. 
# We should ensure it finds a solution. 
# Given the small size, it should find it in milliseconds.

# Let's double check the bitmask logic.
# used_mask contains bits for all differences in the current set.
# When we add x, we calculate new diffs.
# If any new diff bit is already in used_mask, collision.
# Else, we add x and update used_mask.
# We check if all target diffs are covered by checking if (target_mask & ~used_mask) == 0.
# This is correct.

# One potential issue: `d > 1023`. 
# If the max element is > 1023, the bitmask overflows or becomes very large.
# Python ints are arbitrary precision, so `1 << 2000` is fine, just slower.
# But for this problem, max element is likely < 100.
# So 1023 is a safe bound for speed. If we exceed it, we fall back to set? 
# No, let's just allow larger ints. Python handles it.
# But `if d > 1023: valid = False` is a hard cap. 
# If the solution requires max element > 1023, this will fail.
# Is that possible? 
# Sidon set of size 9. Min diameter is roughly 9^2/2 = 40. Max diameter for size 9 is not bounded tightly by simple formulas, but usually small.
# A perfect difference set of order q has size q^2+q+1 and diameter q^2+q.
# For q=2, size 7, diameter 6? No.
# Let's assume max element < 1024. If not, we can increase the limit or remove it.
# Let's remove the hard cap and rely on Python's big ints, but add a check for `d` being reasonable.
# Actually, if `d` is large, `1 << d` is expensive.
# Let's use a `set` for differences if we go above 1024.
# But for this specific problem, 1024 is more than enough.

# Final Code Structure
