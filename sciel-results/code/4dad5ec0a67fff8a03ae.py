"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find a large subset S of [1,256] such that for all a,b in S with a<=b,
    a+b is not a perfect square.
    
    Uses a greedy algorithm with local search (add/remove) to maximize size.
    """
    deadline = time.time() + 85.0  # 5s safety margin before 90s limit
    
    N = 256
    
    # Precompute: for each number x in [1, N], which y in [1, N] would make x+y a perfect square?
    # We need x+y = k^2 for some integer k.
    # k^2 ranges from 1+1=2 to 256+256=512, so k from 2 to 22 (since 22^2=484, 23^2=529>512)
    # Actually k can be 2..22 (k=1 gives 1, but min sum is 2)
    
    # conflict[a] = set of b such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for k in range(2, 23):  # k^2 from 4 to 529
            b = k * k - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Greedy construction: add elements in order of smallest conflict degree
    # First, count conflicts (degree) for each element
    # But conflicts change as we add elements...
    
    # Strategy: Greedy with priority by current degree
    # Start with empty set, repeatedly add the element with fewest conflicts with current set
    # that doesn't conflict with any element in the set
    
    def greedy_construct():
        S = set()
        remaining = set(range(1, N + 1))
        
        while remaining and time.time() < deadline:
            # Find element in remaining with minimum conflicts with S
            best_elem = None
            best_conflicts = float('inf')
            
            for elem in remaining:
                # Check if elem conflicts with any element in S
                # Also check diagonal: elem + elem must not be a perfect square
                if elem in conflict[elem]:  # 2*elem is a perfect square
                    continue
                # Count conflicts with S
                conflicts = 0
                for s in S:
                    if elem in conflict[s]:
                        conflicts += 1
                        break  # just need to know it's invalid
                if conflicts == 0:
                    # Valid to add; count how many it would block
                    block_count = 0
                    for r in remaining:
                        if r != elem and r in conflict[elem]:
                            block_count += 1
                    # Prefer adding elements that block few others
                    if block_count < best_conflicts:
                        best_conflicts = block_count
                        best_elem = elem
            
            if best_elem is None:
                break
            
            S.add(best_elem)
            remaining.discard(best_elem)
            # Remove elements that now conflict with best_elem
            to_remove = set()
            for r in remaining:
                if r in conflict[best_elem]:
                    to_remove.add(r)
            remaining -= to_remove
        
        return S
    
    # Try multiple random orderings for greedy
    import random
    random.seed(42)
    
    best_S = greedy_construct()
    best_size = len(best_S)
    
    # Local search: try to add removed elements, swap, etc.
    # First, let's do a more systematic greedy with different orderings
    
    # Try greedy with different initial orderings
    for trial in range(50):
        if time.time() >= deadline:
            break
        
        # Random permutation of candidates
        candidates = list(range(1, N + 1))
        # Filter out elements where 2*x is a perfect square (diagonal constraint)
        candidates = [x for x in candidates if x not in conflict[x]]
        random.shuffle(candidates)
        
        S = set()
        for elem in candidates:
            if time.time() >= deadline:
                break
            # Check if elem can be added
            valid = True
            for s in S:
                if elem in conflict[s]:
                    valid = False
                    break
            if valid:
                S.add(elem)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Local search on best_S: try to add elements that were excluded
    excluded = set(range(1, N + 1)) - best_S
    # For each excluded element, see if we can remove some elements from S to add it
    for _ in range(200):
        if time.time() >= deadline:
            break
        improved = False
        for elem in list(excluded):
            if time.time() >= deadline:
                break
            # Find all elements in S that conflict with elem
            blockers = set()
            for s in best_S:
                if elem in conflict[s]:
                    blockers.add(s)
            # If we can remove fewer elements than we gain (i.e., len(blockers) < 1, meaning we can add without removing)
            # Actually, we want to swap: remove blockers, add elem
            # Net gain = 1 - len(blockers)
            if len(blockers) == 0:
                best_S.add(elem)
                excluded.discard(elem)
                improved = True
            elif len(blockers) == 1:
                # Try swapping: remove one blocker, add elem
                # Only if this doesn't break other constraints
                # Since we're removing a blocker and adding elem, we need to check
                # that removing the blocker doesn't expose new conflicts
                # Actually, removing an element can only help, so this is safe
                # But we gain 0 net (remove 1, add 1), so only do it if it enables further additions
                pass
        
        if not improved and time.time() >= deadline:
            break
    
    # More aggressive local search: try adding each excluded element by removing minimal blockers
    for _ in range(100):
        if time.time() >= deadline:
            break
        improved = False
        for elem in list(excluded):
            if time.time() >= deadline:
                break
            blockers = set()
            for s in best_S:
                if elem in conflict[s]:
                    blockers.add(s)
            if len(blockers) <= 1:
                # Remove blockers, add elem
                best_S -= blockers
                best_S.add(elem)
                excluded.discard(elem)
                # Update excluded: elements that were only blocked by removed blockers might now be addable
                improved = True
                break  # restart scan after modification
        if not improved:
            break
    
    # Final greedy pass on remaining excluded
    for elem in sorted(excluded):
        if time.time() >= deadline:
            break
        valid = True
        for s in best_S:
            if elem in conflict[s]:
                valid = False
                break
        if valid:
            best_S.add(elem)
            excluded.discard(elem)
    
    # Verify result
    S_list = sorted(best_S)
    for i, a in enumerate(S_list):
        for b in S_list[i:]:
            s = a + b
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid! Remove b
                best_S.discard(b)
                break
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
