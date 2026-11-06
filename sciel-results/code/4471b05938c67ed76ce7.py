"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None
    
    # Precompute bitmasks for sums
    # sum_mask[s] = bitmask of elements x such that x can form sum s with some element
    # We'll use a different approach: track sums as a set, but use bitmasks for fast checks
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def compute_sums(S):
        """Compute all pair sums for a Sidon set."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums
    
    def is_maximal_fast(S, sums):
        """Check if S is maximal: no element from [1,N] can be added."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added: 2x not in sums, and x+a not in sums for all a in S
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True
    
    def count_blocked_fast(S, sums, x):
        """Count how many remaining elements would be blocked if x is added."""
        S_set = set(S)
        new_sums = sums | {2 * x}
        for a in S:
            new_sums.add(x + a)
        
        blocked = 0
        for y in range(1, N + 1):
            if y == x or y in S_set:
                continue
            if 2 * y in new_sums:
                blocked += 1
                continue
            y_blocked = False
            for a in S:
                if (y + a) in new_sums:
                    y_blocked = True
                    break
            if y_blocked:
                blocked += 1
        return blocked
    
    def backtrack_search(S, sums, depth):
        """Backtracking search to find minimal maximal Sidon set."""
        nonlocal best_size, best_set
        
        if time.time() > deadline - 1.0:
            return False
        
        # Check if we've found a better solution
        if len(S) >= best_size:
            return False
        
        # Check if all remaining elements are blocked (maximal)
        if is_maximal_fast(S, sums):
            # We have a maximal Sidon set
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
                return True
            return False
        
        # Find all addable elements
        S_set = set(S)
        candidates = []
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                # Count how many elements this would block
                blocked = count_blocked_fast(S, sums, x)
                candidates.append((blocked, x))
        
        if not candidates:
            return False
        
        # Sort by blocking power (descending)
        candidates.sort(reverse=True)
        
        # Limit branching factor for performance - use more branches
        max_branch = min(len(candidates), 8)
        
        for blocked, x in candidates[:max_branch]:
            # Add x to S
            S.append(x)
            old_sums = set(sums)
            sums.add(2 * x)
            for a in S[:-1]:  # Don't include x itself in the loop
                sums.add(x + a)
            
            # Recurse
            if backtrack_search(S, sums, depth + 1):
                # Restore state
                S.pop()
                sums.clear()
                sums.update(old_sums)
                return True
            
            # Restore state
            S.pop()
            sums.clear()
            sums.update(old_sums)
        
        return False
    
    # Initial greedy to get a good starting point
    def greedy_initial():
        S = []
        S_set = set()
        sums = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_score = -1
            for x in remaining:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in S:
                    if (x + a) in sums:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Count how many elements this would block
                new_sums = sums | {2 * x}
                for a in S:
                    new_sums.add(x + a)
                
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    if 2 * y in new_sums:
                        blocked += 1
                        continue
                    y_blocked = False
                    for a in S:
                        if (y + a) in new_sums:
                            y_blocked = True
                            break
                    if y_blocked:
                        blocked += 1
                
                if blocked > best_score:
                    best_score = blocked
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            sums.add(2 * best_x)
            for a in S[:-1]:
                sums.add(best_x + a)
            remaining.remove(best_x)
        
        return S
    
    initial_set = greedy_initial()
    if initial_set:
        best_size = len(initial_set)
        best_set = list(initial_set)
    
    # Run backtracking search with different random seeds
    for seed in range(20):
        if time.time() > deadline - 3.0:
            break
        
        random.seed(seed)
        
        # Create a shuffled order for backtracking
        S = []
        sums = set()
        
        # Add a few random elements first to break symmetry
        num_initial = random.randint(2, 6)
        for _ in range(num_initial):
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                break
            x = random.choice(candidates)
            
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            S.append(x)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
        
        # Run backtracking
        backtrack_search(S, sums, 0)
    
    if best_set is not None:
        return best_set
    else:
        return initial_set
# EVOLVE-BLOCK-END
