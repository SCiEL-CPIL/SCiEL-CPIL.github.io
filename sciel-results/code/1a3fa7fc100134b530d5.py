"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    seed_elements = [1, 2, 5, 11]
    
    # Precompute: for each pair (i,j) with i<=j in 1..284, store the sum
    # For bitset operations, we'll track which sums are used
    # Sums range from 1+1=2 to 284+284=568, so we need bits 2..568
    
    def check_valid(a):
        """Check if sorted list a is a valid Sidon set."""
        n = len(a)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = a[i] + a[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def greedy_extend_bitset(a, candidates, deadline):
        """Greedy extension using bitset for sum tracking."""
        a = list(a)
        n = len(a)
        # Bitset of used sums: bit s is set if sum s is used
        used_sums = 0
        for i in range(n):
            for j in range(i, n):
                used_sums |= (1 << (a[i] + a[j]))
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in a:
                continue
            # Check if x can be added
            new_sums = 0
            for y in a:
                s = x + y
                if (used_sums >> s) & 1:
                    break
                new_sums |= (1 << s)
            else:
                s2 = 2 * x
                if (used_sums >> s2) & 1:
                    continue
                new_sums |= (1 << s2)
                a.append(x)
                a.sort()
                used_sums |= new_sums
                n = len(a)
        
        return a, used_sums
    
    def backtrack_csp(a, used_sums, remaining, target_size, deadline, depth=0):
        """
        CSP backtracking with forward checking.
        a: current sorted set
        used_sums: bitset of used sums
        remaining: set of candidate elements not yet decided
        target_size: desired final size
        """
        if time.time() > deadline:
            return None
        
        n = len(a)
        if n >= target_size:
            return a[:]
        
        # Forward checking: determine which candidates are still valid
        valid_candidates = []
        for x in remaining:
            if x in a:
                continue
            # Check if x can be added without conflict
            ok = True
            for y in a:
                s = x + y
                if (used_sums >> s) & 1:
                    ok = False
                    break
            if ok:
                s2 = 2 * x
                if (used_sums >> s2) & 1:
                    ok = False
            if ok:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Dynamic variable ordering: choose candidate with fewest remaining valid partners
        # Score each candidate by how many other valid candidates it would block
        best_candidate = None
        best_score = float('inf')
        
        for x in valid_candidates:
            # Count how many other valid candidates would become invalid if we add x
            blocked = 0
            for y in valid_candidates:
                if y == x:
                    continue
                # Check if adding x would block y
                for z in a:
                    s1 = x + z
                    s2 = y + z
                    if s1 == s2:  # This can't happen since x != y
                        blocked += 1
                        break
                    # Check if x+z conflicts with y+z or 2y
                    # Actually, we need to check if y's sums conflict with x's sums
                    # y's new sums: {y+z for z in a} ∪ {2y}
                    # x's new sums: {x+z for z in a} ∪ {2x}
                    # Conflict if any x+z equals any y+w or 2y, or 2x equals y+w or 2y
                    pass
                # Simpler: just count valid candidates that share a sum with x
                x_sums = set()
                for z in a:
                    x_sums.add(x + z)
                x_sums.add(2 * x)
                
                for y2 in valid_candidates:
                    if y2 == x:
                        continue
                    y_sums = set()
                    for z in a:
                        y_sums.add(y2 + z)
                    y_sums.add(2 * y2)
                    if x_sums & y_sums:
                        blocked += 1
            
            if blocked < best_score:
                best_score = blocked
                best_candidate = x
        
        if best_candidate is None:
            return None
        
        x = best_candidate
        new_a = a + [x]
        new_a.sort()
        
        # Update used_sums
        new_used = used_sums
        for y in a:
            new_used |= (1 << (x + y))
        new_used |= (1 << (2 * x))
        
        # Update remaining
        new_remaining = set(remaining)
        new_remaining.discard(x)
        
        result = backtrack_csp(new_a, new_used, new_remaining, target_size, deadline, depth + 1)
        if result is not None:
            return result
        
        # Backtrack: try next candidate
        # For now, just return None to indicate failure
        return None
    
    # Strategy 1: Fast greedy with bitsets
    best_result = list(seed_elements)
    candidates = list(range(12, 285))
    best_result, _ = greedy_extend_bitset(best_result, candidates, deadline)
    
    # Strategy 2: Multiple randomized greedy restarts with bitsets
    random.seed(42)
    num_restarts = 200
    per_restart_deadline = deadline - 10
    
    for r in range(num_restarts):
        if time.time() > per_restart_deadline:
            break
        
        a = list(seed_elements)
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a, _ = greedy_extend_bitset(a, candidates, per_restart_deadline)
        
        if len(a) > len(best_result):
            best_result = a
    
    # Strategy 3: Local search with bitsets
    def local_search_bitset(a, deadline, max_iters=5000):
        """Local search: remove one element, try to add another."""
        a = sorted(a)
        best = list(a)
        best_size = len(a)
        
        n = len(a)
        used_sums = 0
        for i in range(n):
            for j in range(i, n):
                used_sums |= (1 << (a[i] + a[j]))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            if len(a) > len(seed_elements):
                # Remove a random non-seed element
                non_seed_indices = [i for i, x in enumerate(a) if x not in seed_elements]
                if non_seed_indices:
                    idx = random.choice(non_seed_indices)
                    removed = a[idx]
                    a_new = a[:idx] + a[idx+1:]
                    a_new.sort()
                    
                    # Recompute used_sums for a_new
                    n_new = len(a_new)
                    used_new = 0
                    for i in range(n_new):
                        for j in range(i, n_new):
                            used_new |= (1 << (a_new[i] + a_new[j]))
                    
                    # Try to add a random element
                    for _ in range(30):
                        x = random.randint(1, 284)
                        if x in a_new:
                            continue
                        # Check if x can be added
                        ok = True
                        for y in a_new:
                            s = x + y
                            if (used_new >> s) & 1:
                                ok = False
                                break
                        if ok:
                            s2 = 2 * x
                            if (used_new >> s2) & 1:
                                ok = False
                        if ok:
                            a_new.append(x)
                            a_new.sort()
                            used_new |= (1 << (2 * x))
                            for y in a_new[:-1]:
                                used_new |= (1 << (x + y))
                            
                            # Try greedy extension
                            a_new, used_new = greedy_extend_bitset(a_new, list(range(1, 285)), deadline)
                            
                            if len(a_new) > len(a):
                                a = a_new
                                used_sums = used_new
                                improved = True
                                if len(a) > best_size:
                                    best = list(a)
                                    best_size = len(a)
                                break
            
            if not improved and random.random() < 0.05:
                pass
        
        return best
    
    # Apply local search to best result
    if len(best_result) >= 15:
        improved = local_search_bitset(best_result, deadline, max_iters=10000)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Strategy 4: Try to reach higher sizes with more aggressive search
    for target in [19, 20]:
        if len(best_result) >= target:
            break
        if time.time() > deadline - 5:
            break
        
        # Try greedy from best_result
        a = list(best_result)
        a, _ = greedy_extend_bitset(a, list(range(1, 285)), deadline)
        if len(a) > len(best_result):
            best_result = a
    
    # Final verification
    final_a = sorted(best_result)
    assert all(x in final_a for x in seed_elements), "Missing seed elements"
    assert all(1 <= x <= 284 for x in final_a), "Out of range"
    assert len(set(final_a)) == len(final_a), "Duplicates"
    assert check_valid(final_a), "Invalid Sidon set"
    
    return final_a
# EVOLVE-BLOCK-END
