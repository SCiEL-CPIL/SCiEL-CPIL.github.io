import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    
    # Precompute conflict information using bitsets
    # For each candidate x, which existing sums would be created
    # We need: for a set S, x is compatible iff 2x not in sums(S) and x+y not in sums(S) for all y in S
    
    # Strategy: Build a conflict graph where two candidates x,y conflict if they can't both be in a Sidon set
    # Two numbers x,y (x<y) conflict with existing set S if:
    #   - 2x in sums or 2y in sums or x+y in sums
    #   - OR adding both creates: 2x, x+y, 2y all distinct AND x+y not equal to any existing sum
    # Actually the constraint is on the full set's pairwise sums being distinct.
    
    # Better approach: Use the structure that a Sidon set has all pairwise sums distinct.
    # This is equivalent to a Golomb ruler (all pairwise differences distinct) after transformation.
    # For Sidon: sums a_i + a_j for i<=j are all distinct.
    # For Golomb: differences a_j - a_i for i<j are all distinct.
    # They're related but not identical.
    
    # Let's use a direct branch-and-bound with bitset acceleration.
    # Represent the set of used sums as a bitset (sums range from 2 to 568).
    
    def solve_with_bb():
        """Branch and bound with bitset-based sum tracking."""
        # Sums range from 1+1=2 to 284+284=568
        # Use a list of bools or a set for sums
        # For speed, use a set and also track which candidates are still available
        
        best = list(prescribed)
        best_size = len(prescribed)
        
        # Initial sums from prescribed elements
        init_sums = set()
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                init_sums.add(prescribed[i] + prescribed[j])
        
        # Sort prescribed
        prescribed_sorted = sorted(prescribed)
        
        # Candidate pool: all numbers 1..284 not in prescribed
        candidates = [x for x in range(1, max_val + 1) if x not in prescribed_set]
        
        # Precompute for each candidate x, the sums it would add given a set S:
        # adds = {2*x} ∪ {x+y : y in S}
        # x is compatible iff none of these sums are already in the sum set
        
        def can_add(x, S, sums):
            """Check if x can be added to S without creating duplicate sums."""
            if 2 * x in sums:
                return False
            for y in S:
                if x + y in sums:
                    return False
            return True
        
        def branch_and_bound(S, sums, candidates_remaining, depth=0):
            """Recursive B&B. S is sorted list, sums is set of pairwise sums."""
            nonlocal best, best_size
            
            # Time check
            if time.time() > deadline - 2:
                return
            
            # Upper bound: len(S) + len(candidates_remaining)
            if len(S) + len(candidates_remaining) <= best_size:
                return
            
            # Try to add candidates in some order
            # Order: prefer candidates that create fewer conflicts (more flexible)
            # For simplicity, try in descending order (larger numbers first tend to be more flexible)
            
            for i in range(len(candidates_remaining)):
                x = candidates_remaining[i]
                
                if can_add(x, S, sums):
                    # Add x
                    new_S = S + [x]
                    new_S.sort()
                    new_sums = sums.copy()
                    new_sums.add(2 * x)
                    for y in S:
                        new_sums.add(x + y)
                    
                    # Recurse with remaining candidates after i
                    remaining = candidates_remaining[i+1:]
                    
                    if len(new_S) > best_size:
                        best = new_S.copy()
                        best_size = len(new_S)
                    
                    branch_and_bound(new_S, new_sums, remaining, depth + 1)
                
                # Also try skipping x (implicit by continuing the loop)
            
            # The loop above implicitly handles both add and skip by trying each candidate
            # But we need to be more careful - we should try adding x AND not adding x
            # Actually the standard B&B: at each node, for each candidate, decide include or exclude
            # A cleaner formulation:
        
        # Let me restructure: use a proper B&B where at each step we pick the next candidate to consider
        # and branch on include/exclude
        
        def bb_recursive(S, sums, candidate_idx):
            """B&B where candidate_idx is the next candidate to consider from the sorted list."""
            nonlocal best, best_size
            
            if time.time() > deadline - 2:
                return
            
            # Upper bound
            if len(S) + (len(candidates) - candidate_idx) <= best_size:
                return
            
            if candidate_idx >= len(candidates):
                if len(S) > best_size:
                    best = S.copy()
                    best_size = len(S)
                return
            
            x = candidates[candidate_idx]
            
            # Branch 1: Include x
            if can_add(x, S, sums):
                new_S = S + [x]
                new_S.sort()
                new_sums = sums.copy()
                new_sums.add(2 * x)
                for y in S:
                    new_sums.add(x + y)
                
                if len(new_S) > best_size:
                    best = new_S.copy()
                    best_size = len(new_S)
                
                bb_recursive(new_S, new_sums, candidate_idx + 1)
            
            # Branch 2: Exclude x
            bb_recursive(S, sums, candidate_idx + 1)
        
        # Order candidates to improve pruning: try larger values first (they're more likely to fit)
        candidates.sort(reverse=True)
        
        # But we also want to prioritize candidates that don't conflict much
        # For now, just use reverse order
        
        initial_S = list(prescribed_sorted)
        initial_sums = set()
        for i in range(len(initial_S)):
            for j in range(i, len(initial_S)):
                initial_sums.add(initial_S[i] + initial_S[j])
        
        bb_recursive(initial_S, initial_sums, 0)
        
        return best, best_size
    
    # First, try a quick greedy to get a good initial solution for B&B
    def greedy_initial():
        S = sorted(prescribed)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        
        for x in range(1, max_val + 1):
            if x in S:
                continue
            if 2 * x not in sums:
                ok = True
                for y in S:
                    if x + y in sums:
                        ok = False
                        break
                if ok:
                    S.append(x)
                    S.sort()
                    sums.add(2 * x)
                    for y in S[:-1]:
                        sums.add(x + y)
        return S, sums
    
    # Get initial solution via greedy
    greedy_S, greedy_sums = greedy_initial()
    best_size = len(greedy_S)
    best_S = greedy_S.copy()
    
    # Try B&B with a time limit
    if time.time() < deadline - 30:
        bb_best, bb_size = solve_with_bb()
        if bb_size > best_size:
            best_size = bb_size
            best_S = bb_best
    
    # If B&B didn't finish or didn't improve, do local search on best found
    if best_size < 20 and time.time() < deadline - 10:
        # Local search on best_S
        S = best_S.copy()
        S.sort()
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        
        S_set = set(S)
        improved = True
        max_iters = 500
        
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try adds
            for x in range(1, max_val + 1):
                if x in S_set:
                    continue
                if 2 * x not in sums:
                    ok = True
                    for y in S:
                        if x + y in sums:
                            ok = False
                            break
                    if ok:
                        S.append(x)
                        S.sort()
                        sums.add(2 * x)
                        for y in S[:-1]:
                            sums.add(x + y)
                        S_set.add(x)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swaps
            removable = [x for x in S if x not in prescribed_set]
            for r in removable:
                # Remove r
                S.remove(r)
                S_set.discard(r)
                new_sums = set()
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        new_sums.add(S[i] + S[j])
                
                # Try to add something
                for x in range(1, max_val + 1):
                    if x in S_set:
                        continue
                    if 2 * x not in new_sums:
                        ok = True
                        for y in S:
                            if x + y in new_sums:
                                ok = False
                                break
                        if ok:
                            S.append(x)
                            S.sort()
                            S_set.add(x)
                            sums = new_sums
                            sums.add(2 * x)
                            for y in S[:-1]:
                                sums.add(x + y)
                            improved = True
                            break
                
                if not improved:
                    # Restore r
                    S.append(r)
                    S.sort()
                    S_set.add(r)
                    sums = set()
                    for i in range(len(S)):
                        for j in range(i, len(S)):
                            sums.add(S[i] + S[j])
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    return best_S
