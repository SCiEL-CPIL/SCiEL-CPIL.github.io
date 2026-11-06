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
    
    # Use bitset for sum tracking - sums range from 2 to 568
    # sums_bit[i] = 1 if sum i is used
    SUM_MIN = 2
    SUM_MAX = 2 * max_val  # 568
    
    def make_sum_bitset(sums_set):
        """Convert a set of sums to a bitset (list of bools)."""
        bs = [False] * (SUM_MAX + 1)
        for s in sums_set:
            if 0 <= s <= SUM_MAX:
                bs[s] = True
        return bs
    
    def can_add_bitset(x, S, sums_bs):
        """Check if x can be added using bitset for O(1) sum lookups."""
        if sums_bs[2 * x]:
            return False
        for y in S:
            if sums_bs[x + y]:
                return False
        return True
    
    def greedy_initial():
        """Quick greedy to get a good initial solution."""
        S = sorted(prescribed)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        sums_bs = make_sum_bitset(sums)
        S_set = set(S)
        
        for x in range(1, max_val + 1):
            if x in S_set:
                continue
            if can_add_bitset(x, S, sums_bs):
                S.append(x)
                S.sort()
                sums_bs[2 * x] = True
                for y in S[:-1]:
                    sums_bs[x + y] = True
                S_set.add(x)
        return S, sums_bs
    
    # Get initial solution via greedy
    greedy_S, greedy_sums_bs = greedy_initial()
    best_size = len(greedy_S)
    best_S = greedy_S.copy()
    
    # Candidate pool
    candidates = [x for x in range(1, max_val + 1) if x not in prescribed_set]
    candidates.sort(reverse=True)  # Larger values first for better pruning
    
    def bb_recursive(S, sums_bs, candidate_idx, depth=0):
        """Branch and bound with bitset sum tracking and forward checking."""
        nonlocal best, best_size, deadline
        
        # Time check
        if time.time() > deadline - 2:
            return
        
        # Upper bound: can we possibly beat best?
        remaining = len(candidates) - candidate_idx
        if len(S) + remaining <= best_size:
            return
        
        # Forward checking: count how many remaining candidates are still addable
        # This gives a tighter upper bound
        addable_count = 0
        for i in range(candidate_idx, len(candidates)):
            x = candidates[i]
            if can_add_bitset(x, S, sums_bs):
                addable_count += 1
        
        if len(S) + addable_count <= best_size:
            return
        
        if candidate_idx >= len(candidates):
            if len(S) > best_size:
                best = S.copy()
                best_size = len(S)
            return
        
        x = candidates[candidate_idx]
        
        # Branch 1: Include x (if possible)
        if can_add_bitset(x, S, sums_bs):
            new_S = S + [x]
            new_sums_bs = sums_bs.copy()
            new_sums_bs[2 * x] = True
            for y in S:
                new_sums_bs[x + y] = True
            
            if len(new_S) > best_size:
                best = new_S.copy()
                best_size = len(new_S)
            
            bb_recursive(new_S, new_sums_bs, candidate_idx + 1, depth + 1)
        
        # Branch 2: Exclude x
        bb_recursive(S, sums_bs, candidate_idx + 1, depth + 1)
    
    # Run B&B
    if time.time() < deadline - 5:
        initial_S = sorted(prescribed)
        initial_sums = set()
        for i in range(len(initial_S)):
            for j in range(i, len(initial_S)):
                initial_sums.add(initial_S[i] + initial_S[j])
        initial_sums_bs = make_sum_bitset(initial_sums)
        
        bb_recursive(initial_S, initial_sums_bs, 0)
    
    # If B&B didn't reach 20, try local search with swaps
    if best_size < 20 and time.time() < deadline - 10:
        S = best_S.copy()
        S.sort()
        sums_bs = make_sum_bitset(set())
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums_bs[S[i] + S[j]] = True
        S_set = set(S)
        
        improved = True
        iters = 0
        max_iters = 1000
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try adds
            for x in range(1, max_val + 1):
                if x in S_set:
                    continue
                if can_add_bitset(x, S, sums_bs):
                    S.append(x)
                    S.sort()
                    sums_bs[2 * x] = True
                    for y in S[:-1]:
                        sums_bs[x + y] = True
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one non-prescribed, add one new
            removable = [x for x in S if x not in prescribed_set]
            for r in removable:
                # Remove r
                S.remove(r)
                S_set.discard(r)
                
                # Rebuild sums without r
                new_sums_bs = [False] * (SUM_MAX + 1)
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        new_sums_bs[S[i] + S[j]] = True
                
                # Try to add something
                found = False
                for x in range(1, max_val + 1):
                    if x in S_set:
                        continue
                    if can_add_bitset(x, S, new_sums_bs):
                        S.append(x)
                        S.sort()
                        new_sums_bs[2 * x] = True
                        for y in S[:-1]:
                            new_sums_bs[x + y] = True
                        S_set.add(x)
                        sums_bs = new_sums_bs
                        improved = True
                        found = True
                        break
                
                if not found:
                    # Restore r
                    S.append(r)
                    S.sort()
                    S_set.add(r)
                    sums_bs = [False] * (SUM_MAX + 1)
                    for i in range(len(S)):
                        for j in range(i, len(S)):
                            sums_bs[S[i] + S[j]] = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    return best_S
