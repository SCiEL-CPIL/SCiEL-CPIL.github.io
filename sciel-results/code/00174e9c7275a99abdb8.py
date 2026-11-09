import time
import random
import sys
from collections import deque

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin
    
    best_set = [31, 32, 34]
    best_size = 3
    
    # Precompute divisibility relationships for all possible subset sums
    # Max subset sum for 30 numbers from 1..256: at most 30*256 = 7680
    # But realistically with distinct numbers, max sum is around 256+255+...+227 ≈ 7440
    # Let's use a reasonable bound
    MAX_SUM = 8000
    
    # divisors[i] = set of j < i such that i % j == 0
    # This helps us check divisibility quickly
    divisors = [set() for _ in range(MAX_SUM + 1)]
    multiples = [set() for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i * 2, MAX_SUM + 1, i):
            divisors[j].add(i)
            multiples[i].add(j)
    
    # Precompute a bitset of "conflicting" sums for each sum value
    # conflict_mask[s] is a set of sums that conflict with s (divisibility)
    # We'll use a different approach: for each sum, precompute which other sums it conflicts with
    
    def is_valid_full(S):
        """Check if set S is valid using full subset sum enumeration."""
        if not S:
            return True
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        
        non_empty = [s for s in sums if s != 0]
        # Check distinctness (already guaranteed by set)
        # Check divisibility
        non_empty_set = set(non_empty)
        for u in non_empty:
            # Check if any multiple of u is in the set
            for m in multiples[u]:
                if m in non_empty_set:
                    return False
            # Check if any divisor of u is in the set
            for d in divisors[u]:
                if d in non_empty_set:
                    return False
        return True
    
    def try_add_fast(x, sums_set):
        """
        Fast check: can x be added to a set with subset sums sums_set?
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        Uses incremental divisibility checking.
        """
        # Check distinctness: no s in sums_set such that s + x is also in sums_set
        # This means sums_set and {s+x for s in sums_set} must be disjoint
        for s in sums_set:
            if s + x in sums_set:
                return (False, None)
        
        new_sums = set()
        for s in sums_set:
            new_sums.add(s + x)
        
        # Check divisibility only for the NEW sums against all sums
        # A new sum u conflicts if:
        # 1. Some existing sum d divides u (d in divisors[u] and d in sums_set, d != 0)
        # 2. Some existing sum m is a multiple of u (m in multiples[u] and m in sums_set)
        # 3. Some new sum d divides u (d in divisors[u] and d in new_sums, d != 0, d != u)
        # 4. Some new sum m is a multiple of u (m in multiples[u] and m in new_sums, m != u)
        
        all_non_empty = (sums_set | new_sums) - {0}
        
        for u in new_sums:
            if u == 0:
                continue
            # Check against existing sums
            for d in divisors[u]:
                if d in sums_set:
                    return (False, None)
            for m in multiples[u]:
                if m in sums_set:
                    return (False, None)
            # Check against other new sums
            for d in divisors[u]:
                if d in new_sums:
                    return (False, None)
            for m in multiples[u]:
                if m in new_sums:
                    return (False, None)
        
        return (True, sums_set | new_sums)
    
    def greedy_build(candidates, start_S=None, start_sums=None):
        """Greedy build: iterate through candidates, add if valid."""
        if start_S is None:
            S = []
            sums_set = {0}
        else:
            S = list(start_S)
            sums_set = set(start_sums)
        
        for x in candidates:
            if x in S:
                continue
            ok, new_sums = try_add_fast(x, sums_set)
            if ok:
                S.append(x)
                sums_set = new_sums
        
        return S, sums_set
    
    # Strategy 1: Binary signature approach
    def construct_binary_signature():
        """
        Assign each integer a unique power of 2 as its "signature".
        Numbers with unique highest bits ensure subset sum distinctness.
        """
        by_highest_bit = {}
        for n in range(1, 257):
            hb = n.bit_length() - 1
            if hb not in by_highest_bit:
                by_highest_bit[hb] = []
            by_highest_bit[hb].append(n)
        
        best_local = []
        
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            bit_positions = sorted(by_highest_bit.keys(), reverse=True)
            random.shuffle(bit_positions)
            
            S = []
            sums_set = {0}
            
            for bp in bit_positions:
                if time.time() - start_time > time_limit:
                    break
                
                candidates = by_highest_bit[bp][:]
                random.shuffle(candidates)
                
                for x in candidates:
                    ok, new_sums = try_add_fast(x, sums_set)
                    if ok:
                        S.append(x)
                        sums_set = new_sums
                        break
            
            if len(S) > len(best_local):
                best_local = S[:]
        
        return best_local
    
    # Strategy 2: Superincreasing construction
    # If we pick numbers such that each is greater than the sum of all previous,
    # then all subset sums are unique. Then we just need to check divisibility.
    def construct_superincreasing():
        """Build a superincreasing sequence, then filter for divisibility."""
        best_local = []
        
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Start with a random small number
            S = []
            current_sum = 0
            
            # Pick numbers in increasing order, ensuring superincreasing property
            # a[i] > sum(a[0..i-1])
            for _ in range(15):
                if time.time() - start_time > time_limit:
                    break
                
                # Minimum next number to maintain superincreasing
                min_next = current_sum + 1
                max_next = 256
                
                if min_next > max_next:
                    break
                
                # Randomly pick from [min_next, max_next]
                x = random.randint(min_next, max_next)
                
                # Check if adding x maintains validity (divisibility)
                ok, new_sums = try_add_fast(x, {0} | set())
                # Wait, we need to track sums properly
                # For superincreasing, sums are unique, so we just need to track them
                pass
            
            # Actually, let's do it properly
            S = []
            sums_set = {0}
            current_sum = 0
            
            for _ in range(15):
                if time.time() - start_time > time_limit:
                    break
                
                min_next = current_sum + 1
                max_next = 256
                
                if min_next > max_next:
                    break
                
                # Try a few random candidates
                found = False
                for _ in range(10):
                    x = random.randint(min_next, max_next)
                    if x in S:
                        continue
                    ok, new_sums = try_add_fast(x, sums_set)
                    if ok:
                        S.append(x)
                        sums_set = new_sums
                        current_sum += x
                        found = True
                        break
                
                if not found:
                    break
            
            if len(S) > len(best_local):
                best_local = S[:]
        
        return best_local
    
    # Strategy 3: Randomized greedy with bias towards larger numbers
    def randomized_greedy():
        """Randomized greedy with various biases."""
        best_local = []
        
        for restart in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Create candidate list with different biases
            bias = random.random()
            
            if bias < 0.3:
                # Descending order
                candidates = list(range(256, 0, -1))
            elif bias < 0.6:
                # Ascending order
                candidates = list(range(1, 257))
            elif bias < 0.8:
                # Random
                candidates = list(range(1, 257))
                random.shuffle(candidates)
            else:
                # Weighted towards middle
                candidates = list(range(1, 257))
                random.shuffle(candidates)
                # Sort by distance from 128
                candidates.sort(key=lambda x: abs(x - 128))
                random.shuffle(candidates)
            
            S, sums_set = greedy_build(candidates)
            
            if len(S) > len(best_local):
                best_local = S[:]
                # Try to extend
                remaining = [x for x in range(1, 257) if x not in S]
                random.shuffle(remaining)
                S2, sums_set2 = greedy_build(remaining, S, sums_set)
                if len(S2) > len(best_local):
                    best_local = S2[:]
        
        return best_local
    
    # Strategy 4: Local search / hill climbing
    def local_search(start_S):
        """Try to improve a set by swapping elements."""
        if not start_S:
            return []
        
        S = list(start_S)
        
        # Compute sums
        sums_set = {0}
        for x in S:
            new_sums = set()
            for s in sums_set:
                new_sums.add(s + x)
            sums_set = sums_set | new_sums
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            for i in range(len(S)):
                if time.time() - start_time > time_limit:
                    break
                
                removed = S[i]
                S_reduced = S[:i] + S[i+1:]
                
                # Recompute sums without removed
                sums_reduced = {0}
                for x in S_reduced:
                    new_sums = set()
                    for s in sums_reduced:
                        new_sums.add(s + x)
                    sums_reduced = sums_reduced | new_sums
                
                # Try adding each candidate
                for x in range(1, 257):
                    if x in S_reduced:
                        continue
                    ok, new_sums = try_add_fast(x, sums_reduced)
                    if ok:
                        new_S = S_reduced + [x]
                        if len(new_S) > len(S):
                            S = new_S
                            sums_set = new_sums
                            improved = True
                            break
        
        return S
    
    # Run all strategies
    # Strategy 1: Binary signature
    binary_result = construct_binary_signature()
    if len(binary_result) > best_size:
        best_size = len(binary_result)
        best_set = binary_result[:]
    
    # Strategy 2: Superincreasing
    super_result = construct_superincreasing()
    if len(super_result) > best_size:
        best_size = len(super_result)
        best_set = super_result[:]
    
    # Strategy 3: Randomized greedy
    greedy_result = randomized_greedy()
    if len(greedy_result) > best_size:
        best_size = len(greedy_result)
        best_set = greedy_result[:]
    
    # Strategy 4: Local search on best found
    if time.time() - start_time < time_limit - 5:
        ls_result = local_search(best_set)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_set = ls_result[:]
    
    # Final verification
    if len(best_set) > 0:
        if not is_valid_full(best_set):
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
