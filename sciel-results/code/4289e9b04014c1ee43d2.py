import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    N = 256

    def check_valid(seq):
        """Return True if all non-empty consecutive block sums are distinct."""
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def compute_sums(seq):
        """Compute all subarray sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def backtrack(seq, used, sums, depth, max_depth, time_limit):
        """
        Backtracking search with forward checking.
        seq: current sequence (sorted)
        used: set of used numbers
        sums: set of current subarray sums
        depth: current length
        max_depth: target length
        """
        if time.time() > time_limit:
            return None
        
        if depth == max_depth:
            return seq[:]
        
        # Heuristic: Try numbers that are 'compatible' with the current end
        # Since the sequence is strictly increasing, the next number must be > seq[-1] (if seq exists)
        # Actually, we are building the set, then sorting. 
        # Wait, if we build the set, the order is fixed by sorting.
        # So we are choosing the NEXT element to add to the set.
        # The position in the sorted sequence is determined by its value.
        
        # Optimization: 
        # When we add a new number 'x' to the set, it will be inserted at some position 'k' in the sorted sequence.
        # The new subarray sums generated are those that include 'x'.
        # Specifically, if the sorted sequence is S, and we insert x at index k, 
        # the new sums are:
        # 1. Sums ending at x: sum(S[i:k]) + x for i in [0..k] (actually sum(S[i:k+1]))
        # 2. Sums starting at x: sum(S[k:j]) for j in [k+1..n]
        # 3. Sums passing through x: sum(S[i:j]) where i<k<j.
        
        # To make this efficient, we can precompute prefix sums of the current sequence.
        # But the sequence changes.
        
        # Let's stick to a simpler, effective heuristic:
        # We try to add a number 'x' not in 'used'.
        # We check if 'x' can be added without creating duplicate sums.
        
        # To speed up the check, we only check the NEW sums created by adding 'x'.
        # The new sums are those that include 'x'.
        # Let current sorted seq be S.
        # If we insert x, new sorted seq is S'.
        # New sums are { sum(S'[i:j]) | x in S'[i:j] }.
        
        # Instead of generating all, we can check:
        # Does any new sum already exist in 'sums'?
        
        # Let's try a random order of candidates to avoid getting stuck in a specific local optimum.
        # But for backtracking, we need a deterministic order or a randomized restart.
        
        # Candidate generation:
        # We want to find a valid sequence of length max_depth.
        # We'll try to add elements one by one.
        
        # To avoid O(N^2) check for every candidate, we can be smart.
        # But N=256 is small enough that a full check O(N^2) ~ 65k ops is fast if we limit the depth.
        # However, backtracking depth might be large.
        
        # Let's use a randomized greedy backtracking.
        
        # Determine candidates
        # We can bias towards numbers that are "far" from existing sums?
        # Or just random.
        
        # To make it a proper search, we should try candidates in a specific order.
        # Let's shuffle the available numbers.
        
        available = [x for x in range(1, N + 1) if x not in used]
        random.shuffle(available)
        
        for x in available:
            # Quick check: is x too large?
            # If we are building a sequence of length L, the smallest possible sum is 1+...+L.
            # The largest possible sum is (N-L+1)+...+N.
            # This doesn't prune much.
            
            # Check validity of adding x
            new_seq = sorted(seq + [x])
            # Check if new_seq is valid
            # We can optimize this by only checking sums involving x.
            
            # Let's just use the full check for correctness first, then optimize if needed.
            # But full check is O(L^2). If L is ~20, that's 400 ops. Very fast.
            # If L is ~50, 2500 ops. Still fast.
            # The bottleneck is the branching factor.
            
            if check_valid(new_seq):
                # Recurse
                new_sums = compute_sums(new_seq)
                result = backtrack(new_seq, used | {x}, new_sums, depth + 1, max_depth, time_limit)
                if result is not None:
                    return result
        
        return None

    def solve_for_length(L, time_limit):
        """
        Try to find a valid sequence of length L using backtracking.
        """
        # Initial state
        # We can start with an empty sequence.
        # But backtracking from empty is too broad.
        # Let's seed it with a greedy sequence of length L//2 or so?
        # Or just start empty and let the heuristic guide it.
        
        # A better start: Use the greedy_init to get a good starting set of size ~L/2.
        # Then backtrack to extend it.
        
        # However, if we start with a greedy set, we might be stuck in a local optimum of the greedy.
        # Backtracking can undo choices? No, my current backtrack only adds.
        # To allow undoing, we need to pass the state back or use a different structure.
        
        # Let's modify backtrack to allow "undo" by trying different candidates at each step.
        # The current implementation does this: it tries 'x', recurses. If recursion fails, it tries next 'x'.
        # This effectively backtracks.
        
        # But starting from empty with depth L is huge.
        # Let's use a "Hybrid" approach:
        # 1. Greedy build a sequence of length K (e.g., 10).
        # 2. Backtrack to extend to L.
        
        # Let's try starting from a few different greedy seeds.
        
        # Greedy seed
        seed = []
        for x in range(1, N + 1):
            new_seq = sorted(seed + [x])
            if check_valid(new_seq):
                seed = new_seq
                if len(seed) >= L // 2:
                    break
        
        if not seed:
            return None
            
        # Ensure seed is valid (it should be)
        if not check_valid(seed):
            return None
            
        # Backtrack from seed
        used = set(seed)
        sums = compute_sums(seed)
        
        result = backtrack(seed, used, sums, len(seed), L, time_limit)
        return result

    # Main Logic
    best_seq = []
    
    # First, get a strong baseline using greedy
    def greedy_init(order):
        seq = []
        for x in order:
            new_seq = sorted(seq + [x])
            if check_valid(new_seq):
                seq = new_seq
        return seq

    # Try a few greedy inits to find a good starting length
    best_greedy_len = 0
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        seq = greedy_init(order)
        if len(seq) > best_greedy_len:
            best_greedy_len = len(seq)
            best_seq = seq[:]

    # Now, use backtracking to try to extend beyond the greedy limit.
    # We don't know the max length. Let's binary search or increment.
    # Since greedy usually gets close to the optimum, let's try to extend from best_greedy_len.
    
    target_len = best_greedy_len
    max_possible = N
    
    # We will try to find a sequence of length target_len + 1, then +2, etc.
    # But backtracking for length L is expensive.
    # Let's set a time budget for each length attempt.
    
    current_best = best_seq
    current_len = best_greedy_len
    
    # Try to improve
    for attempt_len in range(current_len + 1, min(current_len + 10, N + 1)):
        if time.time() > deadline - 5:
            break
            
        time_left = deadline - 5 - time.time()
        if time_left < 1:
            break
            
        # Allocate time for this attempt
        attempt_budget = min(5.0, time_left)
        
        # Try multiple seeds
        found = None
        for seed_iter in range(5):
            if time.time() > deadline - 5:
                break
                
            # Create a seed
            if seed_iter == 0:
                seed = current_best[:]
            else:
                # Perturb current best
                seed = current_best[:]
                if seed:
                    k = random.randint(2, 5)
                    to_remove = random.sample(seed, min(k, len(seed)))
                    seed = [x for x in seed if x not in set(to_remove)]
                    # Re-greedy fill to stabilize?
                    # Or just use the perturbed seed directly.
                    # Let's re-greedy fill to make it a valid, dense seed.
                    new_seed = []
                    for x in seed:
                        new_seq = sorted(new_seed + [x])
                        if check_valid(new_seq):
                            new_seed = new_seq
                    seed = new_seed
                    
            if not seed:
                continue
                
            # Run backtrack from this seed
            used = set(seed)
            sums = compute_sums(seed)
            
            # We want to reach attempt_len
            # If seed is already >= attempt_len, we don't need to extend?
            # But we want to MAXIMIZE length.
            # So if seed is shorter, we extend.
            
            if len(seed) >= attempt_len:
                # We already have a sequence of this length.
                # But we are trying to find one of length attempt_len.
                # If we have one, we can just use it.
                # But we want to go higher.
                # So this path is only useful if we can extend it further.
                # Let's just try to extend to attempt_len.
                pass
            
            # Backtrack
            result = backtrack(seed, used, sums, len(seed), attempt_len, deadline - 5)
            
            if result is not None:
                found = result
                break
        
        if found is not None:
            current_best = found
            current_len = attempt_len
            # Update best_seq
            best_seq = found
        else:
            # If we failed to find length current_len + 1, maybe we can't go higher?
            # Not necessarily, but it's a strong signal.
            # Let's try one more time with a different seed before giving up on this length.
            # But to save time, let's assume if we can't find L+1, we stop.
            # This is a heuristic.
            break

    return best_seq

# Fallback: If the backtracking is too slow or fails, ensure we return a valid sequence.
# The greedy_init inside construct ensures best_seq is always valid.
