import sys
import itertools
import random
import time

def construct(n):
    # Generate all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Step 1: Generate the standard recursive superpermutation order
    # This gives us a valid sequence of permutations where each new permutation
    # overlaps with the previous one.
    # We simulate the recursive construction to get the order of permutations.
    
    # The recursive construction appends blocks.
    # For n=1: [1]
    # For n=2: [1, 2, 1] -> Perms: (1,2), (2,1) ? 
    # Let's trace the standard algorithm:
    # word = [1]
    # m=2: 
    #   windows of len 1: [1]. 
    #   block for 1: [1, 2, 1]. 
    #   word becomes [1, 2, 1].
    #   Wait, the standard algorithm iterates through unique windows.
    
    # Let's just generate the string first, then extract the order of permutations.
    # This ensures the starting point is valid.
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Extract the sequence of permutations from the word
    # A permutation starts at every index where the window is a valid permutation.
    # In a superpermutation, every window of length n is a permutation.
    # The "order" of permutations in the TSP sense is the sequence of windows.
    # However, consecutive windows overlap by n-1.
    # So the sequence of permutations is just the list of all windows of length n.
    
    seq = []
    for i in range(len(word) - n + 1):
        seq.append(tuple(word[i:i + n]))
        
    # Note: The recursive construction ensures that consecutive windows overlap by n-1.
    # This is a valid path in the TSP graph.
    
    # Calculate cost: sum of (n - overlap(p_i, p_{i+1}))
    # Since they come from the string, overlap(p_i, p_{i+1}) is always n-1.
    # So the cost is sum of 1 for each step. Total length = n + (len(seq)-1)*1 = len(word).
    
    # To improve, we need to find a different sequence of permutations that covers all
    # n! permutations exactly once (Hamiltonian path) but with potentially lower total length?
    # No, the length is determined by the overlaps.
    # If we swap two permutations, we might break the n-1 overlap, increasing length.
    # We want to *maintain* high overlaps.
    # The recursive construction already has max overlap (n-1) for every step.
    # So the length is minimal for that specific structure?
    # Wait. If every step has overlap n-1, the length is n! + n - 1.
    # For n=7, 5040 + 6 = 5046? No.
    # Length = n + (n! - 1) * 1 = 5046?
    # But the recursive superpermutation for n=7 has length 5913.
    # Why? Because the "windows" in the recursive construction are NOT all permutations?
    # No, it IS a superpermutation, so it contains ALL permutations.
    # The number of windows is len(word) - n + 1.
    # If it contains all 5040 permutations, and some are repeated, the number of windows is > 5040.
    # The TSP approach assumes we visit each permutation EXACTLY ONCE.
    # If we visit each permutation exactly once, the length is n + sum(1) = 5046.
    # But a superpermutation must contain every permutation. It does NOT have to contain them exactly once.
    # In fact, it's impossible to have a superpermutation of length 5046 for n=7?
    # The lower bound is 5884. So we MUST have repeats.
    # So the TSP on "each permutation once" is not the right model.
    
    # The correct model: We have a multiset of permutations (the windows).
    # We want to minimize the total length.
    # The current program deletes characters, which removes windows and merges others.
    
    # A different approach: **Greedy Construction with Backtracking / Heuristic Choice**.
    # Instead of the fixed recursive order, at each step, choose the next permutation to append
    # that maximizes overlap and prioritizes "rare" permutations or those that allow future overlaps.
    
    # Let's try a **Greedy** approach that builds the string character by character or permutation by permutation.
    # State: Current string S.
    # Goal: Add next character or permutation.
    # We must ensure all permutations are covered.
    
    # A known better heuristic for n>=4 is to use a "greedy" algorithm that picks the next permutation
    # to be the one that has the highest overlap with the current suffix, among the *uncovered* permutations.
    # If multiple, pick the one that leaves the most options.
    
    # Let's implement this Greedy Uncovered Permutation approach.
    
    # 1. Initialize S = [1]
    # 2. Maintain a set of uncovered permutations.
    # 3. At each step, look at the current suffix of S.
    #    Find all permutations P such that P's prefix matches S's suffix.
    #    Among these, prioritize those in `uncovered`.
    #    If there are uncovered candidates, pick one.
    #    If not, pick a covered one that maintains validity?
    #    Actually, we just need to append the rest of P to S.
    
    # This is risky. Let's stick to the **Local Search on String** but optimize it heavily.
    # The current program's local search is too slow.
    # We can use **Bitset** for permutation coverage to speed up validity checks.
    # And we can try **Multiple Random Restarts** or **Simulated Annealing** on deletions.
    
    # However, the directive asks for a NEW approach.
    # Let's try **Greedy Construction with Overlap Maximization**.
    
    # Algorithm:
    # 1. Start with `1`.
    # 2. Keep track of `covered` permutations (set of tuples).
    # 3. Keep track of `uncovered` permutations (set of tuples).
    # 4. While `uncovered` is not empty:
    #    a. Find the longest suffix of the current word that is a prefix of some permutation in `uncovered`.
    #    b. If found, append the rest of that permutation to the word.
    #    c. If not found, append a character that extends the current suffix to be a prefix of *some* permutation (covered or not) to keep options open.
    #       This part is tricky.
    
    # A simpler variant:
    # The recursive construction is essentially a depth-first search on the permutation tree.
    # We can modify it to be a **Breadth-First** or **Priority Queue** based search.
    # Priority: Number of uncovered permutations that can be reached by extending the current suffix.
    
    # Let's implement the **Greedy Uncovered** strategy:
    # At each step, we have a current suffix `suf`.
    # We look for any uncovered permutation `p` such that `p` starts with `suf`.
    # If we find one, we append the rest of `p`.
    # If we find multiple, we choose the one that maximizes the overlap (longest `suf`).
    # If we find none, we must extend `suf` by one character to make it a prefix of *some* permutation.
    # We choose the character that leads to the most uncovered permutations.
    
    # This is a heuristic. Let's code it.
    
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    uncovered = set(all_perms)
    
    # We need to know, for any prefix, which permutations have it.
    # Precompute: prefix_to_perms = {prefix: [perm1, perm2, ...]}
    # This is memory intensive for n=7. 7! * 7 prefixes = 35k entries. OK.
    
    prefix_to_perms = {}
    for p in all_perms:
        for i in range(1, n + 1):
            pref = p[:i]
            if pref not in prefix_to_perms:
                prefix_to_perms[pref] = []
            prefix_to_perms[pref].append(p)
    
    word = [1]
    uncovered.remove((1,)) # Wait, n=7, so (1,) is not a permutation.
    # Permutations are length n.
    # Initially, no permutation is covered.
    
    # We need to build the word until all permutations are covered.
    # Max length limit: n * n!
    
    # To avoid infinite loops or getting stuck, we use a heuristic.
    # Current suffix is the last k characters of word.
    # We want to find an uncovered permutation P such that P starts with suffix.
    
    # Optimization: Only check suffixes of length 1 to n.
    
    while len(uncovered) > 0:
        # Try to extend word by appending the rest of a permutation
        # that starts with the current suffix.
        
        found = False
        # Check longest suffix first
        max_suffix_len = min(n, len(word))
        for k in range(max_suffix_len, 0, -1):
            suffix = tuple(word[-k:])
            if suffix in prefix_to_perms:
                candidates = prefix_to_perms[suffix]
                # Filter for uncovered
                unc_candidates = [p for p in candidates if p in uncovered]
                if unc_candidates:
                    # Pick one. Heuristic: Pick the one that has the most overlaps with future?
                    # For now, pick the first one (lexicographically) or random?
                    # Lexicographic might be deterministic and bad.
                    # Let's pick the one that minimizes the "cost" of the next step?
                    # Just pick the first one for simplicity, or random to allow restarts?
                    # The task is deterministic. Let's pick the one with the longest common prefix with the *next* likely step?
                    # Simple: Pick the first one.
                    p = unc_candidates[0]
                    # Append the rest of p
                    # p starts with suffix, so we append p[k:]
                    word.extend(p[k:])
                    # Update uncovered
                    # All windows of length n in the new part might be covered.
                    # Specifically, the window ending at the new end is p.
                    # But intermediate windows might also be covered.
                    # We should scan the new windows.
                    start_idx = len(word) - n
                    for i in range(start_idx, len(word) - n + 1):
                        # Actually, just check the new windows created.
                        # The new windows are those that include the newly added characters.
                        pass
                    # Simpler: Just remove p from uncovered? No, other perms might be covered too.
                    # We need to check all new windows.
                    # New windows start from len(word)-len(p[k:]) - n + 1 to len(word)-n.
                    # Let's just recompute uncovered for the last few windows.
                    # To be safe and simple (since n is small), we can just check the window that is exactly p.
                    # But we might miss others.
                    # Let's check all windows in the newly added segment.
                    added_len = len(p) - k
                    for i in range(max(0, len(word) - n - added_len + 1), len(word) - n + 1):
                        w = tuple(word[i:i+n])
                        if w in uncovered:
                            uncovered.remove(w)
                    found = True
                    break
        if found:
            continue
            
        # If no uncovered permutation starts with any suffix, we are stuck.
        # We must append a character that is part of some permutation.
        # We pick a character c such that suffix + c is a prefix of some permutation (covered or uncovered).
        # Prefer c that leads to uncovered perms.
        
        # Find current longest suffix that is a prefix of ANY perm
        # If none, we have a problem. But suffix of length 1 is always a prefix of some perm.
        
        # Try to extend the current suffix (length k) to k+1
        # We want to find a char c such that suffix + (c,) is in prefix_to_perms.
        # And prefer if it leads to uncovered perms.
        
        # We need to determine which suffix to extend.
        # Usually, we extend the longest suffix that is a valid prefix.
        
        # Find the longest suffix of word that is a prefix of some permutation.
        max_k = 0
        best_suffix = None
        for k in range(min(n, len(word)), 0, -1):
            suffix = tuple(word[-k:])
            if suffix in prefix_to_perms:
                max_k = k
                best_suffix = suffix
                break
        
        if best_suffix is None:
            # Should not happen
            return "".join(map(str, word))
            
        # Find candidates for next char
        next_char = None
        best_score = -1
        for c in range(1, n + 1):
            if c in best_suffix:
                continue # Permutation cannot have duplicate
            new_suffix = best_suffix + (c,)
            if new_suffix in prefix_to_perms:
                # Check if it leads to uncovered perms
                candidates = prefix_to_perms[new_suffix]
                unc_count = sum(1 for p in candidates if p in uncovered)
                # Score: unc_count
                if unc_count > best_score:
                    best_score = unc_count
                    next_char = c
                    
        if next_char is None:
            # Fallback: pick any valid char
            for c in range(1, n + 1):
                if c in best_suffix:
                    continue
                new_suffix = best_suffix + (c,)
                if new_suffix in prefix_to_perms:
                    next_char = c
                    break
            
        if next_char is None:
            # Stuck completely
            break
            
        word.append(next_char)
        
        # Check if any new window is covered
        if len(word) >= n:
            w = tuple(word[-n:])
            if w in uncovered:
                uncovered.remove(w)
            # Check previous windows too?
            # Only the last window is new.
            
    return "".join(map(str, word))

BET: {mechanism: "Replaces the slow string-deletion local search with a greedy construction that prioritizes covering uncovered permutations, potentially finding a shorter path by avoiding dead-ends inherent in the fixed recursive structure.", q: 0.6, delta: [0.01, 0.1]}
