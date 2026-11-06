import time
import random
import bisect
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0

    def can_add_descending(S, a):
        """Check if 'a' can be added to S, where all elements in S are > a.
        Condition: No pair b, c in S (b < c) such that a divides (b+c)."""
        if not S:
            return True
        
        # Use numpy for speed if S is large, or just a list of residues
        # Given N=256, a list of residues is fine.
        residues = [x % a for x in S]
        
        # We need to check if there exist i < j such that (residues[i] + residues[j]) % a == 0
        # This is equivalent to: 
        # For each residue r, let count[r] be the number of elements with residue r.
        # If r != (-r) % a:
        #   We need count[r] > 0 AND count[(-r) % a] > 0.
        # If r == (-r) % a (i.e., 2r % a == 0):
        #   We need count[r] >= 2.
        
        counts = [0] * a
        for r in residues:
            counts[r] += 1
            
        for r in range(a):
            partner = (-r) % a
            if r == partner:
                if counts[r] >= 2:
                    return False
            else:
                if counts[r] > 0 and counts[partner] > 0:
                    return False
        return True

    def greedy_descending():
        S = []
        for a in range(N, 0, -1):
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
        return S

    best_set = []
    best_len = 0
    
    S = greedy_descending()
    best_set = S[:]
    best_len = len(S)
    
    # Local Search
    # We will try to improve the set by swapping or adding.
    # Since can_add is now O(N), we can do more iterations.
    
    S_set = set(S)
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # 1. Try to add any missing element
        for a in range(1, N + 1):
            if a in S_set:
                continue
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
                S_set.add(a)
                best_len = len(S)
                improved = True
                break # Restart to check for new additions
        
        if improved:
            continue
            
        # 2. If no direct addition, try swaps
        # We want to remove a subset of S to allow adding a.
        # Heuristic: For each missing a, find conflicts.
        # Try removing 1 element, then 2 elements (if 1 fails).
        
        for a in range(1, N + 1):
            if a in S_set:
                continue
            
            # Identify conflicts for a
            # conflicts are pairs (i, j) in S such that S[i] + S[j] % a == 0
            # We need to break all these pairs.
            
            # Optimization: If a is small, many pairs.
            # Let's just try removing the "most central" element first.
            
            # Count how many conflicts each element is part of
            n = len(S)
            conflict_count = [0] * n
            # We only need to know if we can break all conflicts by removing 1 element.
            # If not, maybe 2?
            
            # For simplicity and speed, let's stick to 1-element removal first.
            # If that fails for all a, try 2-element removal?
            # Or just random swaps?
            
            # Let's refine the 1-element removal.
            # Find all pairs (i, j) with i < j and (S[i] + S[j]) % a == 0
            # If there is only 1 such pair, we must remove S[i] or S[j].
            # If there are multiple pairs, we need a hitting set.
            
            # Heuristic: Remove the element with max degree in the conflict graph.
            
            # To avoid O(N^2) per a, we can be smarter?
            # N is 256, N^2 is 65536.
            # There are at most 256 candidates for a.
            # 256 * 65536 = 16M operations.
            # This is fine in Python if optimized? Maybe slow.
            # But we only do this if direct addition failed.
            
            # Let's compute conflicts for a
            pairs = []
            for i in range(n):
                bi = S[i]
                target = (-bi) % a
                for j in range(i + 1, n):
                    if S[j] % a == target:
                        pairs.append((i, j))
                        
            if not pairs:
                # Should have been added
                continue
                
            if len(pairs) == 1:
                # Only one conflict pair
                i, j = pairs[0]
                # Try removing i
                S_new = S[:]
                S_new.pop(i)
                if can_add_descending(S_new, a):
                    pos = bisect.bisect_left(S_new, a)
                    S_new.insert(pos, a)
                    S = S_new
                    S_set = set(S)
                    improved = True
                    break
                # Try removing j
                S_new = S[:]
                S_new.pop(j)
                if can_add_descending(S_new, a):
                    pos = bisect.bisect_left(S_new, a)
                    S_new.insert(pos, a)
                    S = S_new
                    S_set = set(S)
                    improved = True
                    break
            else:
                # Multiple conflicts
                # Find most frequent index
                freq = {}
                for i, j in pairs:
                    freq[i] = freq.get(i, 0) + 1
                    freq[j] = freq.get(j, 0) + 1
                    
                # Try removing top 3 most frequent
                sorted_indices = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
                for rem_idx in sorted_indices[:3]:
                    S_new = S[:]
                    S_new.pop(rem_idx)
                    if can_add_descending(S_new, a):
                        pos = bisect.bisect_left(S_new, a)
                        S_new.insert(pos, a)
                        S = S_new
                        S_set = set(S)
                        improved = True
                        break
                if improved:
                    break
        
        # 3. Random perturbation if stuck
        if not improved:
            # Try a random swap
            if len(S) > 1 and time.time() - start_time < deadline:
                rem_idx = random.randint(0, len(S) - 1)
                a_candidate = random.randint(1, N)
                if a_candidate in S_set:
                    continue
                S_new = S[:]
                S_new.pop(rem_idx)
                if can_add_descending(S_new, a_candidate):
                    pos = bisect.bisect_left(S_new, a_candidate)
                    S_new.insert(pos, a_candidate)
                    S = S_new
                    S_set = set(S)
                    improved = True

    if len(S) > best_len:
        best_set = S[:]
        best_len = len(S)
        
    return best_set
