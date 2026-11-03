import time
import random
import math
import itertools
import numpy as np

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        if radius >= 1:
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        neigh_set[idx].add(new_idx)
    return neigh, neigh_set, total

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Set Cover Initialization
    uncovered = [True] * total
    rem_count = total
    code = []
    code_set = set()
    
    while rem_count > 0:
        if time.monotonic() > deadline - 60.0:
            break
        best_c = -1
        best_gain = 0
        for c in range(total):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if uncovered[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain == 0:
            break
        code.append(best_c)
        code_set.add(best_c)
        for u in neigh[best_c]:
            if uncovered[u]:
                uncovered[u] = False
                rem_count -= 1
                
    # Ensure we have a valid cover
    if rem_count > 0:
        for u in range(total):
            if uncovered[u]:
                for c in neigh[u]:
                    if c not in code_set:
                        code.append(c)
                        code_set.add(c)
                        for u2 in neigh[c]:
                            if uncovered[u2]:
                                uncovered[u2] = False
                                rem_count -= 1
                        break
                if rem_count == 0:
                    break

    best_code = code[:]
    best_size = len(code)
    
    # 3. Deterministic Local Search with Structured Moves
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(old_c, new_c):
        for u in neigh[old_c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    def try_remove_redundant():
        """Remove all redundant words in one pass."""
        removed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if is_removable(c):
                code.pop(i)
                code_set.discard(c)
                for u in neigh[c]:
                    counts[u] -= 1
                removed = True
        return removed

    def try_best_swap():
        """Find and apply the best 1-for-1 swap that maintains validity."""
        best_old_idx = -1
        best_new_c = -1
        best_gain = 0
        
        for i, old_c in enumerate(code):
            # Check if removing old_c would break coverage
            # For each neighbor u of old_c with count==1, new_c must cover u
            critical_neighbors = []
            for u in neigh[old_c]:
                if counts[u] == 1:
                    critical_neighbors.append(u)
            
            if not critical_neighbors:
                # old_c is redundant, handled by try_remove_redundant
                continue
            
            # Find new_c that covers all critical neighbors and isn't already in code
            for new_c in range(total):
                if new_c == old_c or new_c in code_set:
                    continue
                # Check if new_c covers all critical neighbors
                valid = True
                for u in critical_neighbors:
                    if u not in neigh_set[new_c]:
                        valid = False
                        break
                if valid:
                    # Count how many NEW neighbors new_c covers that aren't already covered
                    gain = 0
                    for u in neigh[new_c]:
                        if counts[u] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_old_idx = i
                        best_new_c = new_c
        
        if best_old_idx >= 0 and best_new_c >= 0:
            old_c = code[best_old_idx]
            for u in neigh[old_c]:
                counts[u] -= 1
            for u in neigh[best_new_c]:
                counts[u] += 1
            code[best_old_idx] = best_new_c
            code_set.discard(old_c)
            code_set.add(best_new_c)
            return True
        return False

    def try_2_for_1_exchange():
        """Try to replace 2 codewords with 1 that covers their union better."""
        # This is expensive, so only try a limited number of pairs
        if len(code) < 2:
            return False
        
        # Sample some pairs to keep it fast
        max_pairs = min(len(code) * 2, 200)
        tried = 0
        best_pair = None
        best_replacement = -1
        best_gain = 0
        
        while tried < max_pairs and time.monotonic() < deadline - 5:
            i = rng.randint(0, len(code) - 1)
            j = rng.randint(0, len(code) - 1)
            if i == j:
                continue
            tried += 1
            
            c1 = code[i]
            c2 = code[j]
            
            # Find neighbors only covered by c1 or c2
            only_c1 = []
            only_c2 = []
            for u in neigh[c1]:
                if counts[u] == 1:
                    only_c1.append(u)
            for u in neigh[c2]:
                if counts[u] == 1:
                    only_c2.append(u)
            
            if not only_c1 and not only_c2:
                # Both are redundant, handled elsewhere
                continue
            
            # Look for a single codeword that covers all of only_c1 and only_c2
            # and provides additional coverage
            all_critical = set(only_c1) | set(only_c2)
            
            # Sample candidate replacements
            for _ in range(50):
                new_c = rng.randint(0, total - 1)
                if new_c == c1 or new_c == c2 or new_c in code_set:
                    continue
                if new_c in (c1, c2):
                    continue
                
                # Check if new_c covers all critical neighbors
                valid = True
                for u in all_critical:
                    if u not in neigh_set[new_c]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate gain: new neighbors covered that weren't before
                gain = 0
                for u in neigh[new_c]:
                    # After removing c1 and c2, what's the count?
                    # This is complex, so approximate
                    pass
                
                if best_replacement == -1 or gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
                    best_replacement = new_c
        
        if best_pair is not None and best_replacement >= 0:
            i, j = best_pair
            c1 = code[i]
            c2 = code[j]
            
            # Remove c1 and c2
            for u in neigh[c1]:
                counts[u] -= 1
            for u in neigh[c2]:
                counts[u] -= 1
            
            # Verify coverage is still valid
            valid = True
            for u in range(total):
                if counts[u] == 0:
                    valid = False
                    break
            
            if valid:
                # Remove from code list (careful with indices)
                if i > j:
                    i, j = j, i
                code.pop(i)
                code.pop(j)
                code_set.discard(c1)
                code_set.discard(c2)
                
                # Add new codeword
                code.append(best_replacement)
                code_set.add(best_replacement)
                for u in neigh[best_replacement]:
                    counts[u] += 1
                return True
            else:
                # Rollback
                for u in neigh[c1]:
                    counts[u] += 1
                for u in neigh[c2]:
                    counts[u] += 1
        
        return False

    # Main local search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.monotonic() < deadline - 5:
        improved = False
        
        # Phase 1: Remove redundant words
        if try_remove_redundant():
            improved = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # Phase 2: Try best swap
        if try_best_swap():
            improved = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # Phase 3: Try 2-for-1 exchange (less frequent)
        if rng.random() < 0.3:
            if try_2_for_1_exchange():
                improved = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0

    # Final cleanup: try to remove redundant words from best_code
    best_set = set(best_code)
    best_counts = [0] * total
    for c in best_code:
        for u in neigh[c]:
            best_counts[u] += 1
            
    final_code = []
    for c in best_code:
        removable = True
        for u in neigh[c]:
            if best_counts[u] == 1:
                removable = False
                break
        if not removable:
            final_code.append(c)
            for u in neigh[c]:
                best_counts[u] -= 1
        else:
            pass
            
    if len(final_code) < len(best_code):
        best_code = final_code

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
