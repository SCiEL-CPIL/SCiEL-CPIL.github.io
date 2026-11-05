import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    # Hard limit is 120s. We aim to finish by 115s to be safe.
    deadline = start_time + 115.0

    total_words = q ** n

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists as numpy arrays
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # --- Phase 1: Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []

    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words

    while uncovered_remaining > 0 and time.time() < deadline - 50.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # Helper to update counts and check redundancy
    # We will maintain cov_count and is_code arrays
    
    def get_unique_points(c, exclude_set):
        """
        Returns list of points covered by c but not by any other word in exclude_set.
        exclude_set is a set of word indices.
        """
        # We need to check cov_count for neighbors of c.
        # If cov_count[x] == 1, it is uniquely covered by c (assuming c is in the set).
        # But cov_count includes all words in current_code.
        # If we are considering removing c, we look at neighbors[x] where cov_count[x] == 1.
        # However, if we are considering a swap or 2-for-1, the context is different.
        pass

    # Optimized Local Search
    # We maintain:
    # current_code: list of ints
    # current_cov_count: numpy array of counts
    # current_is_code: numpy bool array
    
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)

    random.seed(42)
    
    def try_reduce():
        """Attempts to remove redundant codewords. Returns True if any removed."""
        nonlocal current_code, current_size, current_cov_count, current_is_code, current_code_set
        reduced = False
        # Iterate in reverse to allow safe popping
        for i in range(len(current_code) - 1, -1, -1):
            c = current_code[i]
            # Check if all neighbors are covered by >1 codeword
            # If yes, c is redundant
            if np.all(current_cov_count[neighbors[c]] > 1):
                current_cov_count[neighbors[c]] -= 1
                current_is_code[c] = False
                current_code_set.remove(c)
                current_code.pop(i)
                current_size -= 1
                reduced = True
        return reduced

    def try_1_for_1_swap():
        """
        Tries to find a 1-for-1 swap that might enable reduction.
        We pick a random c1, find its unique points.
        Find c2 that covers them.
        Perform swap.
        """
        nonlocal current_code, current_size, current_cov_count, current_is_code, current_code_set
        if current_size == 0:
            return False
        
        # Pick random c1
        c1_idx = random.randint(0, current_size - 1)
        c1 = current_code[c1_idx]
        
        # Find risky points (uniquely covered by c1)
        risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
        
        if not risky_points:
            # c1 is redundant, should have been caught by try_reduce
            # But maybe try_reduce didn't run recently or this is a new state
            return False
            
        # Find c2 covering all risky_points
        # c2 must be in intersection of neighbors of all risky_points
        # And c2 != c1
        # And c2 not in current_code (if we want a swap, c2 must be new)
        
        # Optimization: If risky_points is large, intersection shrinks fast.
        # Start with neighbors of first risky point
        candidates = set(neighbors[risky_points[0]].tolist())
        candidates.discard(c1)
        
        for x in risky_points[1:]:
            candidates &= set(neighbors[x].tolist())
            if not candidates:
                return False
                
        # Filter out existing codewords
        valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]
        
        if not valid_candidates:
            return False
            
        # Pick one randomly
        c2 = random.choice(valid_candidates)
        
        # Perform Swap
        current_cov_count[neighbors[c1]] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        
        current_code[c1_idx] = c2
        current_cov_count[neighbors[c2]] += 1
        current_is_code[c2] = True
        current_code_set.add(c2)
        
        return True

    def try_2_for_1_swap():
        """
        Tries to remove 2 codewords and add 1.
        Reduces size by 1.
        """
        nonlocal current_code, current_size, current_cov_count, current_is_code, current_code_set
        if current_size < 2:
            return False
            
        # Pick two random distinct codewords
        # To avoid O(N^2) scan, we just pick random indices
        idx1 = random.randint(0, current_size - 1)
        idx2 = random.randint(0, current_size - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, current_size - 1)
            
        c1 = current_code[idx1]
        c2 = current_code[idx2]
        
        # We need to find points that are uniquely covered by c1 or c2 
        # relative to the set C \ {c1, c2}.
        
        # Temporarily remove c1 and c2 from counts to see what becomes uncovered
        # But we don't want to modify the global state yet.
        
        # A point x is "risky" for {c1, c2} if:
        # 1. x is in neighbors[c1] or neighbors[c2]
        # 2. In the current state, x is covered by c1 or c2.
        # 3. If we remove c1 and c2, x is no longer covered.
        
        # Let's compute the set of points that would become uncovered.
        # Uncovered if:
        # (x in N(c1) and count[x] == 1 and c1 covers x) OR
        # (x in N(c2) and count[x] == 1 and c2 covers x)
        # OR
        # (x in N(c1) and x in N(c2) and count[x] == 2 and c1,c2 cover x)
        
        # Actually, simpler:
        # Let U be the set of x such that x is in N(c1) U N(c2) and 
        # (count[x] == 1 and (c1==x or c2==x is not right, c1 covers x))
        
        # Let's just iterate through neighbors of c1 and c2.
        risky_set = set()
        
        # Points in N(c1)
        for x in neighbors[c1]:
            if current_cov_count[x] == 1:
                # Only c1 covers it
                risky_set.add(x)
            elif current_cov_count[x] == 2:
                # c1 and one other cover it.
                # If the "one other" is c2, then removing both uncovers it.
                # If the "one other" is not c2, then removing c1,c2 still leaves it covered by the other.
                # So we only care if c2 is the other one.
                # How to check if c2 covers x? x is in N(c2).
                if x in neighbors[c2]:
                    risky_set.add(x)
                    
        # Points in N(c2)
        for x in neighbors[c2]:
            if x in risky_set:
                continue
            if current_cov_count[x] == 1:
                # Only c2 covers it
                risky_set.add(x)
            elif current_cov_count[x] == 2:
                # c2 and one other cover it.
                # If the other is c1, we already handled it (or will, but c1 covers x implies x in N(c1)).
                # If x in N(c1), then count[x] >= 2.
                # If count[x]==2 and x in N(c1) and x in N(c2), then c1 and c2 are the only coverers.
                # We already added it in the first loop if it was in N(c1).
                # If it wasn't in N(c1), then c1 doesn't cover it.
                # So if count[x]==2 and x in N(c2), and x NOT in N(c1), then the other coverer is not c1.
                # So removing c1,c2 leaves it covered.
                pass

        if not risky_set:
            # c1 and c2 are redundant together?
            # This means we can just remove them?
            # No, try_reduce would have removed them if they were individually redundant.
            # If they are not individually redundant, but their union is redundant?
            # That means every point they cover is covered by someone else.
            # In that case, we can remove both.
            # But 2-for-0 is better than 2-for-1.
            # Let's just return False and let try_reduce handle it if possible.
            return False

        # Find c3 that covers all points in risky_set
        # c3 must be in intersection of neighbors of all points in risky_set
        # And c3 not in current_code (or if in current_code, we are doing a swap, not 2-for-1 reduction? 
        # If c3 is in current_code, we are removing c1,c2 and relying on existing c3.
        # But c3 is already in the code. So we just remove c1,c2.
        # This is a 2-for-0 move (reduction by 2).
        
        # Let's check if there is an existing codeword c3 in C \ {c1, c2} that covers risky_set.
        # If yes, we can remove c1 and c2.
        
        # Intersection of neighbors of all risky points
        if not risky_set:
             return False
             
        first_risky = next(iter(risky_set))
        candidates = set(neighbors[first_risky].tolist())
        
        for x in risky_set:
            if x == first_risky:
                continue
            candidates &= set(neighbors[x].tolist())
            if not candidates:
                break
                
        if not candidates:
            return False
            
        # Check if any candidate is already in the code (excluding c1, c2)
        # If so, we can just remove c1, c2.
        for c3 in candidates:
            if c3 in current_code_set and c3 != c1 and c3 != c2:
                # We can remove c1 and c2
                # Perform removal
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                
                current_cov_count[neighbors[c2]] -= 1
                current_is_code[c2] = False
                current_code_set.remove(c2)
                
                # Update list
                # Remove from list
                # Need to find indices
                # This is O(N).
                # To keep code simple, we can rebuild list or just mark and clean up later?
                # Better to update list now.
                # Find index of c1
                if c1 in current_code:
                    current_code.remove(c1)
                if c2 in current_code:
                    current_code.remove(c2)
                current_size -= 2
                return True
                
        # If no existing codeword covers it, we need to ADD a new c3.
        # This is a 2-for-1 move.
        # Remove c1, c2. Add c3.
        valid_candidates = [c3 for c3 in candidates if c3 not in current_code_set]
        
        if not valid_candidates:
            return False
            
        c3 = random.choice(valid_candidates)
        
        # Perform 2-for-1
        current_cov_count[neighbors[c1]] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        
        current_cov_count[neighbors[c2]] -= 1
        current_is_code[c2] = False
        current_code_set.remove(c2)
        
        current_cov_count[neighbors[c3]] += 1
        current_is_code[c3] = True
        current_code_set.add(c3)
        
        # Update list
        current_code.remove(c1)
        current_code.remove(c2)
        current_code.append(c3)
        current_size -= 1
        
        return True

    # Main Loop
    stuck = 0
    last_best_time = start_time
    
    while time.time() < deadline - 10.0:
        
        # 1. Reduction
        if try_reduce():
            stuck = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
                last_best_time = time.time()
            continue # Restart loop to check for more reductions
        
        # 2. Try 2-for-1 (stronger move)
        if try_2_for_1_swap():
            stuck = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
                last_best_time = time.time()
            # After a swap, new redundancies might appear
            continue
            
        # 3. Try 1-for-1
        if try_1_for_1_swap():
            stuck = 0
            # After swap, check for reduction
            continue
            
        # 4. Stuck, Perturb
        stuck += 1
        if stuck > 10:
            # Remove 2 random codewords
            if current_size > 2:
                idx1 = random.randint(0, current_size - 1)
                idx2 = random.randint(0, current_size - 1)
                while idx2 == idx1:
                    idx2 = random.randint(0, current_size - 1)
                    
                c1 = current_code[idx1]
                c2 = current_code[idx2]
                
                # Remove them
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                
                current_cov_count[neighbors[c2]] -= 1
                current_is_code[c2] = False
                current_code_set.remove(c2)
                
                current_code.remove(c1)
                current_code.remove(c2)
                current_size -= 2
                
                # Greedy Repair
                # Find uncovered words
                # We need to identify uncovered words efficiently.
                # We can iterate through all words? 6561 is small.
                # Or maintain a list of uncovered?
                # For simplicity, check cov_count.
                
                # Greedy Add
                while time.time() < deadline - 5.0:
                    # Find word with most uncovered neighbors
                    # potential[x] = number of neighbors of x that are uncovered
                    # We can recompute potential for all x? 
                    # Or just pick a random uncovered word and cover it?
                    
                    # Let's find any uncovered word
                    uncovered_words = np.where(current_cov_count == 0)[0]
                    if len(uncovered_words) == 0:
                        break
                        
                    # Pick one randomly
                    target = uncovered_words[random.randint(0, len(uncovered_words)-1)]
                    
                    # Find best codeword to cover it
                    # We want a codeword c that covers target and many other uncovered words.
                    # This is expensive to compute for all c.
                    # Approximation: Pick c from neighbors[target] that is not in code?
                    # Or just pick random neighbor?
                    
                    # Let's use the potential array updated dynamically?
                    # That's complex.
                    # Simple greedy: Pick c in neighbors[target] that maximizes uncovered neighbors.
                    
                    best_c = -1
                    best_cov = -1
                    for c in neighbors[target]:
                        if c in current_code_set:
                            continue
                        # Count how many neighbors of c are uncovered
                        # This is O(17)
                        count = 0
                        for x in neighbors[c]:
                            if current_cov_count[x] == 0:
                                count += 1
                        if count > best_cov:
                            best_cov = count
                            best_c = c
                            
                    if best_c != -1:
                        current_code.append(best_c)
                        current_code_set.add(best_c)
                        current_is_code[best_c] = True
                        current_cov_count[neighbors[best_c]] += 1
                        current_size += 1
                    else:
                        # Should not happen
                        break
                        
            stuck = 0

    # Final Polish
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    return [list(digits_of[i]) for i in final_code]
