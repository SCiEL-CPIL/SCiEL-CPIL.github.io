import time
import random
import numpy as np

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Precompute all possible triples (a, b, c) with 1 <= a < b < c <= 128
    # We need to check if a+b, a+c, b+c are all in S.
    # To make this fast with NumPy, we precompute the triples and their sums.
    
    # Generate all triples
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                triples.append((a, b, c))
                
    if not triples:
        return list(range(1, n + 1))
        
    triples_np = np.array(triples, dtype=np.int32)
    # Sums: a+b, a+c, b+c
    sums_ab = triples_np[:, 0] + triples_np[:, 1]
    sums_ac = triples_np[:, 0] + triples_np[:, 2]
    sums_bc = triples_np[:, 1] + triples_np[:, 2]
    
    # We will maintain a boolean array `in_set` of size 2*n + 1 (indices 0..256)
    # A triple is "bad" if in_set[a+b] AND in_set[a+c] AND in_set[b+c] are all True.
    
    def is_valid(in_set_bool):
        # in_set_bool is a numpy bool array of size 2*n+1
        # Check all triples at once
        # Note: sums can be up to 256, so array size must be at least 257
        valid_ab = in_set_bool[sums_ab]
        valid_ac = in_set_bool[sums_ac]
        valid_bc = in_set_bool[sums_bc]
        
        # If any triple has all three sums in the set, it's invalid
        bad_triples = np.any(valid_ab & valid_ac & valid_bc)
        return not bad_triples

    def get_bad_elements(in_set_bool, S_list):
        """
        Identify elements that are part of bad triples.
        This is used for removal heuristics.
        Returns a set of elements that appear in any bad triple.
        """
        valid_ab = in_set_bool[sums_ab]
        valid_ac = in_set_bool[sums_ac]
        valid_bc = in_set_bool[sums_bc]
        
        bad_mask = valid_ab & valid_ac & valid_bc
        if not np.any(bad_mask):
            return set()
            
        bad_triples_indices = np.where(bad_mask)[0]
        bad_elements = set()
        for idx in bad_triples_indices:
            a, b, c = triples_np[idx]
            bad_elements.add(a)
            bad_elements.add(b)
            bad_elements.add(c)
            
        return bad_elements

    # --- Phase 1: Structured Construction ---
    # Start with an empty set and add elements in descending order.
    # Adding large numbers first reduces the chance of creating small sums that fall within [1, 128].
    # Specifically, if we add x, the sums x+y will be large.
    
    in_set = np.zeros(2 * n + 1, dtype=bool)
    current_set = set()
    
    # Try adding elements from 128 down to 1
    for val in range(n, 0, -1):
        in_set[val] = True
        if not is_valid(in_set):
            in_set[val] = False
        else:
            current_set.add(val)
            
    best_size = len(current_set)
    best_set = set(current_set)
    
    # --- Phase 2: Local Search Refinement ---
    # Now we try to improve the set using swaps and additions.
    # We use a randomized local search.
    
    # To speed up swap checks, we don't rebuild the full validity check from scratch every time 
    # if we can avoid it, but with NumPy it's fast enough (~0.1ms per check).
    # We can do ~10,000 checks per second.
    
    # Strategy:
    # 1. Try to add any missing element.
    # 2. If stuck, try swaps: remove one element, add another.
    
    # Precompute candidates for addition
    all_elements = list(range(1, n + 1))
    
    # To make the search more effective, we'll do multiple passes.
    # First pass: Greedy addition of any remaining element.
    for val in all_elements:
        if val not in current_set:
            in_set[val] = True
            if is_valid(in_set):
                current_set.add(val)
            else:
                in_set[val] = False
                
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
        
    # Second pass: Randomized Swaps
    # We accept swaps that maintain validity, hoping to escape local maxima.
    # We also try "2-opt" moves: remove 2, add 2? Maybe too complex. Let's stick to 1-1 swaps.
    
    # To avoid getting stuck in a specific local optimum, we can restart from best_set occasionally.
    # But given the time budget, a single long run with many swaps is better.
    
    # We will iterate until time runs out.
    # Each iteration:
    # - Pick a random element in S (x)
    # - Pick a random element not in S (y)
    # - Check if (S - {x}) + {y} is valid.
    # - If valid, accept.
    # - Additionally, try pure additions if possible (though Phase 1 should have caught most).
    
    # Optimization: Pre-compute a list of elements not in S to pick from quickly.
    # Updating this list every iteration is expensive. We can just pick from 1..128 and check membership.
    
    # Let's estimate iterations: 80 seconds * 10,000 checks/sec = 800,000 iterations.
    # This is plenty for 128 elements.
    
    while time.time() < deadline:
        # Check for pure addition first (cheap)
        # This is a bit redundant if we do it in a loop, but let's try a few random adds
        # Actually, let's just do swaps.
        
        x = random.choice(list(current_set))
        y = random.randint(1, n)
        
        if y in current_set:
            continue
            
        # Perform swap
        # Remove x, add y
        in_set[x] = False
        in_set[y] = True
        
        if is_valid(in_set):
            current_set.remove(x)
            current_set.add(y)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Revert
            in_set[x] = True
            in_set[y] = False
            
        # Occasionally, try to add an element if we haven't reached max size
        # But swaps keep size constant. How do we increase size?
        # The initial greedy construction might have left out elements that become addable after swaps.
        # So after every N swaps, try to add a random missing element.
        
        if random.random() < 0.1:
            # Try to add a random missing element
            y_add = random.randint(1, n)
            if y_add not in current_set:
                in_set[y_add] = True
                if is_valid(in_set):
                    current_set.add(y_add)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                else:
                    in_set[y_add] = False

    # Final check: try to add any remaining element in deterministic order
    for val in all_elements:
        if val not in best_set:
            # Check if adding val to best_set is valid
            # We need to reconstruct the boolean array for best_set to check
            # This is O(N) to build array, then O(1) vectorized check.
            # Only do this if we have time.
            if time.time() > deadline - 2.0:
                break
                
            temp_in_set = np.zeros(2 * n + 1, dtype=bool)
            for v in best_set:
                temp_in_set[v] = True
            temp_in_set[val] = True
            if is_valid(temp_in_set):
                best_set.add(val)

    # Ensure return type is list of Python ints
    return list(best_set)
