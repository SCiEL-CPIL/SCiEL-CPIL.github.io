import time
import random

def construct():
    """
    Finds the largest Sidon set (distinct pairwise sums a+b, a<=b) in [1, 284]
    containing {1, 2, 5, 11} using branch-and-bound with bitset pruning.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    prescribed = [1, 2, 5, 11]
    pset = set(prescribed)
    MAXV = 284
    MAX_SUM = 2 * MAXV  # 568
    
    # Precompute: for each pair (x, y) with x <= y, the sum s = x + y
    # Two elements x and y conflict if there exist a, b in the set such that
    # x + y = a + b (with a <= b, x <= y). But since we're building incrementally,
    # we track used sums.
    
    # Bitset for used sums (sums range from 1+1=2 to 284+284=568)
    def sums_bitset_from_list(sl):
        bs = 0
        for i in range(len(sl)):
            x = sl[i]
            for j in range(i, len(sl)):
                bs |= (1 << (x + sl[j]))
        return bs
    
    def can_add(x, sl, bs):
        """Check if x can be added to sorted list sl with current sum bitset bs."""
        for y in sl:
            s = x + y
            if (bs >> s) & 1:
                return False
        if (bs >> (2 * x)) & 1:
            return False
        return True
    
    def add_to_bitset(x, sl, bs):
        """Add x to the sum bitset, given sorted list sl (not containing x)."""
        for y in sl:
            bs |= (1 << (x + y))
        bs |= (1 << (2 * x))
        return bs
    
    # Branch and bound search
    # State: current set (sorted), current sum bitset, next candidate index
    # We'll use a priority queue or recursive DFS with pruning
    
    best_S = list(prescribed)
    best_size = len(prescribed)
    best_bs = sums_bitset_from_list(sorted(prescribed))
    
    # Initial candidates: all numbers 1..284 not in prescribed
    all_candidates = [x for x in range(1, MAXV + 1) if x not in pset]
    
    # For efficient candidate filtering, we'll maintain a list of valid candidates
    # at each level
    
    # DFS with branch and bound
    # We sort candidates and use a recursive approach
    # To avoid deep recursion, we use an explicit stack
    
    # Key insight: we can prune if current_size + remaining_candidates <= best_size
    
    def search(sl, bs, candidates, depth):
        """
        sl: sorted list of current set
        bs: bitset of used sums
        candidates: list of candidate values to consider (all > sl[-1] if sl non-empty, 
                    or all values if sl empty, but we'll filter)
        depth: current depth (for debugging)
        """
        nonlocal best_S, best_size, best_bs
        
        if time.time() > deadline - 1.0:
            return False  # timeout
        
        cur_size = len(sl)
        
        # Prune: even if we add all remaining candidates, can we beat best?
        if cur_size + len(candidates) <= best_size:
            return False
        
        # Try to add candidates in some order
        # Heuristic: try candidates that leave more room? Or just iterate?
        # For now, iterate through candidates and try to add each
        
        for i, x in enumerate(candidates):
            if time.time() > deadline - 1.0:
                return False
            
            if can_add(x, sl, bs):
                # Add x
                new_sl = sl + [x] if not sl or x > sl[-1] else sorted(sl + [x])
                # Ensure sorted: since we process candidates in order, and sl is sorted,
                # if x > sl[-1], then new_sl is sorted. Otherwise need to sort.
                # But candidates are all not in sl, and we'll maintain sorted order.
                # Let's just keep sl sorted and append if x is larger.
                if sl and x > sl[-1]:
                    new_sl = sl + [x]
                else:
                    new_sl = sorted(sl + [x])
                
                new_bs = add_to_bitset(x, sl, bs)
                
                # Update best
                if len(new_sl) > best_size:
                    best_size = len(new_sl)
                    best_S = list(new_sl)
                    best_bs = new_bs
                    if best_size == 20:
                        return True  # optimal found
                
                # Recurse with remaining candidates
                # Remaining candidates: those after x in the candidate list
                # But we need to be careful: candidates should be values not in new_sl
                # Since x is now in new_sl, and all other candidates are different,
                # the remaining candidates are candidates[i+1:]
                # But we also need to ensure they can be added - that's checked in recursion
                
                remaining = candidates[i + 1:]
                if search(new_sl, new_bs, remaining, depth + 1):
                    return True  # optimal found
                
                # Backtrack: we don't need to explicitly remove from bs/sl since
                # we're using new_sl and new_bs for recursion
        
        return False
    
    # Start the search
    initial_sl = sorted(prescribed)
    initial_bs = best_bs
    # Candidates: all numbers not in prescribed, sorted
    # But we should be smart about ordering
    
    # Heuristic: try smaller numbers first? Or larger? 
    # For Sidon sets, spreading out might be better.
    # Let's try a few different orderings.
    
    # Order 1: ascending
    success = search(initial_sl, initial_bs, all_candidates, 0)
    if success or best_size >= 20:
        return best_S
    
    # If we didn't find 20, try a different heuristic
    # Order by "spread" - maybe prioritize numbers that are far from existing ones
    
    # Try descending order
    all_candidates_desc = sorted(all_candidates, reverse=True)
    # Reset best? No, keep the best found so far
    # But the search above may have already explored a lot.
    # Let's try a fresh search with different ordering if time permits.
    
    if time.time() < deadline - 30:
        # Try a beam search as fallback
        beam_width = 50
        beam = [(list(initial_sl), initial_bs, all_candidates)]
        
        for depth in range(16):  # max 16 more elements (4 + 16 = 20)
            if time.time() > deadline - 10:
                break
            
            next_beam = []
            seen = set()
            
            for sl, bs, cands in beam:
                if len(sl) >= best_size:
                    # Try to extend
                    for x in cands:
                        if can_add(x, sl, bs):
                            if sl and x > sl[-1]:
                                new_sl = sl + [x]
                            else:
                                new_sl = sorted(sl + [x])
                            new_bs = add_to_bitset(x, sl, bs)
                            
                            key = tuple(new_sl)
                            if key in seen:
                                continue
                            seen.add(key)
                            
                            if len(new_sl) > best_size:
                                best_size = len(new_sl)
                                best_S = list(new_sl)
                                best_bs = new_bs
                                if best_size == 20:
                                    return best_S
                            
                            # Remaining candidates
                            idx = cands.index(x) if x in cands else -1
                            remaining = cands[idx + 1:] if idx >= 0 else []
                            next_beam.append((new_sl, new_bs, remaining))
            
            if not next_beam:
                break
            
            # Score and prune beam
            def score(state):
                sl, bs, cands = state
                # Count how many candidates can still be added
                count = 0
                for x in cands:
                    if can_add(x, sl, bs):
                        count += 1
                return len(sl) * 1000 + count
            
            next_beam.sort(key=score, reverse=True)
            beam = next_beam[:beam_width]
    
    # Final validation
    result = sorted(best_S)
    sl = result
    seen_sums = set()
    valid = True
    for i in range(len(sl)):
        for j in range(i, len(sl)):
            s = sl[i] + sl[j]
            if s in seen_sums:
                valid = False
                break
            seen_sums.add(s)
        if not valid:
            break
    
    if not valid or not pset.issubset(result):
        return prescribed
    
    return result
