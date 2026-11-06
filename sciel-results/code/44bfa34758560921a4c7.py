"""Small inclusion-maximal Sidon sets in [1,256]"""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all a+b with a<=b distinct)."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            total = a + b
            if total in s:
                return False
            s.add(total)
    return True

def _compute_sums(selected):
    """Compute the set of all a+b sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums):
    """Check if x can be added to the Sidon set."""
    for a in selected:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True

def _get_addable_candidates(selected, sums):
    """Get all elements in [1, N] that can be added to the current Sidon set."""
    sel_set = set(selected)
    candidates = []
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            candidates.append(x)
    return candidates

def _blocking_score(x, selected, sums, candidates):
    """
    Estimate how many other candidates will be blocked if x is added.
    New sums: 2x, and x+a for all a in selected.
    A candidate y is blocked if:
    1. 2y is in new_sums
    2. y+a is in new_sums for some a in selected U {x}
    """
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        new_sums.add(x + a)
    
    blocked_count = 0
    # Check each current candidate
    for y in candidates:
        if y == x:
            continue
        # Check if y becomes non-addable
        # Condition: 2y in new_sums OR (y+a) in new_sums for some a in selected U {x}
        
        if 2 * y in new_sums:
            blocked_count += 1
            continue
            
        # Check sums with existing selected elements
        # Note: y+a for a in selected was already checked to be NOT in sums (old).
        # But now we have new sums.
        # Specifically, we added x+a. So we need to check if y+a == x+b for some a,b in selected?
        # Or y+x == x+a => y=a (impossible since y not in selected)
        # Or y+a == 2x => y = 2x - a
        
        # Actually, simpler: just re-run the addability check logic for y against new_sums and selected U {x}
        # But that's O(K).
        # Let's just check the specific new conflicts.
        # New sums are {2x} U {x+a | a in selected}.
        # Conflicts for y:
        # 1. 2y in new_sums (checked above)
        # 2. y + a in new_sums for a in selected U {x}
        
        # y + x in new_sums?
        # y + x = 2x => y = x (False)
        # y + x = x + a => y = a (False, y not in selected)
        # So y+x is never in new_sums? 
        # Wait, new_sums contains sums of pairs in S_new.
        # Pairs in S_new are (u, v) where u, v in S U {x}.
        # Sums are u+v.
        # We check if y+a is in new_sums.
        # If a in S: y+a is a sum of (y, a). Is (y, a) a pair in S_new? No, y is not in S_new yet.
        # The condition for y to be addable is that NO sum involving y conflicts with existing sums.
        # Existing sums in S_new are {u+v | u,v in S U {x}, u<=v}.
        # y is addable if:
        # 1. 2y not in S_new_sums
        # 2. y+a not in S_new_sums for all a in S U {x}
        
        # We already know y+a not in OLD_sums for all a in S.
        # So we only need to check if y+a is in NEW_sums \ OLD_sums.
        # NEW_sums \ OLD_sums = {2x} U {x+a | a in S}.
        
        # So check if y+a is in {2x} U {x+a' | a' in S} for any a in S U {x}.
        
        # Case a = x:
        # y + x in {2x} U {x+a' | a' in S}
        # y+x = 2x => y=x (False)
        # y+x = x+a' => y=a' (False)
        # So y+x is never a conflict.
        
        # Case a in S:
        # y + a in {2x} U {x+a' | a' in S}
        # 1. y + a = 2x => y = 2x - a
        # 2. y + a = x + a' => y = x + a' - a
        
        # So for a given x, a candidate y is blocked if:
        # 1. 2y = 2x (y=x, false) or 2y = x+a' (y = (x+a')/2)
        # 2. y = 2x - a for some a in S
        # 3. y = x + a' - a for some a, a' in S
        
        # This is getting complex. Let's just do the O(K) check for simplicity, K is small.
        
        is_blocked = False
        # Check against new sums specifically
        # We can just check if _is_addable(y, selected + [x], new_sums) is False
        # But _is_addable iterates over selected.
        # Let's just check the conditions directly.
        
        # Check 2y
        if 2*y in new_sums:
             is_blocked = True
        else:
            # Check y+a for a in selected
            for a in selected:
                if y + a in new_sums:
                    is_blocked = True
                    break
            # Check y+x
            if not is_blocked and y + x in new_sums:
                is_blocked = True
        
        if is_blocked:
            blocked_count += 1
            
    return blocked_count

def _greedy_blocking_search(deadline, rng):
    """
    Build a Sidon set by greedily adding the element that blocks the most other candidates.
    """
    selected = []
    sums = set()
    
    while time.time() < deadline:
        candidates = _get_addable_candidates(selected, sums)
        if not candidates:
            break
            
        if len(candidates) == 1:
            x = candidates[0]
        else:
            # Calculate scores
            scores = []
            for x in candidates:
                # To save time, we can sample or limit the blocking score calculation
                # But N=256 is small.
                score = _blocking_score(x, selected, sums, candidates)
                scores.append((score, x))
            
            # Sort by score descending
            scores.sort(key=lambda item: -item[0])
            
            # Randomize among top few to escape local optima
            top_k = min(3, len(scores))
            # Pick randomly from top_k
            idx = rng.randint(0, top_k - 1)
            x = scores[idx][1]
            
        # Add x
        selected.append(x)
        for a in selected:
            if a != x:
                sums.add(x + a)
        sums.add(2 * x)
        
    return selected, sums

def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            return False
    return True

def _local_search_reduce(selected, sums, deadline):
    """
    Try to reduce the size by removing an element and re-maximalizing with the blocking heuristic.
    """
    best_sel = list(selected)
    best_sums = set(sums)
    best_size = len(best_sel)
    rng = random.Random(42) # Fixed seed for local search stability? Or pass in?
    
    # Try removing each element
    for y in list(best_sel):
        if time.time() > deadline:
            break
        if len(best_sel) <= 1:
            break
        
        # Remove y
        candidate = [x for x in best_sel if x != y]
        cand_sums = _compute_sums(candidate)
        
        # Re-maximalize using the greedy blocking search
        # We need to continue from this state
        # The greedy_blocking_search starts from empty. 
        # We need a version that starts from a given set.
        
        # Let's refactor _greedy_blocking_search to take initial state
        # Or just implement a helper.
        
        # Actually, let's just run the full greedy search from scratch in the main loop.
        # The local search here is tricky because _greedy_blocking_search is a full builder.
        
        # Alternative: Just do the removal and then run a "completion" phase.
        # Completion phase: repeatedly add the best blocking element until maximal.
        
        cand_sel = list(candidate)
        cand_sums_set = set(cand_sums)
        
        # Completion phase
        while time.time() < deadline:
            candidates = _get_addable_candidates(cand_sel, cand_sums_set)
            if not candidates:
                break
                
            if len(candidates) == 1:
                x = candidates[0]
            else:
                scores = []
                for x in candidates:
                    score = _blocking_score(x, cand_sel, cand_sums_set, candidates)
                    scores.append((score, x))
                scores.sort(key=lambda item: -item[0])
                top_k = min(3, len(scores))
                idx = rng.randint(0, top_k - 1)
                x = scores[idx][1]
                
            cand_sel.append(x)
            for a in cand_sel:
                if a != x:
                    cand_sums_set.add(x + a)
            cand_sums_set.add(2 * x)
            
        if len(cand_sel) < best_size:
            if _is_sidon(cand_sel) and _is_maximal(cand_sel, cand_sums_set):
                best_sel = cand_sel
                best_sums = cand_sums_set
                best_size = len(best_sel)
                
    return best_sel, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_selected = None
    best_size = N + 1
    
    rng = random.Random(42)
    
    # Run multiple trials of the greedy blocking search
    attempts = 0
    while time.time() < deadline and attempts < 100:
        attempts += 1
        
        # Slight variation: randomize the tie-breaking or top-k choice
        # The _greedy_blocking_search uses rng internally if we pass it?
        # Currently _greedy_blocking_search creates its own rng? No, I defined it to take rng.
        
        sel, sums = _greedy_blocking_search(deadline, rng)
        
        if not sel:
            continue
            
        # It should be maximal by construction, but verify
        if _is_sidon(sel) and _is_maximal(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = sel
                # Try to improve via local search
                # But local search might be slow. 
                # Given the heuristic is strong, maybe local search is less critical?
                # Let's try a few local search steps if we have time.
                
                # Actually, the local search defined above is O(N * K^2) roughly.
                # With N=256, it's fine.
                
                # But we need to pass rng to local search?
                # Let's just use the current best as is for now, and rely on multiple restarts.
                # The blocking heuristic is quite strong.
                
    if best_selected is None:
        # Fallback
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if _is_addable(x, selected, sums):
                selected.append(x)
                for a in selected:
                    if a != x:
                        sums.add(x + a)
                sums.add(2 * x)
        best_selected = selected
        best_size = len(selected)
        
    return sorted(best_selected)

# EVOLVE-BLOCK-END
