import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    # Start with the full set
    S = set(range(1, n + 1))
    
    # Helper to check validity and count conflicts
    def get_conflicts(S_set):
        """
        Returns a dictionary of element -> count of forbidden triples it is part of.
        A forbidden triple is distinct a,b,c in S such that a+b, a+c, b+c are in S.
        The elements 'involved' in this violation are a, b, c, a+b, a+c, b+c (those that are in S).
        """
        counts = {}
        elems = sorted(S_set)
        m = len(elems)
        
        # Optimization: Precompute sums? No, just iterate.
        # To speed up, we can break early if we find many conflicts, but we need counts.
        
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                ab = a + b
                if ab > n or ab not in S_set:
                    continue
                for k in range(j + 1, m):
                    c = elems[k]
                    ac = a + c
                    if ac > n or ac not in S_set:
                        continue
                    bc = b + c
                    if bc > n or bc not in S_set:
                        continue
                    
                    # Found a forbidden triple (a, b, c)
                    # Increment counts for all involved elements in S
                    for x in (a, b, c, ab, ac, bc):
                        if x in S_set:
                            counts[x] = counts.get(x, 0) + 1
                            
        return counts

    # Phase 1: Greedy Pruning
    # Remove elements with highest conflict count until no conflicts remain
    while time.time() < deadline - 10:
        counts = get_conflicts(S)
        if not counts:
            break
        
        # Find the element with the maximum count
        # If there's a tie, pick the smallest one? Or largest? 
        # Removing larger numbers might free up smaller sums? 
        # Let's just pick the max count.
        max_count = max(counts.values())
        candidates = [x for x, c in counts.items() if c == max_count]
        
        # Remove one candidate. 
        # Heuristic: Removing an element that is a sum (e.g., > n/2) might be better 
        # because it's less likely to be a base for many other triples as a member?
        # Actually, small numbers are more likely to be part of many triples.
        # Let's remove the one with max count. If tie, remove the smallest?
        # Let's try removing the one with max count, breaking ties by value (larger first? smaller first?)
        # Larger numbers are sums more often. Small numbers are bases more often.
        # Let's stick to max count, tie-break by smallest value (most "central").
        to_remove = min(candidates)
        S.discard(to_remove)
        
    # Phase 2: Local Search Improvement
    # Try to improve the set size by swapping elements or adding back removed ones.
    # We keep track of the "removed" set to know what to try adding.
    # Since we started with 1..128, removed = {1..128} - S
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        removed = set(range(1, n + 1)) - S
        
        # 2a. Try to add any single element from 'removed'
        for x in removed:
            if time.time() > deadline - 5:
                break
            if x in S:
                continue
            new_S = S | {x}
            # Check if new_S is valid
            # We can do a fast check: does adding x create any new forbidden triple?
            # A new triple must involve x.
            # Triples involving x: (a, b, x) where a,b in S.
            # Sums: a+b, a+x, b+x must be in new_S.
            # a+b must be in S (since x is new, it can't be a+b unless a+b=x, but a,b,x distinct).
            # a+x must be in new_S -> a+x in S or a+x == x (impossible for a>0) or a+x in {x} (no).
            # So a+x must be in S.
            # b+x must be in S.
            # So we need a,b in S such that a+b in S, a+x in S, b+x in S.
            
            valid = True
            # Iterate over pairs in S
            # Optimization: Only check a,b such that a+x in S and b+x in S.
            # Let A = {a in S | a+x in S}
            # Let B = {b in S | b+x in S}
            # We need a,b in A intersect B (or just in S) such that a+b in S.
            # Actually, a and b must be in S.
            # Condition: a+b in S AND a+x in S AND b+x in S.
            
            # Precompute potential partners
            # This check is O(|S|^2). With |S| ~ 100, it's 10k ops.
            # Doing this for ~20 removed elements is 200k ops. Fast.
            
            for a in S:
                if a + x > n or (a + x) not in S:
                    continue
                for b in S:
                    if b == a:
                        continue
                    if b + x > n or (b + x) not in S:
                        continue
                    if a + b in S:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.add(x)
                improved = True
                break # Restart loop with new S
        
        # 2b. Try swapping: Remove one element y from S, add two from removed?
        # This is more complex. Let's try a simpler swap:
        # For each y in S, remove it. Then try to add as many as possible from removed.
        # If size increases, keep it.
        
        if not improved and time.time() < deadline - 5:
            # Try swaps
            best_swap_gain = 0
            best_swap_y = None
            best_swap_adds = None
            
            # Limit number of swaps tried to save time
            for y in list(S):
                if time.time() > deadline - 5:
                    break
                
                S_trial = S - {y}
                # Try to add elements from removed (excluding y, but y was in S so y is in removed? No, removed is 1..128 - S. So y is NOT in removed. Wait.
                # removed = {1..128} - S. So y is in S, so y is NOT in removed.
                # We want to add elements from 'removed'.
                
                added = []
                # Sort removed by some heuristic? Smallest first?
                # Just iterate.
                for x in removed:
                    if x == y:
                        continue
                    # Check if x can be added to S_trial
                    valid = True
                    for a in S_trial:
                        if a + x > n or (a + x) not in S_trial:
                            continue
                        for b in S_trial:
                            if b == a:
                                continue
                            if b + x > n or (b + x) not in S_trial:
                                continue
                            if a + b in S_trial:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        S_trial.add(x)
                        added.append(x)
                
                gain = len(added) - 1 # Removed 1, added len(added)
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap_y = y
                    best_swap_adds = added
                    # Update S_trial to the final state of this trial?
                    # We need to store the resulting set.
                    # To save memory, we just store the set.
                    S_best_trial = S_trial
                    if gain >= 2: # Significant improvement
                        break # Found a good swap, take it immediately? 
                        # Or keep looking for better? 
                        # Taking the first significant improvement is usually fast enough.
            
            if best_swap_gain > 0 and best_swap_y is not None:
                S = S_best_trial
                improved = True

    return list(S)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verification
    S = set(res)
    valid = True
    for a in S:
        for b in S:
            if a == b: continue
            for c in S:
                if a == c or b == c: continue
                if a+b in S and a+c in S and b+c in S:
                    valid = False
    print(f"Valid: {valid}")

