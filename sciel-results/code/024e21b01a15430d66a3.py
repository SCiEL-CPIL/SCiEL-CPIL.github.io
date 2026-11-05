import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Max possible size is 48, but we can try to go up to 512 numbers if valid
    # But the task says "Maximize cardinality" and "At most 48 integers in [1,512]"
    # Wait, "At most 48 integers" is in the contract text: "Maximize cardinality. At most 48 integers in [1,512]..."
    # This implies the answer is at most 48.
    
    selected = []
    
    # Helper to check if a value 'target' is in subset sums of 'items' of size 'k'
    # This is slow if done naively. We will use a DP bitset for the final verification.
    
    # Greedy construction
    # We iterate candidates from 512 down to 1
    # To speed up, we can skip candidates that are obviously bad?
    
    # We maintain subset sums of 'selected' for all sizes up to len(selected)
    # sums[k] is a set of sums of subsets of 'selected' of size k
    # Initially sums[0] = {0}
    
    # To make it fast, we only update sums when we add an element.
    # But checking validity of a new element x against selected S:
    # 1. x is not mean of subset of S:
    #    For k in 2..len(S): if k*x in sums[k] -> invalid
    # 2. a in S is not mean of subset of S U {x}:
    #    For each a in S:
    #      For j in 1..len(S)-1:
    #         target = (j+1)*a - x
    #         if target in sums_of(S \ {a}, j) -> invalid
    #        
    # Computing sums_of(S \ {a}, j) is the bottleneck.
    # However, if we assume that if target is in sums[S][j], it's likely formed by a subset NOT containing a?
    # No, that's risky.
    
    # Let's try a different approach:
    # Pick numbers that are "far apart".
    # Start with 512, 511, ...
    # Actually, a known construction for mean-free sets is to take numbers from [n/2, n].
    # Let's try picking the largest available numbers.
    
    max_n = 512
    candidates = list(range(max_n, 0, -1))
    
    # We will use a fast check that only considers small subset sizes (2, 3, 4)
    # and then do a full check at the end.
    
    def is_valid_fast(selected, x):
        # Check if x is mean of subset of selected (sizes 2,3,4)
        # We need subset sums of selected of size 2,3,4.
        # We can compute these on the fly or maintain them.
        # For speed, let's just compute combinations for small k.
        # k=2: any pair (a,b) in selected with a+b = 2x
        # k=3: any triple (a,b,c) in selected with a+b+c = 3x
        # k=4: any quadruple...
        
        # This is O(N^k). N is up to 48.
        # 48^4 is ~5M. Too slow for every step if we do it for every candidate.
        # But we only check one candidate at a time.
        # We try candidates from 512 down.
        # If we accept many, N grows.
        
        # Let's maintain sums[2], sums[3], sums[4] as sets.
        pass

    # Better: Maintain sums[0..K] for K=5 or so.
    # And for the "a in S" check, we might miss things.
    # Let's just build the set and verify at the end.
    # If the final set is invalid, we trim.
    
    selected = []
    # To get a large set, we should pick numbers that don't form means.
    # Picking large numbers is good because 2a is large, and sums of other large numbers are large.
    # But if we pick consecutive numbers, we get 3-APs.
    # So we should pick numbers with gaps.
    
    # Heuristic: Pick 512, then 510, 506, ...?
    # Or just try all candidates in descending order and add if it passes a FAST check.
    
    # Fast check:
    # 1. 2x is not sum of 2 elements in selected.
    # 2. 3x is not sum of 3 elements in selected.
    # 3. For each a in selected, 2a is not sum of x and 1 element in selected\{a}.
    #    i.e., 2a - x is not in selected\{a}.
    # 4. For each a in selected, 3a is not sum of x and 2 elements in selected\{a}.
    #    i.e., 3a - x is not sum of 2 elements in selected\{a}.
    
    # This covers small subsets.
    
    # Let's implement this.
    
    # To check "sum of 2 elements in S", we can use a set of sums or just a set of elements.
    # sum of 2 elements: a+b = target -> b = target-a. Check if b in S.
    # sum of 3 elements: a+b+c = target.
    
    # We will maintain:
    # sums2: set of sums of 2 distinct elements
    # sums3: set of sums of 3 distinct elements
    # sums4: set of sums of 4 distinct elements (maybe)
    
    # Updating these sets when adding x:
    # new sums2 = old sums2 U {x + y for y in selected}
    # new sums3 = old sums3 U {x + s2 for s2 in old sums2}
    # This is efficient.
    
    selected = []
    sums2 = set()
    sums3 = set()
    sums4 = set()
    
    for x in range(512, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Check 1: x is not mean of subset of selected
        # k=2: 2x in sums2?
        # k=3: 3x in sums3?
        # k=4: 4x in sums4?
        
        valid = True
        if 2*x in sums2:
            valid = False
        elif 3*x in sums3:
            valid = False
        elif 4*x in sums4:
            valid = False
        
        if valid:
            # Check 2: a in selected is not mean of subset of selected U {x}
            # Subsets involving x:
            # Size 2: {x, y}. Mean (x+y)/2 = a -> x+y = 2a -> y = 2a-x.
            #   y must be in selected, y != a, y != x.
            #   Also y must be distinct from x.
            # Size 3: {x, y, z}. Mean (x+y+z)/3 = a -> x+y+z = 3a -> y+z = 3a-x.
            #   y,z in selected, distinct from a, x.
            #   y+z must be in sums2_of(selected \ {a, x})?
            #   Approximation: check if 3a-x is in sums2.
            #   If it is, it might be formed by a and some other element.
            #   We need to be careful.
            
            for a in selected:
                # Check size 2 subset {x, y}
                y = 2*a - x
                if y == x:
                    continue # y must be other element, but subset elements must be distinct from a and each other?
                    # "subset of at least two OTHER elements".
                    # So if a is the mean, the subset is from A \ {a}.
                    # So y must be in A \ {a}.
                    # If y == a, then 2a-x=a -> a=x, but x is new, so a!=x.
                    # If y == x, then x is in the subset. But the subset must be from A\{a}.
                    # If x is in the subset, it's allowed as long as x != a.
                    # But the subset elements must be distinct? "subset" usually implies distinct elements from the set.
                    # Yes, "distinct positive integers".
                    # So y must be in selected, y != a.
                    # Also y must be distinct from x?
                    # The subset is {x, y}. If x==y, it's not a set of distinct integers?
                    # The set A has distinct integers. So x != y if they are different elements.
                    # But here we are forming a subset of A U {x}.
                    # If y=x, then the subset is {x} (size 1) or {x,x} (invalid).
                    # So y must be != x.
                    
                if y in selected and y != a and y != x:
                    valid = False
                    break
                    
                if not valid: break
                
                # Check size 3 subset {x, y, z}
                # y+z = 3a - x
                # y,z in selected, y!=a, z!=a, y!=z, y!=x, z!=x.
                # Approx: check if 3a-x is in sums2.
                # If yes, there exist u,v in selected with u+v = 3a-x.
                # We need to ensure u,v are not a, not x, and u!=v.
                # This is complex. We'll just check if 3a-x is in sums2.
                # False positives are okay (we reject valid x). False negatives are bad (we accept invalid x).
                # But we will do a full check at the end.
                target3 = 3*a - x
                if target3 in sums2:
                    # Potential violation
                    # We can refine: check if there is a pair in selected \ {a, x} summing to target3.
                    # For now, mark as invalid to be safe?
                    # If we mark as invalid, we might miss good x.
                    # Let's just mark as invalid.
                    valid = False
                    break

        if valid:
            selected.append(x)
            # Update sums
            # New sums2: x + y for y in selected[:-1]
            for y in selected[:-1]:
                sums2.add(x + y)
            # New sums3: x + s2 for s2 in old sums2 (before adding x's pairs)
            # Wait, we need to use the sums2 from BEFORE adding x.
            # So we should compute new_sums3 using old_sums2.
            # But we already updated sums2.
            # We need to store old_sums2.
            
            # Let's redo the update logic properly.
            # This code above is flawed in the update.
            pass

    # The above logic is getting messy. Let's use a simpler, more robust greedy.
    # Just pick 512, 511, ... and check with a full DP for the final set.
    # To get a large set, we should avoid 3-APs.
    # A simple set that is mean-free is powers of 2.
    # But we want max cardinality.
    
    # Let's try a random restart approach with a fast check.
    # Or just return a known good set?
    # No, we must compute.
    
    # Final plan:
    # 1. Greedily add largest numbers, checking only k=2,3 violations for x and a.
    # 2. Verify the final set fully.
    # 3. If invalid, trim from the end until valid.
    
    return selected[:48] # Placeholder

# Let's write the actual code with the full verification at the end.

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    selected = []
    
    # Fast check sums
    sums2 = set()
    sums3 = set()
    
    for x in range(512, 0, -1):
        if time.time() > deadline - 10:
            break
            
        valid = True
        
        # Check if x is mean of subset of selected
        if 2*x in sums2:
            valid = False
        elif 3*x in sums3:
            valid = False
            
        if valid:
            # Check if a in selected becomes mean
            for a in selected:
                # Size 2: {x, y}
                y = 2*a - x
                if y in selected and y != a and y != x:
                    valid = False
                    break
                # Size 3: {x, y, z} -> y+z = 3a-x
                # Check if 3a-x is in sums2
                # Note: sums2 contains sums of pairs from selected.
                # If 3a-x is in sums2, there is a pair (u,v) in selected with u+v=3a-x.
                # We need u,v != a and u,v != x.
                # If we assume this is a violation, we are safe (false positive).
                if 3*a - x in sums2:
                    valid = False
                    break
                    
        if valid:
            selected.append(x)
            # Update sums2 and sums3
            # New pairs with x
            new_sums2 = set()
            for y in selected[:-1]:
                s = x + y
                sums2.add(s)
                new_sums2.add(s)
            
            # New triples with x: x + pair from OLD selected
            # We need to iterate over pairs in selected[:-1]
            # We can use the old sums2? No, sums2 was updated.
            # We need to store old_sums2.
            
            # Let's just recompute sums2 and sums3 from scratch?
            # |selected| is small.
            pass

    # This is too slow/buggy.
    # Let's just return a set of powers of 2 multiplied by a constant?
    # No, let's use a simple greedy with a full check at the end.
    
    # Greedy: pick largest available.
    # Check: is it valid?
    # To check validity of adding x to S:
    # We can just add it, and then check the whole set S U {x} for validity?
    # Full check of a set of size K takes O(K^2 * Sum) time?
    # If K=48, and we do this for each of 48 steps, it's 48 * (48^2 * 25000) ~ 2.7e9. Too slow.
    
    # So we must use the fast check for construction, and full check for the final result.
    
    # The fast check above (k=2,3) is good enough to build a large set.
    # The full check at the end will catch any errors.
    # If the final set is invalid, we trim.
    
    # Let's implement the fast check correctly.
    
    selected = []
    # We will maintain sums2 and sums3 as sets.
    sums2 = set()
    sums3 = set()
    
    for x in range(512, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Check x against selected
        if 2*x in sums2 or 3*x in sums3:
            continue
            
        # Check a in selected against x
        ok = True
        for a in selected:
            # Size 2: y = 2a - x
            y = 2*a - x
            if y in selected and y != a and y != x:
                ok = False
                break
            # Size 3: y+z = 3a - x
            # Check if 3a-x is in sums2
            # If yes, there is a pair in selected summing to it.
            # We assume this is a violation.
            if 3*a - x in sums2:
                ok = False
                break
                
        if ok:
            selected.append(x)
            # Update sums
            # New sums2: x + y for y in selected[:-1]
            # New sums3: x + (y+z) for y,z in selected[:-1]
            # We need old sums2 for sums3 update.
            old_sums2 = set(sums2)
            
            for y in selected[:-1]:
                sums2.add(x + y)
                
            for s2 in old_sums2:
                sums3.add(x + s2)
                
            # Also, sums3 gets new triples from new pairs?
            # No, triples involving x are x + pair.
            # Triples not involving x are already in sums3.
            # So this update is correct.
            
            if len(selected) >= 48:
                break

    # Final Verification
    # If selected is invalid, trim it.
    # We verify the whole set.
    
    def is_valid_set(S):
        n = len(S)
        if n < 2:
            return True
            
        # For each a in S, check if a is mean of subset of S\{a}
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            # Check all subset sizes k >= 2
            # Use DP for subset sums of others
            # Max sum ~ 48 * 512 = 24576
            # We can use a bitset or set of sums for each size.
            
            # Since n is small, we can use combinations for small k?
            # No, k can be large.
            # But we can use DP: dp[k] is a set of sums of size k.
            dp = [set() for _ in range(len(others)+1)]
            dp[0].add(0)
            
            for y in others:
                # Update from high k to low k
                for k in range(len(others)-1, -1, -1):
                    if dp[k]:
                        for s in dp[k]:
                            dp[k+1].add(s + y)
                            
            for k in range(2, len(others)+1):
                if k*a in dp[k]:
                    return False
        return True

    # Verify and trim
    while selected and not is_valid_set(selected):
        selected.pop()
        if time.time() > deadline - 1:
            break

    return selected

# Run construct
construct()
