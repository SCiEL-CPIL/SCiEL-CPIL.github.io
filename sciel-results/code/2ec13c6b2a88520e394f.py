import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 60.0  # 60 seconds limit, keep margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMS
    # lcm[a][b] for 1<=a,b<=128
    # Using math.lcm or manual GCD
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a * b // gcd(a, b)

    # Precompute all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We can store these as frozensets or sorted tuples
    bad_triples = []
    # To speed up, we can just check on the fly if we do it smartly, 
    # but precomputation is O(128^3) ~ 2M ops, which is fast enough in Python? 
    # 2 million iterations might take a few seconds. 
    # Let's optimize: 
    # For each pair (a,b), let L = lcm(a,b). 
    # Any c such that lcm(a,c)=L and lcm(b,c)=L forms a bad triple.
    # This is complex to precompute efficiently without loops.
    # Given N=128, simple triple loop is ~ 350k iterations. 
    # 350k * (3 lcms) is very fast.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((a, b, c))
                    
    # Map element to list of bad triples containing it
    # triple_index[i] = list of indices in bad_triples that contain i
    triple_index = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        triple_index[a].append(idx)
        triple_index[b].append(idx)
        triple_index[c].append(idx)
        
    # Initial Set: All elements
    S = set(elements)
    
    # Remove elements to resolve conflicts
    # We will iteratively find a bad triple fully contained in S and remove one element.
    # To maximize remaining size, we remove the element that appears in the most bad triples *within S*.
    
    # We need to efficiently find a bad triple in S.
    # We can maintain a list of "active" bad triples (those with all 3 elements in S).
    # Initially all are active.
    active_triples = set(range(len(bad_triples)))
    
    # Helper to check if a triple is still in S
    def is_in_s(triple_idx):
        a, b, c = bad_triples[triple_idx]
        return a in S and b in S and c in S
        
    # Iterative removal
    # To avoid O(N) scan for a bad triple every time, we can just pick a random one from a list, 
    # but we need to ensure it's still in S.
    # Given small N, scanning is fine.
    
    while True:
        # Find a bad triple in S
        found = None
        # Randomize order to avoid bias? Or just pick first?
        # Picking the one with max "impact" (degree) is better.
        # Let's find all bad triples in S and pick the one where the min-degree element is removed?
        # Or pick a random bad triple, then remove the element with highest degree in S-conflict graph.
        
        # Efficiently finding one:
        # Iterate through active_triples? No, active_triples is a set of indices.
        # We can just iterate through bad_triples and check membership.
        # 350k checks. If we do this 100 times, 35M checks. Might be slow in Python.
        
        # Optimization: Maintain a list of current bad triples.
        # When we remove an element x, we remove all triples containing x from the active list.
        
        # Let's restart the logic with a list of active triple indices.
        # We'll use a list `active` of indices into bad_triples.
        # When x is removed, we filter `active` to remove any triple containing x.
        
        # Reset for this run
        S = set(elements)
        active = list(range(len(bad_triples)))
        
        # Precompute element -> triple indices map again for speed in removal
        # triple_index is already built.
        
        # We need a way to quickly remove triples containing x from `active`.
        # Since `active` is a list, removing items is O(N).
        # But we can just mark them as inactive?
        # Or use a set for `active`?
        
        # Let's use a set of active triple indices.
        active_set = set(range(len(bad_triples)))
        
        # We also need to know, for a given x, which triples in active_set contain x.
        # We can iterate through triple_index[x] and check if idx is in active_set.
        
        while active_set:
            if time.time() - start_time > limit - 5:
                break
            
            # Pick a random active triple to resolve
            # Using a list from set is O(1) if we use next(iter(set)) but it's biased?
            # Better to pick random index from a list representation?
            # For now, just pick one.
            # To be fair, we can pick the one that allows removing the most "conflicting" element.
            
            # Let's pick a random triple from active_set
            # Converting set to list each time is O(N). 
            # We can maintain a list and a set.
            # Or just use list(active_set)[0] if we don't care about randomness?
            # Randomness helps escape local minima in the "removal order".
            
            # Let's pick a random element from the set.
            # Python's random.choice requires a sequence.
            # We can just do:
            if not active_set:
                break
            idx = random.choice(list(active_set))
            a, b, c = bad_triples[idx]
            
            # Decide which one to remove.
            # Heuristic: Remove the one that is in the most *other* active triples.
            # Count active triples for a, b, c.
            counts = {x: 0 for x in (a, b, c)}
            for x in (a, b, c):
                for t_idx in triple_index[x]:
                    if t_idx in active_set:
                        counts[x] += 1
                        
            # Remove the one with max count
            to_remove = max(counts, key=counts.get)
            
            S.remove(to_remove)
            
            # Update active_set: remove all triples containing to_remove
            # Iterate through triple_index[to_remove]
            for t_idx in triple_index[to_remove]:
                if t_idx in active_set:
                    active_set.remove(t_idx)
                    
        # Now S is valid. 
        # Try to add back elements from the complement.
        complement = [x for x in elements if x not in S]
        
        # Shuffle complement to try adding in random order
        random.shuffle(complement)
        
        for x in complement:
            if time.time() - start_time > limit - 5:
                break
            
            # Check if adding x creates any bad triple
            # We need to check all pairs (a, b) in S such that (a, b, x) is a bad triple.
            # We can iterate through triple_index[x] and check if the other two are in S.
            conflict = False
            for t_idx in triple_index[x]:
                a, b, c = bad_triples[t_idx]
                # The other two elements in the triple
                if a == x:
                    others = (b, c)
                elif b == x:
                    others = (a, c)
                else:
                    others = (a, b)
                
                if others[0] in S and others[1] in S:
                    conflict = True
                    break
            
            if not conflict:
                S.add(x)
                # No need to update active_set here as we are in "add back" phase 
                # and we only care about validity of final S.
                # But if we wanted to continue local search, we'd need to.
        
        # Local Search / Hill Climbing to improve further
        # Try to swap elements: remove one from S, add one from complement, if size increases (impossible) 
        # or if we can add more later? 
        # Actually, the "add back" phase is greedy. 
        # We can try to swap an element in S with one in complement if it allows adding *more* elements later?
        # This is hard to evaluate.
        
        # Simpler: Try to find a pair (u in S, v in complement) such that 
        # removing u and adding v doesn't create conflicts, and then we can add more?
        # Or just check if we can add v by removing u?
        
        # Let's do a simple local search:
        # 1. Try to add any complement element (already done greedily, but maybe order mattered).
        # 2. Try to swap: pick u in S, v in complement. 
        #    If S - {u} + {v} is valid, and allows adding more elements than S - {u}, accept.
        #    This is expensive.
        
        # Given time constraints, the greedy remove-then-add is likely strong.
        # Let's run this multiple times with different random seeds/orders to find the best.
        
        current_best = len(S)
        
        # Store best set
        if not hasattr(construct, '_best_set'):
            construct._best_set = set()
            construct._best_size = 0
            
        if current_best > construct._best_size:
            construct._best_size = current_best
            construct._best_set = set(S)
            
        # If we have time, restart the process with different random choices
        if time.time() - start_time < limit - 10:
            continue
        else:
            break
            
    return list(construct._best_set) if construct._best_set else [1]
