def construct():
    # Approach: Construct a Sidon set (or near-Sidon) that covers [128, 256].
    # A Sidon set has all pair sums distinct (multiplicity 1).
    # If we can find one covering the target interval, peak multiplicity is 1.
    # If not, we minimize the number of collisions.
    
    target_min = 128
    target_max = 256
    n = 128
    
    # We want A subset [0, 128]
    # Start with endpoints to cover min and max sums
    A = [0, 128]
    A_set = set(A)
    sums = set([0, 128, 256])
    
    # We need to cover [128, 256]
    # Uncovered targets
    targets = set(range(target_min, target_max + 1)) - sums
    
    # Greedy selection of next element
    # To maintain Sidon property, new sums x+a must not be in existing sums
    # and x+x must not be in existing sums (unless x is already in A, which it isn't)
    # Also x+a must be distinct for different a in A.
    
    # Candidate elements: 1 to 127
    candidates = list(range(1, 128))
    
    # We will try to add elements one by one.
    # To cover the middle, we might need to be strategic.
    # Simple greedy: pick candidate that maximizes coverage of 'targets'
    # while maintaining low collision count.
    
    max_iter = 100
    for _ in range(max_iter):
        if not targets:
            break
            
        best_candidate = None
        best_new_coverage = -1
        best_new_collisions = float('inf')
        
        for c in candidates:
            if c in A_set:
                continue
                
            # Calculate new sums
            new_sums = []
            valid = True
            for a in A:
                s = c + a
                if s in sums:
                    # Collision! 
                    # If we allow collisions, we track them. 
                    # For Sidon, we disallow.
                    # Let's first try strict Sidon.
                    valid = False
                    break
                new_sums.append(s)
            
            # Check internal collisions among new sums
            # c + a1 == c + a2 => a1 == a2 (distinct in A)
            # c + c is unique.
            # So if no collision with existing sums, new sums are distinct.
            
            if not valid:
                continue
                
            # Count how many targets are covered
            covered = sum(1 for s in new_sums if s in targets)
            
            if covered > best_new_coverage:
                best_new_coverage = covered
                best_candidate = c
                best_new_sums = new_sums
                
        if best_candidate is None:
            # No valid candidate for strict Sidon
            # Break or relax? 
            # If we can't extend Sidon, we might need to accept a set with multiplicity 2.
            # For now, stop if stuck.
            break
            
        # Add best candidate
        A.append(best_candidate)
        A_set.add(best_candidate)
        sums.update(best_new_sums)
        targets -= set(best_new_sums)
        
    # If targets remain, the strict Sidon approach failed to cover all.
    # We can try to add more elements even if they cause collisions, 
    # prioritizing those that cover remaining targets with minimal added multiplicity.
    
    if targets:
        # Relaxation: Allow collisions but minimize max multiplicity
        # Calculate current multiplicities
        from collections import defaultdict
        mult = defaultdict(int)
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                mult[s] += 1
                
        max_mult = max(mult.values()) if mult else 0
        
        # Try to cover remaining targets by adding elements
        # This part is complex. 
        # Given time, if strict Sidon fails, we return the Sidon set found 
        # which likely has peak 1 for the covered part, but misses some targets.
        # But the task REQUIRES covering [128, 256].
        
        # If we missed targets, the solution is invalid.
        # We need a fallback that ensures validity.
        # The fallback is the arithmetic progression which is valid but high peak.
        
        # Let's check if we covered everything.
        if targets:
             # Fallback to a valid set with lower peak than 33 if possible.
             # Try a set of size ~20-25 with random search? 
             # No, let's try a different deterministic construction: 
             # A = {0, 1, ..., k-1} union {N-k+1, ..., N} ?
             
             # Let's use a simple heuristic: 
             # If strict Sidon failed, use the seed's random search logic 
             # but optimized? 
             # Actually, let's just return the arithmetic progression if Sidon fails,
             # but we can do better.
             
             # Let's try to construct a set with peak 2 or 3.
             # A known good structure is taking a grid.
             # e.g. A = {i * 4 + j for i in range(8) for j in range(0, 1)}? No.
             
             # Let's stick to the Sidon set if it works. 
             # If it doesn't, we need a valid set.
             
             # Quick check: Did we cover [128, 256]?
             if targets:
                 # We failed to cover. 
                 # Use a known valid set with low peak.
                 # Try A = list(range(0, 129, 2))? 
                 # Sums are even. Misses odd numbers. Invalid.
                 
                 # Try A = list(range(0, 129, 4))? Sums multiples of 4. Invalid.
                 
                 # We need a set that covers ALL integers.
                 # The arithmetic progression [64, ..., 128] covers all.
                 # Peak 33.
                 
                 # Can we do better?
                 # Let's try A = {0, 128} U {x_i}.
                 
                 # If the greedy Sidon search got stuck, it means we couldn't find a 
                 # collision-free addition. 
                 
                 # Let's just return the best valid set we can find quickly.
                 # We'll use a small random search as a backup, but deterministic.
                 
                 import random
                 random.seed(123)
                 best_A = None
                 best_peak = 33 # Fallback peak
                 
                 for k in range(16, 22):
                     for _ in range(200):
                         cand = random.sample(range(129), k)
                         cand.sort()
                         if cand[0] > 64 or cand[-1] < 128:
                             continue
                         
                         # Check validity and peak
                         A_set_c = set(cand)
                         valid = True
                         max_c = 0
                         for s in range(128, 257):
                             count = 0
                             for a in cand:
                                 if a > s: break
                                 b = s - a
                                 if b < a: continue
                                 if b in A_set_c:
                                     count += 1
                             if count == 0:
                                 valid = False
                                 break
                             if count > max_c:
                                 max_c = count
                         
                         if valid and max_c < best_peak:
                             best_peak = max_c
                             best_A = cand.copy()
                             if best_peak == 1:
                                 return best_A
                 
                 if best_A:
                     return best_A
                 
                 # Final fallback
                 return list(range(64, 129))
             else:
                 # We covered everything with a Sidon set!
                 return A
    else:
        return A
