# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 4096
    
    # Strategy: Greedy construction with intelligent candidate selection.
    # We want to maximize the size of a set S such that all unordered triples
    # (with repetition) have unique sums.
    # 
    # Key insight: The constraint is on the sums of 3 elements.
    # If we pick elements that are "far apart" in some sense, we reduce collisions.
    # A simple greedy that picks the smallest available number works, but might be suboptimal.
    # A better greedy might pick numbers that are "spread out" to allow more elements.
    # 
    # However, the seed program just picks 1, 2, 3, ... until it can't.
    # Let's analyze the seed's performance.
    # The seed picks x from 1 to 512. It returns the list of selected numbers.
    # The score is 1.0, which means it returns a valid set.
    # The task says "Maximize cardinality". The score is value/seed.
    # Wait, the prompt says: "The evaluator measures the raw value (higher is better). The search score is value/seed, so 1.0 equals the seed program".
    # So the "value" is likely the cardinality (size of the set).
    # If the seed returns a set of size k, the score is k/k = 1.0.
    # To get a score > 1.0, we need to return a set of size > k.
    
    # Let's first determine the size of the set returned by the seed program.
    # I can't run the seed program here, but I can simulate its logic quickly or just assume it's the baseline.
    # The seed program loops x from 1 to 512. It stops if adding x causes a collision.
    # Actually, the seed program does NOT stop. It just checks `if len(sums)==len(set(sums))`.
    # If it fails, it does NOT append x. It continues to the next x.
    # So it tries all x from 1 to 512 and adds them if they don't cause a collision with the *currently selected* set.
    # This is a greedy algorithm that tries to add small numbers first.
    
    # Let's implement a faster version of this greedy, and then try to improve it.
    # To check if a new element x can be added to set S, we need to check if any triple sum involving x collides with an existing triple sum.
    # Existing triple sums: all sums of 3 elements from S.
    # New triple sums involving x:
    #   1. x+x+x = 3x
    #   2. x+x+y = 2x+y for y in S
    #   3. x+y+z = x+y+z for y,z in S, y<=z (unordered, but we can just iterate all pairs)
    # We need to check if any of these are in the set of existing triple sums.
    # Also, we need to check if any new triple sums collide with each other.
    #   - 3x vs 2x+y: 3x = 2x+y => x=y. Since x is not in S yet (we're checking if we can add it), and y is in S, this is only possible if x=y, which means x is already in S. But x is a candidate, so x is not in S. So no collision between type 1 and 2.
    #   - 3x vs x+y+z: 3x = x+y+z => 2x = y+z. Possible.
    #   - 2x+y vs 2x+z: y=z. This is the same sum. We are generating unique sums, so we should generate the set of new sums and check for duplicates within the new sums and against the old sums.
    
    # Efficient check:
    # Maintain a set `triple_sums` of all current triple sums.
    # For candidate x:
    #   new_sums = set()
    #   new_sums.add(3*x)
    #   for y in S:
    #       new_sums.add(2*x + y)
    #   for y in S:
    #       for z in S:
    #           if y <= z: # to avoid duplicates in generation? No, just add all.
    #               new_sums.add(x + y + z)
    #   # Check for collisions within new_sums? 
    #   # Wait, the generation above might add the same sum multiple times. 
    #   # E.g., x+y+z where (y,z) and (z,y) are both iterated. 
    #   # But we are adding to a set, so duplicates are removed. 
    #   # The only issue is if two *different* triples produce the same sum. 
    #   # That is exactly what we want to detect. 
    #   # So, if len(new_sums) < number_of_triples_involving_x, then there is a collision among the new triples.
    #   # Number of triples involving x:
    #   #   1 (xxx) + |S| (xxy) + C(|S|+1, 2) (xyz with y,z from S, y<=z? No, unordered triples from {x} U S).
    #   #   Actually, the number of unordered triples from a set of size k is C(k+2, 3).
    #   #   Here, we are adding x to S of size m. The new triples are those that include x.
    #   #   Count = 1 (xxx) + m (xxy) + C(m, 2) (xyz, y<z) + m (xyy? No, x+y+y is same as xyy).
    #   #   Let's list them:
    #   #   (x,x,x)
    #   #   (x,x,y) for y in S
    #   #   (x,y,y) for y in S
    #   #   (x,y,z) for y,z in S, y<z
    #   #   Total = 1 + m + m + C(m, 2) = 1 + 2m + m(m-1)/2 = 1 + 2m + (m^2-m)/2 = 1 + (3m + m^2)/2 = (m^2 + 3m + 2)/2 = (m+1)(m+2)/2.
    #   #   This is C(m+2, 2). Wait.
    #   #   Number of unordered triples with repetition from a set of size m+1 is C(m+1+3-1, 3) = C(m+3, 3).
    #   #   The number of new triples is C(m+3, 3) - C(m+3, 3) ... no.
    #   #   Old triples: C(m+2, 3).
    #   #   New triples: Total triples in S U {x} minus old triples.
    #   #   Total = C(m+1+2, 3) = C(m+3, 3).
    #   #   New = C(m+3, 3) - C(m+2, 3) = C(m+2, 2) = (m+2)(m+1)/2.
    #   #   So if len(new_sums) < (m+2)(m+1)/2, there is a collision.
    #   #   Also, if new_sums intersects with triple_sums, there is a collision.
    
    # This check is O(m^2) per candidate.
    # If we try candidates in increasing order, m grows.
    # The seed program tries x from 1 to 512.
    # If we can do this faster, we can try more candidates or smarter candidates.
    
    # Let's implement this efficient greedy first.
    
    S = []
    triple_sums = set()
    
    # We will try candidates from 1 to n.
    # But maybe we can do better by trying larger numbers first? 
    # Or random order?
    # The seed program tries 1, 2, 3...
    # If we pick large numbers, the sums are large and less likely to collide?
    # But the range is limited to 4096.
    # Let's stick to increasing order first, but with the efficient check.
    
    # Precompute nothing, just loop.
    
    # To beat the seed, we need to find a larger set.
    # The seed's greedy might get stuck because it picks small numbers, which have many small sums that collide.
    # If we skip some small numbers, we might be able to fit more numbers overall.
    
    # Let's try a randomized restart approach.
    # 1. Greedy in increasing order.
    # 2. Greedy in random order.
    # 3. Greedy with some heuristic (e.g., pick the number that minimizes the number of future collisions? Too expensive).
    
    # Let's first implement the efficient greedy in increasing order and see how many we get.
    # Then we can try random restarts.
    
    def try_greedy(order, deadline):
        S = []
        triple_sums = set()
        m = 0
        for x in order:
            if time.time() > deadline:
                break
            # Check if x can be added
            # Generate new sums
            new_sums = set()
            new_sums.add(3*x)
            for y in S:
                new_sums.add(2*x + y)
            for i in range(m):
                for j in range(m):
                    new_sums.add(x + S[i] + S[j])
            
            # Count expected number of unique sums
            expected = (m+1)*(m+2)//2
            
            # Check for internal collisions
            if len(new_sums) < expected:
                continue
            
            # Check for collisions with existing sums
            if new_sums & triple_sums:
                continue
            
            # Add x
            S.append(x)
            triple_sums.update(new_sums)
            m += 1
            
            # Optional: if m reaches some limit, stop? No, we want to maximize.
            # But if m is large, the check becomes slow.
            
        return S
    
    # Try increasing order
    best_S = try_greedy(range(1, n+1), deadline)
    best_size = len(best_S)
    
    # Try random restarts
    # How many restarts can we do?
    # Each restart takes some time.
    # Let's do a few random restarts.
    
    np.random.seed(42)
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5: # Leave 5s margin
            break
        
        # Random order of candidates
        # We don't need to shuffle all 4096, just pick a random subset?
        # No, we should try all candidates, but in random order.
        # Shuffling 4096 elements is fast.
        candidates = list(range(1, n+1))
        np.random.shuffle(candidates)
        
        S = try_greedy(candidates, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    return best_S

# EVOLVE-BLOCK-END
