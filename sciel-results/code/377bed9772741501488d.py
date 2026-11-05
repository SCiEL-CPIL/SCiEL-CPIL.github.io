import time
import random
import sys

# Constants
N = 32
MAX_VAL = 512
TIME_LIMIT = 110.0 # Safety margin from 120s

def construct():
    start_time = time.time()
    
    # Precompute valid indices
    indices = list(range(1, MAX_VAL + 1))
    
    best_score = -1
    best_set = None
    
    # We will run multiple restarts of Simulated Annealing
    # Each restart tries to improve upon the best found so far
    
    def calculate_initial_scores(S):
        """Calculate diff_counts and score for a given set S."""
        diff_counts = [0] * (MAX_VAL + 1)
        in_set = [False] * (MAX_VAL + 1)
        for x in S:
            in_set[x] = True
        
        # For each pair (a, b) in S, check if they form a 3-AP
        # This is O(N^2)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a = S[i]
                b = S[j]
                if a > b:
                    a, b = b, a
                # a < b
                # Case 1: a is middle? No, a < b.
                # If a is start: b is middle? No.
                # Let's just check all 3-APs defined by a,b.
                
                # If a and b are in S, do they form a 3-AP with some c?
                # c = 2a - b (a is middle) -> c < a
                # c = 2b - a (b is middle) -> c > b
                # c = a + (b-a)/2 (b is end, a is start) -> c between a and b
                # c = b + (b-a)/2 (a is end, b is start) -> c < a (wait, if a<b, b is start? no)
                
                # Let's iterate d instead? No, pairs are better.
                
                # 1. a is middle: c = 2a - b. If c >= 1 and c in S, then d = a-b is covered.
                # 2. b is middle: c = 2b - a. If c <= MAX_VAL and c in S, then d = b-a is covered.
                # 3. a is start, b is end: d = (b-a)/2. If (b-a) even, c = a+d. If c in S, d covered.
                
                # Wait, this only counts if the 3rd element is in S.
                # We are building S from scratch.
                
                pass # This function is not strictly needed if we do incremental updates during SA, 
                     # but useful for initialization.
        
        # Better initialization: Greedy
        return None

    def greedy_init():
        """Construct a set using a greedy approach to maximize distinct differences."""
        S = set()
        # Start with some random elements or spread out elements
        # To maximize differences, we want elements that form many 3-APs.
        # A simple heuristic: pick elements that are far apart to allow large d, 
        # but close enough to form small d.
        
        # Let's just start with a random set and polish it.
        # Or a structured set: 
        # Random sampling is usually fine for SA to escape.
        
        # Let's try a few random inits and pick the best initial score
        best_init_S = None
        best_init_score = -1
        
        for _ in range(20):
            S_candidate = set(random.sample(indices, N))
            score = get_score(S_candidate)
            if score > best_init_score:
                best_init_score = score
                best_init_S = S_candidate
        
        return list(best_init_S), best_init_score

    def get_score(S):
        """Calculate the number of distinct differences d for a set S."""
        S_set = set(S)
        covered = set()
        # For each pair (a, b) in S, determine possible d
        # It's faster to iterate over all a, b and check c
        # O(N^2)
        for a in S:
            for b in S:
                if a == b: continue
                # 1. a is middle: a-d, a+d. d = |b-a|/2? No.
                # Let's just check all d? No.
                
                # If a, b in S:
                # If b > a:
                #   d = b - a. Check if a-d in S (a is middle) -> d covered
                #   d = (b - a) / 2. If int, check if a + d in S (b is end) -> d covered
                #   d = b - a. Check if b + d in S (a is start) -> d covered (this is same as first case for different pair)
                
                # Let's be systematic.
                # For each a in S:
                #   For each b in S:
                #     If b > a:
                #       d1 = b - a
                #       if (a - d1) in S_set: covered.add(d1)
                #       if (b + d1) in S_set: covered.add(d1)
                #       
                #       if (b - a) % 2 == 0:
                #         d2 = (b - a) // 2
                #         if (a + d2) in S_set: covered.add(d2)
        
        for a in S:
            for b in S:
                if b <= a: continue
                d = b - a
                # a is middle: a-d, a, a+d. Here a+d = b. So check a-d.
                if (a - d) in S_set:
                    covered.add(d)
                # a is start: a, a+d, a+2d. Here a+d = b. So check a+2d = b+d.
                if (b + d) in S_set:
                    covered.add(d)
                # a is end? No, b > a.
                # b is middle: b-d, b, b+d. Here b-d = a. So check b+d.
                # This is covered by "a is start" case? 
                # If b is middle, then a = b-d, c = b+d. 
                # In the loop above, when we consider pair (a,b), we check if a-d in S (a middle) and b+d in S (a start).
                # The case "b is middle" corresponds to a = b-d. 
                # So we check if b+d in S. This is exactly the second condition: `if (b + d) in S_set`.
                # Wait.
                # If a is start, b is middle: a, b, c. d = b-a. c = b+d.
                # If b is middle, a is start: a, b, c. d = b-a. c = b+d.
                # These are the same geometric configuration.
                
                # What if b is end? a, c, b. d = (b-a)/2. c = a+d.
                if d % 2 == 0:
                    d2 = d // 2
                    if (a + d2) in S_set:
                        covered.add(d2)
                        
        return len(covered)

    # Main SA Loop
    # We will use a single long SA run with multiple restarts if stuck.
    
    # Initialize
    S_list, current_score = greedy_init()
    S_set = set(S_list)
    in_set = [False] * (MAX_VAL + 1)
    for x in S_list:
        in_set[x] = True
        
    # Precompute neighbors for fast updates?
    # For a given x, and current S, the differences affected are those involving x.
    # We will calculate delta score on the fly.
    
    def remove_element(x):
        """Update in_set, S_list, and return list of (d, delta_count) to apply to diff_counts."""
        # This is tricky to do incrementally without maintaining diff_counts.
        # Let's maintain diff_counts.
        pass

    # Let's maintain diff_counts explicitly.
    # diff_counts[d] = number of 3-APs with difference d in current S.
    
    def init_diff_counts(S_list):
        dc = [0] * (MAX_VAL + 1)
        S_set = set(S_list)
        for a in S_list:
            for b in S_list:
                if b <= a: continue
                d = b - a
                # a middle
                if (a - d) in S_set:
                    dc[d] += 1
                # a start (b middle)
                if (b + d) in S_set:
                    dc[d] += 1
                # b end (a start, c middle)
                if d % 2 == 0:
                    d2 = d // 2
                    if (a + d2) in S_set:
                        dc[d2] += 1
        return dc

    diff_counts = init_diff_counts(S_list)
    current_score = sum(1 for d in range(1, MAX_VAL+1) if diff_counts[d] > 0)
    
    best_score = current_score
    best_S = list(S_list)
    
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    
    # We need to be able to quickly calculate the change in score when swapping x_out with x_in.
    # Delta Score = New Score - Old Score
    # New Score depends on new diff_counts.
    
    def get_delta_score(x_out, x_in):
        """
        Calculate the change in score if we remove x_out and add x_in.
        Returns delta_score.
        """
        # We need to know which differences are currently covered by x_out and which will be covered by x_in.
        # This is complex because x_out and x_in might interact with other elements.
        
        # Simpler approach:
        # 1. Remove x_out from S_set and in_set.
        # 2. Update diff_counts by decrementing counts for all d where x_out was part of a 3-AP.
        # 3. Add x_in to S_set and in_set.
        # 4. Update diff_counts by incrementing counts for all d where x_in is part of a 3-AP.
        # 5. Calculate new score.
        # 6. Revert changes.
        
        # This is O(|S|) for each step.
        
        old_score = current_score
        # We can't easily access current_score inside this function if it's a closure variable modified by SA loop.
        # Let's pass state or use nonlocal.
        
        pass

    # Given the complexity of incremental updates and the small N=32, 
    # recalculating the score from scratch for each candidate move might be acceptable if we limit the number of moves.
    # O(N^2) = 1024 operations per move.
    # 120 seconds / 1e-6 sec per op (optimistic) = 1.2e8 ops.
    # 1.2e8 / 1024 = ~100,000 moves.
    # This is a decent number for SA.
    
    # Let's try a simpler approach: 
    # 1. Generate a candidate move (swap).
    # 2. Create a new set S_new.
    # 3. Calculate score(S_new).
    # 4. Accept/reject.
    
    # To make it faster, we can precompute the "contribution" of each element? No, it's pairwise.
    
    # Let's optimize get_score.
    # Using a set for S allows O(1) lookups.
    
    def score_set_fast(S):
        S_set = set(S)
        covered = set()
        # Iterate over all pairs
        # Optimization: only iterate a < b
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                d = b - a
                # a middle: a-d in S
                if (a - d) in S_set:
                    covered.add(d)
                # a start: b+d in S
                if (b + d) in S_set:
                    covered.add(d)
                # b end: a+d/2 in S
                if d % 2 == 0:
                    d2 = d // 2
                    if (a + d2) in S_set:
                        covered.add(d2)
        return len(covered)

    # SA Loop
    moves = 0
    last_improvement = time.time()
    
    while time.time() - start_time < TIME_LIMIT:
        # Pick random x_out from S_list
        idx_out = random.randint(0, N-1)
        x_out = S_list[idx_out]
        
        # Pick random x_in not in S_list
        # To avoid O(N) search for x_in, we can maintain a list of available elements.
        # Or just pick random from 1..512 and check.
        # Probability of collision is 32/512 = 6%. Acceptable.
        x_in = random.randint(1, MAX_VAL)
        if x_in in S_set:
            continue
            
        # Perform swap
        S_list[idx_out] = x_in
        S_set.remove(x_out)
        S_set.add(x_in)
        
        new_score = score_set_fast(S_list)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = list(S_list)
                last_improvement = time.time()
        else:
            # Metropolis acceptance
            delta = new_score - current_score
            if delta < 0:
                accept_prob = pow(2.71828, delta / T)
                if random.random() > accept_prob:
                    # Reject
                    S_list[idx_out] = x_out
                    S_set.remove(x_in)
                    S_set.add(x_out)
            else:
                # Should not happen if delta >= 0 and we didn't accept
                pass
        
        T *= alpha
        if T < T_min:
            # Restart with new random set or higher T?
            # Let's restart with a new random set to explore different basins
            S_list, current_score = greedy_init()
            S_set = set(S_list)
            T = 10.0
            
        moves += 1

    return best_S

# Run a quick test to ensure it works
# print(construct())
