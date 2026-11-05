import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # 5s safety margin
    
    def get_sums(S):
        """Calculate the set of all pairwise sums a+b with a<=b for set S."""
        sums = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def is_maximal(S, S_sums):
        """
        Check if S is inclusion-maximal.
        S is a set of integers. S_sums is the set of pairwise sums of S.
        Returns True if no element in [1, N] \ S can be added.
        """
        S_set = S
        for y in range(1, N + 1):
            if y in S_set:
                continue
            # Check if y can be added
            # Condition to add y: 2y not in S_sums AND (y+a) not in S_sums for all a in S
            if 2 * y in S_sums:
                continue # Blocked by diagonal
            blocked = False
            for a in S_set:
                if y + a in S_sums:
                    blocked = True
                    break
            if not blocked:
                return False # y can be added, so S is not maximal
        return True

    def greedy_construct(order):
        """
        Construct a maximal Sidon set by iterating through order and adding
        if valid. This always produces a maximal set.
        """
        S = set()
        S_sums = set()
        for x in order:
            # Check if x can be added to current S
            # New sums: 2x and x+a for a in S
            # Conflict if 2x in S_sums or (x+a) in S_sums for any a in S
            if 2 * x in S_sums:
                continue
            valid = True
            for a in S:
                if x + a in S_sums:
                    valid = False
                    break
            if valid:
                S.add(x)
                S_sums.add(2 * x)
                for a in S:
                    if a != x: # x is already added, so a != x is implicit if we iterate old S? 
                        # Wait, S was updated. We should iterate over the state BEFORE adding x.
                        # Let's refactor to be safe.
                        pass
                # Refactored update:
                # We need to add x+a for a in OLD_S.
                # Since we added x to S, we can't easily get OLD_S without copying.
                # Let's do it properly.
        # Re-implementing greedy_construct correctly
        S = set()
        S_sums = set()
        for x in order:
            if 2 * x in S_sums:
                continue
            conflict = False
            for a in S:
                if x + a in S_sums:
                    conflict = True
                    break
            if not conflict:
                # Add x
                S.add(x)
                S_sums.add(2 * x)
                for a in S:
                    if a != x:
                        S_sums.add(x + a)
        return S, S_sums

    def prune_and_optimize(S, S_sums, deadline):
        """
        Refine a maximal Sidon set by removing elements if possible,
        and swapping if stuck.
        """
        S = set(S)
        S_sums = get_sums(S) # Ensure consistency
        
        improved = True
        no_improve_count = 0
        max_no_improve = 50 # Reset or swap after this many failed removal attempts in a row?
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Try to remove elements
            # We try removing each element. If successful, restart scan.
            # To speed up, we can pick elements randomly or in a heuristic order.
            # Heuristic: Try removing elements that "contribute" less to blocking?
            # Hard to calculate. Random is fine.
            candidates = list(S)
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                # Try removing x
                new_S = S - {x}
                if not new_S:
                    continue
                
                # Calculate new sums
                # Efficient way: Remove sums involving x from S_sums?
                # Sums involving x are {x+a | a in S} and {2x}.
                # But some of these sums might be shared with other pairs?
                # No, in a Sidon set, all sums are distinct.
                # So sums involving x are unique to x.
                # We can just subtract them.
                
                # However, we need to check maximality of new_S.
                # Maximality check requires S_sums of new_S.
                
                # Let's compute new_S_sums.
                # Since sums are unique, new_S_sums = S_sums - {s | s involves x}
                # But wait, if we remove x, do we lose any sums that don't involve x? No.
                # Do we lose sums that involve x? Yes.
                # Are there sums in S_sums that involve x and another element a, where a is also removed?
                # We are removing only one element x at a time.
                
                # So, sums_to_remove = {x + a for a in S} (includes 2x since x in S)
                # Note: S is the OLD set.
                sums_to_remove = set()
                for a in S:
                    sums_to_remove.add(x + a)
                
                new_S_sums = S_sums - sums_to_remove
                
                # Check if new_S is maximal
                if is_maximal(new_S, new_S_sums):
                    S = new_S
                    S_sums = new_S_sums
                    improved = True
                    no_improve_count = 0
                    break # Restart scan with new S
                
            if improved:
                continue
            
            # Phase 2: If no removal worked, try a swap
            # Pick random x in S, y not in S
            # Check if S' = S - {x} + {y} is Sidon and Maximal
            if len(S) > 1:
                x = random.choice(list(S))
                y = random.randint(1, N)
                if y in S:
                    continue # Retry swap generation
                
                # Check if y can be added to S - {x}
                # Let S_temp = S - {x}
                # We need to check if S_temp U {y} is Sidon.
                # This is equivalent to:
                # 1. y not in S_temp (true by selection)
                # 2. 2y not in S_sums(S_temp)
                # 3. y+a not in S_sums(S_temp) for all a in S_temp
                
                # First, compute S_sums(S_temp)
                # S_sums(S_temp) = S_sums - {s | s involves x}
                sums_inv_x = set()
                for a in S:
                    sums_inv_x.add(x + a)
                S_temp_sums = S_sums - sums_inv_x
                
                # Check Sidon validity of adding y to S_temp
                valid_sidon = True
                if 2 * y in S_temp_sums:
                    valid_sidon = False
                else:
                    for a in S_temp_sums: # No, iterate a in S_temp
                        pass
                    for a in (S - {x}):
                        if y + a in S_temp_sums:
                            valid_sidon = False
                            break
                
                if valid_sidon:
                    # Calculate new sums for S_new = S_temp U {y}
                    # New sums = S_temp_sums U {2y} U {y+a for a in S_temp}
                    new_S_sums_candidate = set(S_temp_sums)
                    new_S_sums_candidate.add(2 * y)
                    for a in (S - {x}):
                        new_S_sums_candidate.add(y + a)
                    
                    S_new = (S - {x}) | {y}
                    
                    # Check Maximality
                    if is_maximal(S_new, new_S_sums_candidate):
                        S = S_new
                        S_sums = new_S_sums_candidate
                        improved = True
                        no_improve_count = 0
                        break # Restart scan with new S
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    # Stuck hard. Maybe break and try another restart?
                    # Or just continue with random swaps?
                    # Let's just continue, random swaps might eventually work.
                    pass

        return S, S_sums

    # Main logic
    best_S = None
    best_size = N + 1
    
    # Phase 1: Generate initial solutions
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline: break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S, S_sums = greedy_construct(order)
        
        # Refine
        S, S_sums = prune_and_optimize(S, S_sums, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = S
            
        # If we found a very small set, we can stop early?
        # Theoretical lower bound for maximal Sidon set in [1,N] is approx sqrt(N)?
        # For N=256, sqrt(256)=16.
        # If we hit 16 or 17, it's likely optimal or near-optimal.
        if best_size <= 17:
             break

    if best_S is None:
        # Fallback
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, _ = greedy_construct(order)
        return list(S)
        
    return list(best_S)
