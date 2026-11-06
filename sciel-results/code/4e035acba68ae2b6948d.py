import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit
    
    best_set = None
    best_size = N + 1
    
    # We will run a few iterations with different random tie-breaking seeds
    # to explore the space slightly, but primarily rely on the heuristic.
    
    for attempt in range(10):
        if time.time() > deadline - 2.0:
            break
            
        S = []
        sums = set()
        # A: set of available candidates
        A = set(range(1, N + 1))
        
        while A:
            if time.time() > deadline - 1.0:
                break
            
            # If only one candidate left, must take it
            if len(A) == 1:
                x = next(iter(A))
                S.append(x)
                sums.add(2 * x)
                for y in S[:-1]:
                    sums.add(x + y)
                break
            
            # Evaluate each candidate x in A
            best_x = None
            best_block_count = -1
            candidates = list(A)
            
            # Optional: Randomize order to break ties differently in different attempts
            if attempt > 0:
                random.shuffle(candidates)
            
            for x in candidates:
                # Simulate adding x
                # New sums: x+y for y in S, and 2x
                new_sums = set()
                for y in S:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                # Check how many other candidates z in A \ {x} are blocked
                block_count = 0
                for z in A:
                    if z == x:
                        continue
                    
                    # Check if z is still valid if x is added
                    # z is invalid if:
                    # 1. 2z is in sums U new_sums
                    # 2. There exists y in S U {x} such that z+y is in sums U new_sums
                    
                    is_valid = True
                    
                    # Check diagonal
                    if 2 * z in sums or 2 * z in new_sums:
                        is_valid = False
                    else:
                        # Check cross sums
                        # For each y in S, check if z+y in sums U new_sums
                        for y in S:
                            s_val = z + y
                            if s_val in sums or s_val in new_sums:
                                is_valid = False
                                break
                        if is_valid:
                            # Check with y = x
                            s_val = z + x
                            if s_val in sums or s_val in new_sums:
                                is_valid = False
                            
                    if not is_valid:
                        block_count += 1
                
                # Heuristic: Maximize block count.
                # Tie-breaking: Prefer larger x? Or random?
                # Larger x tends to be "heavier" in sums.
                # Let's prefer larger x in case of tie to push towards high end,
                # which often leaves smaller numbers blocked by many large sums.
                if block_count > best_block_count or (block_count == best_block_count and (best_x is None or x > best_x)):
                    best_block_count = block_count
                    best_x = x
            
            if best_x is None:
                break
                
            # Commit to best_x
            S.append(best_x)
            sums.add(2 * best_x)
            for y in S[:-1]:
                sums.add(best_x + y)
            
            # Update A: remove best_x and any z that is now invalid
            A.discard(best_x)
            to_remove = set()
            for z in A:
                # Check if z is valid w.r.t new S
                # 2z not in sums
                # z+y not in sums for all y in S
                if 2 * z in sums:
                    to_remove.add(z)
                    continue
                for y in S:
                    if z + y in sums:
                        to_remove.add(z)
                        break
            A.difference_update(to_remove)
            
        # End while A
        
        if S:
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
        
        # If we found a very small set, we can stop early
        if best_size <= 15: # Theoretical lower bound is ~sqrt(N) ~ 16
            break

    if best_set:
        return best_set
    else:
        # Fallback to simple greedy if something went wrong or timeout
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
# EVOLVE-BLOCK-END
