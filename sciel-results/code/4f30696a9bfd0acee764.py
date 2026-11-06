import time
import random

def construct():
    """
    Finds the largest Sidon set (distinct pairwise sums) in [1, 284]
    containing {1, 2, 5, 11}.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    prescribed = [1, 2, 5, 11]
    pset = set(prescribed)
    MAXV = 284
    
    def get_sums(S):
        sums = set()
        sl = sorted(S)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def is_valid_add(x, S, used_sums):
        if x in S:
            return False
        for y in S:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True

    def count_valid_adds(S, used_sums):
        count = 0
        for x in range(1, MAXV + 1):
            if x not in S and is_valid_add(x, S, used_sums):
                count += 1
        return count

    # Initial state
    S0 = set(prescribed)
    sums0 = get_sums(S0)
    
    best_S = list(S0)
    best_size = len(S0)
    
    # Beam Search Phase
    beam_width = 100
    max_depth = 25
    rng = random.Random(123)
    
    beam = [(S0, sums0)]
    
    for depth in range(max_depth):
        if time.time() > deadline - 40: # Leave 40s for local search
            break
            
        next_beam = []
        seen_states = set()
        
        for S, used_sums in beam:
            valid_candidates = []
            for x in range(1, MAXV + 1):
                if x in S:
                    continue
                if is_valid_add(x, S, used_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                continue
                
            # Sample candidates to expand
            # If many, sample to keep beam diverse and small
            if len(valid_candidates) > 10:
                candidates_to_try = rng.sample(valid_candidates, 10)
            else:
                candidates_to_try = valid_candidates
                
            for x in candidates_to_try:
                new_S = S | {x}
                new_sums = used_sums | {x + y for y in S} | {2 * x}
                
                key = tuple(sorted(new_S))
                if key in seen_states:
                    continue
                seen_states.add(key)
                
                next_beam.append((new_S, new_sums))
                
                if len(new_S) > best_size:
                    best_size = len(new_S)
                    best_S = list(new_S)
        
        if not next_beam:
            break
            
        if len(next_beam) > beam_width:
            # Score by number of remaining options
            def score_state(state):
                S, used_sums = state
                return count_valid_adds(S, used_sums)
            
            next_beam.sort(key=score_state, reverse=True)
            beam = next_beam[:beam_width]
        else:
            beam = next_beam

    # Local Search Phase with restarts
    ls_deadline = deadline - 2.0
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > ls_deadline:
            break
        
        # Start from best_S, but perturb for restarts > 0
        if restart == 0:
            cur_S = set(best_S)
            cur_sums = get_sums(cur_S)
        else:
            # Perturb: remove 1-2 random non-prescribed elements
            cur_S = set(best_S)
            removable = [x for x in cur_S if x not in pset]
            if not removable:
                break
            num_remove = min(2, len(removable))
            to_remove = random.Random(1000 + restart).sample(removable, num_remove)
            for x in to_remove:
                cur_S.discard(x)
            cur_sums = get_sums(cur_S)
        
        cur_size = len(cur_S)
        ls_rng = random.Random(456 + restart)
        
        # Local search loop
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < ls_deadline:
            # Try to add
            valid_adds = [x for x in range(1, MAXV + 1) if x not in cur_S and is_valid_add(x, cur_S, cur_sums)]
            
            if valid_adds:
                # Pick a valid addition - prefer ones that leave more room
                if len(valid_adds) > 1:
                    # Score by how many additional elements can be added after this one
                    best_x = valid_adds[0]
                    best_score = -1
                    for x in valid_adds[:min(20, len(valid_adds))]:
                        temp_S = cur_S | {x}
                        temp_sums = cur_sums | {x + y for y in cur_S} | {2 * x}
                        score = count_valid_adds(temp_S, temp_sums)
                        if score > best_score:
                            best_score = score
                            best_x = x
                    x = best_x
                else:
                    x = valid_adds[0]
                
                cur_S.add(x)
                cur_sums = get_sums(cur_S)
                cur_size += 1
                no_improve_count = 0
                if cur_size > best_size:
                    best_size = cur_size
                    best_S = list(cur_S)
                continue
            
            # No valid additions, try swaps
            removable = [x for x in cur_S if x not in pset]
            if not removable:
                break
            
            # Try multiple swap attempts
            found_swap = False
            shuffled_removable = removable[:]
            ls_rng.shuffle(shuffled_removable)
            
            for y in shuffled_removable:
                temp_S = cur_S - {y}
                temp_sums = get_sums(temp_S)
                
                valid_adds_temp = [x for x in range(1, MAXV + 1) if x not in temp_S and is_valid_add(x, temp_S, temp_sums)]
                
                if valid_adds_temp:
                    # Perform swap - prefer additions that leave more room
                    if len(valid_adds_temp) > 1:
                        best_x = valid_adds_temp[0]
                        best_score = -1
                        for x in valid_adds_temp[:min(10, len(valid_adds_temp))]:
                            new_S = temp_S | {x}
                            new_sums = temp_sums | {x + z for z in temp_S} | {2 * x}
                            score = count_valid_adds(new_S, new_sums)
                            if score > best_score:
                                best_score = score
                                best_x = x
                        x = best_x
                    else:
                        x = valid_adds_temp[0]
                    
                    cur_S = temp_S | {x}
                    cur_sums = get_sums(cur_S)
                    found_swap = True
                    break
            
            if not found_swap:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0
        
        # After local search, if we found something better, update best
        if len(cur_S) > best_size:
            best_size = len(cur_S)
            best_S = list(cur_S)
    
    return best_S
