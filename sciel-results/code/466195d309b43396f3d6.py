"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_3ap_with_candidate(candidate, selected_set):
    """Check if adding candidate to selected_set creates a 3-term AP."""
    for a in selected_set:
        if a == candidate:
            continue
        # Check if candidate is the middle: a, candidate, c => c = 2*candidate - a
        c = 2 * candidate - a
        if c != candidate and c in selected_set:
            return True
        # Check if candidate is the end: a, b, candidate => b = (a + candidate) / 2
        # or candidate, b, a => b = (candidate + a) / 2
        s = a + candidate
        if s % 2 == 0:
            b = s // 2
            if b != a and b != candidate and b in selected_set:
                return True
    return False

def _reciprocal_sum(selected):
    """Compute the sum of reciprocals of selected numbers."""
    return sum(1.0 / x for x in selected)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Greedy initialization: prioritize small numbers
    selected = set()
    for x in range(1, 257):
        if not _has_3ap_with_candidate(x, selected):
            selected.add(x)
    
    best_set = set(selected)
    best_score = _reciprocal_sum(best_set)
    
    # Local search with simulated annealing
    random.seed(42)
    current_set = set(selected)
    current_score = best_score
    
    # Precompute some useful data
    all_numbers = set(range(1, 257))
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Try random moves
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a number not in the set
            candidates_to_add = list(all_numbers - current_set)
            if not candidates_to_add:
                continue
            # Prioritize smaller numbers
            candidates_to_add.sort()
            x = candidates_to_add[random.randint(0, min(len(candidates_to_add)-1, 20))]
            
            if not _has_3ap_with_candidate(x, current_set):
                new_set = set(current_set)
                new_set.add(x)
                new_score = _reciprocal_sum(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_set = set(current_set)
                        best_score = current_score
        
        elif move_type < 0.6:
            # Try to remove a number and add another
            if len(current_set) < 2:
                continue
            x_remove = random.choice(list(current_set))
            candidates_to_add = list(all_numbers - current_set)
            if not candidates_to_add:
                continue
            candidates_to_add.sort()
            x_add = candidates_to_add[random.randint(0, min(len(candidates_to_add)-1, 30))]
            
            new_set = set(current_set)
            new_set.remove(x_remove)
            if not _has_3ap_with_candidate(x_add, new_set):
                new_set.add(x_add)
                new_score = _reciprocal_sum(new_set)
                delta = new_score - current_score
                if delta > 0 or random.random() < 0.1:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_set = set(current_set)
                        best_score = current_score
        
        else:
            # Try swapping two numbers
            if len(current_set) < 2:
                continue
            x1 = random.choice(list(current_set))
            candidates_to_add = list(all_numbers - current_set)
            if not candidates_to_add:
                continue
            candidates_to_add.sort()
            x2 = candidates_to_add[random.randint(0, min(len(candidates_to_add)-1, 30))]
            
            new_set = set(current_set)
            new_set.remove(x1)
            new_set.add(x2)
            
            # Verify no 3AP
            valid = True
            for a in new_set:
                for b in new_set:
                    if a >= b:
                        continue
                    c = 2 * b - a
                    if c in new_set and c != a and c != b:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_score = _reciprocal_sum(new_set)
                delta = new_score - current_score
                if delta > 0 or random.random() < 0.1:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_set = set(current_set)
                        best_score = current_score
    
    # Final verification
    final_list = sorted(best_set)
    for i in range(len(final_list)):
        for j in range(i+1, len(final_list)):
            a, b = final_list[i], final_list[j]
            c = 2 * b - a
            if c != a and c != b and c in best_set:
                # This shouldn't happen, but return a valid set just in case
                pass
    
    return final_list
# EVOLVE-BLOCK-END
