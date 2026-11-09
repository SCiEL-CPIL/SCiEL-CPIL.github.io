"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(S_set):
    """Check if set S has no three distinct elements a,b,c with all pair sums in S."""
    S_list = sorted(S_set)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j + 1, n):
                c = S_list[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def _count_violations(S_set, S_list):
    """Count number of forbidden triples."""
    count = 0
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j + 1, n):
                c = S_list[k]
                if a + c in S_set and b + c in S_set:
                    count += 1
    return count

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    n_max = 128
    
    # Start with known valid set: {63, ..., 128}, size 66
    best_set = set(range(63, n_max + 1))
    best_size = len(best_set)
    
    # Try greedy addition from below
    # For each candidate to add, check if the set remains valid
    candidates = list(range(1, 63))
    random.shuffle(candidates)
    
    current_set = set(best_set)
    current_list = sorted(current_set)
    
    # Try adding elements one by one
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        trial_set = current_set | {cand}
        trial_list = sorted(trial_set)
        if _is_valid(trial_set):
            current_set = trial_set
            current_list = trial_list
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Local search: try swaps (remove one, add another)
    if time.time() < deadline - 5:
        all_nums = set(range(1, n_max + 1))
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            # Try random swaps
            if current_set:
                to_remove = random.choice(sorted(current_set))
            else:
                break
            possible_adds = [x for x in all_nums - current_set if x != to_remove]
            if not possible_adds:
                break
            to_add = random.choice(possible_adds)
            trial_set = (current_set - {to_remove}) | {to_add}
            if len(trial_set) >= len(current_set):  # only accept if same or larger
                if _is_valid(trial_set):
                    current_set = trial_set
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
    
    # Final greedy pass on best set
    if time.time() < deadline - 2:
        current_set = set(best_set)
        for cand in range(1, 63):
            if time.time() > deadline - 2:
                break
            if cand in current_set:
                continue
            trial_set = current_set | {cand}
            if _is_valid(trial_set):
                current_set = trial_set
                best_set = set(current_set)
                best_size = len(current_set)
    
    result = sorted(best_set)
    # Ensure Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
