import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    best_size = 0
    best_set = set(prescribed_set)
    best_sums = set()
    best_origins = defaultdict(set)

    # Pre-compute sums for prescribed set
    def get_sums(s):
        sums = set()
        origins = defaultdict(set)
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                a, b = lst[i], lst[j]
                s_val = a + b
                sums.add(s_val)
                origins[s_val].add((a, b))
        return sums, origins

    initial_sums, initial_origins = get_sums(prescribed_set)
    best_sums = initial_sums
    best_origins = initial_origins

    # Helper to check if x can be added to a set with given sums
    def can_add(x, s_set, s_sums):
        if x in s_set:
            return False
        # Check x + y for all y in s_set
        for y in s_set:
            if x + y in s_sums:
                return False
        # Check 2x
        if 2 * x in s_sums:
            return False
        return True

    # Helper to update sums and origins when adding x
    def add_to_sums(x, s_set, s_sums, s_origins):
        new_sums = set(s_sums)
        new_origins = {k: set(v) for k, v in s_origins.items()} # Deep copy
        for y in s_set:
            s_val = x + y
            new_sums.add(s_val)
            new_origins[s_val].add((min(x,y), max(x,y)))
        s_val = 2 * x
        new_sums.add(s_val)
        new_origins[s_val].add((x, x))
        return new_sums, new_origins

    def greedy_fill(current_set, current_sums, current_origins, candidates):
        """Greedily add elements from candidates to current_set."""
        for x in candidates:
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                current_set.add(x)
                current_sums, current_origins = add_to_sums(x, current_set, current_sums, current_origins)
        return current_set, current_sums, current_origins

    def repair_search(current_set, current_sums, current_origins, time_limit):
        """
        Repair-based local search.
        Attempts to add random elements. If conflict, removes a conflicting element.
        """
        best_local_set = set(current_set)
        best_local_sums = current_sums
        best_local_origins = current_origins
        start = time.time()

        all_candidates = list(range(1, 285))

        while time.time() - start < time_limit:
            # Try to add a random candidate
            candidate = random.choice(all_candidates)
            if candidate in best_local_set:
                continue

            # Check for conflict
            conflict_found = False
            conflicting_pair = None
            for y in best_local_set:
                s_val = candidate + y
                if s_val in best_local_sums:
                    # Conflict found. Get the pair that caused it
                    # best_local_origins[s_val] contains pairs (a,b)
                    # We need to remove one of a or b
                    conflict_found = True
                    # Pick a random pair from the origins
                    pairs = list(best_local_origins[s_val])
                    if pairs:
                        a, b = random.choice(pairs)
                        conflicting_pair = (a, b)
                    break
            
            if not conflict_found:
                # Check 2*candidate
                if 2 * candidate in best_local_sums:
                    conflict_found = True
                    pairs = list(best_local_origins[2 * candidate])
                    if pairs:
                        a, b = random.choice(pairs)
                        conflicting_pair = (a, b)

            if not conflict_found:
                # No conflict, add it
                best_local_set.add(candidate)
                best_local_sums, best_local_origins = add_to_sums(candidate, best_local_set, best_local_sums, best_local_origins)
            else:
                # Conflict found. Try to resolve by removing one of the conflicting elements
                # Only remove if it's not prescribed
                if conflicting_pair:
                    a, b = conflicting_pair
                    removable_candidates = []
                    if a not in prescribed_set:
                        removable_candidates.append(a)
                    if b not in prescribed_set:
                        removable_candidates.append(b)
                    
                    if removable_candidates:
                        to_remove = random.choice(removable_candidates)
                        # Remove it
                        temp_set = set(best_local_set)
                        temp_set.remove(to_remove)
                        temp_sums, temp_origins = get_sums(temp_set)
                        
                        # Try to add candidate again
                        if can_add(candidate, temp_set, temp_sums):
                            temp_set.add(candidate)
                            temp_sums, temp_origins = add_to_sums(candidate, temp_set, temp_sums, temp_origins)
                            
                            # Greedily refill to try to maintain/increase size
                            # Use a subset of candidates to speed up? No, just greedy fill all.
                            refill_cands = [x for x in all_candidates if x not in temp_set]
                            random.shuffle(refill_cands)
                            temp_set, temp_sums, temp_origins = greedy_fill(temp_set, temp_sums, temp_origins, refill_cands)
                            
                            if len(temp_set) > len(best_local_set):
                                best_local_set = temp_set
                                best_local_sums = temp_sums
                                best_local_origins = temp_origins
                            elif len(temp_set) == len(best_local_set) and random.random() < 0.2:
                                # Accept same size to explore
                                best_local_set = temp_set
                                best_local_sums = temp_sums
                                best_local_origins = temp_origins

        return best_local_set, best_local_sums, best_local_origins

    # Main search loop
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1

        # 1. Random Greedy Construction
        current_set = set(prescribed_set)
        current_sums, current_origins = get_sums(current_set)

        candidates_list = list(range(12, 285))
        random.shuffle(candidates_list)

        current_set, current_sums, current_origins = greedy_fill(current_set, current_sums, current_origins, candidates_list)

        # 2. Repair-based Local Search
        remaining = deadline - time.time()
        if remaining > 3:
            repair_time = min(10.0, max(1.0, remaining * 0.5))
            current_set, current_sums, current_origins = repair_search(current_set, current_sums, current_origins, repair_time)

        # 3. Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            best_sums, best_origins = get_sums(best_set)

            if best_size >= 20:
                break

    # Final validation before returning
    final_sums, _ = get_sums(best_set)
    if len(final_sums) != len(best_set) * (len(best_set) + 1) // 2:
        return sorted(prescribed)

    return sorted(list(best_set))
