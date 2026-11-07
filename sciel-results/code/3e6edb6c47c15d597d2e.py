import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin before 60s hard limit
    sys.setrecursionlimit(10000)
    
    # Validity check: all subset sums must be distinct and no two divide each other
    def is_valid_subset_sums(sums_set):
        """Check if the set of sums satisfies the no-divisibility condition."""
        # Check for duplicates (shouldn't happen with set, but just in case)
        # Check divisibility: no two distinct elements where one divides the other
        sums_list = sorted(sums_set)
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if sums_list[j] % sums_list[i] == 0:
                    return False
        return True

    # Try to add an element to a set, checking validity
    def try_add_element(current_elements, current_sums, candidate):
        """
        Try to add candidate to current_elements.
        Returns (new_elements, new_sums) if valid, else None.
        """
        if candidate in current_elements:
            return None
        
        # Generate new subset sums
        new_sums = set()
        for s in current_sums:
            ns = s + candidate
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        
        if candidate in current_sums or candidate in new_sums:
            return None
        new_sums.add(candidate)
        
        all_sums = current_sums.union(new_sums)
        
        # Check divisibility: for each new sum, check against all existing sums
        # We need to check: for any a in new_sums, b in all_sums, a != b, not (a % b == 0 or b % a == 0)
        # But we already know current_sums is internally valid.
        # So we only need to check interactions between new_sums and (current_sums U new_sums)
        
        for s in new_sums:
            for t in all_sums:
                if s == t:
                    continue
                if t % s == 0 or s % t == 0:
                    return None
        
        new_elements = current_elements.union({candidate})
        return (new_elements, all_sums)

    # Branch and bound to find maximum valid set
    def branch_and_bound(remaining, current_elements, current_sums, best_size, best_set, deadline):
        """
        Branch and bound search for maximum valid set.
        remaining: sorted list of remaining candidate elements
        current_elements: set of elements in current solution
        current_sums: set of subset sums
        best_size: current best size found
        best_set: current best set
        deadline: time limit
        Returns (best_size, best_set)
        """
        if time.time() > deadline:
            return (best_size, best_set)
        
        # Pruning: if current + remaining can't beat best, prune
        if len(current_elements) + len(remaining) <= best_size:
            return (best_size, best_set)
        
        if not remaining:
            if len(current_elements) > best_size:
                best_size = len(current_elements)
                best_set = set(current_elements)
            return (best_size, best_set)
        
        # Try to add the first remaining element
        candidate = remaining[0]
        rest = remaining[1:]
        
        # Branch 1: Include candidate
        result = try_add_element(current_elements, current_sums, candidate)
        if result is not None:
            new_elements, new_sums = result
            best_size, best_set = branch_and_bound(
                rest, new_elements, new_sums, best_size, best_set, deadline
            )
        
        # Branch 2: Exclude candidate
        best_size, best_set = branch_and_bound(
            rest, current_elements, current_sums, best_size, best_set, deadline
        )
        
        return (best_size, best_set)
    
    # Greedy construction with randomization
    def greedy_construct(candidates, current_elements=None, current_sums=None, deadline=None):
        """
        Greedily construct a valid set by trying elements in the given order.
        Returns (set of elements, set of subset sums)
        """
        if current_elements is None:
            current_elements = set()
        if current_sums is None:
            current_sums = set()
        
        for c in candidates:
            if deadline is not None and time.time() > deadline:
                break
            result = try_add_element(current_elements, current_sums, c)
            if result is not None:
                current_elements, current_sums = result
        
        return (current_elements, current_sums)
    
    # Local search: try to improve a set by swapping elements
    def local_search(initial_set, candidates, deadline):
        """
        Try to improve the initial set by removing elements and trying to add others.
        Returns the best set found.
        """
        best_set = set(initial_set)
        best_sums = None
        
        # Compute sums for best_set
        if best_set:
            all_sums = set()
            elements = sorted(best_set)
            for mask in range(1, 1 << len(elements)):
                s = 0
                for i in range(len(elements)):
                    if mask & (1 << i):
                        s += elements[i]
                if s in all_sums:
                    return best_set  # Invalid, shouldn't happen
                all_sums.add(s)
            best_sums = all_sums
        else:
            best_sums = set()
        
        best_size = len(best_set)
        
        # Try to add elements from candidates not in best_set
        pool = [c for c in candidates if c not in best_set]
        random.shuffle(pool)
        
        # Try adding one element at a time
        for c in pool:
            if time.time() > deadline:
                break
            result = try_add_element(best_set, best_sums, c)
            if result is not None:
                new_set, new_sums = result
                if len(new_set) > best_size:
                    best_set = new_set
                    best_sums = new_sums
                    best_size = len(best_set)
                    # Update pool
                    pool = [x for x in pool if x not in best_set]
        
        # Try removing one element and adding another
        for remove_candidate in list(best_set):
            if time.time() > deadline:
                break
            
            # Compute sums without remove_candidate
            reduced_set = best_set - {remove_candidate}
            if not reduced_set:
                reduced_sums = set()
            else:
                elements = sorted(reduced_set)
                reduced_sums = set()
                for mask in range(1, 1 << len(elements)):
                    s = 0
                    for i in range(len(elements)):
                        if mask & (1 << i):
                            s += elements[i]
                    reduced_sums.add(s)
            
            # Try to add elements from pool
            pool = [c for c in candidates if c not in reduced_set]
            random.shuffle(pool)
            
            for add_candidate in pool:
                if time.time() > deadline:
                    break
                result = try_add_element(reduced_set, reduced_sums, add_candidate)
                if result is not None:
                    new_set, new_sums = result
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_sums = new_sums
                        best_size = len(best_set)
        
        return best_set
    
    # Candidate pool
    # Start with 3-256, as 1 and 2 are problematic
    candidates = list(range(3, 257))
    
    # Phase 1: Multiple greedy passes with different orderings
    best_set = set()
    best_sums = set()
    best_size = 0
    
    # Sorted ascending
    result = greedy_construct(sorted(candidates), deadline=deadline)
    if len(result[0]) > best_size:
        best_set, best_sums = result
        best_size = len(best_set)
    
    # Sorted descending
    result = greedy_construct(sorted(candidates, reverse=True), deadline=deadline)
    if len(result[0]) > best_size:
        best_set, best_sums = result
        best_size = len(best_set)
    
    # Random shuffles
    for i in range(20):
        if time.time() > deadline - 5.0:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_construct(shuffled, deadline=deadline)
        if len(result[0]) > best_size:
            best_set, best_sums = result
            best_size = len(best_set)
    
    # Phase 2: Local search
    if time.time() < deadline - 5.0:
        improved = local_search(best_set, candidates, deadline - 5.0)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
            # Recompute sums
            elements = sorted(best_set)
            best_sums = set()
            for mask in range(1, 1 << len(elements)):
                s = 0
                for i in range(len(elements)):
                    if mask & (1 << i):
                        s += elements[i]
                best_sums.add(s)
    
    # Phase 3: Branch and bound on a reduced candidate set
    # Use the top candidates by some heuristic
    if time.time() < deadline - 2.0:
        # Try to find a better set using B&B
        # Start with best_set and try to add more elements
        # Or use B&B from scratch on a reduced candidate set
        
        # Reduced candidate set: elements that could potentially be added
        # Sort by how "compatible" they are with best_set
        remaining_candidates = [c for c in candidates if c not in best_set]
        
        # Try to add elements one by one using B&B
        # This is complex, so let's just try a few random subsets
        
        for i in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Random subset of candidates
            if remaining_candidates:
                sample_size = min(len(remaining_candidates), 15)
                sample = random.sample(remaining_candidates, sample_size)
                sample.sort()
                
                # B&B on this sample
                best_size_bb, best_set_bb = branch_and_bound(
                    sample, best_set, best_sums, best_size, set(best_set), deadline - 2.0
                )
                
                if best_size_bb > best_size:
                    best_set = best_set_bb
                    best_size = best_size_bb
                    # Recompute sums
                    elements = sorted(best_set)
                    best_sums = set()
                    for mask in range(1, 1 << len(elements)):
                        s = 0
                        for i in range(len(elements)):
                            if mask & (1 << i):
                                s += elements[i]
                        best_sums.add(s)
    
    # Final verification
    if best_set:
        elements = sorted(best_set)
        all_sums = set()
        valid = True
        for mask in range(1, 1 << len(elements)):
            s = 0
            for i in range(len(elements)):
                if mask & (1 << i):
                    s += elements[i]
            if s in all_sums:
                valid = False
                break
            all_sums.add(s)
        
        if valid:
            # Check divisibility
            sums_list = sorted(all_sums)
            for i in range(len(sums_list)):
                for j in range(i + 1, len(sums_list)):
                    if sums_list[j] % sums_list[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return [int(x) for x in sorted(best_set)]
    
    # Fallback: return a small valid set
    # Try {3}
    return [3]
