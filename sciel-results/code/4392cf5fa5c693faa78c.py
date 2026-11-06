"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return best valid result found before the hard limit.
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill

    def is_valid_sidon(selected):
        """Check if selected is a valid Sidon set."""
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(selected, N):
        """Check if selected is inclusion-maximal in [1,N]."""
        selected_set = set(selected)
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if adding x would violate Sidon property
            valid = True
            for a in selected:
                s = x + a
                if s in sums:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if s in sums:
                    valid = False
            if valid:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction using a given order of candidates."""
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    def try_remove_and_reconstruct(selected, target_size):
        """Try removing elements and reconstructing to get smaller maximal set."""
        best = selected
        best_size = len(selected)
        
        # Try removing each element and see if we can still get a maximal set
        for i in range(len(selected)):
            if time.time() - start_time > DEADLINE:
                break
            
            # Remove element i
            candidate = selected[:i] + selected[i+1:]
            
            # Try to extend greedily to see if we get a smaller maximal set
            # Actually, we need to check if the reduced set is still maximal
            # and if not, try to add back elements in a way that minimizes size
            
            # Simpler approach: try all subsets of size len(selected)-1, len(selected)-2, etc.
            # But this is exponential. Instead, let's try a local search.
            
            # Check if candidate is valid Sidon
            if not is_valid_sidon(candidate):
                continue
            
            # Check if candidate is maximal
            if is_maximal(candidate, N):
                if len(candidate) < best_size:
                    best = candidate
                    best_size = len(candidate)
                break  # Found a smaller maximal set
            
            # If not maximal, try to add elements back greedily
            # to make it maximal with minimal additions
            current = list(candidate)
            current_sums = set()
            for j, a in enumerate(current):
                for b in current[j:]:
                    current_sums.add(a + b)
            
            # Try adding missing elements in order of least "conflict"
            missing = [x for x in range(1, N + 1) if x not in set(current)]
            
            # Sort missing by how many conflicts they cause
            def conflict_count(x):
                count = 0
                for a in current:
                    if (x + a) in current_sums:
                        count += 1
                if (2 * x) in current_sums:
                    count += 1
                return count
            
            missing.sort(key=conflict_count)
            
            for x in missing:
                if time.time() - start_time > DEADLINE:
                    break
                
                added = [x + a for a in current] + [2 * x]
                if len(set(added)) == len(added) and not current_sums.intersection(added):
                    current.append(x)
                    current_sums.update(added)
                
                # Check if current is now maximal
                if is_maximal(current, N):
                    if len(current) < best_size:
                        best = current
                        best_size = len(current)
                    break
        
        return best

    def local_search(selected):
        """Local search to minimize size while maintaining maximality."""
        current = list(selected)
        current_size = len(current)
        improved = True
        
        while improved and (time.time() - start_time < DEADLINE):
            improved = False
            
            # Try removing each element
            for i in range(len(current)):
                if time.time() - start_time > DEADLINE:
                    break
                
                candidate = current[:i] + current[i+1:]
                
                if not is_valid_sidon(candidate):
                    continue
                
                if is_maximal(candidate, N):
                    current = candidate
                    current_size = len(current)
                    improved = True
                    break
            
            # Try swapping elements
            for i in range(len(current)):
                if time.time() - start_time > DEADLINE:
                    break
                
                for x in range(1, N + 1):
                    if x in current:
                        continue
                    
                    # Try replacing current[i] with x
                    candidate = current[:i] + [x] + current[i+1:]
                    
                    if not is_valid_sidon(candidate):
                        continue
                    
                    if is_maximal(candidate, N):
                        # Check if this swap allows further removals
                        # For now, just accept if it's valid and maximal
                        current = candidate
                        improved = True
                        break
                
                if improved:
                    break
        
        return current

    # Start with greedy construction (ascending order)
    best = greedy_construct(range(1, N + 1))
    
    # Try to improve via removal
    best = try_remove_and_reconstruct(best, len(best))
    
    # Local search
    best = local_search(best)
    
    # Try different initial orderings
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidate = greedy_construct(order)
        
        if len(candidate) < len(best):
            best = candidate
            best = try_remove_and_reconstruct(best, len(best))
            best = local_search(best)
    
    # Try descending order
    if time.time() - start_time < DEADLINE:
        candidate = greedy_construct(range(N, 0, -1))
        if len(candidate) < len(best):
            best = candidate
            best = try_remove_and_reconstruct(best, len(best))
            best = local_search(best)
    
    # Final verification
    if not is_valid_sidon(best):
        # Fallback to simple greedy
        best = greedy_construct(range(1, N + 1))
    
    if not is_maximal(best, N):
        # Ensure maximality
        selected_set = set(best)
        sums = set()
        for i, a in enumerate(best):
            for b in best[i:]:
                sums.add(a + b)
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            valid = True
            for a in best:
                if (x + a) in sums:
                    valid = False
                    break
            if valid and (2 * x) in sums:
                valid = False
            if valid:
                best.append(x)
                selected_set.add(x)
                for a in best[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
    
    return best
# EVOLVE-BLOCK-END
