"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find the largest Sidon set containing {1, 2, 5, 11} in [1, 284].
    
    Approach:
    1. Greedy construction with conflict minimization.
    2. Randomized restarts to escape local optima.
    """
    
    mandatory = [1, 2, 5, 11]
    start_time = time.time()
    time_limit = 110  # seconds

    def get_initial_sums(mandatory):
        sums = set()
        for i in range(len(mandatory)):
            for j in range(i, len(mandatory)):
                sums.add(mandatory[i] + mandatory[j])
        return sums

    def is_compatible(x, current_sums, current_set):
        if x in current_set:
            return False
        if 2 * x in current_sums:
            return False
        for y in current_set:
            if x + y in current_sums:
                return False
        return True

    def get_new_sums(x, current_set):
        new_sums = {2 * x}
        for y in current_set:
            new_sums.add(x + y)
        return new_sums

    def count_conflicts(x, current_set, current_sums, candidates):
        """
        Estimate how many candidates x blocks.
        """
        new_sums = get_new_sums(x, current_set)
        blocked = 0
        for y in candidates:
            if y == x or y in current_set:
                continue
            
            # Check if y is compatible with current_set U {x}
            if 2 * y in new_sums or 2 * y in current_sums:
                blocked += 1
                continue
                
            conflict = False
            if y + x in current_sums:
                conflict = True
            else:
                for s in current_set:
                    if y + s in current_sums or y + s in new_sums:
                        conflict = True
                        break
            
            if conflict:
                blocked += 1
        return blocked

    best_set = set(mandatory)
    best_size = len(mandatory)
    
    random.seed(42)
    
    # Run multiple randomized greedy trials
    for trial in range(200):
        if time.time() - start_time > time_limit:
            break
        
        current_set = set(mandatory)
        current_sums = get_initial_sums(mandatory)
        candidates = list(range(12, 285))
        
        # Shuffle candidates to introduce randomness
        random.shuffle(candidates)
        
        while candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Filter compatible candidates
            compatible = []
            for x in candidates:
                if is_compatible(x, current_sums, current_set):
                    compatible.append(x)
                    
            if not compatible:
                break
            
            # Rank compatible candidates by conflict count
            # To save time, if many, we can sample, but 280 is small
            scored = []
            for x in compatible:
                conflicts = count_conflicts(x, current_set, current_sums, candidates)
                scored.append((conflicts, x))
            
            scored.sort(key=lambda item: item[0])
            
            # Pick from the top few to allow some randomness
            k = min(5, len(scored))
            best_candidate = scored[random.randint(0, k-1)][1]
            
            current_set.add(best_candidate)
            new_sums = get_new_sums(best_candidate, current_set - {best_candidate})
            current_sums |= new_sums
            
            # Remove best_candidate and prune incompatible ones
            new_candidates = []
            for x in candidates:
                if x != best_candidate and is_compatible(x, current_sums, current_set):
                    new_candidates.append(x)
            candidates = new_candidates
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        if best_size == 20: # Theoretical max
            break

    return sorted(best_set)

# EVOLVE-BLOCK-END
