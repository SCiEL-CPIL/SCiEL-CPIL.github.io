"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large Sidon set (B2 set) in [1, 284] containing {1, 2, 5, 11}.
    Uses simulated annealing with efficient conflict counting.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    prescribed = [1, 2, 5, 11]
    N = 284
    target_size = 20
    
    # Precompute all valid positions for each size
    # A Sidon set is a B2 sequence: all pairwise sums a_i + a_j (i <= j) are distinct.
    
    # We'll use simulated annealing to find a set of size 20.
    # State: a set of integers in [1, 284]
    # Objective: maximize the number of distinct pairwise sums (or minimize conflicts)
    
    # Efficient conflict counting:
    # For a set S, the number of "bad" pairs (sums that appear more than once)
    # can be tracked with a difference array.
    
    # Let's use a different approach: start with the prescribed elements and try to add more.
    # But the greedy approach only gets 16. We need a smarter search.
    
    # Approach: Simulated Annealing
    # State: a set of k integers in [1, 284]
    # Energy: number of duplicate sums (we want 0)
    # Moves: add an element, remove an element, or replace an element
    
    # For efficiency, we'll track the count of each sum in a dictionary.
    
    def compute_energy(s, sum_counts):
        """Compute the energy (number of duplicate sums) given sum_counts."""
        # sum_counts maps sum -> count
        # Energy = sum over all sums of max(0, count - 1)
        energy = 0
        for c in sum_counts.values():
            if c > 1:
                energy += c - 1
        return energy
    
    def update_sum_counts(s, x, sign, sum_counts):
        """Update sum_counts when adding (sign=1) or removing (sign=-1) element x from set s."""
        for y in s:
            sm = x + y
            sum_counts[sm] = sum_counts.get(sm, 0) + sign
            if sum_counts[sm] == 0:
                del sum_counts[sm]
        # Also handle 2*x
        sm = 2 * x
        sum_counts[sm] = sum_counts.get(sm, 0) + sign
        if sum_counts[sm] == 0:
            del sum_counts[sm]
    
    # Start with a random valid configuration of size 20 that includes the prescribed elements.
    # We need to find an initial set of 20 elements including {1, 2, 5, 11} that has few conflicts.
    
    # Let's try multiple random starts and pick the best one.
    
    best_set = None
    best_energy = float('inf')
    
    # Try to build a good initial set using a greedy approach with randomization
    def build_initial_set():
        s = list(prescribed)
        s.sort()
        sum_counts = {}
        # Initialize sum counts
        for i, x in enumerate(s):
            for y in s[i:]:
                sm = x + y
                sum_counts[sm] = sum_counts.get(sm, 0) + 1
        
        # Try to add more elements
        for _ in range(100):
            candidates = [x for x in range(1, N+1) if x not in s]
            random.shuffle(candidates)
            added = False
            for x in candidates:
                # Check if adding x creates no new conflicts
                new_conflicts = 0
                for y in s:
                    sm = x + y
                    if sum_counts.get(sm, 0) >= 1:
                        new_conflicts += 1
                sm2 = 2 * x
                if sum_counts.get(sm2, 0) >= 1:
                    new_conflicts += 1
                
                if new_conflicts == 0:
                    # Add x
                    s.append(x)
                    s.sort()
                    update_sum_counts(s, x, 1, sum_counts)
                    added = True
                    if len(s) == target_size:
                        break
            if not added:
                break
        
        return s, sum_counts
    
    # Try multiple initial constructions
    for attempt in range(50):
        if time.time() > deadline:
            break
        s, sum_counts = build_initial_set()
        energy = compute_energy(s, sum_counts)
        if energy < best_energy:
            best_energy = energy
            best_set = s[:]
            if energy == 0:
                break
    
    if best_set is None or len(best_set) < 20:
        # If we couldn't build a set of size 20, use what we have
        if best_set:
            return best_set
        else:
            return prescribed
    
    # Now do simulated annealing on the set of size 20
    s = best_set[:]
    s.sort()
    sum_counts = {}
    for i, x in enumerate(s):
        for y in s[i:]:
            sm = x + y
            sum_counts[sm] = sum_counts.get(sm, 0) + 1
    
    energy = compute_energy(s, sum_counts)
    best_s = s[:]
    best_e = energy
    
    if energy == 0:
        return s
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 1e-6
    
    while T > min_T and time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Replace an element
            idx = random.randint(0, len(s)-1)
            old_x = s[idx]
            candidates = [x for x in range(1, N+1) if x not in s]
            if not candidates:
                continue
            new_x = random.choice(candidates)
            
            # Remove old_x
            s.remove(old_x)
            update_sum_counts(s, old_x, -1, sum_counts)
            # Add new_x
            s.append(new_x)
            s.sort()
            update_sum_counts(s, new_x, 1, sum_counts)
            
            new_energy = compute_energy(s, sum_counts)
            delta_e = new_energy - energy
            
            if delta_e <= 0 or random.random() < pow(2.718281828, -delta_e / T):
                energy = new_energy
                if energy < best_e:
                    best_e = energy
                    best_s = s[:]
                    if energy == 0:
                        break
            else:
                # Revert
                s.remove(new_x)
                update_sum_counts(s, new_x, -1, sum_counts)
                s.append(old_x)
                s.sort()
                update_sum_counts(s, old_x, 1, sum_counts)
                
        elif move_type < 0.7:
            # Swap two elements
            if len(s) < 2:
                continue
            idx1, idx2 = random.sample(range(len(s)), 2)
            x1, x2 = s[idx1], s[idx2]
            candidates = [x for x in range(1, N+1) if x not in s]
            if not candidates:
                continue
            new_x1 = random.choice(candidates)
            # Choose new_x2 different from new_x1
            candidates2 = [x for x in candidates if x != new_x1]
            if not candidates2:
                continue
            new_x2 = random.choice(candidates2)
            
            # Remove x1, x2
            s.remove(x1)
            s.remove(x2)
            update_sum_counts(s, x1, -1, sum_counts)
            update_sum_counts(s, x2, -1, sum_counts)
            # Add new_x1, new_x2
            s.append(new_x1)
            s.append(new_x2)
            s.sort()
            update_sum_counts(s, new_x1, 1, sum_counts)
            update_sum_counts(s, new_x2, 1, sum_counts)
            
            new_energy = compute_energy(s, sum_counts)
            delta_e = new_energy - energy
            
            if delta_e <= 0 or random.random() < pow(2.718281828, -delta_e / T):
                energy = new_energy
                if energy < best_e:
                    best_e = energy
                    best_s = s[:]
                    if energy == 0:
                        break
            else:
                # Revert
                s.remove(new_x1)
                s.remove(new_x2)
                update_sum_counts(s, new_x1, -1, sum_counts)
                update_sum_counts(s, new_x2, -1, sum_counts)
                s.append(x1)
                s.append(x2)
                s.sort()
                update_sum_counts(s, x1, 1, sum_counts)
                update_sum_counts(s, x2, 1, sum_counts)
                
        else:
            # Single element move: pick a random element and move it to a nearby position
            idx = random.randint(0, len(s)-1)
            old_x = s[idx]
            delta = random.randint(-5, 5)
            if delta == 0:
                continue
            new_x = old_x + delta
            if new_x < 1 or new_x > N or new_x in s:
                continue
            
            # Remove old_x
            s.remove(old_x)
            update_sum_counts(s, old_x, -1, sum_counts)
            # Add new_x
            s.append(new_x)
            s.sort()
            update_sum_counts(s, new_x, 1, sum_counts)
            
            new_energy = compute_energy(s, sum_counts)
            delta_e = new_energy - energy
            
            if delta_e <= 0 or random.random() < pow(2.718281828, -delta_e / T):
                energy = new_energy
                if energy < best_e:
                    best_e = energy
                    best_s = s[:]
                    if energy == 0:
                        break
            else:
                # Revert
                s.remove(new_x)
                update_sum_counts(s, new_x, -1, sum_counts)
                s.append(old_x)
                s.sort()
                update_sum_counts(s, old_x, 1, sum_counts)
        
        T *= cooling
    
    # Return the best set found
    if best_e == 0:
        return best_s
    
    # If we couldn't achieve energy 0, return the best set we found
    # But we need to make sure it's valid (no duplicate sums)
    # If best_e > 0, the set is not valid, so we need to find a smaller valid set
    
    # Let's try to extract a valid subset from the best set
    # Start with the prescribed elements and greedily add elements that don't create conflicts
    
    def extract_valid_set(s):
        s = sorted(s)
        result = list(prescribed)
        sum_counts = {}
        for i, x in enumerate(result):
            for y in result[i:]:
                sm = x + y
                sum_counts[sm] = sum_counts.get(sm, 0) + 1
        
        for x in s:
            if x in result:
                continue
            # Check if adding x creates no conflicts
            valid = True
            for y in result:
                sm = x + y
                if sum_counts.get(sm, 0) >= 1:
                    valid = False
                    break
            if valid:
                sm2 = 2 * x
                if sum_counts.get(sm2, 0) >= 1:
                    valid = False
            if valid:
                result.append(x)
                result.sort()
                update_sum_counts(result, x, 1, sum_counts)
        
        return result
    
    valid_set = extract_valid_set(best_s)
    
    # Also try to build a valid set from scratch using the best_s as a guide
    # Sort candidates by their frequency in best_s
    candidate_order = {}
    for x in best_s:
        candidate_order[x] = candidate_order.get(x, 0) + 1
    
    def build_valid_set_with_order():
        result = list(prescribed)
        result.sort()
        sum_counts = {}
        for i, x in enumerate(result):
            for y in result[i:]:
                sm = x + y
                sum_counts[sm] = sum_counts.get(sm, 0) + 1
        
        # Sort all candidates by their frequency in best_s (descending)
        all_candidates = [x for x in range(1, N+1) if x not in result]
        all_candidates.sort(key=lambda x: -candidate_order.get(x, 0))
        
        for x in all_candidates:
            # Check if adding x creates no conflicts
            valid = True
            for y in result:
                sm = x + y
                if sum_counts.get(sm, 0) >= 1:
                    valid = False
                    break
            if valid:
                sm2 = 2 * x
                if sum_counts.get(sm2, 0) >= 1:
                    valid = False
            if valid:
                result.append(x)
                result.sort()
                update_sum_counts(result, x, 1, sum_counts)
        
        return result
    
    valid_set2 = build_valid_set_with_order()
    
    # Return the larger valid set
    if len(valid_set2) > len(valid_set):
        return valid_set2
    else:
        return valid_set

# EVOLVE-BLOCK-END
