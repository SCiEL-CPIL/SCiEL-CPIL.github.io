import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20-second safety margin
    
    # Precompute for speed: we'll use a list-based approach for counts
    # counts[s] = number of representations of s as a+b with a<=b from the set
    
    def can_add_fast(x, current_set, counts):
        # Check 2x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x+y for y in current_set where y >= x (to avoid double counting)
        # Actually we need to check all y, but we only add x+y where x<=y or y<=x
        # Since we're adding x, we need to check x+y for all y in current_set
        # But if y < x, then y+x is the same sum as x+y, and we should have already
        # counted it when y was added (if x was already in the set at that time)
        # Wait, no. When we add x, we need to check all sums x+y for y in current_set.
        # The sum x+y has representations from existing pairs plus the new pair (min(x,y), max(x,y)).
        # We need to ensure that after adding, the count doesn't exceed 2.
        # So we check: for each y in current_set, if counts[x+y] >= 2, we can't add x.
        # But also need to check 2x.
        for y in current_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def update_counts_fast(x, current_set, counts):
        # Add representation for 2x
        counts[2*x] = counts.get(2*x, 0) + 1
        # Add representations for x+y
        for y in current_set:
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def remove_counts_fast(x, current_set, counts):
        # Remove representation for 2x
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        # Remove representations for x+y
        for y in current_set:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    def greedy_construct(order, deadline):
        """Greedy construction given an ordering of candidates."""
        current_set = set()
        counts = {}
        for x in order:
            if time.time() > deadline:
                break
            if can_add_fast(x, current_set, counts):
                current_set.add(x)
                update_counts_fast(x, current_set, counts)
        return current_set, counts

    def local_search(current_set, counts, deadline):
        """Local search with swaps and additions."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            for x in range(1, 1001):
                if time.time() > deadline:
                    break
                if x not in current_set and can_add_fast(x, current_set, counts):
                    current_set.add(x)
                    update_counts_fast(x, current_set, counts)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add another
            for remove_x in list(current_set):
                if time.time() > deadline:
                    break
                
                # Temporarily remove
                old_set = current_set - {remove_x}
                remove_counts_fast(remove_x, old_set, counts)
                current_set.discard(remove_x)
                
                found_swap = False
                for add_x in range(1, 1001):
                    if add_x == remove_x or add_x in current_set:
                        continue
                    if can_add_fast(add_x, current_set, counts):
                        current_set.add(add_x)
                        update_counts_fast(add_x, current_set, counts)
                        found_swap = True
                        break
                
                if not found_swap:
                    # Put back
                    current_set.add(remove_x)
                    update_counts_fast(remove_x, current_set, counts)
                else:
                    improved = True
                    break
        
        return current_set, counts

    def simulated_annealing(current_set, counts, deadline):
        """Simulated annealing for local optima escape."""
        if time.time() > deadline:
            return current_set, counts
        
        temp = 50.0
        cooling = 0.95
        best_set = current_set.copy()
        best_counts = counts.copy()
        best_size = len(current_set)
        
        current_size = len(current_set)
        
        while time.time() < deadline and temp > 0.1:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3 and len(current_set) > 0:
                # Remove a random element
                remove_x = random.choice(list(current_set))
                old_set = current_set - {remove_x}
                remove_counts_fast(remove_x, old_set, counts)
                current_set.discard(remove_x)
                new_size = len(current_set)
                
                # Accept if improvement or with probability based on temperature
                delta = new_size - current_size
                if delta >= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    current_size = new_size
                    if current_size > best_size:
                        best_set = current_set.copy()
                        best_counts = counts.copy()
                        best_size = current_size
                else:
                    # Revert
                    current_set.add(remove_x)
                    update_counts_fast(remove_x, current_set, counts)
                    
            elif move_type < 0.6:
                # Add a random element
                candidates = [x for x in range(1, 1001) if x not in current_set]
                if candidates:
                    add_x = random.choice(candidates)
                    if can_add_fast(add_x, current_set, counts):
                        current_set.add(add_x)
                        update_counts_fast(add_x, current_set, counts)
                        new_size = len(current_set)
                        delta = new_size - current_size
                        if delta >= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                            current_size = new_size
                            if current_size > best_size:
                                best_set = current_set.copy()
                                best_counts = counts.copy()
                                best_size = current_size
                        else:
                            # Revert
                            remove_counts_fast(add_x, current_set - {add_x}, counts)
                            current_set.discard(add_x)
                    else:
                        # Can't add, so try a swap instead
                        if len(current_set) > 0:
                            remove_x = random.choice(list(current_set))
                            old_set = current_set - {remove_x}
                            remove_counts_fast(remove_x, old_set, counts)
                            current_set.discard(remove_x)
                            
                            if can_add_fast(add_x, current_set, counts):
                                current_set.add(add_x)
                                update_counts_fast(add_x, current_set, counts)
                                new_size = len(current_set)
                                delta = new_size - current_size
                                if delta >= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                                    current_size = new_size
                                    if current_size > best_size:
                                        best_set = current_set.copy()
                                        best_counts = counts.copy()
                                        best_size = current_size
                                else:
                                    # Revert both
                                    remove_counts_fast(add_x, current_set - {add_x}, counts)
                                    current_set.discard(add_x)
                                    current_set.add(remove_x)
                                    update_counts_fast(remove_x, current_set, counts)
                            else:
                                # Revert
                                current_set.add(remove_x)
                                update_counts_fast(remove_x, current_set, counts)
            
            else:
                # Swap two elements
                if len(current_set) >= 2:
                    candidates_out = list(current_set)
                    candidates_in = [x for x in range(1, 1001) if x not in current_set]
                    if candidates_in:
                        remove_x = random.choice(candidates_out)
                        add_x = random.choice(candidates_in)
                        
                        old_set = current_set - {remove_x}
                        remove_counts_fast(remove_x, old_set, counts)
                        current_set.discard(remove_x)
                        
                        if can_add_fast(add_x, current_set, counts):
                            current_set.add(add_x)
                            update_counts_fast(add_x, current_set, counts)
                            new_size = len(current_set)
                            delta = new_size - current_size
                            if delta >= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                                current_size = new_size
                                if current_size > best_size:
                                    best_set = current_set.copy()
                                    best_counts = counts.copy()
                                    best_size = current_size
                            else:
                                # Revert
                                remove_counts_fast(add_x, current_set - {add_x}, counts)
                                current_set.discard(add_x)
                                current_set.add(remove_x)
                                update_counts_fast(remove_x, current_set, counts)
                        else:
                            # Revert
                            current_set.add(remove_x)
                            update_counts_fast(remove_x, current_set, counts)
            
            temp *= cooling
        
        if best_size > len(current_set):
            return best_set, best_counts
        return current_set, counts

    best_set = set()
    best_size = 0
    
    # Strategy 1: Multiple random restarts with greedy + local search
    while time.time() < deadline - 30:
        # Create a random ordering with some bias
        order = list(range(1, 1001))
        random.shuffle(order)
        
        current_set, counts = greedy_construct(order, deadline - 30)
        
        # Local search
        if time.time() < deadline - 20:
            current_set, counts = local_search(current_set, counts, deadline - 20)
        
        if len(current_set) > best_size:
            best_set = current_set.copy()
            best_size = len(current_set)
    
    # Strategy 2: Simulated annealing from the best found
    if time.time() < deadline - 10 and best_size > 0:
        # Reconstruct counts for best_set
        counts = {}
        for x in best_set:
            update_counts_fast(x, best_set - {x}, counts)
        
        sa_set, sa_counts = simulated_annealing(best_set, counts, deadline - 10)
        
        if len(sa_set) > best_size:
            best_set = sa_set.copy()
            best_size = len(sa_set)
    
    return list(best_set)
