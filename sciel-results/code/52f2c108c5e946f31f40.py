import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    fixed = [1, 2, 5, 11]
    fixed_set = set(fixed)
    max_val = 284
    target_size = 20

    best_set = None
    best_size = len(fixed)

    def get_sums(a):
        sums = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
        return sums

    def count_collisions(a):
        sums = []
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.append(a[i] + a[j])
        return len(sums) - len(set(sums))

    def incremental_collision_count(a, added):
        """
        a: list of existing elements (without 'added')
        added: the element to add
        Returns number of NEW collisions introduced by adding 'added'
        """
        existing_sums = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                existing_sums.add(a[i] + a[j])
        
        new_sums = set()
        for x in a:
            new_sums.add(x + added)
        new_sums.add(added + added)
        
        collisions = 0
        for s in new_sums:
            if s in existing_sums:
                collisions += 1
        # Also check collisions among the new sums themselves
        new_sums_list = list(new_sums)
        seen = set()
        for s in new_sums_list:
            if s in seen:
                collisions += 1
            seen.add(s)
        return collisions

    def random_initial_state(size):
        a = list(fixed)
        remaining = [x for x in range(1, max_val + 1) if x not in fixed_set]
        random.shuffle(remaining)
        a.extend(remaining[:max(0, size - len(fixed))])
        random.shuffle(a)
        return a

    def greedy_construct(max_restarts=200):
        nonlocal best_set, best_size
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            a = list(fixed)
            a_set = set(a)
            sums = set()
            for i in range(len(a)):
                for j in range(i, len(a)):
                    sums.add(a[i] + a[j])
            
            candidates = [x for x in range(1, max_val + 1) if x not in fixed_set]
            random.shuffle(candidates)
            
            for c in candidates:
                if len(a) >= target_size:
                    break
                # Check if c can be added
                valid = True
                for x in a:
                    if x + c in sums:
                        valid = False
                        break
                if not valid:
                    continue
                if c + c in sums:
                    continue
                # Also need to ensure c is not in a_set (already ensured by candidates)
                
                # Add c
                a.append(c)
                a_set.add(c)
                for x in a[:-1]:
                    sums.add(x + c)
                sums.add(c + c)
            
            if len(a) > best_size:
                best_size = len(a)
                best_set = list(a)
                if best_size >= target_size:
                    return True
        return best_size >= target_size

    def local_search_improve(a, max_iters=5000):
        """
        Try to improve a valid Sidon set by swapping elements.
        Returns improved set or original if no improvement.
        """
        a = list(a)
        a_set = set(a)
        current_cost = count_collisions(a)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if current_cost == 0:
                break
            
            improved = False
            # Try removing one element and adding another
            non_fixed_indices = [i for i, x in enumerate(a) if x not in fixed_set]
            if not non_fixed_indices:
                break
            
            for idx in non_fixed_indices:
                old_val = a[idx]
                # Try replacing with a random value
                for _ in range(10):
                    new_val = random.randint(1, max_val)
                    if new_val in a_set and new_val != old_val:
                        continue
                    # Check if new_val can replace old_val
                    # Create temp set
                    temp = a[:idx] + a[idx+1:]
                    temp_set = set(temp)
                    if new_val in temp_set:
                        continue
                    
                    # Check collisions in temp + new_val
                    new_a = temp + [new_val]
                    new_cost = count_collisions(new_a)
                    if new_cost < current_cost:
                        a = new_a
                        a_set = set(a)
                        current_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try random mutation
                idx = random.randint(0, len(a) - 1)
                if a[idx] in fixed_set:
                    continue
                new_val = random.randint(1, max_val)
                if new_val in a_set and new_val != a[idx]:
                    continue
                a[idx] = new_val
                a_set = set(a)
                current_cost = count_collisions(a)
        
        return a, current_cost

    def simulated_annealing(size=20, max_iters=100000):
        nonlocal best_set, best_size
        
        a = random_initial_state(size)
        current_cost = count_collisions(a)
        
        if current_cost == 0 and len(a) > best_size:
            best_size = len(a)
            best_set = list(a)
            return True
        
        T = 50.0
        T_min = 0.01
        cooling = 0.9999
        
        best_cost = current_cost
        best_a = list(a)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose a move
            move_type = random.random()
            new_a = list(a)
            
            if move_type < 0.8:
                # Move a non-fixed element
                non_fixed_indices = [i for i, x in enumerate(a) if x not in fixed_set]
                if not non_fixed_indices:
                    T *= cooling
                    continue
                idx = random.choice(non_fixed_indices)
                old_val = a[idx]
                new_val = random.randint(1, max_val)
                if new_val in a and new_val != old_val:
                    continue
                new_a[idx] = new_val
            elif move_type < 0.9:
                # Remove a non-fixed element
                non_fixed_indices = [i for i, x in enumerate(a) if x not in fixed_set]
                if non_fixed_indices:
                    idx = random.choice(non_fixed_indices)
                    new_a.pop(idx)
            else:
                # Add a random element
                if len(a) < size + 2:
                    new_val = random.randint(1, max_val)
                    if new_val not in a:
                        new_a.append(new_val)
            
            if new_a == a:
                T *= cooling
                continue
            
            new_cost = count_collisions(new_a)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                a = new_a
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_a = list(a)
                
                if current_cost == 0 and len(a) > best_size:
                    best_size = len(a)
                    best_set = list(a)
                    if best_size >= target_size:
                        return True
            
            T *= cooling
        
        # After SA, try to extend best_a if it's valid
        if best_cost == 0 and len(best_a) < target_size:
            improved_a, _ = local_search_improve(best_a, max_iters=1000)
            if count_collisions(improved_a) == 0 and len(improved_a) > best_size:
                best_size = len(improved_a)
                best_set = list(improved_a)
                if best_size >= target_size:
                    return True
        
        return best_size >= target_size

    # Strategy:
    # 1. Greedy construction (fast, often finds good solutions)
    # 2. Simulated Annealing (more robust, can escape local optima)
    # 3. Local search refinement
    
    if greedy_construct():
        return sorted(best_set)
    
    if simulated_annealing():
        return sorted(best_set)
    
    # Final fallback: more greedy restarts
    for _ in range(500):
        if time.time() > deadline:
            break
        a = list(fixed)
        a_set = set(a)
        sums = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
        
        candidates = [x for x in range(1, max_val + 1) if x not in fixed_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if len(a) >= target_size:
                break
            valid = True
            for x in a:
                if x + c in sums:
                    valid = False
                    break
            if not valid:
                continue
            if c + c in sums:
                continue
            
            a.append(c)
            a_set.add(c)
            for x in a[:-1]:
                sums.add(x + c)
            sums.add(c + c)
        
        if len(a) > best_size:
            best_size = len(a)
            best_set = list(a)
            if best_size >= target_size:
                return sorted(best_set)
    
    if best_set:
        return sorted(best_set)
    return sorted(fixed)
