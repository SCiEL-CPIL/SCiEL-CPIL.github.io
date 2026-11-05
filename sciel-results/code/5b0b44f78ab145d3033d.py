"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    range_1_N = range(1, N + 1)
    
    def get_sums(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def is_valid_sidon(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def can_add(x, selected, sums):
        if 2 * x in sums:
            return False
        for a in selected:
            if x + a in sums:
                return False
        return True

    def count_addable(selected, sums):
        """Returns list of elements that can be added."""
        addable = []
        for x in range_1_N:
            if x in selected:
                continue
            if can_add(x, selected, sums):
                addable.append(x)
        return addable

    def is_maximal(selected, sums):
        return len(count_addable(selected, sums)) == 0

    def random_greedy(seed):
        random.seed(seed)
        candidates = list(range_1_N)
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            if can_add(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        # Ensure maximality
        addable = count_addable(selected, sums)
        while addable:
            x = addable[0]
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
            addable = count_addable(selected, sums)
            
        return selected, sums

    # Simulated Annealing
    def simulated_annealing(initial_selected, initial_sums, time_limit):
        current_selected = list(initial_selected)
        current_sums = set(initial_sums)
        
        # Calculate initial energy
        if not is_valid_sidon(current_selected):
            current_energy = 10000
        else:
            addable = count_addable(current_selected, current_sums)
            if addable:
                current_energy = 100 + len(addable)
            else:
                current_energy = len(current_selected)
        
        best_selected = list(current_selected)
        best_sums = set(current_sums)
        best_energy = current_energy
        
        T = 10.0
        cooling_rate = 0.995
        
        steps = 0
        while time.time() < time_limit:
            steps += 1
            
            # Generate neighbor
            new_selected = list(current_selected)
            new_sums = set(current_sums)
            
            move_type = random.randint(0, 2)
            
            if move_type == 0 and len(current_selected) > 0:
                # Remove random element
                idx = random.randint(0, len(new_selected) - 1)
                removed = new_selected[idx]
                new_selected.pop(idx)
                # Recalculate sums
                new_sums = get_sums(new_selected)
                
            elif move_type == 1:
                # Add random missing element
                missing = [x for x in range_1_N if x not in new_selected]
                if not missing:
                    continue
                x = random.choice(missing)
                if can_add(x, new_selected, new_sums):
                    new_selected.append(x)
                    new_sums.add(2 * x)
                    for a in new_selected[:-1]:
                        new_sums.add(x + a)
                else:
                    # If can't add, maybe skip or accept penalty? 
                    # Let's just skip invalid additions to keep state valid Sidon?
                    # Or allow invalid states? 
                    # Allowing invalid states helps escape local minima.
                    # Let's allow it but with high energy.
                    new_selected.append(x)
                    # Sums will be recalculated or we just mark invalid
                    # For simplicity, let's just recalculate sums and check validity later
                    new_sums = get_sums(new_selected)

            else:
                # Swap
                if len(current_selected) > 0:
                    idx = random.randint(0, len(new_selected) - 1)
                    old_x = new_selected[idx]
                    missing = [x for x in range_1_N if x not in new_selected]
                    if missing:
                        new_x = random.choice(missing)
                        new_selected[idx] = new_x
                        new_sums = get_sums(new_selected)
                    else:
                        continue
                else:
                    continue
            
            # Calculate new energy
            if not is_valid_sidon(new_selected):
                # Count violations roughly
                sums = get_sums(new_selected)
                # Recount to find violations is expensive. 
                # Just use a high penalty.
                new_energy = 10000 + random.randint(0, 100)
            else:
                addable = count_addable(new_selected, new_sums)
                if addable:
                    new_energy = 100 + len(addable)
                else:
                    new_energy = len(new_selected)
            
            # Acceptance
            if new_energy < current_energy:
                current_selected = new_selected
                current_sums = new_sums
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_selected = list(current_selected)
                    best_sums = set(current_sums)
                    best_energy = current_energy
                    
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    accept_prob = math.exp(-delta / T)
                    if random.random() < accept_prob:
                        current_selected = new_selected
                        current_sums = new_sums
                        current_energy = new_energy
            
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
                
        return best_selected, best_sums

    # Initial solution
    init_selected, init_sums = random_greedy(42)
    
    # Run SA
    time_limit = time.time() + 25 # Leave 5s for final checks
    best_selected, best_sums = simulated_annealing(init_selected, init_sums, time_limit)
    
    # Final verification
    if not is_valid_sidon(best_selected):
        # Fallback to initial
        return init_selected
    
    if not is_maximal(best_selected, best_sums):
        # Extend to maximal
        addable = count_addable(best_selected, best_sums)
        while addable:
            x = addable[0]
            best_selected.append(x)
            best_sums.add(2 * x)
            for a in best_selected[:-1]:
                best_sums.add(x + a)
            addable = count_addable(best_selected, best_sums)
            
    return best_selected
# EVOLVE-BLOCK-END
