import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill
    
    prescribed = {1, 2, 5, 11}
    N = 284
    TARGET_SIZE = 20  # Max possible size

    def get_sums(s):
        """Returns a set of all pairwise sums a+b where a<=b in s."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums

    def is_valid(s):
        """Check if s is a valid Sidon set."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def greedy_construct():
        """Construct a valid Sidon set greedily."""
        s = sorted(prescribed)
        sums = get_sums(s)
        
        # Try to add numbers from 1 to 284
        for x in range(1, N + 1):
            if x in s:
                continue
            # Check if x can be added
            ok = True
            new_sums = set()
            for y in s:
                val = x + y
                if val in sums:
                    ok = False
                    break
                new_sums.add(val)
            if ok:
                # Check x + x
                if x + x in sums:
                    ok = False
                else:
                    new_sums.add(x + x)
            
            if ok:
                s.append(x)
                sums.update(new_sums)
                
        return s

    # 1. Get a baseline valid set
    best_set = greedy_construct()
    best_size = len(best_set)
    
    if best_size >= TARGET_SIZE:
        return best_set

    # 2. Simulated Annealing to improve size
    # We want to maximize size.
    # Score = size - penalty * conflicts
    
    penalty = 50.0  # High penalty to prioritize validity
    
    def get_score(s):
        """Returns a score where higher is better.
        Valid sets have score = size.
        Invalid sets have score = size - penalty * conflicts.
        """
        sums = set()
        conflicts = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                val = s[i] + s[j]
                if val in sums:
                    conflicts += 1
                else:
                    sums.add(val)
        return len(s) - penalty * conflicts

    current_set = best_set[:]
    current_score = get_score(current_set)
    
    temperature = 10.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Pre-shuffled list of candidates for random selection
    candidates = list(range(1, N + 1))
    
    steps = 0
    while time.time() < deadline:
        if temperature < min_temp:
            # Restart with a new random state or the best known valid set
            # Let's restart from the best known valid set but with some random perturbation
            current_set = best_set[:]
            # Randomly remove some non-prescribed elements to allow re-adding
            non_prescribed = [x for x in current_set if x not in prescribed]
            if non_prescribed:
                to_remove = random.sample(non_prescribed, min(3, len(non_prescribed)))
                current_set = [x for x in current_set if x not in to_remove]
            current_score = get_score(current_set)
            temperature = 10.0
            
        # Generate a neighbor
        move_type = random.random()
        
        if len(current_set) < TARGET_SIZE and move_type < 0.5:
            # Try to Add
            # Pick a random candidate not in current_set
            x = random.choice(candidates)
            if x not in current_set:
                new_set = current_set + [x]
            else:
                continue
        elif len(current_set) > len(prescribed) and move_type < 0.8:
            # Try to Remove
            non_prescribed = [i for i, x in enumerate(current_set) if x not in prescribed]
            if non_prescribed:
                idx = random.choice(non_prescribed)
                new_set = current_set[:idx] + current_set[idx+1:]
            else:
                continue
        else:
            # Try to Swap
            non_prescribed = [i for i, x in enumerate(current_set) if x not in prescribed]
            if not non_prescribed:
                continue
            idx = random.choice(non_prescribed)
            old_val = current_set[idx]
            new_val = random.choice(candidates)
            if new_val == old_val or new_val in current_set:
                continue
            new_set = current_set[:]
            new_set[idx] = new_val
        
        new_score = get_score(new_set)
        
        # Check if new set is valid and better than best
        # We can quickly check validity if new_score == len(new_set)
        if new_score == len(new_set):
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set[:]
                if best_size >= TARGET_SIZE:
                    return best_set
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / temperature):
            current_set = new_set
            current_score = new_score
            
        temperature *= cooling_rate
        steps += 1
        
        # Optional: If we have been stuck for a long time, force a restart
        # But the temperature restart handles this.
        
    return best_set
