"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score_set_bitset(s):
    """
    Fast scoring using bitset operations.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in s.
    """
    if not s:
        return 0
    s_set = set(s)
    max_val = max(s_set)
    # Create a bitset where bit i is 1 if i in s
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    count = 0
    # d can range from 1 to max_val // 2
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
            
    return count

def _construct_random():
    """
    Generate a random set of 32 distinct integers in 1..512.
    """
    return random.sample(range(1, 513), 32)

def _simulated_annealing(initial_set, deadline, start_temp=10.0, cooling=0.9999, min_temp=0.01):
    """
    Optimize the set using Simulated Annealing.
    """
    current_set = set(initial_set)
    current_score = _score_set_bitset(current_set)
    best_set = list(current_set)
    best_score = current_score
    
    temp = start_temp
    steps = 0
    
    # Precompute list of all possible elements for faster random selection?
    # Or just sample.
    all_elements = list(range(1, 513))
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random element to remove
        elem_to_remove = random.choice(list(current_set))
        current_set.remove(elem_to_remove)
        
        # Pick a random element to add
        # To ensure distinctness, we can sample until we find one not in set
        # Or just sample from all_elements and check
        new_elem = random.choice(all_elements)
        attempts = 0
        while new_elem in current_set and attempts < 10:
            new_elem = random.choice(all_elements)
            attempts += 1
        
        if new_elem in current_set:
            # Rollback if we failed to find a unique element (unlikely)
            current_set.add(elem_to_remove)
            continue
            
        current_set.add(new_elem)
        
        new_score = _score_set_bitset(current_set)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Rollback
            current_set.remove(new_elem)
            current_set.add(elem_to_remove)
        
        temp *= cooling
        steps += 1
        
        # Safety break if too many steps without improvement? 
        # SA handles this via temperature, but we can add a stagnation check if needed.
        # For now, rely on time and temp.
        
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin
    
    best_set = None
    best_score = -1
    
    # Strategy: Multi-start Simulated Annealing
    # We will run SA for a fixed time slice for each start.
    # Or run SA until deadline, but restart if we get stuck?
    # Given the small size, we can do multiple short runs or one long run with restarts.
    
    # Let's do a few runs.
    # 1. Start from a random set.
    # 2. Start from a greedy set (if we keep the old one, but we replaced it. Let's make a simple greedy).
    
    # Simple Greedy Initialization
    def _construct_greedy():
        current_set = set()
        # Start with some random spread
        for _ in range(10):
            current_set.add(random.randint(1, 512))
        
        for step in range(22):
            best_elem = None
            best_score = -1
            # Try to add an element that maximizes score
            # To speed up, we can sample candidates instead of all 512
            candidates = random.sample(range(1, 513), 100)
            for elem in candidates:
                if elem in current_set:
                    continue
                test_set = current_set | {elem}
                score = _score_set_bitset(test_set)
                if score > best_score:
                    best_score = score
                    best_elem = elem
            if best_elem is not None:
                current_set.add(best_elem)
        return list(current_set)

    # Run 1: From Random
    if time.time() < deadline:
        init_set = _construct_random()
        s, sc = _simulated_annealing(init_set, min(deadline, start_time + 30.0))
        if sc > best_score:
            best_score = sc
            best_set = s

    # Run 2: From Greedy
    if time.time() < deadline:
        init_set = _construct_greedy()
        s, sc = _simulated_annealing(init_set, min(deadline, start_time + 60.0))
        if sc > best_score:
            best_score = sc
            best_set = s

    # Run 3: From Best Found so far (Perturbed)
    if time.time() < deadline and best_set is not None:
        perturbed = set(best_set)
        # Remove 2, Add 2
        for _ in range(2):
            if perturbed:
                to_remove = random.choice(list(perturbed))
                perturbed.remove(to_remove)
        while len(perturbed) < 32:
            new_elem = random.randint(1, 512)
            if new_elem not in perturbed:
                perturbed.add(new_elem)
        
        s, sc = _simulated_annealing(list(perturbed), deadline)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    # Run 4: Another Random Restart
    if time.time() < deadline:
        init_set = _construct_random()
        s, sc = _simulated_annealing(init_set, deadline)
        if sc > best_score:
            best_score = sc
            best_set = s

    if best_set is None:
        # Fallback
        best_set = _construct_random()
        
    return best_set

# EVOLVE-BLOCK-END
