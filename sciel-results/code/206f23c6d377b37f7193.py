import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def check_and_add(x, S, pair_sums, used_sums):
        """
        Check if x can be added to S.
        S: list of selected numbers
        pair_sums: list of sums a+b for a<=b in S
        used_sums: set of all triple sums a+b+c for a<=b<=c in S
        Returns True if added, False otherwise.
        """
        # Generate new triple sums
        # Type 2: x + a + b (a,b in S)
        # Type 3: x + x + a (a in S)
        # Type 4: x + x + x
        
        # Check Type 2 against used_sums
        for ps in pair_sums:
            if (x + ps) in used_sums:
                return False
                
        # Check Type 3 against used_sums
        for s in S:
            if (2 * x + s) in used_sums:
                return False
                
        # Check Type 4 against used_sums
        if (3 * x) in used_sums:
            return False
            
        # Check internal collisions
        # Type 2 vs Type 3: x + ps == 2x + s  => ps == x + s
        # We can just check if any x+ps is in the set of 2x+s
        type3_sums = {2 * x + s for s in S}
        for ps in pair_sums:
            if (x + ps) in type3_sums:
                return False
            if (x + ps) == 3 * x:
                return False
                
        # If all pass, add
        for ps in pair_sums:
            used_sums.add(x + ps)
        for s in S:
            used_sums.add(2 * x + s)
        used_sums.add(3 * x)
        
        # Update pair_sums
        pair_sums.append(2 * x)
        for s in S:
            pair_sums.append(x + s)
            
        S.append(x)
        return True

    def run_greedy(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        S = []
        pair_sums = []
        used_sums = set()
        
        candidates = list(range(1, 4097))
        if randomize:
            random.shuffle(candidates)
            
        for x in candidates:
            if check_and_add(x, S, pair_sums, used_sums):
                pass
        
        return S

    best_set = []
    
    # 1. Run deterministic greedy
    best_set = run_greedy(randomize=False)
    best_size = len(best_set)
    
    # 2. Run randomized greedy
    run_count = 0
    while time.time() < deadline:
        run_count += 1
        current_set = run_greedy(randomize=True, seed=run_count)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set

# EVOLVE-BLOCK-END
