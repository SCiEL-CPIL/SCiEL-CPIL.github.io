import random
import time

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    all_nums = list(range(2, 41))
    composites = [x for x in all_nums if x not in primes]
    
    # Precompute primes for factorization if needed, but direct product check is easier
    
    def is_valid(S):
        """
        Check if all 2^k subset products of S are distinct.
        """
        k = len(S)
        if k == 0:
            return True
        # Use a set to store products
        # Start with empty product
        prods = {1}
        for x in S:
            # Create new products by multiplying x with all existing
            # We can do this in-place if we iterate over a copy
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            prods.update(new_prods)
        
        return len(prods) == (1 << k)
    
    # Initial state
    current_set = set(primes)
    best_set = set(primes)
    best_size = len(primes)
    
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    # Heuristic: Sort composites by value? Or random?
    # Smaller composites might be "denser" in prime space?
    # Let's just shuffle.
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        candidates = [x for x in all_nums if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                break # Found one, restart add search
        
        if not added:
            # 2. Try to swap to escape local optimum
            # If we can't add, maybe we need to change the composition
            # Try random swaps
            if current_set and candidates:
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    y = random.choice(list(current_set))
                    x = random.choice(candidates)
                    new_set = (current_set - {y}) | {x}
                    if is_valid(new_set):
                        # Accept swap
                        # We don't update best_size here because size is same
                        # But this might enable future adds
                        current_set = new_set
                        break # Try adds again in next main loop iteration
            
            # 3. If stuck for a while, restart from best
            # Simple heuristic: if no add and no swap helped, restart
            # To avoid infinite loops of useless swaps, we can restart from best_set
            # periodically.
            # For simplicity, just keep swapping. The randomization should eventually find a path.
            # However, if we are at size 12 and can't get to 13, we might loop forever.
            # Let's add a restart mechanism.
            
            # If we haven't improved in a long time, restart
            # We don't track "last improvement time" easily in this simple loop.
            # Let's just restart from best_set if we are stuck at a size < best_size?
            # No, current_set size is always >= best_size? No, swaps keep size same.
            # So current_size is constant during the "stuck" phase.
            
            # Let's restart from best_set if we are stuck.
            # How to detect stuck?
            # If we did 100 swaps and no adds, restart.
            # We don't have a counter here.
            
            # Alternative: Randomized Restart
            # If time > 20s, start restarting more frequently.
            
            # Simple approach: Just keep trying.
            # The probability of finding a valid add increases if the set is "less constrained".
            # Swaps change the set.
            
        # Small sleep to avoid burning CPU? No, we need to use the time.
    
    return list(best_set)
