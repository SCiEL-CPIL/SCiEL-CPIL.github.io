import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    # 1. Fast Greedy Initialization
    # This gives a strong lower bound for best_len
    def fast_greedy():
        values = []
        products = set()
        for x in range(2, 129):
            # Calculate new suffix products
            # If we append x to values, the new products are:
            # x, x*values[-1], x*values[-1]*values[-2], ...
            tails = []
            prod = x
            valid = True
            for a in reversed(values):
                prod *= a
                if prod in products:
                    valid = False
                    break
                tails.append(prod)
            
            if not valid:
                continue
            
            # Check if any tail is already in products
            # tails are distinct from each other
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        
        return values

    best_seq = fast_greedy()
    best_len = len(best_seq)
    
    # 2. Beam Search
    # State: (seq, prods, sufs)
    # seq: list of ints
    # prods: set of ints (all block products)
    # sufs: list of ints (suffix products)
    
    K = 300  # Beam width
    
    # Initial state
    # seq: [], prods: set(), sufs: []
    beam = [ ([], set(), []) ]
    
    # We will iterate x from 2 to 128
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        
        for seq, prods, sufs in beam:
            # Pruning: Can this state possibly beat best_len?
            # Max additional elements we can add is (128 - x)
            if len(seq) + (128 - x) < best_len:
                continue
                
            # Option 1: Exclude x
            # We keep the state as is.
            # However, we might have multiple identical states in the beam?
            # The beam is a list. We should probably deduplicate or just let the sorting handle it.
            # Actually, if we exclude x, the state remains (seq, prods, sufs).
            # If we include x, the state becomes (new_seq, new_prods, new_sufs).
            
            # Option 2: Include x
            # Calculate new tails
            new_tails = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                new_tails.append(prod)
            
            # Check collision
            # If any new tail is in prods, collision
            # Also, new_tails are distinct from each other because seq is strictly increasing and >= 2
            if not prods.intersection(new_tails):
                new_prods = prods.union(new_tails)
                new_seq = seq + [x]
                new_sufs = new_tails
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    
                new_beam.append((new_seq, new_prods, new_sufs))
                
            # Add the "exclude" state to new_beam
            # We can just append the current state
            new_beam.append((seq, prods, sufs))
            
        if not new_beam:
            break
            
        # Prune and Sort
        # We want to keep the "best" K states.
        # Sorting key:
        # 1. Length (descending)
        # 2. Last element (ascending) - smaller last element leaves more room
        # 3. Sum of elements (ascending) - smaller sum might be "safer"?
        
        def sort_key(state):
            seq = state[0]
            if not seq:
                last = 0
                s = 0
            else:
                last = seq[-1]
                s = sum(seq)
            return (-len(seq), last, s)
            
        new_beam.sort(key=sort_key)
        
        # Keep top K
        # Note: new_beam contains both "skip" and "include" states.
        # There might be duplicates if multiple parents lead to the same "skip" state?
        # No, each parent is unique in the beam.
        # But "skip" states from different parents are different unless the parents were identical.
        # Since we start with 1 state and branch, states are unique paths.
        
        beam = new_beam[:K]
        
        # Early stopping check
        # If the best possible length from the current beam is less than best_len, we can stop?
        # max_len_in_beam = max(len(s[0]) for s in beam)
        # if max_len_in_beam + (128 - x) < best_len:
        #     break
        # This is already handled in the pruning step for the next iteration,
        # but we can check here to break the loop early if no state in the beam
        # can possibly beat the current best_len with the remaining numbers.
        
        max_possible = 0
        for s in beam:
            possible = len(s[0]) + (128 - x)
            if possible > max_possible:
                max_possible = possible
                
        if max_possible < best_len:
            break

    return best_seq
