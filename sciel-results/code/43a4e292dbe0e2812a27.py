import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0 # 90s limit, leave 5s margin
    
    # 1. Greedy Initialization
    selected = []
    reachable = 1 # Bit i set means sum i is reachable
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # 2. Local Search with Directed Moves
    # We maintain the current set and its reachable bitset.
    current_set = best_set.copy()
    current_reachable = reachable
    
    all_elements = list(range(1, 101))
    
    # Helper to check if adding x is valid given current_reachable
    # If we add x, the new reachable sums are R | (R << x).
    # This is valid iff (R | (R << x)) does not have bit 200 set.
    # Since R doesn't have bit 200 set (invariant), we just need to check
    # if (R << x) has bit 200 set.
    # (R << x) has bit 200 set iff R has bit (200 - x) set.
    def can_add(x, R):
        if x > 200: return False # x itself is 200? No, max is 100.
        # If x == 200, bit 200 of (R<<200) depends on bit 0 of R. R always has bit 0.
        # But x is in [1,100].
        # Check if R has bit (200-x) set.
        return not ((R >> (200 - x)) & 1)
        
    # Helper to update reachable bitset
    def update_reachable(R, x):
        return R | (R << x)
        
    # Helper to remove x from R? 
    # Removing is hard with bitsets. 
    # For swap moves (remove a, add b, c), we might need to recompute R for the removed element.
    # However, we can just recompute R from scratch if we do a swap, as it's O(N) which is small (N<=100).
    # Or we can maintain a count of how many subsets sum to each value? No, too complex.
    # Let's stick to:
    # 1. Try to add single elements.
    # 2. If stuck, try swap: remove one, add two.
    #    To check validity of swap (remove a, add b, c):
    #    We need to know if S \ {a} U {b, c} is valid.
    #    We can recompute the bitset for S \ {a} U {b, c} from scratch. 
    #    Since N is small (<=100), recomputing is fast enough if we don't do it millions of times.
    #    But we can be smarter.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single missing element
        missing = [x for x in all_elements if x not in current_set]
        random.shuffle(missing)
        
        for x in missing:
            if can_add(x, current_reachable):
                # Add x
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
        
        # Phase 2: If no single addition worked, try swaps (remove 1, add 2)
        # This allows us to trade one "blocking" element for two "safe" ones.
        if not improved:
            # Try a few random swaps to escape local optima
            # We need to be careful with time.
            # Let's try up to 1000 random swap attempts in this phase
            for _ in range(1000):
                if time.time() > deadline: break
                
                # Pick a random element to remove
                if not current_set: break
                a = random.choice(list(current_set))
                
                # Pick two random elements to add (from missing)
                missing_now = [x for x in all_elements if x not in current_set]
                if len(missing_now) < 2: break
                
                # Optimization: don't pick the same x twice
                b, c = random.sample(missing_now, 2)
                
                # Check if S' = S \ {a} U {b, c} is valid
                # We can construct the new set and check validity
                # To speed up, we can check if adding b and c to (S \ {a}) causes sum 200.
                # Let R_a be the reachable sums of S \ {a}.
                # We don't have R_a stored. 
                # Recomputing R_a is O(|S|). 
                # Checking if b,c can be added to R_a:
                # We can just simulate:
                # Start with R=1.
                # For each x in S \ {a}, R |= R << x.
                # Then check if (R | (R<<b) | (R<<c) | (R<<(b+c))) has bit 200.
                # Actually, simpler: just build the set and run the full validity check?
                # Full validity check is O(N^2) or O(N * 2^N)? No, bitset is O(N * W) where W is word size.
                # N=100, W=300 bits. Very fast.
                
                new_set = current_set.copy()
                new_set.remove(a)
                new_set.add(b)
                new_set.add(c)
                
                # Validity check
                r = 1
                valid = True
                for x in new_set:
                    r |= (r << x)
                    if (r >> 200) & 1:
                        valid = False
                        break
                
                if valid:
                    # Accept the swap
                    current_set = new_set
                    # Update reachable bitset for current_set
                    current_reachable = 1
                    for x in current_set:
                        current_reachable |= (current_reachable << x)
                    
                    improved = True
                    if len(current_set) > best_size:
                        best_set = current_set.copy()
                        best_size = len(current_set)
                    # Restart phase 1? Or just continue?
                    # Since we improved, let's break out of swap loop and go back to single additions
                    break

    return list(best_set)
