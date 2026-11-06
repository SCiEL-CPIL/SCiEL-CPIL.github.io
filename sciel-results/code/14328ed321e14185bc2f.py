import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    # Universe: 51 to 100
    U = list(range(51, 101))
    U_set = set(U)
    
    # Precompute pairs for each sum
    # Max sum of 2 elements in U is 100+99=199
    # Min sum is 51+52=103
    # Target sums for x in U: 200-x ranges from 100 to 149
    pairs_by_sum = {s: [] for s in range(100, 150)}
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            s = U[i] + U[j]
            if 100 <= s <= 149:
                pairs_by_sum[s].append((U[i], U[j]))
                
    def is_valid_addition(x, S):
        """Check if adding x to S creates a sum of 200."""
        target = 200 - x
        if target not in pairs_by_sum:
            return True
        for a, b in pairs_by_sum[target]:
            if a in S and b in S:
                return False
        return True

    def is_valid_set(S):
        """Check if S has no subset summing to 200."""
        # Since S subset of U, only triples can sum to 200 (pairs < 200, singles < 200)
        # We can just check all pairs? No, check if any pair sums to 200-x for any x in S.
        # Simpler: iterate through all triples? 50C3 is 19600. Fast enough.
        # But we can use the pairs_by_sum logic.
        # For each x in S, check if any pair in S sums to 200-x.
        # This counts each bad triple 3 times.
        # We just need to know if ANY exists.
        S_list = list(S)
        S_set = set(S_list)
        for x in S_list:
            target = 200 - x
            if target in pairs_by_sum:
                for a, b in pairs_by_sum[target]:
                    if a in S_set and b in S_set:
                        return False
        return True

    # Greedy initialization
    def greedy_init():
        S = set()
        random.shuffle(U)
        for x in U:
            if is_valid_addition(x, S):
                S.add(x)
        return S

    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local Search
    current_set = set(best_set)
    current_size = best_size
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random element to add
        candidates = [x for x in U if x not in current_set]
        if not candidates:
            break
        
        x = random.choice(candidates)
        
        # Find conflicts
        target = 200 - x
        conflicts = []
        if target in pairs_by_sum:
            for a, b in pairs_by_sum[target]:
                if a in current_set and b in current_set:
                    conflicts.append((a, b))
        
        if not conflicts:
            # Can add x directly
            current_set.add(x)
            current_size += 1
            if current_size > best_size:
                best_set = set(current_set)
                best_size = current_size
                T = 1.0 # Reset temperature?
            continue
        
        # We must remove elements to resolve conflicts
        # Get all elements involved in conflicts
        involved = set()
        for a, b in conflicts:
            involved.add(a)
            involved.add(b)
            
        # Try removing each element in involved
        # We want to remove the minimum number of elements.
        # Since conflicts are pairs, removing 1 element might fix multiple conflicts if it's shared.
        # We try single removals first.
        found_improvement = False
        for y in involved:
            # Try removing y
            new_set = current_set - {y}
            new_set.add(x)
            
            # Check if new_set is valid
            # Only need to check if x creates new conflicts with new_set
            # And if removing y didn't break anything else (it didn't, it only removed)
            # But we must ensure no OTHER triples exist in new_set.
            # Since current_set was valid, and we only removed y and added x,
            # the only new potential conflicts are those involving x.
            # Conflicts involving x are resolved if we removed at least one element from each conflicting pair.
            # Does removing y resolve ALL conflicts?
            resolved = True
            for a, b in conflicts:
                if y != a and y != b:
                    # This conflict is NOT resolved by removing y
                    resolved = False
                    break
            
            if resolved:
                # Check if x creates new conflicts with the REMAINING set
                # Actually, if resolved is True, then for every pair (a,b) in conflicts, y is in it.
                # So no pair from conflicts remains.
                # Are there other pairs in current_set that sum to target?
                # No, because current_set was valid, so no pair in current_set summed to target?
                # Wait. current_set was valid means no triple in current_set sums to 200.
                # It does NOT mean no pair in current_set sums to 200-x.
                # If a pair in current_set summed to 200-x, then x + that pair would be a bad triple.
                # So yes, the conflicts list contains ALL pairs in current_set summing to target.
                # So if we remove y and it hits all conflicts, then no pair in new_set sums to target.
                # So x is safe.
                
                new_size = current_size - 1 + 1 # -1 for y, +1 for x
                if new_size > current_size:
                    current_set = new_set
                    current_size = new_size
                    found_improvement = True
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                    break
                elif new_size == current_size:
                    # Accept with probability based on temperature
                    if random.random() < T:
                        current_set = new_set
                        current_size = new_size
                        found_improvement = True
                        if current_size > best_size:
                            best_set = set(current_set)
                            best_size = current_size
                        break
                else:
                    # new_size < current_size, skip
                    pass
        
        if not found_improvement:
            # Try removing 2 elements?
            # This is more expensive. Only do if time permits.
            # For now, just cool down
            T *= cooling
            
    return list(best_set)
