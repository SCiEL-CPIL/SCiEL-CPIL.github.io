import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def try_greedy(start=1, end=4096, step=1):
        selected = []
        existing_sums = set()
        
        # Pre-allocate or just use list
        for x in range(start, end + 1, step):
            if time.time() > deadline:
                return selected
            
            # Check if x can be added
            valid = True
            
            # 1. x, x, x -> 3x
            if 3 * x in existing_sums:
                valid = False
            
            # 2. x, x, y -> 2x + y
            if valid:
                for y in selected:
                    if 2 * x + y in existing_sums:
                        valid = False
                        break
            
            # 3. x, y, z -> x + y + z
            if valid:
                k = len(selected)
                for i in range(k):
                    y = selected[i]
                    # y <= z
                    for j in range(i, k):
                        z = selected[j]
                        if x + y + z in existing_sums:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                # Update existing_sums
                # Add 3x
                existing_sums.add(3 * x)
                # Add 2x + y
                for y in selected[:-1]: # exclude x itself as y? No, y is from old set
                    existing_sums.add(2 * x + y)
                # Add x + y + z
                k_old = len(selected) - 1
                for i in range(k_old):
                    y = selected[i]
                    for j in range(i, k_old):
                        z = selected[j]
                        existing_sums.add(x + y + z)
        
        return selected

    # Standard greedy
    best_set = try_greedy(1, 4096, 1)
    
    # Try to improve? 
    # The greedy is deterministic. 
    # Maybe try starting with different initial sets?
    # Or try removing the smallest element and re-greedy?
    
    # Given the fast execution, we can try a few variations.
    # Variation 2: Start with 2?
    # If we skip 1, can we get more?
    # Let's check if 1 is always in the optimal set.
    # For B_h, 1 is usually included.
    
    # Let's try a "repair" step: 
    # If we have a set S, can we swap an element a with a larger b?
    # This is complex.
    
    # Let's just run the greedy. It's the most robust baseline.
    # The prompt mentions a score of 1.778 for a similar strategy.
    # If the seed was slow and timed out returning a partial set, this will be much better.
    # If the seed finished, this might be the same or slightly better if the seed was suboptimal.
    # But the seed's logic is identical, just slower. 
    # Wait, the seed uses `combinations_with_replacement` and recomputes ALL sums.
    # My version is incremental.
    # The result of a greedy algorithm is deterministic. 
    # If the seed didn't time out, it would produce the same set.
    # If it did time out, it produced a smaller set.
    # The score 1.0 suggests it might have timed out or the baseline is just the seed's output.
    
    # Let's add a simple local search: 
    # Try to remove the largest element and see if we can add two smaller ones? No, we want to maximize count.
    # Try to remove an element and re-greedy the rest?
    
    # Actually, a different ordering might help.
    # What if we try to pick numbers that are "sparse"?
    # The greedy picks the smallest valid number. This is standard.
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
