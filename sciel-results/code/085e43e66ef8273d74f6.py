import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    # 1. Precompute LCM compatibility
    # adj[i] is a bitmask of j such that lcm(i, j) > N
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = i * j // g
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute compatibility bitmasks for faster checks
    full_mask = (1 << (N + 1)) - 2
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_adj[i] = full_mask & ~adj[i] & ~(1 << i)

    def is_compatible_with_set(x, mask):
        """Check if x is compatible with all elements in the set represented by mask."""
        # x is compatible with y if bit y is set in comp_adj[x]
        # We need (mask & ~comp_adj[x]) == 0
        return (mask & ~comp_adj[x]) == 0

    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def local_search(initial_set):
        """
        Local search to maximize the size of a valid set.
        Uses Add and Swap heuristics.
        """
        current_set = set(initial_set)
        current_mask = 0
        for v in current_set:
            current_mask |= (1 << v)
        
        best_size = len(current_set)
        best_set = list(current_set)
        
        # Order candidates by value to prefer smaller numbers (often more compatible)
        candidates_order = list(range(1, N + 1))
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Try to Add
            # Find all x not in current_set that are compatible with current_set
            added_any = False
            # Sort candidates to try "safer" ones first? 
            # Smaller numbers are generally compatible with more numbers.
            for x in candidates_order:
                if (current_mask >> x) & 1:
                    continue
                if is_compatible_with_set(x, current_mask):
                    current_set.add(x)
                    current_mask |= (1 << x)
                    improved = True
                    added_any = True
                    break # Restart loop to check for more additions
            
            if added_any:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                continue
            
            # 2. Try to Swap (1-for-1) followed by Add
            # This allows escaping local optima where no single addition works,
            # but replacing one element allows adding two.
            # To be efficient, we only do this if the set size is stuck.
            
            # Iterate through elements in current_set to remove
            # We pick the one that "blocks" the most additions? 
            # Or just iterate.
            # To save time, we limit the number of swap attempts per iteration.
            
            swap_found = False
            # Try removing each y in current_set
            # To avoid O(N^3) behavior, we sample or limit.
            # Given N=512, and set size ~25, 25*512 = 12800 checks per "round".
            # Each check involves is_compatible_with_set (O(1) bitmask op).
            # So 12800 ops is very fast.
            
            for y in list(current_set):
                if (time.time() - start_time) > time_limit:
                    return best_set
                
                # Tentatively remove y
                temp_set = current_set - {y}
                temp_mask = current_mask & ~(1 << y)
                
                # Try to add x to temp_set
                # We want to find an x that is compatible with temp_set
                # AND that allows adding another z later?
                # Simple heuristic: Just find any valid x.
                # If we find x, we set current = temp + x.
                # Then the next "Add" phase might find z.
                
                # To be smarter: Prefer x that is compatible with many potential z's?
                # For now, just find the first valid x.
                for x in candidates_order:
                    if x == y:
                        continue
                    if (temp_mask >> x) & 1:
                        continue
                    if is_compatible_with_set(x, temp_mask):
                        # Check if this swap leads to a larger set after adding
                        # We don't know yet, so we just accept the swap
                        # and let the Add phase do the work.
                        # However, if x is "worse" than y, we might oscillate.
                        # To prevent oscillation, we only accept if x < y? 
                        # Or we track visited states?
                        # Let's just accept and rely on the Add phase to grow the set.
                        # If the set doesn't grow, we might swap back.
                        # To avoid this, we only swap if x is "better" in some metric?
                        # Metric: Number of compatible neighbors in the remaining universe?
                        
                        # Let's just do it. The Add phase will try to grow.
                        current_set = temp_set
                        current_set.add(x)
                        current_mask = temp_mask
                        current_mask |= (1 << x)
                        
                        improved = True
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)

        return best_set

    # 3. Generate Initial Solutions
    # Try divisors of highly composite numbers <= 512
    candidates = [360, 480, 420, 336, 300, 240, 210, 180]
    best_initial = []
    
    for m in candidates:
        divs = get_divisors(m)
        # Filter divisors <= 512 (all are)
        # Check validity (should be valid by construction)
        # We assume divisors of M are pairwise LCM <= M <= 512
        if len(divs) > len(best_initial):
            best_initial = divs
            
    # Also try {1..22} or similar simple sets if divisors are small
    simple_set = list(range(1, 23))
    if len(simple_set) > len(best_initial):
        best_initial = simple_set

    # 4. Run Local Search
    result = local_search(best_initial)
    
    # 5. Final Validation
    result_set = set(result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                # Fallback to a known valid set if error
                return best_initial
                
    return sorted(result)

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
