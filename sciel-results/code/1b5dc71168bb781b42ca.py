import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIMEOUT = 55.0  # Safety margin for 60s limit
    
    n = 256
    candidates = list(range(1, n + 1))
    
    # Precompute suffix sums of reciprocals for upper bound
    # suffix_sum[i] = sum(1/j for j in range(i, n+1))
    # Note: candidates are 1..n, so index i corresponds to value i
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / i

    def is_valid_addition(x, s_set):
        # s_set is a set of selected numbers
        # Check if adding x creates a 3-AP with any elements in s_set
        for a in s_set:
            # Case 1: x is start. x, a, c => 2a = x + c => c = 2a - x
            c = 2 * a - x
            if c in s_set:
                return False
            # Case 2: x is middle. a, x, c => 2x = a + c => c = 2x - a
            c = 2 * x - a
            if c in s_set:
                return False
            # Case 3: x is end. a, b, x => 2b = a + x => b = (a + x) / 2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in s_set:
                    return False
        return True

    # Greedy initialization to get a good lower bound
    def greedy_solve():
        selected = []
        s_set = set()
        for x in range(1, n + 1):
            if is_valid_addition(x, s_set):
                selected.append(x)
                s_set.add(x)
        return selected, sum(1.0 / x for x in selected)

    best_set, best_weight = greedy_solve()
    
    # Branch and Bound
    # We use a stack to avoid recursion depth issues and allow easier time checking
    # State: (idx, s_set_frozenset, current_weight)
    # Using frozenset for s_set makes it hashable and immutable, but copying is expensive.
    # Since we are doing DFS, we can use a list and backtrack (add/remove).
    # However, a stack of states is safer for time limits if we want to abort.
    # Let's use recursion with a global time check.
    
    def search(idx, s_set, current_weight):
        nonlocal best_weight, best_set
        
        # Time check
        if time.time() - start_time > TIMEOUT:
            return

        # Upper bound check
        if current_weight + suffix_sum[idx] <= best_weight:
            return

        if idx > n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(s_set)
            return

        x = candidates[idx - 1]
        
        # Branch 1: Include x
        if is_valid_addition(x, s_set):
            s_set.add(x)
            search(idx + 1, s_set, current_weight + 1.0 / x)
            s_set.remove(x)
        
        # Branch 2: Exclude x
        search(idx + 1, s_set, current_weight)

    # Start search from root
    # We can potentially start with a partial solution from greedy?
    # No, B&B explores the full space. The greedy result is just a bound.
    # However, we can prune branches that cannot beat the greedy result.
    
    search(1, set(), 0.0)
    
    return best_set

# EVOLVE-BLOCK-END
