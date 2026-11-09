import time
import random
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 128] containing no three distinct elements 
    whose three pairwise least common multiples are equal. Maximizes cardinality.
    
    Uses multi-restart randomized greedy construction with local search polish.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    N = 128
    
    # 1. Precompute LCMs
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
                
    # 2. Identify Bad Triples and build conflict structure
    bad_triples = []
    # For each pair (a,b), store which triples contain both a and b
    # This allows O(1) check: adding c conflicts if any triple (a,b,c) is active
    pair_to_triples = {}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                lac = lcm_table[a][c]
                if lab == lac:
                    lbc = lcm_table[b][c]
                    if lab == lbc:
                        idx = len(bad_triples)
                        bad_triples.append((a, b, c))
                        # Register in pair_to_triples for all three pairs
                        pair_to_triples.setdefault((a, b), []).append(idx)
                        pair_to_triples.setdefault((a, c), []).append(idx)
                        pair_to_triples.setdefault((b, c), []).append(idx)
    
    # Also build num_to_triples for local search
    num_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        num_to_triples[a].append(idx)
        num_to_triples[b].append(idx)
        num_to_triples[c].append(idx)
    
    # 3. Helper: check if adding element x to current set creates a conflict
    # A conflict exists if there are two elements a, b in the set such that (a,b,x) is a bad triple
    def can_add(x, in_set):
        for t_idx in num_to_triples[x]:
            a, b, c = bad_triples[t_idx]
            # Find the other two elements
            others = [a, b, c]
            others.remove(x)
            if in_set[others[0]] and in_set[others[1]]:
                return False
        return True
    
    # 4. Greedy construction: add elements in a given order, skipping conflicts
    def greedy_construct(order, in_set):
        for x in order:
            if not in_set[x] and can_add(x, in_set):
                in_set[x] = True
    
    # 5. Local search: try to add back elements that were skipped
    def local_search_add_back(in_set):
        improved = True
        while improved:
            improved = False
            for i in range(1, N + 1):
                if not in_set[i] and can_add(i, in_set):
                    in_set[i] = True
                    improved = True
    
    # 6. Local search: try swaps - remove one element, add two others
    def local_search_swaps(in_set, max_swaps=50):
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            for remove_elem in range(1, N + 1):
                if not in_set[remove_elem]:
                    continue
                # Try removing remove_elem and see if we can add 2+ new elements
                backup = in_set[remove_elem]
                in_set[remove_elem] = False
                
                added_count = 0
                added_elems = []
                for add_elem in range(1, N + 1):
                    if add_elem == remove_elem or in_set[add_elem]:
                        continue
                    if can_add(add_elem, in_set):
                        in_set[add_elem] = True
                        added_count += 1
                        added_elems.append(add_elem)
                
                if added_count >= 2:
                    # Keep the swap
                    swaps_done += 1
                    improved = True
                else:
                    # Revert
                    in_set[remove_elem] = backup
                    for elem in added_elems:
                        in_set[elem] = False
        
        return in_set
    
    # 7. Main: Multi-restart randomized greedy
    best_set = None
    best_size = 0
    
    # First, try a deterministic order (ascending)
    in_set = [False] * (N + 1)
    order = list(range(1, N + 1))
    greedy_construct(order, in_set)
    local_search_add_back(in_set)
    size = sum(in_set[1:])
    if size > best_size:
        best_size = size
        best_set = in_set[:]
    
    # Now do randomized restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Create a random order with some bias
        # Mix: 50% random, 50% sorted by some heuristic
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        in_set = [False] * (N + 1)
        greedy_construct(order, in_set)
        local_search_add_back(in_set)
        size = sum(in_set[1:])
        
        if size > best_size:
            best_size = size
            best_set = in_set[:]
            # Try swaps on the best solution found so far
            if best_size > 0 and time.time() < deadline - 5:
                local_search_swaps(best_set, max_swaps=20)
                new_size = sum(best_set[1:])
                if new_size > best_size:
                    best_size = new_size
        
        # If we've done enough restarts or are running low on time, do final polish
        if restarts >= 500 or time.time() > deadline - 10:
            break
    
    # Final polish on the best solution
    if best_set is not None and time.time() < deadline - 2:
        local_search_add_back(best_set)
        if time.time() < deadline - 2:
            local_search_swaps(best_set, max_swaps=30)
    
    # Construct result
    if best_set is None:
        result = [1]
    else:
        result = [i for i in range(1, N + 1) if best_set[i]]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# We must ensure the EVOLVE-BLOCK structure is respected if required by the harness,
# but the prompt says "Keep the function names and signatures of the seed program".
# The seed program had the function inside the EVOLVE-BLOCK.
# I will place the implementation inside the block to be safe, or just define it.
# The prompt says: "Output the COMPLETE program in a single python code block."
# And "Keep the program's entry-point function name(s) ... EXACTLY as in the current program".
# The current program has `def construct():` inside `# EVOLVE-BLOCK-START`.
# I will structure the output to match this convention.
# EVOLVE-BLOCK-END
