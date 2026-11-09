"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm, gcd
    from itertools import combinations
    
    n = 256
    numbers = list(range(2, n + 1))
    num_count = len(numbers)
    
    # Build incompatibility: conflict[i][j] = True if lcm(numbers[i], numbers[j]) <= 256
    # Use adjacency list for efficiency
    conflict = [set() for _ in range(num_count)]
    for i in range(num_count):
        for j in range(i + 1, num_count):
            a, b = numbers[i], numbers[j]
            if lcm(a, b) <= n:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Weights: 1/x
    weights = [1.0 / numbers[i] for i in range(num_count)]

    def solve_subproblem(forced_in, forbidden):
        """
        Solve MWIS on the graph induced by nodes not in forbidden, 
        with nodes in forced_in already selected.
        forbidden: set of indices that cannot be selected (conflicts with forced_in)
        forced_in: set of indices already selected
        """
        available = [i for i in range(num_count) if i not in forbidden and i not in forced_in]
        
        # Greedy initialization on available nodes
        # Sort by weight descending
        available_sorted = sorted(available, key=lambda i: -weights[i])
        
        selected = set(forced_in)
        
        for i in available_sorted:
            if not (conflict[i] & selected):
                selected.add(i)
        
        # Local search on the full selected set
        def total_weight(sel):
            return sum(weights[i] for i in sel)
        
        current_weight = total_weight(selected)
        
        improved = True
        # Limit iterations to ensure speed, but allow enough for convergence
        max_iters = 50
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try adding a single unselected vertex
            for i in available:
                if i not in selected:
                    if not (conflict[i] & selected):
                        new_w = current_weight + weights[i]
                        if new_w > current_weight + 1e-15:
                            selected.add(i)
                            current_weight = new_w
                            improved = True
            
            # Try swap: remove one selected (from available part, not forced), add one unselected
            # We should not remove forced_in nodes
            removable = [i for i in selected if i not in forced_in]
            for i in removable:
                temp_selected = selected - {i}
                best_gain = 0
                best_add = None
                for j in available:
                    if j not in temp_selected:
                        if not (conflict[j] & temp_selected):
                            gain = weights[j] - weights[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_add = j
                if best_add is not None and best_gain > 1e-15:
                    selected.discard(i)
                    selected.add(best_add)
                    current_weight += best_gain
                    improved = True

        return selected, current_weight

    best_result = set()
    best_score = 0.0

    # Case 1: No forced nodes
    sel, score = solve_subproblem(set(), set())
    if score > best_score:
        best_score = score
        best_result = sel

    # Case 2: One forced node
    # We only need to check small numbers as forcing large numbers is less likely to change the core structure significantly 
    # but might be optimal if the number itself is very large? No, large numbers have small weight.
    # However, forcing a small number excludes many conflicts, potentially allowing a better tail.
    # Let's check all nodes, but prioritize small ones.
    for i in range(num_count):
        # If i is already in best_result, skip? No, best_result might be suboptimal for this specific forced set.
        # But if we force i, we must exclude its conflicts.
        forbidden = conflict[i]
        forced = {i}
        sel, score = solve_subproblem(forced, forbidden)
        if score > best_score:
            best_score = score
            best_result = sel

    # Case 3: Two forced nodes (compatible pairs)
    # This is O(N^2) * solve_time. N=255. 255^2 ~ 65000. 
    # Solve time is small if we optimize. 
    # To be safe, we can restrict pairs to small numbers or just run if time permits.
    # Given 60s limit, and Python overhead, 65000 * (greedy+local) might be too slow.
    # Let's restrict pairs to indices < 20 (numbers 2..21) or similar.
    # Small numbers have the most impact on the conflict graph structure.
    small_indices = list(range(20)) 
    
    for i in small_indices:
        for j in small_indices:
            if i >= j:
                continue
            # Check if i and j are compatible
            if j in conflict[i]:
                continue
            
            forbidden = conflict[i] | conflict[j]
            forced = {i, j}
            sel, score = solve_subproblem(forced, forbidden)
            if score > best_score:
                best_score = score
                best_result = sel

    result = [numbers[i] for i in best_result]
    result.sort()
    return result
# EVOLVE-BLOCK-END
