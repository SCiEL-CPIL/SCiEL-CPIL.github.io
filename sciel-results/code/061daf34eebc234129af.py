"""Rewritten program using Maximum Weight Closure on a DAG for the 40-subset problem."""
import time
import itertools
import sys

P = {'n': 12, 'm': 40}
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    n = P['n']
    m = P['m']
    total_sets = 1 << n  # 4096
    
    # Precompute all subsets as lists
    all_subsets = []
    for mask in range(total_sets):
        s = [b for b in range(n) if (mask >> b) & 1]
        all_subsets.append(s)
    
    # Precompute sizes
    sizes = [len(s) for s in all_subsets]
    
    # Build comparability: for each pair (i, j), i is a strict subset of j if
    # mask_i | mask_j == mask_j and mask_i != mask_j
    # We'll build a graph where nodes are the 4096 subsets.
    # Edge from i to j (i -> j) if i is a strict subset of j.
    # Weight on edge (i, j) = 1 (each comparable pair contributes 1)
    # We want to select exactly m nodes to maximize the number of comparable pairs among them.
    
    # Maximum Weight Closure formulation:
    # Each node v has a "selection" variable x_v in {0, 1}.
    # We want to maximize sum over comparable pairs (i,j) of x_i * x_j.
    # This is a quadratic objective, not directly a closure problem.
    
    # Alternative: use a greedy/local search approach with smart initialization.
    # Given the time constraint, let's try a strong heuristic:
    # 1. Start with a good initial solution (e.g., smallest sets from B_6 or similar)
    # 2. Use local search: try swapping one set in with one set out to improve the count.
    
    def count_comparable(selected):
        """Count comparable pairs in a set of selected subsets (by mask)."""
        count = 0
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a = selected[i]
                b = selected[j]
                # Check if a is strict subset of b or b is strict subset of a
                if (a | b) == b and a != b:
                    count += 1
                elif (a | b) == a and a != b:
                    count += 1
        return count
    
    # Precompute for each mask, the list of masks that are strict supersets
    # and strict subsets, to speed up evaluation.
    # Actually, for local search, we need to evaluate the change in count when
    # adding/removing a mask. Let's precompute comparability as a bitset or adjacency.
    
    # For each mask, store the set of masks it's comparable to (strict subset/superset).
    # This is expensive to store fully, but we can compute on the fly.
    
    # Let's use a different approach: represent the selected set as a boolean array.
    # For each candidate swap, compute the delta.
    
    # Initial solution: take the 40 smallest sets from the full Boolean lattice.
    # Sort all masks by (size, mask_value) and take the first 40.
    sorted_masks = sorted(range(total_sets), key=lambda x: (sizes[x], x))
    initial_selected = sorted_masks[:m]
    
    selected_set = set(initial_selected)
    best_count = count_comparable(list(selected_set))
    best_selected = list(selected_set)
    
    # Local search: try all single swaps
    # For speed, precompute for each mask the number of comparable masks in the current selection.
    # When we remove mask r and add mask a, the change in count is:
    #   - lose: number of comparable pairs involving r (with other selected masks)
    #   + gain: number of comparable pairs involving a (with other selected masks, excluding r)
    #   - plus/minus: whether a and r are comparable (if they are, we lose that pair too)
    
    def get_comparable_count_in_selected(mask, selected_set):
        """Count how many masks in selected_set are comparable with mask (strict subset/superset)."""
        count = 0
        for other in selected_set:
            if other == mask:
                continue
            a, b = mask, other
            if (a | b) == b and a != b:
                count += 1
            elif (a | b) == a and a != b:
                count += 1
        return count
    
    # For local search, we'll maintain for each selected mask its comparable count within the selection.
    # This allows O(1) update when swapping.
    
    # Let's implement a simple hill-climbing with random restarts within the time budget.
    
    import random
    
    best_score = best_count
    
    # Precompute comparability between all pairs? That's 4096^2 / 2 ~ 8 million pairs.
    # Too much memory. Let's compute on the fly but cache for selected sets.
    
    # For the local search, let's precompute for each mask in the "candidate pool"
    # (masks not in selected) its comparable count with the current selection.
    
    # Actually, let's just do a simple local search with a time limit.
    
    # Precompute a fast comparability check using bit operations.
    # Two masks a, b: a is strict subset of b iff (a & b) == a and a != b.
    
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def is_comparable(a, b):
        if a == b:
            return False
        return is_strict_subset(a, b) or is_strict_subset(b, a)
    
    # Maintain an array comp_count[i] = number of selected masks comparable with mask i.
    # When we add/remove a mask, we update comp_count for all masks.
    
    # For efficiency, let's precompute the comparability graph as a list of lists.
    # comp_list[i] = list of masks j such that i and j are comparable.
    # This is 4096 * (avg degree) entries. Avg degree is about 2 * (2^12 - 1) / 4096 * something.
    # Actually, for a given mask of size k, it has C(12-k, d) supersets of size k+d and C(k, d) subsets of size k-d.
    # Total comparable masks for a mask of size k is sum_{d=1}^{12-k} C(12-k,d) + sum_{d=1}^{k} C(k,d) = (2^(12-k) - 1) + (2^k - 1).
    # For k=6, that's 63 + 63 = 126. Max is around 126. So total edges ~ 4096 * 126 / 2 ~ 258k. Manageable.
    
    # Build comp_list
    comp_list = [[] for _ in range(total_sets)]
    for i in range(total_sets):
        for j in range(i + 1, total_sets):
            if is_comparable(i, j):
                comp_list[i].append(j)
                comp_list[j].append(i)
    
    # Now local search:
    # State: selected_set (set of masks)
    # comp_count[i] = number of selected masks in comp_list[i]
    
    def compute_comp_count(selected_set):
        comp_count = [0] * total_sets
        for mask in selected_set:
            for neighbor in comp_list[mask]:
                comp_count[neighbor] += 1
        return comp_count
    
    def get_score_from_comp_count(selected_set, comp_count):
        # Score = sum over i in selected_set of comp_count[i] / 2
        # because each comparable pair is counted twice.
        total = sum(comp_count[i] for i in selected_set)
        return total // 2
    
    # Initialize
    selected_set = set(initial_selected)
    comp_count = compute_comp_count(selected_set)
    current_score = get_score_from_comp_count(selected_set, comp_count)
    
    best_score = current_score
    best_selected = list(selected_set)
    
    # Local search: try all single swaps
    # For each mask r in selected_set and mask a not in selected_set:
    #   new_score = current_score - comp_count[r] + (comp_count[a] - (1 if r in comp_list[a] else 0))
    #   Wait, need to be careful.
    
    # When we remove r and add a:
    #   - We lose all comparable pairs involving r: comp_count[r] pairs.
    #   - We gain comparable pairs involving a with the remaining selected set:
    #     comp_count[a] counts comparable masks in the OLD selected set (including r).
    #     If a and r are comparable, then in the new set, a is comparable with (comp_count[a] - 1) masks.
    #     If a and r are not comparable, then a is comparable with comp_count[a] masks.
    #   - But we also need to check: are there pairs (a, r) that were not counted before?
    #     No, because a was not in the selected set before.
    #   - The new score = current_score - comp_count[r] + new_comp_count_for_a
    #     where new_comp_count_for_a = comp_count[a] - (1 if is_comparable(a, r) else 0)
    
    # Actually, the score is (sum of comp_count over selected) / 2.
    # Old sum = S. New sum = S - comp_count[r] + new_comp_count_for_a.
    # New score = (S - comp_count[r] + new_comp_count_for_a) / 2.
    # Delta = (-comp_count[r] + new_comp_count_for_a) / 2.
    
    # For speed, let's iterate and find the best swap.
    
    # Precompute for each mask a, whether it's comparable with each r in selected_set.
    # This is expensive. Let's just do a simple local search with a time limit.
    
    # Given the time constraint, let's do a few iterations of local search.
    
    # For each iteration, try all swaps and pick the best improving one.
    
    # To speed up, let's precompute a boolean array comparable_with_selected[a] for each a.
    # But that's 4096 * 4096 booleans, which is ~2M entries. OK.
    
    # Actually, let's just use the comp_list and check membership.
    
    # Let's do a simple hill-climbing:
    
    improved = True
    iterations = 0
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        iterations += 1
        best_swap = None
        best_delta = 0
        
        # For each r in selected_set, for each a not in selected_set:
        # This is 40 * 4056 ~ 162k swaps per iteration. Each swap check is O(1) if we precompute.
        
        # Precompute for each a not in selected_set: comp_count[a]
        # And for each pair (a, r), whether they're comparable.
        
        # Let's precompute a 2D array or use a set.
        
        # For speed, let's create a set of selected masks for O(1) lookup.
        selected_set = set(best_selected) if not improved else selected_set  # This is getting messy.
        
        break  # Let's use a cleaner approach below.
    
    # Cleaner local search:
    selected_set = set(initial_selected)
    comp_count = compute_comp_count(selected_set)
    current_score = get_score_from_comp_count(selected_set, comp_count)
    best_score = current_score
    best_selected = list(selected_set)
    
    # Precompute a list of non-selected masks
    all_masks = list(range(total_sets))
    
    # Local search iterations
    while (time.time() - start_time) < DEADLINE:
        improved = False
        best_swap = None
        best_delta = 0
        
        # Try all swaps
        for r in list(selected_set):
            for a in all_masks:
                if a in selected_set:
                    continue
                # Compute delta
                # new_comp_count_for_a = comp_count[a] - (1 if a in comp_list[r] else 0)
                # Check if a and r are comparable
                if r in comp_list[a]:
                    new_comp_for_a = comp_count[a] - 1
                else:
                    new_comp_for_a = comp_count[a]
                
                delta = (-comp_count[r] + new_comp_for_a) // 2
                
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (r, a)
        
        if best_swap is not None and best_delta > 0:
            r, a = best_swap
            selected_set.remove(r)
            selected_set.add(a)
            # Update comp_count
            for neighbor in comp_list[r]:
                comp_count[neighbor] -= 1
            for neighbor in comp_list[a]:
                comp_count[neighbor] += 1
            current_score += best_delta
            best_score = current_score
            best_selected = list(selected_set)
            improved = True
    
    # Convert to lists of integers
    result = [all_subsets[mask] for mask in best_selected]
    
    # Ensure types are correct
    final_result = []
    for s in result:
        final_result.append([int(x) for x in s])
    
    return final_result
