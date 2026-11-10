"""Rewritten program: greedy + local search with swaps and deadline guard."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 192
K = 3
MAX_SIZE = 48
TIME_LIMIT = 50.0  # safety margin before 60s limit


def _valid_candidate(x, selected_set, selected_list):
    """Check if x can be added to selected without violating the constraint."""
    n_sel = len(selected_list)
    if n_sel < K:
        # If there are fewer than 3 other elements, no sum of 3 distinct others exists yet for x?
        # Wait, if n_sel < 3, we can't form a sum of 3 distinct OTHERS for x.
        # But for existing a, we need to check if a divides sum of 3 from (selected - {a}) + {x}.
        # If n_sel == 2, selected - {a} has 1 element, + x makes 2. Can't form 3.
        # If n_sel == 1, selected - {a} has 0, + x makes 1. Can't form 3.
        # So if n_sel < 3, x is always valid? 
        # Let's check: sum of 3 distinct OTHERS.
        # If I add x to a set of size m, the new set has size m+1.
        # For x: others are m elements. If m < 3, no triple. Valid.
        # For a in selected: others are (m-1) elements from selected + 1 (x) = m elements.
        # If m < 3, no triple. Valid.
        return True

    # Check x: x must not divide sum of any 3 distinct elements from selected_list
    # Precompute sums of pairs to speed up? Or just iterate.
    # Number of triples is C(m, 3).
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            for k in range(j + 1, n_sel):
                if (selected_list[i] + selected_list[j] + selected_list[k]) % x == 0:
                    return False
    
    # Check each existing a: a must not divide sum of any 3 distinct elements from (selected - {a}) + {x}
    # The new triples involving x are: (x, u, v) where u,v in selected, u != a, v != a, u != v
    for i, a in enumerate(selected_list):
        # We need to check triples (x, u, v) where u, v are in selected_list, u != v, u != a, v != a.
        # This is equivalent to choosing 2 distinct elements from selected_list excluding a.
        # Let's iterate over pairs (u, v) from selected_list, ensuring neither is a.
        for j in range(n_sel):
            if j == i:
                continue
            u = selected_list[j]
            for k in range(j + 1, n_sel):
                if k == i:
                    continue
                v = selected_list[k]
                if (x + u + v) % a == 0:
                    return False
    return True


def _is_valid(selected_list):
    """Full validation of a set."""
    n = len(selected_list)
    for i, a in enumerate(selected_list):
        others = [v for j, v in enumerate(selected_list) if j != i]
        for t in combinations(others, K):
            if sum(t) % a == 0:
                return False
    return True


def _greedy_construct():
    """Greedy construction from large to small."""
    selected = []
    selected_set = set()
    for x in range(N, 0, -1):
        if _valid_candidate(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected


def _local_search(initial, deadline):
    """Local search with swaps to improve set size."""
    best = list(initial)
    best_set = set(best)
    
    while time.time() < deadline:
        improved = False
        n_best = len(best)
        
        # Try to add any missing element
        # Iterate in reverse order to prefer larger numbers? Or random?
        # Original used reverse. Let's stick to it but check time.
        for x in range(N, 0, -1):
            if x in best_set:
                continue
            if _valid_candidate(x, best_set, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        # We iterate over indices to remove.
        for i in range(n_best - 1, -1, -1):
            if time.time() >= deadline:
                return best
            
            a = best[i]
            # Try to find an element x not in best that can replace a
            # To speed up, we can pre-check x against the rest of the set excluding a.
            # We will reuse the logic from _valid_candidate but adapted for swap.
            
            # Create a list of others (best without a)
            # Instead of creating a new list every time, we can just iterate carefully.
            # But _valid_candidate expects a list.
            # Let's just call a helper that checks swap validity efficiently.
            
            # Optimization: Check if x is "compatible" with best - {a}
            # 1. x must not divide sum of any 3 from best - {a}
            # 2. Each b in best - {a} must not divide sum of any 3 from (best - {a} - {b}) + {x}
            
            # Let's try candidates x in descending order
            for x in range(N, 0, -1):
                if x in best_set:
                    continue
                
                ok = True
                others_count = n_best - 1
                
                # Check 1: x vs triples in others
                # If others_count < 3, skip
                if others_count >= 3:
                    # Iterate triples in best excluding index i
                    # This is expensive if done naively.
                    # Let's just use the existing _valid_candidate logic by constructing the hypothetical set?
                    # No, that's too slow.
                    
                    # We can optimize by checking x against sums of 3 from best excluding a.
                    # We can precompute pair sums of best?
                    # For now, let's do a direct check but break early.
                    
                    # To avoid O(N^3) inside O(N^2) loop, we need to be careful.
                    # However, N=192, SetSize ~48.
                    # Swaps are tried only when no addition is possible.
                    # Number of swaps tried per iteration is limited by time.
                    
                    # Let's use a slightly faster inline check for x vs others
                    # We need to check if any sum of 3 distinct elements from others is divisible by x.
                    # We can iterate pairs (u, v) from others and check (u+v+w) % x == 0 for w in others?
                    # No, that's O(N^3).
                    
                    # Let's just rely on _valid_candidate for the *new* element x against the *old* set (minus a)?
                    # _valid_candidate(x, set_others, list_others) checks:
                    # 1. x vs triples in list_others.
                    # 2. Each a' in list_others vs triples in (list_others - {a'}) + {x}.
                    # This is exactly what we need!
                    
                    # So we can just call _valid_candidate(x, best_set_without_a, best_without_a_list)
                    # But we don't have best_without_a_list readily available.
                    # Constructing it is O(N).
                    
                    # Let's construct it.
                    others_list = [best[j] for j in range(n_best) if j != i]
                    others_set = best_set - {a}
                    
                    if _valid_candidate(x, others_set, others_list):
                        # Swap is valid
                        best[i] = x
                        best_set.discard(a)
                        best_set.add(x)
                        improved = True
                        break
                
                if time.time() >= deadline:
                    return best
            
            if improved:
                break
        
        # If no improvement, try removing an element that's causing issues
        if not improved:
            # Try removing the smallest element to potentially allow adding larger ones
            if len(best) > 1:
                # Find index of min
                min_val = min(best)
                idx = best.index(min_val)
                best.pop(idx)
                best_set.discard(min_val)
                # Don't mark as improved to allow trying other removals or additions in next loop
                # But we need to avoid infinite loop of removing same element if it can't be added back.
                # Actually, if we remove it, we might add a different one.
                # To prevent oscillation, we can keep a record of removed elements?
                # Simple heuristic: just remove and continue.
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Phase 1: Greedy construction
    selected = _greedy_construct()
    
    # Phase 2: Local search
    if time.time() < deadline - 2.0:
        selected = _local_search(selected, deadline)
    
    # Ensure validity
    if not _is_valid(selected):
        # Fallback: rebuild greedily
        selected = _greedy_construct()
    
    return selected


# EVOLVE-BLOCK-END
