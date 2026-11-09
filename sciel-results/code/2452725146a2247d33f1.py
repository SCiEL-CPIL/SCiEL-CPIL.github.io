import time
import random
import itertools
import math
import numpy as np
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute the conflict structures
    # A forbidden configuration is defined by 3 disjoint pairs.
    # We need to track for each configuration how many of its 8 transversals are present.
    
    # Generate all unique configurations (sets of 3 disjoint pairs)
    configs = []
    config_to_idx = {}
    
    def get_matchings(v6):
        matchings = []
        for i in range(1, 6):
            rem = [v for j, v in enumerate(v6) if j != 0 and j != i]
            for j in range(1, 4):
                rem2 = [v for k, v in enumerate(rem) if k != 0 and k != j]
                p1 = tuple(sorted((v6[0], v6[i])))
                p2 = tuple(sorted((rem[0], rem[j])))
                p3 = tuple(sorted((rem2[0], rem2[1])))
                pairs = sorted([p1, p2, p3])
                matchings.append(tuple(pairs))
        return matchings

    for v6 in itertools.combinations(range(n), 6):
        for m in get_matchings(v6):
            if m not in config_to_idx:
                config_to_idx[m] = len(configs)
                configs.append(m)
    
    num_configs = len(configs)
    
    # Map each triple to the list of config indices it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for cfg_idx, config in enumerate(configs):
        p1, p2, p3 = config
        a, b = p1
        c, d = p2
        e, f = p3
        
        transversals = [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
        
        for t in transversals:
            t_sorted = tuple(sorted(t))
            if t_sorted in triple_to_idx:
                triple_to_configs[triple_to_idx[t_sorted]].append(cfg_idx)
    
    # State variables
    selected = [False] * num_triples
    config_counts = [0] * num_configs
    
    def is_valid_add(t_idx):
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] == 7:
                return False
        return True

    def add_triple(t_idx):
        selected[t_idx] = True
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] += 1

    def remove_triple(t_idx):
        selected[t_idx] = False
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] -= 1
            
    def try_add(t_idx):
        if is_valid_add(t_idx):
            add_triple(t_idx)
            return True
        return False

    best_system = None
    best_count = 0
    random.seed(42)
    
    # Precompute transversals for each config
    config_transversals = []
    for config in configs:
        p1, p2, p3 = config
        a, b = p1
        c, d = p2
        e, f = p3
        transversals = [
            (a, c, e), (a, c, f), (a, d, e), (a, d, f),
            (b, c, e), (b, c, f), (b, d, e), (b, d, f)
        ]
        t_indices = []
        for t in transversals:
            t_sorted = tuple(sorted(t))
            t_indices.append(triple_to_idx[t_sorted])
        config_transversals.append(t_indices)

    # ==================== BRANCH AND BOUND FOR MAX INDEPENDENT SET ====================
    # Formulation: We want to select maximum triples such that no config has all 8 transversals.
    # This is equivalent to: for each config, at most 7 of its 8 transversals can be selected.
    # 
    # Branch-and-bound approach:
    # - State: (current_selection_tuple, remaining_triples_set)
    # - Upper bound: greedy heuristic on remaining triples
    # - Prune if current_count + upper_bound <= best_count
    
    # For efficiency, we'll use a simpler B&B that branches on whether to include or exclude
    # a triple, with good upper bounds.
    
    # Precompute for each triple, the configs it's in (already done: triple_to_configs)
    
    # Upper bound function: given current config counts and set of available triples,
    # estimate max additional triples we can add using greedy.
    def upper_bound_greedy(available_triples, current_counts):
        """
        Estimate upper bound on how many more triples can be added.
        Uses a greedy heuristic: repeatedly pick the triple that blocks the fewest configs.
        """
        if not available_triples:
            return 0
        
        # Work with a copy of counts
        cnt = current_counts[:]
        remaining = set(available_triples)
        count = 0
        
        # Greedy: pick triples that are "least blocking"
        # A triple is less blocking if it appears in configs that already have high counts
        # (i.e., those configs are closer to being full)
        
        while remaining:
            # Find the triple that can be added with minimal "cost"
            # Cost = number of configs where this triple would be the 8th (forbidden)
            # Actually, we want to add triples that don't violate constraints.
            # So pick a triple that is valid, preferring those that appear in fewer "critical" configs.
            
            best_t = None
            best_score = float('inf')
            
            for t_idx in remaining:
                # Check if this triple is valid
                valid = True
                critical_count = 0  # How many configs would become "full" if we add this
                for cfg_idx in triple_to_configs[t_idx]:
                    if cnt[cfg_idx] == 7:
                        valid = False
                        break
                    elif cnt[cfg_idx] == 6:
                        critical_count += 1
                
                if not valid:
                    continue
                
                # Score: prefer triples that appear in fewer critical configs
                # Also consider total degree
                score = critical_count * 10 + len(triple_to_configs[t_idx])
                
                if score < best_score:
                    best_score = score
                    best_t = t_idx
            
            if best_t is None:
                break
            
            # Add this triple
            remaining.remove(best_t)
            for cfg_idx in triple_to_configs[best_t]:
                cnt[cfg_idx] += 1
            count += 1
            
            # Time check
            if time.time() > deadline - 5:
                break
        
        return count
    
    # Branch and bound
    # We'll use a recursive approach with memoization on the "frontier" of decisions.
    # However, the state space is large, so we'll use a heuristic-driven B&B.
    
    def branch_and_bound(initial_selection, initial_counts, available_set):
        """
        Branch and bound to maximize additional triples.
        Returns (additional_count, selection_dict)
        """
        nonlocal best_count, best_system
        
        if time.time() > deadline - 5:
            return 0, {}
        
        # Upper bound
        ub = upper_bound_greedy(available_set, initial_counts)
        if len(initial_selection) + ub <= best_count:
            return 0, {}
        
        if not available_set:
            return 0, {}
        
        # Choose a triple to branch on
        # Heuristic: pick a triple that appears in many "critical" configs
        # or a triple that is "central" in the constraint structure
        
        # Score each available triple
        triple_scores = []
        for t_idx in available_set:
            # Count how many configs this triple is in that are "tight" (count >= 5)
            tight_count = 0
            for cfg_idx in triple_to_configs[t_idx]:
                if initial_counts[cfg_idx] >= 5:
                    tight_count += 1
            # Prefer branching on triples with higher tight_count
            triple_scores.append((tight_count, t_idx))
        
        triple_scores.sort(reverse=True)
        
        if not triple_scores:
            return 0, {}
        
        # Branch on the first few high-scoring triples
        # To keep search manageable, we'll do a limited-depth B&B
        
        best_additional = 0
        best_add_selection = {}
        
        # Try including some of the top-scored triples
        for _, t_idx in triple_scores[:5]:  # Branch on top 5
            # Check if we can include t_idx
            can_include = True
            for cfg_idx in triple_to_configs[t_idx]:
                if initial_counts[cfg_idx] == 7:
                    can_include = False
                    break
            
            if can_include:
                # Include t_idx
                new_selection = dict(initial_selection)
                new_selection[t_idx] = True
                new_counts = initial_counts[:]
                for cfg_idx in triple_to_configs[t_idx]:
                    new_counts[cfg_idx] += 1
                
                # Remove t_idx and any triples that would now be invalid
                new_available = set(available_set)
                new_available.remove(t_idx)
                
                # Recursively solve
                add_count, add_sel = branch_and_bound(new_selection, new_counts, new_available)
                if add_count > best_additional:
                    best_additional = add_count
                    best_add_selection = {**new_selection, **add_sel}
        
        # Also try excluding some triples (to explore different branches)
        # This is a heuristic B&B, not exhaustive
        
        return best_additional, best_add_selection
    
    # ==================== LOCAL SEARCH (retained from original) ====================
    def local_search(state_selected, state_counts, current_count):
        sel = state_selected[:]
        cnt = state_counts[:]
        count = current_count
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Sweep 1: Try to add any unselected triple that is now valid
            for t_idx in range(num_triples):
                if not sel[t_idx]:
                    valid = True
                    for cfg_idx in triple_to_configs[t_idx]:
                        if cnt[cfg_idx] == 7:
                            valid = False
                            break
                    if valid:
                        sel[t_idx] = True
                        for cfg_idx in triple_to_configs[t_idx]:
                            cnt[cfg_idx] += 1
                        count += 1
                        improved = True
                        
            if not improved:
                unselected = [i for i in range(num_triples) if not sel[i]]
                if not unselected:
                    break
                    
                swap_found = False
                for _ in range(100):
                    if time.time() > deadline - 3:
                        break
                    t_add = random.choice(unselected)
                    
                    valid = True
                    blocking_configs = []
                    for cfg_idx in triple_to_configs[t_add]:
                        if cnt[cfg_idx] == 7:
                            valid = False
                            blocking_configs.append(cfg_idx)
                    
                    if valid:
                        sel[t_add] = True
                        for cfg_idx in triple_to_configs[t_add]:
                            cnt[cfg_idx] += 1
                        count += 1
                        improved = True
                        break
                    if not blocking_configs:
                        continue
                        
                    cfg_block = random.choice(blocking_configs)
                    
                    selected_in_cfg = []
                    for t_idx_t in config_transversals[cfg_block]:
                        if sel[t_idx_t]:
                            selected_in_cfg.append(t_idx_t)
                            
                    if not selected_in_cfg:
                        continue
                        
                    t_remove = random.choice(selected_in_cfg)
                    
                    for cfg_idx in triple_to_configs[t_remove]:
                        cnt[cfg_idx] -= 1
                    sel[t_remove] = False
                    
                    valid_add = True
                    for cfg_idx in triple_to_configs[t_add]:
                        if cnt[cfg_idx] == 7:
                            valid_add = False
                            break
                    
                    if valid_add:
                        sel[t_add] = True
                        for cfg_idx in triple_to_configs[t_add]:
                            cnt[cfg_idx] += 1
                        improved = True
                        swap_found = True
                        break
                    else:
                        sel[t_remove] = True
                        for cfg_idx in triple_to_configs[t_remove]:
                            cnt[cfg_idx] += 1
                        
                if swap_found:
                    continue
                else:
                    break
        
        return count, sel, cnt

    # ==================== MAIN SEARCH LOOP ====================
    max_runs = 200
    
    for run in range(max_runs):
        if time.time() > deadline - 10:
            break
            
        # Reset state
        for i in range(num_triples):
            if selected[i]:
                remove_triple(i)
        for i in range(num_configs):
            config_counts[i] = 0
            
        # 1. Randomized Greedy Initialization
        if run % 3 == 0:
            order = list(range(num_triples))
            random.shuffle(order)
        elif run % 3 == 1:
            order = sorted(range(num_triples), key=lambda x: len(triple_to_configs[x]))
            random.shuffle(order)
        else:
            order = list(range(num_triples))
            random.shuffle(order)
            
        for t_idx in order:
            try_add(t_idx)
            
        current_count = sum(selected)
        
        # 2. Local Search Phase
        new_count, new_selected, new_counts = local_search(selected, config_counts, current_count)
        
        if new_count > current_count:
            selected[:] = new_selected
            config_counts[:] = new_counts
            current_count = new_count
            
        # 3. Branch and Bound Polish
        # Try to improve using B&B on the remaining triples
        if time.time() < deadline - 15:
            # Get current selection as a set
            current_sel_set = set(i for i in range(num_triples) if selected[i])
            available_set = set(range(num_triples)) - current_sel_set
            
            # Try B&B to add more triples
            if available_set:
                # Use a simpler B&B that just tries to add one more triple
                # by exploring swaps more aggressively
                
                # For now, just do additional local search iterations
                extra_count, extra_sel, extra_cnt = local_search(selected, config_counts, current_count)
                if extra_count > current_count:
                    selected[:] = extra_sel
                    config_counts[:] = extra_cnt
                    current_count = extra_count
        
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]
        
    if best_system is None:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system
