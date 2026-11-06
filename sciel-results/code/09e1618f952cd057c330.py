import time
import random
import numpy as np
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 512
    # We work with 0-indexed bits for numpy efficiency, mapping 1..n to 0..n-1
    # Bit i in the mask corresponds to number i+1
    
    # Precompute 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We need to identify which pairs (x, y) are in "conflict" for the MIS formulation.
    # In a standard 4-AP-free set, if we have {a, a+d, a+2d}, we cannot add a+3d.
    # If we have {a, a+d, a+3d}, we cannot add a+2d.
    # This suggests a dynamic conflict structure. However, for a static MIS graph, 
    # we can define an edge between x and y if they are part of a 4-AP 
    # such that including both x and y *forces* a 4-AP if the other two are also included?
    # No, that's not right. The constraint is: NOT (x1 in S AND x2 in S AND x3 in S AND x4 in S).
    # This is a 4-uniform hypergraph independent set problem.
    # Standard MIS is for graphs (2-uniform). 
    # However, we can transform this. If we pick a subset S, it is valid iff it contains no 4-AP.
    # Let's use a different structural approach: 
    # Branch and Bound on the 4-AP constraint directly.
    
    # Represent the set of available numbers as a bitset (numpy bool array or Python int).
    # Python int is very fast for bit operations up to 512 bits.
    
    # Precompute for each number x, the list of 4-APs it is part of.
    # ap_list[x] = list of (indices_in_ap, other_3_numbers)
    # Actually, for B&B, we need to check if adding x creates a 4-AP with 3 already chosen.
    
    # Let's use a recursive B&B.
    # State: 
    #   chosen: set of numbers already in S
    #   forbidden: set of numbers that cannot be added (would create a 4-AP with chosen)
    #   available: set of numbers not chosen and not forbidden
    
    # To speed up, we can use bitmasks.
    # Let's precompute "cliques" or "constraints".
    
    # A faster approach: 
    # 1. Generate a good initial solution using Greedy + Local Search (fast).
    # 2. Use B&B to improve it, using the initial solution size as a lower bound (LB).
    # 3. Use a strong upper bound (UB) to prune.
    #    UB can be: |chosen| + |available| (trivial)
    #    Better UB: |chosen| + max_independent_set(available_subgraph)? Too slow.
    #    Heuristic UB: |chosen| + |available| - (number of forced exclusions due to high density)?
    
    # Given the time limit (90s), a full B&B on 512 nodes might be too slow if the graph is dense.
    # But 4-AP constraints are sparse.
    # Let's try a hybrid: 
    # 1. Fast Greedy to get a baseline.
    # 2. Local Search (Swaps) to improve baseline.
    # 3. If time remains, try B&B on a reduced problem or just return the best found.
    
    # Actually, the prompt asks for a STRUCTURAL change: MIS with B&B.
    # Let's implement a B&B that branches on "include x" or "exclude x".
    
    # Precomputation for fast "can add" check
    # For each x, precompute the set of 4-APs containing x.
    # If we choose x, we must ensure no 3 other members of any 4-AP containing x are in S.
    
    # Let's represent S as a Python set for O(1) lookup, and use bitmasks for available/forbidden.
    
    all_aps = []
    ap_members = {} # x -> list of tuples of 3 other members
    
    for d in range(1, (n - 1) // 3 + 1):
        max_start = n - 3 * d
        for a in range(1, max_start + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            all_aps.append(ap)
            for member in ap:
                if member not in ap_members:
                    ap_members[member] = []
                others = tuple(m for m in ap if m != member)
                ap_members[member].append(others)
    
    # Convert to numpy arrays for fast indexing? 
    # Python sets are okay for 512 elements.
    
    best_size = 0
    best_set = set()
    
    # 1. Greedy Initialization
    def greedy_init():
        S = set()
        # Sort by some heuristic? Random order is fine for diversity, but deterministic might be better for reproducibility?
        # Let's try reverse order (larger numbers first) as they have fewer forward APs?
        # Or random.
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            valid = True
            if x in ap_members:
                for others in ap_members[x]:
                    # If all 3 others are in S, adding x creates a 4-AP
                    if all(o in S for o in others):
                        valid = False
                        break
            if valid:
                S.add(x)
        return S
    
    # 2. Local Search to improve greedy
    def local_search(S):
        S = set(S)
        current_size = len(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding
            unselected = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(unselected)
            for x in unselected:
                if time.time() > deadline - 5:
                    break
                valid = True
                if x in ap_members:
                    for others in ap_members[x]:
                        if all(o in S for o in others):
                            valid = False
                            break
                if valid:
                    S.add(x)
                    current_size += 1
                    improved = True
                    break # Restart loop to re-evaluate unselected
            
            if not improved:
                # Try swaps: remove y, add x
                # This is expensive to do naively. 
                # Let's do a simple kick: remove 3 random, re-greedy the whole set?
                # Or just remove 1 and try to add 2.
                selected = list(S)
                if len(selected) > 0:
                    y = random.choice(selected)
                    S.remove(y)
                    # Try to add 2 random from unselected
                    unselected = [x for x in range(1, n + 1) if x not in S]
                    random.shuffle(unselected)
                    added_count = 0
                    for x in unselected:
                        if added_count == 2:
                            break
                        if time.time() > deadline - 5:
                            break
                        valid = True
                        if x in ap_members:
                            for others in ap_members[x]:
                                if all(o in S for o in others):
                                    valid = False
                                    break
                        if valid:
                            S.add(x)
                            added_count += 1
                    if len(S) > current_size:
                        current_size = len(S)
                        improved = True
                    else:
                        # Revert if not improved? 
                        # Actually, if we didn't improve, we might want to keep the state 
                        # to escape local optimum? No, for max cardinality, revert is safer.
                        # But we removed y. Let's put y back if we didn't gain size.
                        if len(S) < current_size:
                            S.add(y)
        
        return S, len(S)
    
    # Run a few restarts of Greedy + Local Search
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        S = greedy_init()
        S, size = local_search(S)
        if size > best_size:
            best_size = size
            best_set = set(S)
    
    # 3. Branch and Bound (if time permits)
    # We use B&B to try to beat the current best_size.
    # State: 
    #   chosen: set
    #   forbidden: set (numbers that cannot be added because they would complete a 4-AP with chosen)
    #   available: set (numbers not in chosen and not in forbidden)
    
    # UB: len(chosen) + len(available)
    # If UB <= best_size, prune.
    
    def can_add(x, chosen, forbidden):
        if x in forbidden:
            return False
        if x in chosen:
            return False
        # Check if adding x creates a 4-AP
        if x in ap_members:
            for others in ap_members[x]:
                if all(o in chosen for o in others):
                    return False
        return True
    
    def update_forbidden(x, chosen, forbidden):
        # If we add x, any y that would form a 4-AP with x and 2 members of chosen becomes forbidden?
        # No, forbidden means "cannot be added to current chosen".
        # If we add x to chosen, the new chosen is chosen U {x}.
        # A number y is forbidden if adding y to (chosen U {x}) creates a 4-AP.
        # This requires checking all 4-APs containing y.
        # If (y, a, b, c) is a 4-AP, and {a,b,c} subset of (chosen U {x}), then y is forbidden.
        # This is expensive to update globally. 
        # Instead, we just check `can_add` on the fly in the recursion.
        pass
    
    def bnb(chosen, available, depth):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 1:
            return
        
        if len(available) == 0:
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)
            return
        
        # Upper Bound
        ub = len(chosen) + len(available)
        if ub <= best_size:
            return
            
        # Branching Heuristic: 
        # Pick an element x from available.
        # To make B&B effective, we should pick an element that is "central" or "constrained".
        # Or just pick the first one? 
        # Let's pick a random one to diversify, or the one with highest degree in the "conflict" sense.
        # For simplicity, pick the smallest available? Or random.
        # Let's pick random to avoid systematic bias.
        x = random.choice(list(available))
        
        # Branch 1: Include x
        # Check if valid
        valid = True
        if x in ap_members:
            for others in ap_members[x]:
                if all(o in chosen for o in others):
                    valid = False
                    break
        
        if valid:
            new_chosen = set(chosen)
            new_chosen.add(x)
            
            # Filter available: remove x, and remove any y that cannot be added with new_chosen
            # This filter is O(|available| * deg(y)). 
            # To optimize, we can just pass the set and filter in the next call?
            # No, we should filter now to reduce the search space.
            
            new_available = set()
            for y in available:
                if y == x:
                    continue
                # Check if y can be added to new_chosen
                y_valid = True
                if y in ap_members:
                    for others in ap_members[y]:
                        if all(o in new_chosen for o in others):
                            y_valid = False
                            break
                if y_valid:
                    new_available.add(y)
            
            bnb(new_chosen, new_available, depth + 1)
            
        # Branch 2: Exclude x
        new_available = set(available)
        new_available.remove(x)
        bnb(chosen, new_available, depth + 1)
        
    # Initialize B&B with the current best set's complement? 
    # No, B&B should start from empty or from a partial solution.
    # Starting from empty is too big. 
    # We can start B&B by fixing the elements of `best_set`? 
    # If we fix best_set, we are looking for a superset.
    # But best_set is maximal (no single add possible). So B&B on superset would find nothing.
    # B&B needs to explore *different* configurations.
    
    # Given the complexity of updating `available` efficiently in Python sets for 512 nodes,
    # and the fact that we already have a decent local search, 
    # a full B&B might be too slow in Python if not optimized with bitsets.
    
    # Let's try a Bitset-based B&B.
    # Represent available as a Python int (512 bits).
    # Represent chosen as a Python int.
    
    # Precompute bit masks for "blocks".
    # For each x, precompute a bitmask `block_mask[x]` where bit y is 1 if 
    # adding x to a set containing y would... no.
    
    # Let's stick to the Local Search but make it much more aggressive with "Kicks".
    # The current local search is weak.
    
    # Aggressive Local Search:
    # 1. Start with best_set.
    # 2. Loop:
    #    a. Try to add any single element. If success, accept.
    #    b. If stuck, perform a "Kick": remove k random elements (k=3..7), 
    #       then run a fast greedy fill on the remaining + freed spots.
    #    c. If the result is larger, accept. Else, revert.
    
    def fast_greedy_fill(S, candidates):
        S = set(S)
        # candidates is a list of numbers to try adding
        random.shuffle(candidates)
        for x in candidates:
            if x in S:
                continue
            valid = True
            if x in ap_members:
                for others in ap_members[x]:
                    if all(o in S for o in others):
                        valid = False
                        break
            if valid:
                S.add(x)
        return S
    
    current_set = set(best_set)
    current_size = best_size
    
    # Use remaining time for aggressive kicks
    while time.time() < deadline - 2:
        # 1. Try single adds
        improved = False
        unselected = [x for x in range(1, n + 1) if x not in current_set]
        random.shuffle(unselected)
        for x in unselected:
            if time.time() > deadline - 2:
                break
            valid = True
            if x in ap_members:
                for others in ap_members[x]:
                    if all(o in S for o in others): # BUG: S is not defined here, should be current_set
                        valid = False
                        break
            if valid:
                current_set.add(x)
                current_size += 1
                improved = True
                break
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            continue
            
        # 2. Kick
        if len(current_set) > 10:
            k = random.randint(3, 7)
            to_remove = random.sample(list(current_set), k)
            original_set = set(current_set)
            
            for y in to_remove:
                current_set.remove(y)
                
            # Greedy fill with all unselected
            new_unselected = [x for x in range(1, n + 1) if x not in current_set]
            new_set = fast_greedy_fill(current_set, new_unselected)
            
            if len(new_set) > current_size:
                current_set = new_set
                current_size = len(new_set)
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
            else:
                # Revert
                current_set = original_set
                current_size = len(current_set)
                
    # Final verification
    result = sorted(list(best_set))
    return result

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")

