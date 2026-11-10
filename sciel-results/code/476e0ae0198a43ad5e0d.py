import time
import random

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22
    lcm_all = 232792560
    recip_int = {i: lcm_all // i for i in range(1, 23)}

    # Precompute all subset sums for the full set as a sorted tuple for fast membership
    # Actually, let's just build on the fly but use a set.
    # To speed up "can_add", we maintain the set of subset sums.

    def can_add(e, cur_sums):
        v = recip_int[e]
        # We need to check if {s + v for s in cur_sums} intersects cur_sums
        # This is equivalent to: exists s, t in cur_sums such that s + v = t => t - s = v
        # So v must not be in {t - s for t, s in cur_sums, t != s}
        # But checking all pairs is O(N^2).
        # The set approach is O(N).
        for s in cur_sums:
            if (s + v) in cur_sums:
                return False
        return True

    def add_element(e, cur_sums):
        v = recip_int[e]
        return cur_sums | {s + v for s in cur_sums}

    # Greedy build with a specific order
    def greedy_build(ordering, cur_sums=None):
        if cur_sums is None:
            cur_sums = {0}
        S = []
        for e in ordering:
            if time.time() > deadline:
                break
            if can_add(e, cur_sums):
                S.append(e)
                cur_sums = add_element(e, cur_sums)
        return S, cur_sums

    # Start with a very good initial seed.
    # Known good sets often contain large numbers.
    # Let's try to find a large set quickly.
    
    # Strategy 1: Randomized Greedy with Restart
    # We want to maximize size.
    # A key insight: "Dissociated" sets over integers (where reciprocals are mapped to integers)
    # are related to Sidon sets or B_2 sets in some contexts, but here it's about sum-free subsets.
    # Actually, if S is dissociated, no element is in the span of the others with coefficients in {-1, 0, 1}.
    
    # Let's use a smarter heuristic for ordering.
    # Smaller reciprocals (larger i) are "smaller" integers.
    # Adding large integers first might be better because they create larger gaps?
    # Or small integers first?
    
    # Let's try multiple random orders and keep the best.
    best_set = [1]
    best_sums = {0, lcm_all} # sums for {1}
    
    # Initial quick pass: Descending order (22, 21, ...)
    S_init, sums_init = greedy_build(list(range(22, 0, -1)))
    if len(S_init) > len(best_set):
        best_set = S_init
        best_sums = sums_init

    # Local Search: Swap
    # We will try to swap elements in best_set with elements outside to see if we can
    # then add more, or if the swap itself allows a larger set.
    # Actually, a simple swap that keeps size same doesn't help directly unless it enables expansion.
    
    # Better Strategy: Simulated Annealing / Random Local Search
    # Objective: Maximize len(S) subject to S being dissociated.
    # Moves:
    # 1. Add an element (if valid) -> always accept (increases size)
    # 2. Remove an element (if valid) -> accept with low prob (to escape local opt)
    # 3. Swap an element in S with one not in S -> accept if it allows subsequent adds or with prob.
    
    # Given the 60s limit, we can do many iterations.
    
    current_set = list(best_set)
    current_sums = set(best_sums)
    
    # Helper to check if a set is valid
    def is_valid(S):
        # Rebuild sums to be safe? No, we maintain invariants.
        # But if we do random swaps, we must update sums correctly.
        pass

    # Let's refine the search loop.
    # We will maintain `current_set` and `current_sums`.
    
    # To make swaps efficient, we need to remove an element from current_sums.
    # This is harder.
    # Removal from a subset-sum set:
    # If S = A ∪ {e}, then Sums(S) = Sums(A) ∪ (Sums(A) + v_e).
    # To remove e, we need to identify which sums are unique to e.
    # A sum x is in Sums(A) iff x can be formed without e.
    # This is effectively solving the subset sum problem for A.
    # However, since N is small (<=22), we can rebuild sums from scratch after a swap.
    # Rebuild cost: O(2^k * k). For k=10, 1000*10 = 10k ops. Very fast.
    # So we can afford to rebuild sums after every move.
    
    def build_sums_from_set(S):
        sums = {0}
        for e in S:
            v = recip_int[e]
            sums = sums | {s + v for s in sums}
        return sums

    # Restart the search with a robust local search
    best_size = len(best_set)
    best_result = list(best_set)
    
    # Randomized Local Search
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Initialization
        if restart == 0:
            cur_set = list(best_result)
        else:
            # Start from a random subset or random permutation greedy
            # Sometimes restarting from scratch is better
            if random.random() < 0.5:
                # Random greedy
                order = list(range(1, 23))
                random.shuffle(order)
                cur_set, _ = greedy_build(order)
            else:
                # Perturb best
                cur_set = list(best_result)
                # Remove 1-2 random elements
                for _ in range(random.randint(1, 2)):
                    if cur_set:
                        idx = random.randint(0, len(cur_set)-1)
                        cur_set.pop(idx)
        
        cur_sums = build_sums_from_set(cur_set)
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try all possible additions
            remaining = [e for e in range(1, 23) if e not in cur_set]
            for e in remaining:
                if time.time() > deadline - 5:
                    break
                if can_add(e, cur_sums):
                    cur_set.append(e)
                    cur_sums = add_element(e, cur_sums)
                    improved = True
                    # Don't break, keep trying to add more if possible?
                    # can_add checks against current cur_sums which is updated.
                    # So the loop continues and checks the rest of remaining.
                    # But note: `remaining` is a fixed list. If we added an element,
                    # the condition for other elements might change.
                    # Since we are in a loop over `remaining`, and we update `cur_sums`,
                    # the check `can_add` is always fresh.
                    # However, if we add e, it might prevent adding f.
                    # This is a greedy expansion. It's fine.
            
            # If we improved, check if it's new best
            if len(cur_set) > best_size:
                best_size = len(cur_set)
                best_result = list(cur_set)
            
            # If no addition was possible, try a swap
            if not improved:
                # Try swapping one element out and one in
                # This might allow further expansion.
                swapped = False
                # We can't just pick random ones.
                # Let's try removing each element and see if we can add two new ones?
                # Or just 1 in, 1 out to change the structure?
                
                # A simpler move: Remove one element, then try to expand.
                # This allows "escaping" a local optimum where no single addition works,
                # but removing a "blocking" element allows multiple additions.
                
                # Try removing each element in cur_set
                for i in range(len(cur_set)):
                    if time.time() > deadline - 5:
                        break
                    
                    e_remove = cur_set[i]
                    new_set = cur_set[:i] + cur_set[i+1:]
                    new_sums = build_sums_from_set(new_set)
                    
                    # Try to expand new_set
                    old_size = len(cur_set)
                    improved_set, improved_sums = greedy_build([e for e in range(1, 23) if e not in new_set], new_sums)
                    
                    # Note: greedy_build returns (S, sums) where S is the list of added elements.
                    # We need to combine new_set + improved_set.
                    final_set = new_set + improved_set
                    final_sums = build_sums_from_set(final_set)
                    
                    if len(final_set) > len(cur_set):
                        cur_set = final_set
                        cur_sums = final_sums
                        improved = True
                        if len(cur_set) > best_size:
                            best_size = len(cur_set)
                            best_result = list(cur_set)
                        # After a successful "remove and re-expand", we should restart the inner while loop
                        # because the set has changed significantly.
                        break # Break the "for i" loop, go back to "while improved"
                
                # If no removal+expansion worked, try a pure swap (1 out, 1 in) to maintain size
                # but hopefully change the "shape" to allow future additions.
                # This is less likely to help immediately but might help later.
                # Given time, maybe skip pure swaps unless stuck for a long time.
                # For now, if no improvement, we accept the current local optimum for this restart.
                # The outer loop will try a new restart/perturbation.

    # Final check and return
    if not best_result:
        best_result = [1]
        
    # Verify best_result is valid
    if not is_dissociated(best_result):
        # Fallback
        best_result = [1]

    return sorted(best_result)

def is_dissociated(S):
    """Check if S has distinct reciprocal subset sums."""
    # Re-implement here to be self-contained in the verification, 
    # or use the helper from construct? 
    # The function is_dissociated is not defined in the global scope in the previous snippet, 
    # but I used it in the final check. I need to define it or inline it.
    # Let's inline it to avoid dependency issues if I defined it inside construct.
    lcm_all = 232792560
    recip_int = {i: lcm_all // i for i in range(1, 23)}
    sums = {0}
    for e in S:
        v = recip_int[e]
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True
