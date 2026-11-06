"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def is_sidon(state):
        """Check if state is a Sidon set."""
        sums = set()
        n = len(state)
        for i in range(n):
            a = state[i]
            if 2 * a in sums:
                return False
            sums.add(2 * a)
            for j in range(i + 1, n):
                b = state[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_sums(state):
        """Compute the set of all sums a+b for a<=b in state."""
        sums = set()
        n = len(state)
        for i in range(n):
            a = state[i]
            sums.add(2 * a)
            for j in range(i + 1, n):
                sums.add(state[i] + state[j])
        return sums

    def is_maximal(state, sums):
        """Check if state is inclusion-maximal.
        A state is maximal if no element x in [1, N] \\ state can be added 
        without creating a sum collision.
        Adding x creates new sums: {2x} U {x+y for y in state}.
        Collision if any of these are in 'sums'.
        """
        state_set = set(state)
        for x in range(1, N + 1):
            if x in state_set:
                continue
            
            # Check if x can be added
            # New sums: 2*x, and x+y for all y in state
            if 2 * x in sums:
                continue
            
            can_add = True
            for y in state:
                if x + y in sums:
                    can_add = False
                    break
            
            if can_add:
                return False
        return True

    def greedy_construct(order):
        """Build a Sidon set greedily using the given order of candidates."""
        state = []
        sums = set()
        state_set = set()
        
        for x in order:
            # Check if x can be added
            if 2 * x in sums:
                continue
            
            # Check x + y for all y in state
            valid = True
            for y in state:
                if x + y in sums:
                    valid = False
                    break
            
            if valid:
                state.append(x)
                state_set.add(x)
                sums.add(2 * x)
                for y in state[:-1]: # y is the last one added, x is current. Wait, state[:-1] is all previous.
                    sums.add(x + y)
        
        return state, sums, state_set

    def local_minimize(state, sums, state_set):
        """Try to remove elements to minimize cardinality while preserving maximality."""
        # We want to remove elements.
        # If we remove an element, the set is still Sidon (subset of Sidon is Sidon).
        # But we must ensure the new set is still MAXIMAL.
        # If the new set is not maximal, it means some element x could be added.
        # If x is the element we just removed, that's fine (it wasn't in the set).
        # But if x is some OTHER element, then the set wasn't maximal before either?
        # No. Removing an element can make a set NON-maximal?
        # Actually, removing an element from a maximal set usually makes it non-maximal 
        # because the removed element itself can now be re-added (since its sums are gone).
        # Wait. If S is maximal, and we remove x to get S' = S \ {x}.
        # Is S' maximal?
        # S' is Sidon.
        # Can we add x to S'? Yes, because x was in S, and S was Sidon, so x + y for y in S' 
        # and 2x were all distinct and distinct from sums of S'. 
        # So x can be added to S'. Thus S' is NOT maximal.
        
        # Therefore, simply removing elements from a maximal set will ALWAYS make it non-maximal 
        # (because the removed element becomes addable).
        
        # So, "Local Removal" in the sense of "remove and keep" is impossible if we require 
        # the result to be maximal.
        
        # UNLESS: We remove an element x, and then ADD another element y?
        # Or, we look for a subset S' of S such that S' is maximal.
        # If S is maximal, any proper subset S' is NOT maximal (because elements in S\S' can be added? 
        # Not necessarily. If S' is a subset of S, and S is Sidon, S' is Sidon.
        # If x in S\S', can x be added to S'?
        # The sums of S' are a subset of sums of S.
        # The new sums introduced by x are {2x} U {x+y for y in S'}.
        # Since S is Sidon, 2x and x+y (for y in S) are all distinct and not in sums(S).
        # Since sums(S') is a subset of sums(S), these new sums are also not in sums(S').
        # So YES, any element x in S\S' can be added to S'.
        # Therefore, ANY proper subset of a maximal Sidon set is NOT maximal.
        
        # Conclusion: A maximal Sidon set is "minimal" in the sense that no proper subset is maximal.
        # So, if we have a maximal Sidon set, we CANNOT remove any element and remain maximal.
        
        # This implies that the "minimize cardinality" part of the task is about finding a 
        # MAXIMAL set with the smallest possible size, not by shrinking an existing one, 
        # but by finding a different maximal set that is smaller.
        
        # So the strategy must be:
        # 1. Generate a maximal Sidon set (e.g., via greedy).
        # 2. Try to find a smaller maximal Sidon set.
        
        # How to find a smaller one?
        # We can try to "swap" elements or use a different construction order.
        # Or, we can start with a smaller set and try to make it maximal?
        # But making a set maximal usually requires adding elements, increasing size.
        
        # Wait, the problem is: "Construct a Sidon subset... that cannot be extended... and minimize its cardinality."
        # This is the "Smallest Maximal Sidon Set" problem.
        
        # Key Insight: 
        # Any maximal Sidon set S has the property that no proper subset is maximal.
        # So we can't just prune a maximal set. We must find a different maximal set.
        
        # Strategy:
        # 1. Greedy construction with random order tends to produce larger sets.
        # 2. To get smaller sets, we might need to be "selective" in the greedy phase?
        #    But greedy always adds if possible. To get a smaller maximal set, we might need to 
        #    SKIP some addable elements? 
        #    If we skip an addable element x, the set might still become maximal later?
        #    If we skip x, and later the set becomes maximal without x, then x is not in the set.
        #    But if x is addable at the end, the set is not maximal.
        #    So, if we skip x, we MUST ensure that x is NOT addable at the end.
        #    This is a constraint.
        
        # Alternative Strategy: 
        # Use a heuristic that favors "dense" sums to block more candidates?
        # Or, use local search on the set of elements to find a smaller maximal set.
        
        # Let's try a different approach:
        # 1. Start with a random set of size k (small, e.g., 5-10).
        # 2. Check if it's Sidon. If not, fix it or discard.
        # 3. Check if it's maximal. If not, add elements to make it maximal.
        #    This will increase size.
        # 4. The goal is to find a maximal set with small size.
        
        # Actually, there is a known result: The smallest maximal Sidon sets are often constructed 
        # by specific patterns.
        
        # Let's try a "Reverse Greedy" or "Backtracking" approach?
        # Given time limits, let's stick to:
        # 1. Multiple Greedy restarts with random orders.
        # 2. Record the smallest maximal set found.
        # 3. Additionally, try to "perturb" a found maximal set:
        #    - Remove a random element x.
        #    - The set is no longer maximal (x can be added).
        #    - To make it maximal again, we must add some OTHER element y that blocks x?
        #    - No, adding y doesn't block x unless y creates a sum collision with x.
        #    - If we add y such that x+y is in sums, then x is blocked.
        #    - So: Remove x, add y. If the new set is Sidon and Maximal, we have a new candidate.
        #    - Size remains the same. This is a "swap".
        #    - Can we reduce size?
        #    - If we remove x, and the set becomes maximal WITHOUT adding anything?
        #    - As proved above, NO. A proper subset of a maximal set is never maximal.
        
        # So, the size of a maximal Sidon set is fixed by its structure. 
        # We cannot shrink a maximal set. We can only find a different maximal set that is smaller.
        
        # Therefore, the search is purely about finding a maximal set with minimal cardinality.
        # Greedy with random order is a good heuristic. 
        # To improve, we can bias the random order to prefer elements that create "crowded" sums?
        # Or simply run many restarts.
        
        # Let's implement:
        # 1. Loop for time.
        # 2. Generate random permutation of [1..N].
        # 3. Greedy construct.
        # 4. Verify maximality (greedy should produce maximal sets, but let's check).
        # 5. If maximal and size < best_size, update best.
        
        # Optimization:
        # Instead of full permutation, just random sample? No, we need to consider all elements to ensure maximality.
        # Greedy on a permutation of ALL elements ensures maximality: 
        # At the end, for any x not in set, it was considered at some point.
        # If it was addable then, it would have been added.
        # If it wasn't addable then, is it addable now?
        # Adding more elements only adds more sums, making it HARDER to add new elements.
        # So if x was not addable at step k, it is not addable at step N.
        # Thus, greedy on a full permutation ALWAYS produces a maximal Sidon set.
        
        # So the algorithm is simply:
        # Repeat:
        #   perm = random.shuffle(range(1, N+1))
        #   S = greedy(perm)
        #   if len(S) < best_size:
        #       best = S
        #       best_size = len(S)
        
        # This is fast. Greedy is O(N * |S|). |S| is small (~sqrt(N) ~ 16). N=256.
        # 256 * 16 ~ 4000 ops per restart.
        # We can do thousands of restarts in 55 seconds.
        
        pass # Placeholder for logic inside construct

    best_state = None
    best_size = float('inf')
    
    # We can also try sorted orders or other heuristics
    # But random is a good baseline.
    
    # To make it better, we can try "Greedy with Bias":
    # Sometimes starting with small numbers creates many sums.
    # Sometimes starting with large numbers?
    # Let's just use random permutations.
    
    while time.time() < deadline:
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        state, sums, state_set = greedy_construct(perm)
        
        # Greedy on full permutation is always maximal
        # But let's double check just in case of implementation error
        if not is_maximal(state, sums):
            # Should not happen, but if it does, try to fix?
            # Or just skip.
            continue
            
        if len(state) < best_size:
            best_size = len(state)
            best_state = list(state)
            
            # If we find a very small one, we can break?
            # Theoretical lower bound for maximal Sidon set in [1, N]?
            # Max Sidon set size is ~sqrt(N).
            # Minimal maximal size is likely close to sqrt(N) or slightly less.
            # For N=256, sqrt(256)=16.
            # If we find 10 or 11, that's good.
            if best_size <= 10:
                break

    if best_state is None:
        # Fallback: simple greedy ascending
        perm = list(range(1, N + 1))
        best_state, _, _ = greedy_construct(perm)
        
    return best_state

# EVOLVE-BLOCK-END

