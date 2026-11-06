"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256
DEADLINE = 55.0  # Safety margin for 60s limit

# EVOLVE-BLOCK-START

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, full_set):
    """Check if selected is inclusion-maximal within full_set."""
    if not _is_sidon(selected):
        return False
    
    sums = _compute_sums(selected)
    selected_set = set(selected)
    
    for x in full_set:
        if x in selected_set:
            continue
        # Check if adding x keeps it Sidon
        # New sums to check: 2*x and x+a for all a in selected
        if 2 * x in sums:
            continue
        valid = True
        for a in selected:
            if x + a in sums:
                valid = False
                break
        if valid:
            return False
    return True

def _count_conflicts(x, selected, sums):
    """Count how many existing sums are blocked by x."""
    # A sum s is blocked if s == x + a for some a in selected, or s == 2x
    # Actually, we want to know how many *potential* additions x would prevent.
    # Or simpler: how many elements in [1..N] would x conflict with?
    # Let's use a heuristic: count how many elements y in [1..N] (y not in selected)
    # would form a conflict with x given the current sums?
    # This is expensive. 
    # Alternative heuristic: remove x if it allows the most other elements to be added?
    # Let's stick to: remove x if it is "redundant" for maximality.
    # An element x is redundant if removing it still leaves the set maximal?
    # No, removing an element makes the set *less* maximal (easier to add things).
    # We want to remove elements that are "forcing" the set to be large?
    # Actually, we want to find a small set that is maximal.
    # A small maximal set is one where every element is "essential" for blocking some other element?
    # No, maximal means you can't add ANY element.
    # So every element in the set is not necessarily essential for maximality, 
    # but the *combination* of them blocks all others.
    
    # Strategy: Randomly remove an element, then try to "repair" maximality by adding others?
    # Or: Start with a valid maximal set, and try to swap elements for "cheaper" ones?
    
    # Let's use a local search on the set itself.
    # Move: Remove x, Add y (where y was previously blocked).
    # But we want to minimize size. So primarily remove.
    # If we remove x, the set might no longer be maximal (some z can now be added).
    # If we can add z and the size doesn't increase (or we can remove another), we might improve.
    # But adding increases size. We want to *decrease* size.
    # This implies the current set has "redundant" elements that don't contribute to blocking?
    # Wait. If a set is maximal, can we remove an element and still be maximal?
    # If we remove x, the set S' = S \ {x} is a subset. 
    # If S' is still maximal, then x was redundant. 
    # But if S is maximal, S' is NOT maximal (because x can be added back to S' to form S, which is Sidon).
    # So you can never just remove an element from a maximal set and remain maximal.
    # Therefore, to reduce size, you must REMOVE x and ADD some other elements to block what x was blocking?
    # No, removing x unblocks some elements. To remain maximal, you must block those newly unblocked elements.
    # So: Remove x. Identify elements z that can now be added. 
    # If we can add a single z that blocks all of them, size stays same.
    # If we can add nothing (impossible, x was blocking itself? No).
    # Actually, if we remove x, x itself becomes addable. So we MUST block x.
    # So we need to add some y such that y conflicts with x.
    # If we add y, size +1. Net change 0.
    # To reduce size, we need to remove x and add NOTHING? Impossible.
    # Or remove x, and the new set is maximal without adding? Impossible, x can be added.
    
    # Conclusion: You cannot reduce the size of a maximal Sidon set by simply removing elements.
    # You must perform a swap: Remove x, Add y. If y blocks more "potential" elements than x did, 
    # maybe we can remove *another* element?
    
    # Better strategy: Randomized construction.
    # 1. Pick a random permutation.
    # 2. Greedily build Sidon set.
    # 3. The result is maximal by construction (greedy saturation).
    # 4. Try to "prune" or "optimize" the set? 
    # The seed program does greedy ascending. 
    # The observed improvement (1.23) used "local search removes redundant elements... plus multiple random order restarts".
    # Wait, "removes redundant elements from greedy Sidon sets while preserving maximality".
    # How? 
    # If S is maximal, and we remove x, S\{x} is not maximal.
    # Unless... the definition of maximal in the task allows for non-greedy maximal sets?
    # Yes. A set is maximal if no element can be added.
    # If I have a maximal set S, and I remove x, I get S'. S' is not maximal.
    # But maybe S' can be made maximal by adding FEWER elements than x was "costing"?
    # No, x is 1 element.
    
    # Let's look at the "redundant elements" claim.
    # Perhaps the greedy set contains elements that are not "critical" for blocking specific numbers, 
    # and by swapping them for "more efficient" blockers, we can eventually drop some?
    # 
    # Let's try a different angle: 
    # Start with a small random Sidon set.
    # Try to make it maximal by adding elements.
    # Greedy addition will make it maximal.
    # The size is determined by how many additions were needed.
    # To minimize size, we want to start with a set that blocks as many numbers as possible with few elements.
    # This is a covering problem.
    
    # Strategy:
    # 1. Generate many random small Sidon sets (e.g., size 10-20).
    # 2. For each, greedily complete it to a maximal set.
    # 3. Keep the smallest maximal set found.
    # 4. Use local search: swap elements in the final maximal set to see if we can find a "hole" 
    #    that allows removing an element? (We established you can't just remove).
    #    But maybe a swap allows a subsequent removal?
    
    # Let's implement:
    # 1. Multiple random restarts with random initial seeds.
    # 2. Greedy completion.
    # 3. Local search: Try to swap an element x in the set with an element y not in the set.
    #    If S' = (S \ {x}) U {y} is Sidon, check if it's maximal.
    #    If it is, and len(S') == len(S), it's a neutral move.
    #    If we can find a sequence of moves that leads to a smaller set?
    #    Actually, if S is maximal, and we swap x for y, S' is Sidon. 
    #    Is S' maximal? Not necessarily.
    #    If S' is maximal, we have another maximal set of same size.
    #    How to get smaller?
    #    Maybe some maximal sets are smaller than others.
    #    Random restarts with different random orders often find smaller maximal sets in such combinatorial problems.
    
    def _greedy_complete(initial_set, full_range):
        """Complete initial_set to a maximal Sidon set using greedy addition."""
        selected = list(initial_set)
        if not _is_sidon(selected):
            return None
        sums = _compute_sums(selected)
        selected_set = set(selected)
        
        # Try to add elements from full_range in random order? Or sorted?
        # Random order might find different maximal sets.
        candidates = [x for x in full_range if x not in selected_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if 2 * x in sums:
                continue
            valid = True
            for a in selected:
                if x + a in sums:
                    valid = False
                    break
            if valid:
                selected.append(x)
                selected_set.add(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        # Verify maximality
        if _is_maximal(selected, full_range):
            return selected
        return None

    def _random_initial_set(k):
        """Generate a random Sidon set of size k."""
        if k == 0:
            return []
        # Pick k random numbers and check if Sidon
        # This is slow if k is large, but we start small.
        # Use a greedy random pick instead.
        selected = []
        sums = set()
        available = list(range(1, N + 1))
        random.shuffle(available)
        
        for x in available:
            if len(selected) == k:
                break
            if 2 * x in sums:
                continue
            valid = True
            for a in selected:
                if x + a in sums:
                    valid = False
                    break
            if valid:
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        return selected

    start_time = time.time()
    best_set = None
    best_size = float('inf')
    
    full_range = range(1, N + 1)
    
    # First, try the seed's ascending greedy to have a baseline
    seed_set = []
    sums_seed = set()
    for x in range(1, N + 1):
        added = [x + a for a in seed_set] + [2 * x]
        if len(set(added)) == len(added) and not sums_seed.intersection(added):
            seed_set.append(x)
            sums_seed.update(added)
    if _is_maximal(seed_set, full_range):
        best_set = seed_set
        best_size = len(seed_set)
        
    # Now try random restarts
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1
        # Start with a small random Sidon set
        # Size of initial set: random between 5 and 20
        k = random.randint(5, 20)
        initial = _random_initial_set(k)
        
        # Complete it
        completed = _greedy_complete(initial, full_range)
        
        if completed:
            if len(completed) < best_size:
                best_size = len(completed)
                best_set = completed
                # If we find a very small one, we might want to do local search
                # But for now, just keep updating.
                
        # Also try completing the empty set (pure random greedy)
        if attempts % 2 == 0:
            completed_empty = _greedy_complete([], full_range)
            if completed_empty and len(completed_empty) < best_size:
                best_size = len(completed_empty)
                best_set = completed_empty

    if best_set is None:
        # Fallback to seed
        return seed_set
    return best_set

# EVOLVE-BLOCK-END
