import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(S):
        sums = get_sums(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if adding x preserves Sidon property
            new_sums = set()
            valid = True
            # Check sums involving x
            for a in S:
                s = x + a
                if s in sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            
            s = 2 * x
            if s in sums or s in new_sums:
                continue
                
            # If we get here, x can be added without conflict.
            # Thus S is not maximal.
            return False
        return True

    def greedy_construct(randomize_order=False):
        if randomize_order:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            
        selected = []
        sums = set()
        
        for x in candidates:
            added = [x + a for a in selected] + [2 * x]
            # Check for internal conflicts in 'added'
            if len(set(added)) != len(added):
                continue
            # Check against existing sums
            if sums.intersection(added):
                continue
            
            selected.append(x)
            sums.update(added)
            
        return selected

    best_set = greedy_construct(randomize_order=False)
    if not best_set:
        return [1] # Fallback, though greedy usually finds something
    
    best_size = len(best_set)
    
    # Try multiple random restarts of construction + pruning
    while time.time() < deadline:
        # 1. Construct a maximal Sidon set
        S = greedy_construct(randomize_order=True)
        if not S:
            continue
        
        # The greedy construction ensures S is Sidon.
        # Does it ensure S is maximal?
        # Greedy picks x if it doesn't conflict. 
        # If we skip x, it's because it conflicted with some chosen a or 2x.
        # This implies x cannot be added to the *current* set.
        # However, maximality requires that x cannot be added to S.
        # Since S is the final set, and x was skipped due to conflict with elements in S,
        # adding x to S would indeed cause a conflict.
        # So greedy output is always maximal.
        
        current_size = len(S)
        if current_size >= best_size:
            continue
            
        # 2. Prune to minimize size while maintaining maximality
        # We want to remove elements. 
        # Removing an element y from S makes S' = S \ {y}.
        # S' is Sidon.
        # Is S' maximal?
        # S' is maximal if no element can be added.
        # Note: y itself might be addable to S'.
        # If y is addable to S', then S' is not maximal.
        # Also, other elements might become addable.
        
        # Strategy: Try to remove elements one by one.
        # If removing y results in a non-maximal set, try to "fix" it?
        # Or just reject the removal.
        # A better heuristic: 
        # Iterate through elements. For each element y, try removing it.
        # If the resulting set is maximal, keep it removed.
        # This is a greedy pruning.
        
        S_set = set(S)
        # We need to be careful with order of removal.
        # Randomize removal order?
        
        # Let's try a simple pass:
        # Sort S. Try to remove each element.
        # If removal keeps it maximal, remove it.
        
        # To be more effective, we can randomize the order of checking removals.
        removal_order = list(S)
        random.shuffle(removal_order)
        
        S_current = list(S)
        S_current_set = set(S_current)
        
        for y in removal_order:
            if time.time() > deadline:
                break
            if y not in S_current_set:
                continue
            
            # Try removing y
            S_trial = [x for x in S_current if x != y]
            
            # Check if S_trial is maximal
            if is_maximal(S_trial):
                S_current = S_trial
                S_current_set.discard(y)
                
        if len(S_current) < best_size:
            best_set = S_current
            best_size = len(S_current)
            
            # If we reach a very small size, we might stop early or continue
            # Theoretical lower bound for maximal Sidon set in [1, N] is roughly sqrt(N) or similar?
            # For N=256, sqrt(256)=16. 
            # Maximal sets can be smaller than maximum sets.
            # Let's see if we can get close to 16.
            if best_size <= 18:
                 # Continue searching for a bit more?
                 pass

    return best_set
# EVOLVE-BLOCK-END
