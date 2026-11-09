import itertools
import time

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    # A is sorted
    n = len(A)
    # Check from largest down? No, we want to minimize, so we want to know the max.
    # But for optimization, we can just check if size K is possible.
    # Let's return the exact max.
    # Optimization: If we only care about < 4, we check size 4.
    # But we need the value for scoring.
    # Since n=12 is small, we can check all subsets of size k for k=12..1.
    # But we can prune.
    
    # Actually, just checking size 4 and 5 is enough to distinguish 3 vs 4 vs 5+.
    # If no 4-subset is Sidon, max is 3 (since 3-subsets are almost always Sidon? No, 3-subsets can fail).
    # Wait, if no 4-subset is Sidon, is it possible that no 3-subset is Sidon?
    # A 3-subset $\{a,b,c\}$ fails if $a+b = a+c$ (impossible if distinct), $a+a=b+c$, etc.
    # $2a = b+c$ is the main collision for 3 elements.
    # So if we avoid 3-term APs, 3-subsets are Sidon.
    # So if we have no Sidon 4-subset, we likely have Sidon 3-subsets, so max is 3.
    
    # So the goal is: Find A such that no 4-subset is Sidon.
    
    best_val = 12
    for k in range(12, 0, -1):
        found = False
        for sub in itertools.combinations(A, k):
            if is_sidon(sub):
                found = True
                break
        if found:
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    # Initial candidate
    A = list(range(12))
    current_val = max_sidon_size(A)
    best_A = A
    best_val = current_val
    
    # Local search
    # We want to reduce current_val.
    # If current_val is 4, we try to break all Sidon 4-subsets.
    
    while time.time() < deadline:
        if best_val == 3:
            break # Can't do better than 3 (min is 1? No, 1 is trivial. 2 is trivial. 3 is likely min for 12 elements)
            
        # Get a random Sidon subset of size best_val (or best_val-1 if we want to probe?)
        # Actually, if best_val is 4, we know there is at least one Sidon 4-subset.
        # We pick a random Sidon 4-subset and try to break it.
        
        # Find all Sidon 4-subsets
        sidon_4s = [sub for sub in itertools.combinations(A, 4) if is_sidon(sub)]
        
        if not sidon_4s:
            # Then max is 3
            # Update best
            if best_val > 3:
                best_val = 3
                best_A = A
            break
        
        # Pick a random one
        import random
        target = random.choice(sidon_4s)
        
        # Try to modify A to break 'target'
        # We can swap an element in 'target' with an element outside 'target'
        # Or swap any element in A with another.
        
        # Strategy: Pick an element x in target. Pick a new value y in [0, 128] \ A.
        # Replace x with y.
        # Check if new A has smaller max_sidon_size.
        
        # To speed up, we can just check if the specific 'target' is broken.
        # But we might create new Sidon subsets elsewhere.
        
        # Let's try random swaps.
        improved = False
        for _ in range(100): # try 100 swaps per iteration
            i = random.randint(0, 11)
            # Pick a new value not in A
            # To keep it simple, pick from a pool or random in [0, 128]
            # Better: pick from a set of "good" values?
            # Let's just pick random in [0, 128]
            
            new_val = random.randint(0, 128)
            if new_val in A:
                continue
                
            # Swap A[i] with new_val
            old_val = A[i]
            A[i] = new_val
            A.sort()
            
            # Evaluate new A
            # We can shortcut: if the specific 'target' is no longer Sidon, it's a step forward.
            # But we need to ensure we didn't increase the max size.
            # Since we are minimizing, if max size stays 4, it's a tie.
            # If it becomes 5, it's worse.
            # If it becomes 3, it's better.
            
            # Fast check:
            # 1. Is there any Sidon 5-subset? (Unlikely if we were at 4, but possible)
            # 2. Are there any Sidon 4-subsets?
            
            # Let's just compute max_sidon_size for the new A.
            # It's fast enough.
            new_val_size = max_sidon_size(A)
            
            if new_val_size < best_val:
                best_val = new_val_size
                best_A = A
                improved = True
                break # restart loop with new best
            elif new_val_size == best_val:
                # Keep the change if it breaks the target?
                # Maybe just keep exploring.
                pass
            else:
                # Revert
                A[i] = old_val
                A.sort()
                
        if improved:
            continue
            
        # If no improvement found in this batch, maybe try a different strategy?
        # For now, just keep going.
        
    return best_A
