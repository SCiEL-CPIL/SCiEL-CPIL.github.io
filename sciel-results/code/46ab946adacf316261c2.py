import time
import random

N = 256

def _is_valid_sidon(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    if not _is_valid_sidon(S):
        return False
    S_set = set(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x creates a collision
        # New sums involving x: x + y for y in S, and x + x
        # Collision if (x + y) in sums OR (x + x) in sums
        # Note: if x+y == x+x implies y=x (impossible as x not in S)
        # if x+y == u+v (existing)
        # if x+x == u+v (existing)
        
        # Optimization: check against existing sums
        conflict = False
        # Check x+x
        if 2 * x in sums:
            conflict = True
        else:
            for y in S:
                if x + y in sums:
                    conflict = True
                    break
        if not conflict:
            return False
    return True

def _make_maximal(S, N):
    # Ensure S is Sidon first
    if not _is_valid_sidon(S):
        # Remove conflicts? Or just restart?
        # For this specific fallback, we assume input is Sidon.
        pass
        
    S = list(S)
    S_set = set(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
            
    # Greedily add unblocked elements?
    # Actually, to make it maximal, we MUST add every unblocked element?
    # No, adding one element might block others.
    # But to be maximal, NO element can be added.
    # So we must add elements until none can be added.
    # This will result in a maximal set.
    
    changed = True
    while changed:
        changed = False
        # Recalculate sums? Or update incrementally?
        # Recalculating is O(k^2). k is small.
        S_set = set(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
                
        for x in range(1, N + 1):
            if x in S_set:
                continue
            conflict = False
            if 2 * x in sums:
                conflict = True
            else:
                for y in S:
                    if x + y in sums:
                        conflict = True
                        break
            if not conflict:
                S.append(x)
                S_set.add(x)
                changed = True
                # We can break and restart loop to recalculate sums
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # Candidate 1: Powers of 2
    S_powers = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Candidate 2: Randomized small sets
    best_set = None
    best_size = float('inf')
    
    # Try powers of 2
    if _is_maximal(S_powers, N):
        return S_powers
    else:
        # Make it maximal by adding necessary elements
        S_completed = _make_maximal(S_powers, N)
        if len(S_completed) < best_size:
            best_size = len(S_completed)
            best_set = S_completed
    
    # Try other bases? 3, 5, 7?
    for base in [3, 5, 7, 10]:
        if time.time() > deadline:
            break
        S_base = []
        x = 1
        while x <= N:
            S_base.append(x)
            x *= base
        # Ensure Sidon (powers of base > 2 are Sidon? 
        # 1, 3, 9, 27... sums are distinct?
        # 1+9=10, 3+3=6. Distinct.
        # 1+27=28, 3+9=12. Distinct.
        # Generally yes for base >= 3.
        if _is_valid_sidon(S_base):
            if _is_maximal(S_base, N):
                if len(S_base) < best_size:
                    best_size = len(S_base)
                    best_set = S_base
            else:
                S_completed = _make_maximal(S_base, N)
                if len(S_completed) < best_size:
                    best_size = len(S_completed)
                    best_set = S_completed
    
    # Try random sets
    # Size k around 10-15
    for _ in range(50):
        if time.time() > deadline:
            break
        k = random.randint(8, 12)
        S_rand = random.sample(range(1, N + 1), k)
        if _is_valid_sidon(S_rand):
            # Check if maximal
            if _is_maximal(S_rand, N):
                if len(S_rand) < best_size:
                    best_size = len(S_rand)
                    best_set = S_rand
            else:
                # Don't complete random sets if they are too large already?
                # Completion might blow up size.
                # Only complete if current size + estimated additions < best_size
                # Estimate: number of holes.
                pass

    if best_set is None:
        # Fallback to seed logic
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
        
    return best_set
