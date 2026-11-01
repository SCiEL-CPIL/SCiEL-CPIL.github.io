import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321"

    # Phase 1: Greedy Construction
    # Generate all permutations
    all_perms = list(permutations(range(1, n + 1)))
    # Map prefix to list of indices of perms starting with that prefix
    # We need prefixes of length 1 to n-1
    
    # To save memory and time, we only store what's necessary for the greedy step.
    # We need to find an uncovered perm P such that P[0:k] == current_tail[-k:]
    
    # Precompute:
    # prefix_map[k][prefix] = list of perm indices
    prefix_map = [{} for _ in range(n)]
    for i, p in enumerate(all_perms):
        for k in range(1, n):
            pref = p[:k]
            if pref not in prefix_map[k]:
                prefix_map[k][pref] = []
            prefix_map[k][pref].append(i)
    
    covered = [False] * len(all_perms)
    
    # Start with a random permutation
    start_idx = random.randint(0, len(all_perms) - 1)
    current_perm = all_perms[start_idx]
    word = list(current_perm)
    covered[start_idx] = True
    remaining = len(all_perms) - 1
    
    # Greedy loop
    while remaining > 0:
        if time.time() > deadline - 30:
            break
            
        # Try to find the next permutation
        # Check k from n-1 down to 1
        found = False
        for k in range(n - 1, 0, -1):
            if len(word) < k:
                continue
            tail = tuple(word[-k:])
            candidates = prefix_map[k].get(tail, [])
            
            # Find an uncovered candidate
            for idx in candidates:
                if not covered[idx]:
                    # Append the rest of this permutation
                    perm = all_perms[idx]
                    # The overlap is k, so we append perm[k:]
                    word.extend(perm[k:])
                    covered[idx] = True
                    remaining -= 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            # This should not happen if the graph is connected enough,
            # but if it does, we might be stuck.
            # In a superpermutation construction, we should always find a path.
            # If we are stuck, it means we entered a dead-end cycle.
            # To fix this, we would need backtracking or a different start.
            # For now, break to avoid infinite loop.
            break
            
    # If we didn't cover all, we have a problem. 
    # However, the greedy method on the permutation graph (de Bruijn-like) 
    # usually works if we allow smaller overlaps.
    # Actually, the greedy method described above ALWAYS finds a path 
    # because we can always fall back to k=1? 
    # Wait, if k=1, tail is (x,). Candidates are all perms starting with x.
    # There are (n-1)! such perms.
    # As long as there is an uncovered one, we can append it.
    # So it will always finish.

    # Phase 2: Local Search to Shorten
    # Convert to list of ints for efficiency
    arr = word
    L = len(arr)
    
    # Build coverage counts
    # perm_count[perm] = count
    perm_count = {}
    for i in range(L - n + 1):
        perm = tuple(arr[i:i + n])
        perm_count[perm] = perm_count.get(perm, 0) + 1
        
    # Helper to check if removal at i is safe
    # Affected windows: j in [i-n+1, i]
    def can_remove(i):
        for j in range(max(0, i - n + 1), min(i, L - n) + 1):
            perm = tuple(arr[j:j + n])
            if perm_count.get(perm, 0) <= 1:
                return False
        return True

    # Iterative removal
    # We scan and remove.
    # Since removal changes indices, we scan from left to right, 
    # but if we remove, we stay at the same index (to check the new element at that pos).
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        i = 0
        while i < len(arr) - n + 1: # Only need to check positions that are part of windows?
            # Actually, removing a char at i affects windows [i-n+1, i].
            # We can try removing any char.
            if can_remove(i):
                arr.pop(i)
                L -= 1
                changed = True
                # Update perm_count for affected windows?
                # For simplicity and correctness, rebuild perm_count periodically or just check carefully.
                # Rebuilding is O(L*n). L ~ 6000. n=7. 42000 ops.
                # If we do this every removal, and we have ~1000 removals, it's 42M ops. OK.
                perm_count = {}
                for j in range(len(arr) - n + 1):
                    perm = tuple(arr[j:j + n])
                    perm_count[perm] = perm_count.get(perm, 0) + 1
                # Do not increment i, check same position again
            else:
                i += 1
                
    return ''.join(str(x) for x in arr)
