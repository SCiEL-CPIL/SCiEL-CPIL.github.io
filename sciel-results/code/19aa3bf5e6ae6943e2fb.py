import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s buffer
    
    def is_sidon_and_covers(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def search(max_val, target_size):
        # Backtracking search for a Sidon set of size target_size
        # containing 0, with elements <= max_val, covering 1..30
        S = [0]
        used_diffs = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if len(S) == target_size:
                if is_sidon_and_covers(S):
                    return S[:]
                return None
            
            # Pruning: check if we can cover remaining diffs
            missing = [d for d in range(1, 31) if d not in used_diffs]
            slots_left = target_size - len(S)
            # Max new diffs from adding 'slots_left' elements to 'len(S)' existing
            # Each new element x adds |S| diffs.
            # But diffs must be distinct.
            # Rough bound: len(missing) <= slots_left * len(S) + slots_left*(slots_left-1)/2 ?
            # Actually, just check if len(missing) is too large.
            
            # Heuristic: try candidates in increasing order to find small diameter solutions
            # But we also need to ensure we CAN reach max_val? No, we are bounded by max_val.
            # We want to find A solution. To minimize max, we should search with increasing max_val.
            
            last = S[-1]
            # Candidates must be > last
            # Also, to cover large diffs, we might need large elements.
            
            # Optimized candidate generation:
            # We need to cover missing diffs.
            # If d is missing, either d is a diff between existing elements (impossible by def),
            # or d = x - s_i for some new x, or d = s_j - x (impossible if x > last),
            # or d = x - y where x, y are new.
            
            # Simple strategy: iterate x from last+1 to max_val
            # This might be slow if max_val is large.
            # But for n=9, max_val is likely < 100.
            
            for x in range(last + 1, max_val + 1):
                if time.time() > deadline:
                    return None
                
                # Check if x can be added
                valid = True
                new_diffs = []
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Tentatively add
                S.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Backtrack
                S.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return None
        
        return backtrack(1)

    # Try sizes 9, 10, 11...
    # Start with small max_val and increase?
    # Or just try to find the best one.
    
    best_solution = None
    best_max = float('inf')
    
    # We know n must be at least 9.
    for n in range(9, 13):
        # Try to find a solution with small max_val
        # Start max_val from a lower bound.
        # Lower bound for max_val:
        # If 0 in S, and we need diff 30, max element must be at least 30?
        # Not necessarily. 10 and 40 gives 30.
        # But generally, small sets have small diameters.
        
        # Let's try max_val from 40 to 200 in steps.
        # Or just run search with a reasonably high max_val and hope it finds a small one?
        # No, backtracking with increasing x will find the smallest x sequence first?
        # Yes, if we iterate x from last+1 upwards, the first solution found for a fixed n
        # will have the lexicographically smallest sequence, which tends to have small max.
        
        # Let's try n=9 first.
        # If found, check its max.
        # Then try to see if we can do better?
        # Given time, finding one good one is key.
        
        # Let's set a high max_val for the search to allow flexibility,
        # but the search order (ascending x) will naturally find small diameter ones first.
        
        sol = search(200, n)
        if sol:
            curr_max = max(sol)
            if curr_max < best_max:
                best_max = curr_max
                best_solution = sol
                # If we found a size 9 solution, it's likely optimal or near optimal.
                # We can break if we are confident.
                if n == 9:
                    # Try to verify if we can do even smaller?
                    # The search already tried smallest x first.
                    # So this is the min-diameter size-9 set?
                    # Not strictly, because different structures might have smaller max.
                    # But for now, this is a strong candidate.
                    pass
        
        if time.time() > deadline:
            break

    if best_solution:
        return best_solution
    
    # Fallback: Greedy or simple construction if search failed
    # The seed program is a fallback, but we want to be better.
    # If search fails, return a valid set from a known construction or the seed logic.
    
    # Simple greedy fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

BET: {mechanism: "Backtracking search with pruning for small Sidon sets (n=9-11) finds near-optimal diameter much faster than greedy, exploiting the small search space for n=9", q: 0.9, delta: [0.3, 0.8]}
