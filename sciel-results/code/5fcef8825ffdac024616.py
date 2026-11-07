import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (90s limit, leave buffer)
    
    # Baseline: [63, 128], size 66, verified valid
    baseline = list(range(63, 129))
    best_set = set(baseline)
    best_size = 66
    
    def is_valid_set(S):
        """Check if set S has no violating triple."""
        S_list = sorted(S)
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if a + b in S_set and a + c in S_set and b + c in S_set:
                        return False
        return True
    
    def count_violations(S):
        """Count number of violating triples in S."""
        S_list = sorted(S)
        n = len(S_list)
        S_set = set(S_list)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if a + b in S_set and a + c in S_set and b + c in S_set:
                        count += 1
        return count
    
    def violating_triples(S):
        """Return list of violating triples."""
        S_list = sorted(S)
        n = len(S_list)
        S_set = set(S_list)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if a + b in S_set and a + c in S_set and b + c in S_set:
                        triples.append((a, b, c))
        return triples
    
    # Try to add elements from 1..62 to the baseline
    # For each candidate element, check if adding it keeps the set valid
    # If not, try swapping: add candidate, remove another element
    
    candidates = list(range(62, 0, -1))  # Try larger candidates first (closer to baseline)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in best_set:
            continue
        
        # Try adding cand
        new_set = best_set | {cand}
        viol = violating_triples(new_set)
        
        if not viol:
            # Valid! Update best
            best_set = new_set
            best_size = len(best_set)
            continue
        
        # If there are violations, try to fix by removing one element
        # Find elements that appear in violations
        viol_elements = set()
        for t in viol:
            viol_elements.update(t)
        
        for rem in viol_elements:
            if rem == cand:
                continue  # Don't remove the candidate we just added
            trial_set = new_set - {rem}
            if len(trial_set) > best_size:
                # Check if valid
                if is_valid_set(trial_set):
                    best_set = trial_set
                    best_size = len(best_set)
                    break
            elif len(trial_set) == best_size:
                # Same size, but maybe enables future additions
                # Only accept if it seems "better" in some heuristic
                pass
    
    # Local search: try swapping elements
    if time.time() < deadline - 5:
        # Try to improve by removing one element and adding a different one
        for remove in list(best_set):
            if time.time() > deadline:
                break
            for add in range(1, 129):
                if time.time() > deadline:
                    break
                if add in best_set:
                    continue
                trial = (best_set - {remove}) | {add}
                if len(trial) != len(best_set):
                    continue
                if is_valid_set(trial):
                    # Same size, different composition - might help future adds
                    # Only accept if it's "better" (lower min element or something)
                    # Actually, just keep the best
                    pass
    
    # Final validation
    if not is_valid_set(best_set):
        # Fallback to baseline
        return baseline
    
    result = list(best_set)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    return result

def is_valid(S):
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if a + b in S_set and a + c in S_set and b + c in S_set:
                    return False
    return True
