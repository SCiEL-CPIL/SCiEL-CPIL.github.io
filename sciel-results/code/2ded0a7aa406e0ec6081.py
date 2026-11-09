"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    deadline = time.time() + 85.0 # Safety margin
    
    def is_valid_full(S):
        """Check if set S is fully valid."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j, m):
                    if S_list[i] < S_list[j] and S_list[j] <= S_list[k]:
                        if (S_list[j] + S_list[k]) % S_list[i] == 0:
                            return False
        return True

    def is_valid_addition_fast(S, x):
        """Fast check if adding x to S is valid using precomputed sums."""
        S_list = sorted(S)
        m = len(S_list)
        
        # Precompute all valid sums b+c where b,c in S, b<=c
        # For each potential a, check if any sum is divisible by a
        
        # Case 1: x is smallest (a=x)
        # Check if x divides any b+c where b,c in S and x < b <= c
        for i in range(m):
            if S_list[i] > x:
                for j in range(i, m):
                    if (S_list[i] + S_list[j]) % x == 0:
                        return False
        
        # Case 2: x is middle (b=x)
        # Check if a divides x+c for any a,c in S with a < x <= c
        for i in range(m):
            if S_list[i] < x:
                for j in range(m):
                    if S_list[j] >= x and (x + S_list[j]) % S_list[i] == 0:
                        return False
        
        # Case 3: x is largest (c=x)
        # Check if a divides b+x for any a,b in S with a < b <= x
        for i in range(m):
            for j in range(m):
                if S_list[i] < S_list[j] <= x:
                    if (S_list[j] + x) % S_list[i] == 0:
                        return False
        
        return True

    best_set = set()
    best_size = 0
    
    # Try multiple starting thresholds
    for start_threshold in [171, 175, 180, 185, 190, 192]:
        if time.time() > deadline - 10:
            break
            
        current_set = set(range(start_threshold, n + 1))
        
        # Verify initial set is valid
        if not is_valid_full(current_set):
            continue
            
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        # Candidate pool: numbers not in current set, try smaller first (more likely to be addable)
        candidates = list(range(1, start_threshold))
        random.shuffle(candidates)
        
        additions_made = 0
        attempts = 0
        stuck_count = 0
        
        while time.time() < deadline - 5:
            if not candidates:
                break
            
            x = candidates.pop()
            
            if is_valid_addition_fast(current_set, x):
                current_set.add(x)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                additions_made += 1
                stuck_count = 0
            else:
                attempts += 1
                stuck_count += 1
                if stuck_count > 200:
                    # Try reshuffling remaining candidates
                    candidates = list(set(range(1, n+1)) - current_set)
                    random.shuffle(candidates)
                    stuck_count = 0
                    if not candidates:
                        break
        
        # Try some random restarts with different orderings if time permits
        if time.time() < deadline - 15:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                current_set2 = set(range(start_threshold, n + 1))
                candidates2 = list(range(1, start_threshold))
                random.shuffle(candidates2)
                
                for x in candidates2:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_addition_fast(current_set2, x):
                        current_set2.add(x)
                
                if len(current_set2) > best_size:
                    best_set = set(current_set2)
                    best_size = len(current_set2)
    
    # Final verification
    if best_set and not is_valid_full(best_set):
        # Fallback to pure tail
        best_set = set(range(192, 257))
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
